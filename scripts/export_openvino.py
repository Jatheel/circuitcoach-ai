"""
CircuitCoach AI - export the model to OpenVINO (FP32 and INT8) for fast CPU inference,
and measure how much accuracy INT8 quantisation costs.

Why OpenVINO INT8: the w8a32 LiteRT file only stores int8 *weights* and computes in float,
so it is smaller but not faster on a laptop CPU (see docs/results/export_report.json).
OpenVINO INT8 quantises weights *and* activations (post-training, calibrated on real
ElectroCom61 training images via NNCF), which is what actually speeds up Intel CPUs.

Exports:
  circuitcoach-yolo26n_openvino_model/       FP32 reference
  circuitcoach-yolo26n_int8_openvino_model/  INT8, calibrated with --data

Then evaluates PyTorch, OpenVINO FP32 and OpenVINO INT8 on the same split and writes
openvino_report.json (mAP, drop vs PyTorch in points, size, CPU inference ms).

Usage (Kaggle, dataset attached and cleaned with prepare_dataset.py):
  pip install ultralytics==8.4.215 openvino
  python export_openvino.py --weights best.pt --data ec61_clean/data.yaml --out /kaggle/working/openvino
  python export_openvino.py ... --upload      # + push both folders and the report to Hugging Face
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

STEM = "circuitcoach-yolo26n"
REPO_NAME = "circuitcoach-yolo26n"


def folder_size_mb(p):
    p = Path(p)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 2)


def evaluate(weights, data, split, imgsz, out, tag, device):
    m = YOLO(str(weights), task="detect").val(
        data=data, split=split, imgsz=imgsz, batch=1 if tag != "pytorch" else 32,
        device=device, plots=False, verbose=False,
        project=str(out / "val"), name=tag, exist_ok=True,
    )
    return {
        "file": Path(weights).name,
        "size_MB": folder_size_mb(weights),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "inference_ms": round(float(m.speed["inference"]), 2),
        "device": device,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained best.pt")
    ap.add_argument("--data", required=True, help="data.yaml (INT8 calibration + evaluation)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/openvino"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the training images used for INT8 calibration")
    ap.add_argument("--pt-device", default="0", help="device for the PyTorch eval (0 = GPU)")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pt = args.out / f"{STEM}.pt"
    if Path(args.weights).resolve() != pt.resolve():
        shutil.copy2(args.weights, pt)  # exports are written next to this file

    print("[export] OpenVINO FP32 ...")
    fp32 = Path(YOLO(str(pt)).export(format="openvino", imgsz=args.imgsz))
    print("[export] OpenVINO INT8 (post-training quantisation, calibrated on --data) ...")
    int8 = Path(YOLO(str(pt)).export(format="openvino", imgsz=args.imgsz, quantize=8,
                                     data=args.data, fraction=args.fraction))

    print(f"[eval] comparing formats on the {args.split} split (OpenVINO runs on CPU) ...")
    pt_dev = int(args.pt_device) if args.pt_device.isdigit() else args.pt_device
    rows = {
        "pytorch": evaluate(pt, args.data, args.split, args.imgsz, args.out, "pytorch", pt_dev),
        "openvino_fp32": evaluate(fp32, args.data, args.split, args.imgsz, args.out,
                                  "openvino_fp32", "cpu"),
        "openvino_int8": evaluate(int8, args.data, args.split, args.imgsz, args.out,
                                  "openvino_int8", "cpu"),
    }
    base = rows["pytorch"]["mAP50"]
    for r in rows.values():
        r["drop_vs_pytorch_points"] = round((base - r["mAP50"]) * 100, 2)
    int8_drop = round((rows["openvino_fp32"]["mAP50"] - rows["openvino_int8"]["mAP50"]) * 100, 2)

    report = {"split": args.split, "imgsz": args.imgsz, "calibration_fraction": args.fraction,
              "formats": rows, "int8_drop_vs_fp32_points": int8_drop,
              "note": "inference_ms here is the Kaggle CPU from the validator; laptop CPU "
                      "speed comes from scripts/benchmark_speed.py"}
    (args.out / "openvino_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'format':14s} {'size_MB':>8s} {'mAP50':>7s} {'mAP50-95':>9s} "
          f"{'drop(pts)':>10s} {'infer_ms':>9s}  device")
    for tag, r in rows.items():
        print(f"{tag:14s} {r['size_MB']:8.2f} {r['mAP50']:7.3f} {r['mAP50_95']:9.3f} "
              f"{r['drop_vs_pytorch_points']:10.2f} {r['inference_ms']:9.1f}  {r['device']}")
    print(f"\nINT8 vs FP32: -{int8_drop} mAP50 points")
    print(f"FP32 folder: {fp32}\nINT8 folder: {int8}")
    print(f"Download both folders + {args.out / 'openvino_report.json'}")

    if args.upload:
        from huggingface_hub import HfApi, whoami
        token = os.environ.get("HF_TOKEN")
        if not token:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        repo_id = f"{whoami(token=token)['name']}/{REPO_NAME}"
        api = HfApi(token=token)
        for folder in (fp32, int8):
            api.upload_folder(folder_path=str(folder), path_in_repo=folder.name, repo_id=repo_id,
                              commit_message=f"Add {folder.name}")
        api.upload_file(path_or_fileobj=str(args.out / "openvino_report.json"),
                        path_in_repo="metrics/openvino_report.json", repo_id=repo_id,
                        commit_message="Add OpenVINO FP32/INT8 export check")
        print(f"[done] uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
