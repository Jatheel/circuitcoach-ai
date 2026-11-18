"""
CircuitCoach AI - export the model to ONNX for the in-browser web demo, check it, and
(optionally) upload it to the Hugging Face model repo.

The ONNX file embeds NMS, so its output is (1, 300, 6):
  [x1, y1, x2, y2, confidence, class_id] in 640x640 input pixels.
The browser only has to filter by confidence and undo the letterbox.

Checks:
  1. test-split mAP of the ONNX model vs PyTorch
  2. box-by-box agreement with PyTorch at conf 0.4 on 25 test images

Usage (Kaggle):
  python export_onnx.py --weights best.pt --data data.yaml --out /kaggle/working/onnx           # check only
  python export_onnx.py --weights best.pt --data data.yaml --out /kaggle/working/onnx --upload  # + upload
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

REPO_NAME = "circuitcoach-yolo26n"
ONNX_NAME = "circuitcoach-yolo26n.onnx"


def iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/onnx"))
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pt = args.out / "circuitcoach-yolo26n.pt"
    shutil.copy2(args.weights, pt)

    print("[export] ONNX with embedded NMS ...")
    onnx = Path(YOLO(str(pt)).export(format="onnx", imgsz=640, nms=True, simplify=True))
    if onnx.name != ONNX_NAME:
        onnx = Path(shutil.move(onnx, args.out / ONNX_NAME))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    io = {"inputs": [(i.name, i.shape) for i in sess.get_inputs()],
          "outputs": [(o.name, o.shape) for o in sess.get_outputs()]}
    print("[onnx] inputs:", io["inputs"], "| outputs:", io["outputs"])

    print("[eval] test split: PyTorch vs ONNX ...")
    res = {}
    for tag, w, kw in [("pytorch", pt, {"batch": 32, "device": 0}),
                       ("onnx", onnx, {"batch": 1, "device": "cpu"})]:
        m = YOLO(str(w), task="detect").val(data=args.data, split="test", imgsz=640, plots=False,
                                            verbose=False, project=str(args.out / "val"), name=tag,
                                            exist_ok=True, **kw)
        res[tag] = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                    "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}

    print("[check] box agreement at conf 0.4 on 25 test images ...")
    root = Path(args.data).parent
    imgs = sorted((root / "test/images").glob("*"))[:25]
    m_pt, m_ox = YOLO(str(pt)), YOLO(str(onnx), task="detect")
    ious, n_pt, n_ox = [], 0, 0
    for im in imgs:
        a = m_pt.predict(str(im), conf=0.4, verbose=False)[0].boxes
        b = m_ox.predict(str(im), conf=0.4, verbose=False)[0].boxes
        A, ca = a.xyxy.cpu().numpy(), a.cls.cpu().numpy()
        B, cb = b.xyxy.cpu().numpy(), b.cls.cpu().numpy()
        n_pt += len(A); n_ox += len(B)
        for i in range(len(A)):
            same = np.where(cb == ca[i])[0]
            if len(same):
                ious.append(float(iou(A[i:i + 1], B[same])[0].max()))
    agree = {"boxes_pytorch": n_pt, "boxes_onnx": n_ox, "matched": len(ious),
             "mean_iou": round(float(np.mean(ious)), 4) if ious else None}

    report = {"file": ONNX_NAME, "size_MB": round(onnx.stat().st_size / 1e6, 2), "io": io,
              "test": res, "agreement_conf0.4": agree}
    (args.out / "onnx_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'format':8s} {'mAP50':>7s} {'mAP50-95':>9s} {'P':>6s} {'R':>6s}")
    for tag, r in res.items():
        print(f"{tag:8s} {r['mAP50']:7.3f} {r['mAP50_95']:9.3f} {r['precision']:6.3f} {r['recall']:6.3f}")
    print(f"\nboxes at conf 0.4: pytorch={n_pt} onnx={n_ox} matched={len(ious)} "
          f"mean IoU={agree['mean_iou']}")
    print(f"ONNX file: {onnx} ({report['size_MB']} MB)")

    if args.upload:
        from huggingface_hub import HfApi, whoami
        token = os.environ.get("HF_TOKEN")
        if not token:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        repo_id = f"{whoami(token=token)['name']}/{REPO_NAME}"
        api = HfApi(token=token)
        api.upload_file(path_or_fileobj=str(onnx), path_in_repo=ONNX_NAME, repo_id=repo_id,
                        commit_message="Add ONNX export with embedded NMS for the web demo")
        api.upload_file(path_or_fileobj=str(args.out / "onnx_report.json"),
                        path_in_repo="metrics/onnx_report.json", repo_id=repo_id,
                        commit_message="Add ONNX export check")
        print(f"[done] uploaded: https://huggingface.co/{repo_id}/blob/main/{ONNX_NAME}")


if __name__ == "__main__":
    main()
