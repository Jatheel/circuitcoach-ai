"""
CircuitCoach AI - export the trained model for phones (LiteRT .tflite) and check
that the exported files keep their accuracy.

Exports two LiteRT files:
  float32 : full-accuracy reference           -> circuitcoach-yolo26n.tflite
  w8a32   : int8 weights, float activations,   -> circuitcoach-yolo26n_w8a32.tflite
            about 1/4 of the size, no calibration data needed

Then evaluates PyTorch, float32 and w8a32 on the same split and picks the file to ship:
  ship w8a32 if it loses <= --max-drop mAP50 points against float32, otherwise float32.
  float32 itself should be within --max-drop points of PyTorch, or the export is broken.

Usage:
  python export_models.py --weights best.pt --data data.yaml --split test --out release
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

STEM = "circuitcoach-yolo26n"


def evaluate(weights, data, split, imgsz, out, tag, device):
    m = YOLO(str(weights), task="detect").val(
        data=data, split=split, imgsz=imgsz, batch=1 if str(weights).endswith(".tflite") else 32,
        device=device, plots=False, verbose=False,
        project=str(out / "val"), name=tag, exist_ok=True,
    )
    return {
        "file": Path(weights).name,
        "size_MB": round(Path(weights).stat().st_size / 1e6, 2),
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/release"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-drop", type=float, default=2.0, help="allowed mAP50 drop, in points")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pt = args.out / f"{STEM}.pt"
    shutil.copy2(args.weights, pt)  # exports are written next to this file

    print("[export] LiteRT float32 ...")
    f32 = Path(YOLO(str(pt)).export(format="litert", imgsz=args.imgsz))
    print("[export] LiteRT w8a32 ...")
    w8 = Path(YOLO(str(pt)).export(format="litert", imgsz=args.imgsz, quantize="w8a32"))

    # keep the .tflite files at the top of the release folder with clear names
    final = {}
    for tag, f in [("float32", f32), ("w8a32", w8)]:
        name = f"{STEM}.tflite" if tag == "float32" else f"{STEM}_w8a32.tflite"
        dst = args.out / name
        if f.resolve() != dst.resolve():
            shutil.copy2(f, dst)
        final[tag] = dst

    print(f"[eval] comparing formats on the {args.split} split (tflite runs on CPU) ...")
    rows = {
        "pytorch": evaluate(pt, args.data, args.split, args.imgsz, args.out, "pytorch", 0),
        "float32": evaluate(final["float32"], args.data, args.split, args.imgsz, args.out,
                            "float32", "cpu"),
        "w8a32": evaluate(final["w8a32"], args.data, args.split, args.imgsz, args.out,
                          "w8a32", "cpu"),
    }
    base = rows["pytorch"]["mAP50"]
    for r in rows.values():
        r["drop_vs_pytorch_points"] = round((base - r["mAP50"]) * 100, 2)

    export_ok = rows["float32"]["drop_vs_pytorch_points"] <= args.max_drop
    w8_drop = round((rows["float32"]["mAP50"] - rows["w8a32"]["mAP50"]) * 100, 2)
    ship = "w8a32" if w8_drop <= args.max_drop else "float32"

    report = {"split": args.split, "max_drop_points": args.max_drop, "formats": rows,
              "float32_export_ok": export_ok, "w8a32_drop_vs_float32_points": w8_drop,
              "ship": ship, "ship_file": final[ship].name}
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'format':8s} {'size_MB':>8s} {'mAP50':>7s} {'mAP50-95':>9s} "
          f"{'drop(pts)':>10s} {'infer_ms':>9s}  device")
    for tag, r in rows.items():
        print(f"{tag:8s} {r['size_MB']:8.2f} {r['mAP50']:7.3f} {r['mAP50_95']:9.3f} "
              f"{r['drop_vs_pytorch_points']:10.2f} {r['inference_ms']:9.1f}  {r['device']}")
    print(f"\nfloat32 export check: {'OK' if export_ok else 'FAILED - drop too large, re-export'}")
    print(f"w8a32 vs float32: -{w8_drop} mAP50 points")
    print(f"Ship: {ship}  ->  {final[ship].name}")
    print(f"Release folder: {args.out}")


if __name__ == "__main__":
    main()
