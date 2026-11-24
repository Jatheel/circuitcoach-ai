"""
CircuitCoach AI - CPU speed benchmark across model formats on the same video frames.

For each model file: 10 warm-up frames, then --frames timed frames on CPU at imgsz 640,
repeated --repeats times with the order of formats alternated (median pass reported).
Reports per-stage time from Ultralytics (preprocess / inference / postprocess), wall-clock
mean, p50 and p95 per frame, FPS, file size, and box agreement with the PyTorch model at
conf 0.4. Optionally joins test-split mAP50 from the export reports, so speed and accuracy
cost are in one table.

Every format sees exactly the same frames: up to --unique frames are read once from the
video into memory (long side resized to --max-side) and cycled.

Fair input size: Ultralytics predicts .pt models with rect=True, so a 16:9 frame becomes a
640x384 tensor, while the exported files have a fixed 640x640 input. By default every format
is run at 640x640 (rect=False) so they do the same work; --rect shows PyTorch's native speed.

Usage (laptop plugged in, power mode "Best performance", other apps closed):
  python scripts/benchmark_speed.py --video clip.mp4 --frames 200 \
    --weights circuitcoach-yolo26n.pt circuitcoach-yolo26n.onnx \
              circuitcoach-yolo26n_openvino_model circuitcoach-yolo26n_int8_openvino_model \
    --accuracy docs/results/export_report.json docs/results/onnx_report.json \
               docs/results/openvino_report.json \
    --power "plugged in, Best performance" --out docs/results/speed_cpu

.pt / .onnx / .tflite names that are not on disk are downloaded from Hugging Face.
A .tflite file is skipped with a note if TensorFlow / LiteRT cannot load on this machine.
"""

import argparse
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_detect import resolve_weights  # noqa: E402

AGREE_FRAMES = 25
AGREE_CONF = 0.4
RECT = False  # set from --rect in main()


def iou(a, b):
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def versions():
    out = {"python": platform.python_version()}
    for pkg in ("ultralytics", "torch", "opencv-python", "onnxruntime", "openvino"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def size_mb(p):
    p = Path(p)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 2)


def read_frames(video, n, max_side):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"could not open video: {video}")
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    if not frames:
        sys.exit(f"no frames read from {video}")
    return frames


def load_accuracy(paths):
    """file name -> test-split mAP50 / mAP50-95, from export_report / onnx_report /
    openvino_report JSONs."""
    acc = {}

    def walk(d):
        if isinstance(d, dict):
            if "file" in d and "mAP50" in d:
                acc[d["file"]] = {"mAP50": d["mAP50"], "mAP50_95": d.get("mAP50_95")}
            for v in d.values():
                walk(v)

    for p in paths or []:
        d = json.loads(Path(p).read_text())
        walk(d)
        if "file" in d and isinstance(d.get("test"), dict) and "onnx" in d["test"]:  # onnx_report
            acc[d["file"]] = {"mAP50": d["test"]["onnx"]["mAP50"],
                              "mAP50_95": d["test"]["onnx"]["mAP50_95"]}
    return acc


def label(path):
    n = Path(path).name
    if n.endswith("_int8_openvino_model"):
        return "OpenVINO INT8"
    if n.endswith("_openvino_model"):
        return "OpenVINO FP32"
    if n.endswith("_w8a32.tflite"):
        return "LiteRT w8a32"
    return {".pt": "PyTorch", ".onnx": "ONNX Runtime", ".tflite": "LiteRT FP32"}.get(
        Path(n).suffix, n)


def bench(model, frames, warmup, n, imgsz):
    for i in range(warmup):
        model.predict(frames[i % len(frames)], imgsz=imgsz, conf=AGREE_CONF, device="cpu",
                      rect=RECT, verbose=False)
    wall, pre, inf, post = [], [], [], []
    for i in range(n):
        t = time.perf_counter()
        r = model.predict(frames[i % len(frames)], imgsz=imgsz, conf=AGREE_CONF, device="cpu",
                          rect=RECT, verbose=False)[0]
        wall.append((time.perf_counter() - t) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    wall = np.array(wall)
    return {"preprocess_ms": round(float(np.mean(pre)), 2),
            "inference_ms": round(float(np.mean(inf)), 2),
            "postprocess_ms": round(float(np.mean(post)), 2),
            "total_ms_mean": round(float(wall.mean()), 2),
            "total_ms_p50": round(float(np.percentile(wall, 50)), 2),
            "total_ms_p95": round(float(np.percentile(wall, 95)), 2),
            "fps": round(1000.0 / float(wall.mean()), 1)}


def predictions(model, frames, imgsz):
    out = []
    for f in frames[:AGREE_FRAMES]:
        b = model.predict(f, imgsz=imgsz, conf=AGREE_CONF, device="cpu", rect=RECT,
                          verbose=False)[0].boxes
        out.append((b.xyxy.cpu().numpy(), b.cls.cpu().numpy()))
    return out


def agreement(ref, other):
    ious, n_ref, n_oth = [], 0, 0
    for (A, ca), (B, cb) in zip(ref, other):
        n_ref += len(A); n_oth += len(B)
        for i in range(len(A)):
            same = np.where(cb == ca[i])[0]
            if len(same):
                ious.append(float(iou(A[i:i + 1], B[same])[0].max()))
    return {"boxes_pytorch": n_ref, "boxes": n_oth, "matched_same_class": len(ious),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=200, help="timed frames per format")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed passes per format, order alternated; median pass is reported")
    ap.add_argument("--unique", type=int, default=100, help="distinct frames kept in memory")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rect", action="store_true",
                    help="let .pt use a rectangular input (default: 640x640 square for all)")
    ap.add_argument("--accuracy", nargs="*", default=[], help="export report JSONs (for mAP50)")
    ap.add_argument("--power", default="not recorded", help='e.g. "plugged in, Best performance"')
    ap.add_argument("--out", type=Path, default=Path("docs/results/speed_cpu"))
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    global RECT
    RECT = args.rect

    frames = read_frames(args.video, args.unique, args.max_side)
    h, w = frames[0].shape[:2]
    print(f"[frames] {len(frames)} distinct frames ({w}x{h}) from {args.video}")
    acc = load_accuracy(args.accuracy)

    # 1) load every model once; check it runs and how its boxes compare with PyTorch
    rows, models, ref = [], {}, None
    for wpath in args.weights:
        name = Path(wpath).name
        try:
            path = resolve_weights(wpath)
            model = YOLO(path, task="detect")
            model.predict(frames[0], imgsz=args.imgsz, device="cpu", rect=RECT, verbose=False)
        except (Exception, SystemExit) as e:  # e.g. no TFLite runtime on this machine
            print(f"[skip] {name}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            rows.append({"format": label(name), "file": name, "skipped": str(e)[:300]})
            continue
        row = {"format": label(name), "file": name, "size_MB": size_mb(path)}
        preds = predictions(model, frames, args.imgsz)
        if ref is None and name.endswith(".pt"):
            ref = preds
        if ref is not None:
            row["agreement_vs_pytorch_conf0.4"] = agreement(ref, preds)
        if name in acc:
            row.update({"test_mAP50": acc[name]["mAP50"], "test_mAP50_95": acc[name]["mAP50_95"]})
        rows.append(row)
        models[name] = (model, row)

    # 2) time them in --repeats passes, alternating the order (forward, reverse, ...) so a
    #    laptop that heats up and throttles does not always penalise the same format.
    #    Each format reports the pass with its median mean-time; the spread is kept.
    passes = {n: [] for n in models}
    order = list(models)
    for p in range(args.repeats):
        seq = order if p % 2 == 0 else order[::-1]
        for name in seq:
            print(f"[bench] pass {p + 1}/{args.repeats} {label(name)} ({name}) ...")
            res = bench(models[name][0], frames, args.warmup, args.frames, args.imgsz)
            passes[name].append(res)
            print(f"        {res['total_ms_mean']:.1f} ms/frame, {res['fps']:.1f} FPS")
    for name, (_, row) in models.items():
        runs = sorted(passes[name], key=lambda r: r["total_ms_mean"])
        row.update(runs[len(runs) // 2])
        means = [r["total_ms_mean"] for r in passes[name]]
        row["passes_total_ms_mean"] = means
        row["spread_ms"] = round(max(means) - min(means), 2)

    ok = [r for r in rows if "skipped" not in r]
    base = next((r["total_ms_mean"] for r in ok if r["file"].endswith(".pt")), None)
    for r in ok:
        if base:
            r["speedup_vs_pytorch"] = round(base / r["total_ms_mean"], 2)

    report = {
        "machine": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(),
                    "torch_threads": torch.get_num_threads(), "os": platform.platform(),
                    "power": args.power},
        "versions": versions(),
        "setup": {"video": Path(args.video).name, "frame_size": f"{w}x{h}",
                  "distinct_frames": len(frames), "warmup": args.warmup,
                  "timed_frames": args.frames, "repeats": args.repeats,
                  "order": "alternating forward/reverse, median pass reported",
                  "imgsz": args.imgsz, "device": "cpu",
                  "input": "rect for .pt" if RECT else f"{args.imgsz}x{args.imgsz} letterbox, all formats",
                  "conf": AGREE_CONF},
        "results": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2))

    md = ["| Format | Size | Test mAP50 | Pre + infer + post (ms) | Total ms (p50 / p95) | FPS "
          "| Speed-up | Same boxes as .pt |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "skipped" in r:
            md.append(f"| {r['format']} | – | – | skipped on this machine | – | – | – | – |")
            continue
        a = r.get("agreement_vs_pytorch_conf0.4")
        agree = f"{a['matched_same_class']}/{a['boxes_pytorch']}" if a else "–"
        mp = f"{r['test_mAP50']:.3f}" if "test_mAP50" in r else "–"
        md.append(f"| {r['format']} | {r['size_MB']:.1f} MB | {mp} | "
                  f"{r['preprocess_ms']:.1f} + {r['inference_ms']:.1f} + {r['postprocess_ms']:.1f} | "
                  f"{r['total_ms_p50']:.1f} / {r['total_ms_p95']:.1f} | **{r['fps']:.1f}** | "
                  f"{r.get('speedup_vs_pytorch', 1.0):.2f}× | {agree} |")
    table = "\n".join(md)
    (args.out / "benchmark.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nCPU: {report['machine']['cpu']} | power: {args.power} | "
          f"{args.frames} frames, input {report['setup']['input']}\n")
    print(table)
    print(f"\n[done] {args.out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
