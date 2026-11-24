"""
CircuitCoach AI - live video detection and tracking with OpenCV.

Reads frames from a webcam, a video file or a phone camera (DroidCam), runs the model on
each frame, tracks every part with ByteTrack so it keeps a stable ID, and draws boxes,
labels, FPS and a per-class count of unique parts with OpenCV.

Pipeline per frame:
  read (cv2.VideoCapture) -> optional CLAHE (uneven lighting) -> sharpness check
  (variance of the Laplacian, blurry frames are skipped) -> model.track / model.predict
  -> draw (cv2.rectangle, cv2.putText) -> show (cv2.imshow) / save (cv2.VideoWriter)

Usage:
  python scripts/live_detect.py --source 0                                   # laptop webcam
  python scripts/live_detect.py --source 1                                   # DroidCam client
  python scripts/live_detect.py --source http://192.168.1.20:4747/video      # DroidCam over Wi-Fi
  python scripts/live_detect.py --source clip.mp4 --save outputs/demo.mp4    # file -> annotated file
  python scripts/live_detect.py --source 0 --show-sharpness                  # calibrate --blur-thresh
  python scripts/live_detect.py --source 0 --blur-thresh 80 --clahe

Keys: q / Esc quit, p pause, s save a snapshot (PNG).

Weights default to circuitcoach-yolo26n.pt from the Hugging Face model repo. --weights also
accepts a local .pt, .onnx or *_openvino_model folder.
"""

import argparse
import colorsys
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HF_REPO = "JatheelMHD/circuitcoach-yolo26n"
HF_FILES = {"circuitcoach-yolo26n.pt", "circuitcoach-yolo26n.onnx",
            "circuitcoach-yolo26n.tflite", "circuitcoach-yolo26n_w8a32.tflite"}
SHARPNESS_WIDTH = 640  # frames are resized to this width before the Laplacian, so the
                       # threshold does not depend on the camera resolution


# ----------------------------------------------------------------------------- helpers
# Pure functions / small classes: no model or camera needed (see tests/test_live_detect.py).

def sharpness(frame):
    """Variance of the Laplacian on a grey copy resized to SHARPNESS_WIDTH.
    High = sharp edges, low = blurry. Typical sharp desk frames score in the hundreds."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w != SHARPNESS_WIDTH:
        gray = cv2.resize(gray, (SHARPNESS_WIDTH, max(1, round(h * SHARPNESS_WIDTH / w))),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(frame):
    """Contrast Limited Adaptive Histogram Equalisation on the lightness channel only
    (LAB colour space), so uneven lighting is evened out without shifting colours."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def class_color(cls_id):
    """Fixed, well-spread BGR colour per class id (golden-ratio hue steps)."""
    h = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class FpsMeter:
    """Frames per second, smoothed with an exponential moving average."""

    def __init__(self, alpha=0.1):
        self.alpha, self.fps, self.last = alpha, None, None

    def tick(self, now=None):
        now = time.perf_counter() if now is None else now
        if self.last is not None and now > self.last:
            inst = 1.0 / (now - self.last)
            self.fps = inst if self.fps is None else self.alpha * inst + (1 - self.alpha) * self.fps
        self.last = now
        return self.fps


class UniqueCounter:
    """Counts unique tracked parts per class.

    A track ID is counted once, and only after it has been seen in `min_hits` frames, so
    one-frame flickers do not inflate the count. IDs of None (tracker not confirmed yet)
    are ignored. An ID switch, or a part leaving and re-entering the frame, still gives a
    new ID and therefore a second count."""

    def __init__(self, min_hits=5):
        self.min_hits = min_hits
        self.hits = defaultdict(int)
        self.counted = {}  # track id -> class name it was counted as

    def update(self, ids, names):
        for tid, name in zip(ids, names):
            if tid is None:
                continue
            self.hits[tid] += 1
            if tid not in self.counted and self.hits[tid] >= self.min_hits:
                self.counted[tid] = name

    def counts(self):
        out = defaultdict(int)
        for name in self.counted.values():
            out[name] += 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def total(self):
        return len(self.counted)


def draw_label(img, text, x, y, color, scale=0.5, thickness=1):
    """Text on a filled background box with its top-left corner at (x, y)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(max(0, min(x, img.shape[1] - tw - 4)))
    y = int(max(0, min(y, img.shape[0] - th - base - 4)))
    cv2.rectangle(img, (x, y), (x + tw + 4, y + th + base + 4), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(img, text, (x + 2, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0) if lum > 140 else (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(img, dets, names):
    """dets: list of (x1, y1, x2, y2, conf, cls_id, track_id or None). Draws in place."""
    for x1, y1, x2, y2, conf, cls_id, tid in dets:
        color = class_color(int(cls_id))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)
        label = f"{names[int(cls_id)]} {conf:.2f}"
        if tid is not None:
            label = f"#{tid} {label}"
        draw_label(img, label, p1[0], p1[1] - 20 if p1[1] >= 20 else p1[1] + 2, color)
    return img


def draw_panel(img, lines, right=False, max_lines=16):
    """Semi-transparent text panel in the top-left (or top-right) corner."""
    lines = lines[:max_lines]
    if not lines:
        return img
    scale, lh, pad = 0.55, 22, 8
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad
    w, h = min(w, img.shape[1]), min(h, img.shape[0])
    x0 = img.shape[1] - w if right else 0
    roi = img[0:h, x0:x0 + w]
    roi[:] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + pad, (i + 1) * lh), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def parse_source(s):
    """'0' -> 0 (webcam index); anything else (file path, http/rtsp URL) stays a string."""
    return int(s) if str(s).isdigit() else s


def is_live(source):
    return isinstance(source, int) or str(source).lower().startswith(("http://", "https://",
                                                                       "rtsp://"))


def resolve_weights(w):
    """Local path if it exists; otherwise download a known file name from the HF repo."""
    if w and Path(w).exists():
        return str(w)
    name = Path(w).name if w else "circuitcoach-yolo26n.pt"
    if name not in HF_FILES:
        sys.exit(f"weights not found: {w}")
    from huggingface_hub import hf_hub_download
    print(f"[weights] downloading {name} from {HF_REPO} ...")
    return hf_hub_download(HF_REPO, name)


def load_model(weights):
    from ultralytics import YOLO
    w = resolve_weights(weights)
    return YOLO(w, task="detect"), w


def extract_dets(result, min_conf):
    """Ultralytics Results -> list of (x1, y1, x2, y2, conf, cls, track_id|None) >= min_conf."""
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    ids = b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(b)
    return [(*xyxy[i], float(conf[i]), int(cls[i]), ids[i])
            for i in range(len(b)) if conf[i] >= min_conf]


# ----------------------------------------------------------------------------- writer

class LazyWriter:
    """cv2.VideoWriter created from the first frame's exact size (a size mismatch gives an
    empty file). For live sources without --save-fps, the first `probe` frames are buffered
    and the writer's fps is set from the measured processing rate, so the saved clip plays
    at real speed instead of sped up."""

    def __init__(self, path, fps=None, probe=30):
        self.path, self.fps, self.probe = Path(path), fps, probe
        self.writer, self.buffer, self.t0 = None, [], None

    def _open(self, frame, fps):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      float(fps), (w, h))
        if not self.writer.isOpened():
            sys.exit(f"could not open video writer for {self.path}")
        self.fps = fps
        print(f"[save] {self.path} {w}x{h} @ {fps:.1f} fps")

    def write(self, frame):
        if self.writer is None and self.fps:
            self._open(frame, self.fps)
        if self.writer is not None:
            self.writer.write(frame)
            return
        if self.t0 is None:
            self.t0 = time.perf_counter()
        self.buffer.append(frame.copy())
        if len(self.buffer) >= self.probe:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        dt = time.perf_counter() - self.t0
        fps = (len(self.buffer) - 1) / dt if dt > 0 and len(self.buffer) > 1 else 10.0
        self._open(self.buffer[0], max(1.0, min(fps, 60.0)))
        for f in self.buffer:
            self.writer.write(f)
        self.buffer = []

    def release(self):
        if self.writer is None:
            self._flush()
        if self.writer is not None:
            self.writer.release()


# ----------------------------------------------------------------------------- main

def open_capture(source):
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # faster to open on Windows
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"could not open source: {source!r}")
    if is_live(source):
        # keep only the newest frame so a slow model does not fall further and further behind
        # a 30 fps camera (some backends ignore this)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index, video file or stream URL")
    ap.add_argument("--weights", default=None, help=".pt, .onnx or *_openvino_model folder "
                                                    "(default: download .pt from Hugging Face)")
    ap.add_argument("--conf", type=float, default=0.4, help="min confidence to draw / count")
    ap.add_argument("--track-conf", type=float, default=0.1,
                    help="confidence passed to the tracker; kept low so ByteTrack's second "
                         "association pass sees low-score boxes")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-track", action="store_true", help="plain detection, no IDs/counts")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--min-hits", type=int, default=5, help="frames before an ID is counted")
    ap.add_argument("--save", default=None, help="write the annotated video here (.mp4)")
    ap.add_argument("--save-fps", type=float, default=None, help="fps of the saved video")
    ap.add_argument("--summary", default=None, help="write the JSON summary here")
    ap.add_argument("--no-show", action="store_true", help="no window (headless)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--blur-thresh", type=float, default=0.0,
                    help="skip inference when sharpness is below this (0 = off)")
    ap.add_argument("--show-sharpness", action="store_true", help="show the sharpness score")
    ap.add_argument("--clahe", action="store_true", help="CLAHE on each frame (uneven light)")
    args = ap.parse_args()

    source = parse_source(args.source)
    model, weights = load_model(args.weights)
    names = model.names
    track = not args.no_track
    print(f"[model] {weights} | {len(names)} classes | tracking: "
          f"{args.tracker if track else 'off'} | device: {args.device}")

    cap = open_capture(source)
    live = is_live(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    save_fps = args.save_fps or (None if live else (src_fps if 1 <= src_fps <= 120 else 30.0))
    writer = LazyWriter(args.save, fps=save_fps) if args.save else None

    counter = UniqueCounter(args.min_hits)
    fps_meter = FpsMeter()
    dets, infer_ms, n_frames, n_skipped, n_snap = [], 0.0, 0, 0, 0
    fps_hist, paused, t_start = [], False, time.perf_counter()
    win = "CircuitCoach AI - live"
    if not args.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if live:
                        print("[source] no frame from the camera/stream; stopping")
                    break
                n_frames += 1
                if args.clahe:
                    frame = apply_clahe(frame)

                score = sharpness(frame) if (args.blur_thresh > 0 or args.show_sharpness) else None
                blurry = args.blur_thresh > 0 and score < args.blur_thresh
                if blurry:
                    n_skipped += 1  # keep the previous detections on screen
                else:
                    if track:
                        r = model.track(frame, persist=True, tracker=args.tracker,
                                        conf=args.track_conf, imgsz=args.imgsz,
                                        device=args.device, verbose=False)[0]
                    else:
                        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                          device=args.device, verbose=False)[0]
                    infer_ms = r.speed.get("inference", 0.0)
                    dets = extract_dets(r, args.conf)
                    if track:
                        counter.update([d[6] for d in dets], [names[d[5]] for d in dets])

                view = draw_detections(frame.copy(), dets, names)
                fps = fps_meter.tick()
                if fps:
                    fps_hist.append(fps)
                info = [f"FPS {fps:5.1f}" if fps else "FPS  ...", f"infer {infer_ms:5.1f} ms",
                        f"parts in view {len(dets)}"]
                if score is not None and args.show_sharpness:
                    info.append(f"sharpness {score:6.0f}")
                if blurry:
                    info.append("BLURRY - skipped")
                draw_panel(view, info)
                if track:
                    draw_panel(view, [f"unique parts: {counter.total()}"] +
                               [f"{n}: {c}" for n, c in counter.counts().items()], right=True)
                if writer:
                    writer.write(view)

            if not args.no_show:
                cv2.imshow(win, view)
                key = cv2.waitKey(1 if not paused else 50) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("s"):
                    n_snap += 1
                    snap = Path("outputs") / f"snapshot_{int(time.time())}_{n_snap}.png"
                    snap.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap), view)
                    print(f"[snapshot] {snap}")
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and n_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    summary = {
        "source": str(source), "weights": Path(weights).name, "tracking": track,
        "conf": args.conf, "min_hits": args.min_hits, "clahe": args.clahe,
        "blur_thresh": args.blur_thresh, "frames": n_frames, "skipped_blurry": n_skipped,
        "mean_fps": round(float(np.mean(fps_hist)), 2) if fps_hist else None,
        "wall_fps": round(n_frames / elapsed, 2) if elapsed > 0 else None,
        "unique_parts": counter.total() if track else None,
        "unique_per_class": counter.counts() if track else None,
        "saved_video": str(args.save) if args.save else None,
    }
    print(json.dumps(summary, indent=2))
    out = args.summary or (str(Path(args.save).with_name(Path(args.save).stem + "_summary.json"))
                           if args.save else None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=2))
        print(f"[summary] {out}")


if __name__ == "__main__":
    main()
