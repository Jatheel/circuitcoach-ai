"""
CircuitCoach AI - turn held-out test images into a video for the live pipeline, the demo
clip and the CPU benchmark (when no real parts or camera are at hand).

Each image gets a slow zoom-and-pan ("Ken Burns") so the tracker has motion to follow, with
a short cross-fade between images. Images with the most labelled parts are used first, so
the demo shows busy scenes.

These are in-lab images from the ElectroCom61 test split (never used for training): label
any demo made from them as "held-out test images", not as a real-world camera test.
Dataset: ElectroCom61, Sayeedi et al., CC BY 4.0 - credit it wherever the video is posted.

Usage (Kaggle, after prepare_dataset.py):
  python make_test_video.py --images ec61_clean/test/images --out /kaggle/working/test_clip.mp4
  python make_test_video.py --images ec61_clean/test/images --out clip.mp4 --n 12 --size 1080
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def n_labels(img_path):
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not lbl.exists():
        return 0
    return sum(1 for line in lbl.read_text().splitlines() if line.strip())


def ken_burns(img, size, n_frames, zoom_to, rng):
    """Frames that zoom from the full image to `zoom_to`x towards a random point."""
    h, w = img.shape[:2]
    s = min(h, w)  # centre square crop (the dataset images are square already)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    cx_end, cy_end = rng.uniform(0.35, 0.65) * s, rng.uniform(0.35, 0.65) * s
    zoom_in = rng.random() < 0.5  # alternate between zooming in and out
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        t = t * t * (3 - 2 * t)  # ease in/out
        if not zoom_in:
            t = 1 - t
        z = 1 + (zoom_to - 1) * t
        cw = s / z
        cx = s / 2 + (cx_end - s / 2) * t
        cy = s / 2 + (cy_end - s / 2) * t
        x0 = float(np.clip(cx - cw / 2, 0, s - cw))
        y0 = float(np.clip(cy - cw / 2, 0, s - cw))
        m = np.array([[size / cw, 0, -x0 * size / cw], [0, size / cw, -y0 * size / cw]])
        yield cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path, help="e.g. ec61_clean/test/images")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=12, help="number of images (12 x 2.5 s = 30 s)")
    ap.add_argument("--seconds", type=float, default=2.5, help="seconds per image")
    ap.add_argument("--fade", type=float, default=0.4, help="cross-fade seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=960, help="square output size in pixels")
    ap.add_argument("--zoom", type=float, default=1.35, help="max zoom factor")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    imgs = [p for p in sorted(args.images.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise SystemExit(f"no images in {args.images}")
    # busiest scenes first (ties broken by a seeded shuffle), then shown in random order
    rng.shuffle(imgs)
    imgs = sorted(imgs, key=n_labels, reverse=True)[:args.n]
    rng.shuffle(imgs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (args.size, args.size))
    per, fade = round(args.seconds * args.fps), round(args.fade * args.fps)
    prev_tail, total = None, 0
    for k, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        frames = list(ken_burns(img, args.size, per, args.zoom, rng))
        if prev_tail is not None and fade:
            for i in range(fade):  # cross-fade from the last frame of the previous image
                a = (i + 1) / (fade + 1)
                vw.write(cv2.addWeighted(prev_tail, 1 - a, frames[0], a, 0)); total += 1
        for f in frames:
            vw.write(f); total += 1
        prev_tail = frames[-1]
        print(f"[{k + 1}/{len(imgs)}] {p.name} ({n_labels(p)} labelled parts)")
    vw.release()
    print(f"[done] {args.out}  {total} frames, {total / args.fps:.1f} s, "
          f"{args.size}x{args.size} @ {args.fps} fps")


if __name__ == "__main__":
    main()
