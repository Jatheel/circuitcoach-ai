"""
CircuitCoach AI - evaluate a trained YOLO model on one split.

Outputs (in --out):
  metrics.json      overall precision, recall, mAP50, mAP50-95, speed
  per_class.csv     per-class images, boxes, P, R, mAP50, mAP50-95
  confusions.csv    most common "true class -> predicted class" mix-ups,
                    plus missed parts and false boxes on background
  failures.csv      test images ranked by number of errors
  failures/         the worst images with the model's predictions drawn
  + Ultralytics plots (confusion matrix, PR curve, val_batch*_pred.jpg ...)

Usage:
  python evaluate.py --weights best.pt --data data.yaml --split test --out eval/test_v2
  # later, for the phone test set:
  python evaluate.py --weights best.pt --data phone/data.yaml --split test --out eval/phone_v2
"""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def xywhn_to_xyxyn(b):
    cx, cy, w, h = b
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and b (M,4), xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(gt_cls, gt_box, pr_cls, pr_box, pr_conf, iou_thr=0.5):
    """Greedy matching, highest confidence first.
    Returns (true positives, wrong-class hits [(true, pred)], missed true classes,
    false-positive predicted classes)."""
    ious = iou_matrix(pr_box, gt_box)
    gt_used = np.zeros(len(gt_cls), bool)
    tp, wrong, fps = 0, [], []
    for p in np.argsort(-np.asarray(pr_conf)) if len(pr_conf) else []:
        cand = [g for g in range(len(gt_cls)) if not gt_used[g] and ious[p, g] >= iou_thr]
        same = [g for g in cand if gt_cls[g] == pr_cls[p]]
        if same:
            g = max(same, key=lambda g: ious[p, g])
            gt_used[g] = True
            tp += 1
        elif cand:
            g = max(cand, key=lambda g: ious[p, g])
            gt_used[g] = True
            wrong.append((int(gt_cls[g]), int(pr_cls[p])))
        else:
            fps.append(int(pr_cls[p]))
    missed = [int(gt_cls[g]) for g in range(len(gt_cls)) if not gt_used[g]]
    return tp, wrong, missed, fps


def read_labels(path):
    cls, box = [], []
    if path.exists():
        for ln in path.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                cls.append(int(p[0]))
                box.append(xywhn_to_xyxyn([float(v) for v in p[1:]]))
    return cls, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence used for the failure review (mAP uses all boxes)")
    ap.add_argument("--n-failures", type=int, default=12)
    args = ap.parse_args()

    meta = yaml.safe_load(Path(args.data).read_text())
    names = meta["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    nc = len(names)
    root = Path(meta.get("path", Path(args.data).parent))
    img_dir = root / meta[args.split]
    lbl_dir = Path(str(img_dir).replace("/images", "/labels"))
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    if args.out.exists():
        shutil.rmtree(args.out)
    model = YOLO(args.weights)

    # 1. standard metrics (all confidences, as mAP requires)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                  plots=True, project=str(args.out.parent), name=args.out.name,
                  exist_ok=True, verbose=False)
    box = m.box
    overall = {
        "weights": str(args.weights), "split": args.split, "images": len(images),
        "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4), "mAP50_95": round(float(box.map), 4),
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in m.speed.items()},
    }

    # boxes and images per class, counted from the labels
    n_boxes, n_imgs = Counter(), Counter()
    gts = {}
    for img in images:
        c, b = read_labels(lbl_dir / f"{img.stem}.txt")
        gts[img] = (c, b)
        n_boxes.update(c)
        n_imgs.update(set(c))

    rows = []
    idx = list(map(int, box.ap_class_index))
    for k, c in enumerate(idx):
        p, r, ap50, ap = box.class_result(k)
        rows.append([c, names[c], n_imgs[c], n_boxes[c], round(float(p), 4),
                     round(float(r), 4), round(float(ap50), 4), round(float(ap), 4)])
    rows.sort(key=lambda r: r[6])
    with open(args.out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "images", "boxes", "precision", "recall",
                    "mAP50", "mAP50_95"])
        w.writerows(rows)
    absent = [names[c] for c in range(nc) if c not in idx]

    # 2. failure review at a fixed confidence
    wrong_pairs, missed, fps, per_image = Counter(), Counter(), Counter(), []
    worst = []
    for r in model.predict([str(p) for p in images], imgsz=args.imgsz, conf=args.conf,
                           stream=True, verbose=False):
        img = Path(r.path)
        gc, gb = gts.get(img, ([], []))
        pc = r.boxes.cls.cpu().numpy().astype(int).tolist()
        pb = r.boxes.xyxyn.cpu().numpy().tolist()
        pf = r.boxes.conf.cpu().numpy().tolist()
        tp, wrong, miss, fp = match_image(gc, gb, pc, pb, pf)
        wrong_pairs.update(wrong)
        missed.update(miss)
        fps.update(fp)
        errors = len(wrong) + len(miss) + len(fp)
        per_image.append([img.name, len(gc), tp, len(wrong), len(miss), len(fp), errors,
                          "; ".join(f"{names[t]}->{names[p]}" for t, p in wrong),
                          "; ".join(names[c] for c in miss), "; ".join(names[c] for c in fp)])
        if errors:
            worst.append((errors, img.name, r.plot()))
            worst = sorted(worst, key=lambda x: -x[0])[: args.n_failures]

    per_image.sort(key=lambda r: -r[6])
    with open(args.out / "failures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_boxes", "correct", "wrong_class", "missed",
                    "false_boxes", "errors", "wrong_class_detail", "missed_detail",
                    "false_box_detail"])
        w.writerows(per_image)

    fail_dir = args.out / "failures"
    fail_dir.mkdir(exist_ok=True)
    from PIL import Image
    for rank, (err, name, arr) in enumerate(worst, 1):
        Image.fromarray(arr[..., ::-1]).save(fail_dir / f"{rank:02d}_{err}err_{name}")

    with open(args.out / "confusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "true_class", "predicted_class", "count"])
        for (t, p), n in wrong_pairs.most_common():
            w.writerow(["wrong_class", names[t], names[p], n])
        for c, n in missed.most_common():
            w.writerow(["missed", names[c], "", n])
        for c, n in fps.most_common():
            w.writerow(["false_box", "", names[c], n])

    total_gt = sum(n_boxes.values())
    overall["failure_review"] = {
        "conf": args.conf, "iou": 0.5, "true_boxes": total_gt,
        "correct": total_gt - sum(missed.values()) - sum(wrong_pairs.values()),
        "wrong_class": sum(wrong_pairs.values()), "missed": sum(missed.values()),
        "false_boxes": sum(fps.values()),
        "images_with_errors": sum(1 for r in per_image if r[6]),
    }
    overall["classes_absent_from_split"] = absent
    (args.out / "metrics.json").write_text(json.dumps(overall, indent=2))

    # 3. summary
    print(f"\n=== {args.split} split | {len(images)} images | {total_gt} boxes ===")
    print(f"P={overall['precision']:.3f}  R={overall['recall']:.3f}  "
          f"mAP50={overall['mAP50']:.3f}  mAP50-95={overall['mAP50_95']:.3f}")
    print(f"Speed (ms/img): {overall['speed_ms_per_image']}")
    fr = overall["failure_review"]
    print(f"\nAt conf {args.conf}: correct={fr['correct']}  wrong class={fr['wrong_class']}  "
          f"missed={fr['missed']}  false boxes={fr['false_boxes']}  "
          f"images with errors={fr['images_with_errors']}/{len(images)}")
    print("\n10 weakest classes (mAP50):")
    for r in rows[:10]:
        print(f"  {r[1]:32s} boxes={r[3]:3d}  P={r[4]:.2f}  R={r[5]:.2f}  "
              f"mAP50={r[6]:.3f}  mAP50-95={r[7]:.3f}")
    print("\nTop mix-ups (true -> predicted):")
    for (t, p), n in wrong_pairs.most_common(10):
        print(f"  {names[t]} -> {names[p]}: {n}")
    print("\nMost missed:", ", ".join(f"{names[c]}={n}" for c, n in missed.most_common(8)) or "none")
    print("Most false boxes:", ", ".join(f"{names[c]}={n}" for c, n in fps.most_common(8)) or "none")
    if absent:
        print("Classes with no boxes in this split:", absent)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
