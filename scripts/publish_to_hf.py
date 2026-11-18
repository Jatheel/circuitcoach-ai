"""
CircuitCoach AI - dataset preparation for ElectroCom61 (YOLO format).

What it does
1. Merges the original train/valid/test splits into one pool.
2. Checks every image (can it be opened?) and every label line
   (5 numbers, valid class id, box inside the image). Bad lines are dropped;
   polygon lines are converted to boxes; corrupt images are dropped.
3. Finds exact duplicates (same file bytes) and near-duplicates
   (64-bit difference hash), plus Roboflow copies of the same source photo.
4. Re-splits 75/15/10 by duplicate GROUP, so near-identical photos never
   end up on both sides of the train/test line. The split is stratified so
   rare classes still appear in val and test where possible.
5. Writes a clean YOLO dataset, data.yaml, audit.json and per_class_counts.csv.

Usage (Kaggle):
    python prepare_dataset.py                       # finds the dataset by itself
    python prepare_dataset.py --near-dup-bits 4     # stricter near-duplicate match
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ORIG_SPLITS = ["train", "valid", "val", "test"]


# ---------------------------------------------------------------- helpers
def find_raw_root(search_root: Path) -> Path:
    yamls = sorted(search_root.rglob("data.yaml"))
    if not yamls:
        raise SystemExit(f"[error] no data.yaml found under {search_root}")
    return yamls[0].parent


def dhash64(path: Path) -> int:
    """64-bit difference hash: shrink to 9x8 grey, compare neighbours."""
    with Image.open(path) as im:
        g = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def source_key(name: str) -> str:
    """Roboflow names copies 'photo_jpg.rf.<hash>.jpg' - same photo, same key."""
    return name.split(".rf.")[0] if ".rf." in name else name


def parse_label_file(path: Path, nc: int, stats: Counter):
    """Return a list of clean 'cls cx cy w h' lines."""
    if not path.exists():
        stats["missing_label_files"] += 1
        return []
    clean = []
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            stats["bad_lines_not_numbers"] += 1
            continue
        if not 0 <= cls < nc:
            stats["bad_lines_class_id"] += 1
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> box
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            stats["polygons_converted"] += 1
        else:
            stats["bad_lines_wrong_length"] += 1
            continue
        # clip to the image, then reject boxes that are empty or far outside
        x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x1 - x0 <= 0.001 or y1 - y0 <= 0.001:
            stats["bad_lines_empty_or_outside"] += 1
            continue
        if (x1 - x0) < w - 0.02 or (y1 - y0) < h - 0.02:
            stats["boxes_clipped"] += 1
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        clean.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return clean


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=None,
                    help="folder holding data.yaml (default: search /kaggle/input)")
    ap.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/ec61_clean"))
    ap.add_argument("--near-dup-bits", type=int, default=6,
                    help="dHash Hamming distance counted as near-duplicate (lower = stricter)")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.75, 0.15, 0.10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = args.raw or find_raw_root(args.search_root)
    meta = yaml.safe_load((raw / "data.yaml").read_text())
    names = meta["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    nc = len(names)
    print(f"[scan] dataset root: {raw}")
    print(f"[scan] {nc} classes, first five: {names[:5]}")

    # 1. pool every image from every original split
    items = []
    for split in ORIG_SPLITS:
        img_dir = raw / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append({"img": p, "lbl": raw / split / "labels" / f"{p.stem}.txt",
                              "orig_split": split})
    print(f"[scan] {len(items)} images found")

    # 2. validate images and labels
    stats = Counter()
    kept = []
    for it in items:
        try:
            with Image.open(it["img"]) as im:
                im.verify()
            it["dhash"] = dhash64(it["img"])
            it["md5"] = md5(it["img"])
        except Exception:
            stats["corrupt_images_dropped"] += 1
            continue
        it["lines"] = parse_label_file(it["lbl"], nc, stats)
        if not it["lines"]:
            stats["images_without_boxes"] += 1
        kept.append(it)
    items = kept

    # 3a. exact duplicates: keep the first copy only
    seen, unique = {}, []
    for it in items:
        if it["md5"] in seen:
            stats["exact_duplicates_dropped"] += 1
            continue
        seen[it["md5"]] = True
        unique.append(it)
    items = unique
    n = len(items)

    # 3b. group near-duplicates and Roboflow copies of the same source photo
    uf = UnionFind(n)
    by_source = defaultdict(list)
    for i, it in enumerate(items):
        by_source[source_key(it["img"].name)].append(i)
    for idx in by_source.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
            stats["same_source_links"] += 1

    hashes = np.array([it["dhash"] for it in items], dtype=np.uint64)
    near_pairs = 0
    for i in range(n - 1):
        x = np.bitwise_xor(hashes[i + 1:], hashes[i])
        dist = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(axis=1)
        for k in np.nonzero(dist <= args.near_dup_bits)[0]:
            uf.union(i, i + 1 + int(k))
            near_pairs += 1
    stats["near_duplicate_pairs"] = near_pairs

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    groups = list(groups.values())
    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"[dups] exact dropped={stats['exact_duplicates_dropped']} "
          f"near pairs={near_pairs} groups={len(groups)} largest_group={sizes[0]}")
    if sizes[0] > max(25, 0.02 * n):
        print(f"[warn] largest group holds {sizes[0]} images (>2% of data). Similar "
              f"backgrounds may be chaining photos together; re-run with --near-dup-bits 4")

    # 4. grouped, stratified 75/15/10 split
    split_names = ["train", "val", "test"]
    total_boxes = Counter()
    for it in items:
        for ln in it["lines"]:
            total_boxes[int(ln.split()[0])] += 1
    target = {s: {c: total_boxes[c] * r for c in total_boxes}
              for s, r in zip(split_names, args.ratios)}
    target_imgs = {s: n * r for s, r in zip(split_names, args.ratios)}
    have = {s: Counter() for s in split_names}
    have_imgs = Counter()

    def group_classes(g):
        cnt = Counter()
        for i in g:
            for ln in items[i]["lines"]:
                cnt[int(ln.split()[0])] += 1
        return cnt

    rng = random.Random(args.seed)
    rng.shuffle(groups)
    gcls = [group_classes(g) for g in groups]
    # rarest classes first, so they are spread across all three splits
    order = sorted(range(len(groups)),
                   key=lambda k: min((total_boxes[c] for c in gcls[k]), default=10**9))

    assignment = {}
    for k in order:
        g, cls = groups[k], gcls[k]

        def score(s):
            img_need = (target_imgs[s] - have_imgs[s]) / max(target_imgs[s], 1)
            if not cls:
                return img_need
            cls_need = sum((target[s][c] - have[s][c]) / max(target[s][c], 1e-9)
                           for c in cls) / len(cls)
            return cls_need + 0.5 * img_need

        s = max(split_names, key=score)
        assignment[k] = s
        have_imgs[s] += len(g)
        have[s].update(cls)

    # 5. write the clean dataset
    if args.out.exists():
        shutil.rmtree(args.out)
    used_names = set()
    for k, g in enumerate(groups):
        s = assignment[k]
        for i in g:
            it = items[i]
            name = it["img"].name
            if name in used_names:  # same file name in two original splits
                name = f"{it['orig_split']}_{name}"
            used_names.add(name)
            img_out = args.out / s / "images" / name
            lbl_out = args.out / s / "labels" / f"{Path(name).stem}.txt"
            img_out.parent.mkdir(parents=True, exist_ok=True)
            lbl_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(it["img"], img_out)
            lbl_out.write_text("\n".join(it["lines"]) + ("\n" if it["lines"] else ""))

    data_yaml = {"path": str(args.out.resolve()), "train": "train/images",
                 "val": "val/images", "test": "test/images",
                 "names": {i: nm for i, nm in enumerate(names)}}
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    with open(args.out / "per_class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "train_boxes", "val_boxes", "test_boxes", "total"])
        for c, nm in enumerate(names):
            w.writerow([c, nm, have["train"][c], have["val"][c], have["test"][c], total_boxes[c]])

    missing_val = [names[c] for c in range(nc) if have["val"][c] == 0]
    missing_test = [names[c] for c in range(nc) if have["test"][c] == 0]
    audit = {
        "source": str(raw), "near_dup_bits": args.near_dup_bits, "seed": args.seed,
        "ratios": args.ratios, "images_scanned": len(kept) + stats["corrupt_images_dropped"],
        "images_written": n, "groups": len(groups), "largest_groups": sizes[:10],
        "split_images": dict(have_imgs),
        "split_boxes": {s: sum(have[s].values()) for s in split_names},
        "classes_missing_from_val": missing_val, "classes_missing_from_test": missing_test,
        "checks": dict(stats),
    }
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2))

    # summary
    print(f"[done] wrote {n} images to {args.out}")
    print(f"       train {have_imgs['train']} | val {have_imgs['val']} | test {have_imgs['test']}")
    print(f"[checks] {dict(stats)}")
    rare = sorted(range(nc), key=lambda c: have["train"][c])[:10]
    print("[info] 10 classes with the fewest training boxes: "
          + ", ".join(f"{names[c]}={have['train'][c]}" for c in rare))
    print(f"[info] classes missing from val: {missing_val or 'none'}")
    print(f"[info] classes missing from test: {missing_test or 'none'}")
    print(f"[info] data.yaml -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
