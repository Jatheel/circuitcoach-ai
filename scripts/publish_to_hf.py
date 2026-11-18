"""
CircuitCoach AI - build the release folder + model card and upload it to Hugging Face.

Collects:
  release/  circuitcoach-yolo26n.pt, circuitcoach-yolo26n.tflite (float32),
            circuitcoach-yolo26n_w8a32.tflite, export_report.json
  eval/     metrics.json, per_class.csv, confusions.csv, confusion_matrix_normalized.png
  train run args.yaml, results.csv, results.png
  dataset   audit.json, per_class_counts.csv, data.yaml (class names)
and writes README.md (the model card) from the real numbers.

Usage (Kaggle):
  python publish_to_hf.py --release /kaggle/working/release --eval /kaggle/working/eval/test_v2 \
      --train-run <.../runs/yolo26n_ec61_v2> --dataset-dir <.../ec61_clean> --no-upload   # check first
  python publish_to_hf.py ... (same arguments, without --no-upload)                      # publish
The token is read from the HF_TOKEN environment variable or the Kaggle secret HF_TOKEN.
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import yaml

REPO_NAME = "circuitcoach-yolo26n"

CITATION = """@article{sayeedi2025electrocom61,
  title   = {ElectroCom61: A multiclass dataset for detection of electronic components},
  author  = {Sayeedi, Md. Faiyaz Abdullah and Osmani, Anas Mohammad Ishfaqul Muktadir and
             Rahman, Taimur and Deepti, Jannatul Ferdous and Rahman, Raiyan and Islam, Salekul},
  journal = {Data in Brief},
  volume  = {59},
  pages   = {111331},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.111331}
}"""

FAILURE_NOTES = [
    "**Classification is reliable:** at confidence 0.25, only {wrong} of {total} test boxes "
    "received the wrong class; most errors are missed parts ({missed}) or extra boxes ({fp}).",
    "**Look-alike small parts cause most errors:** small dark round parts (buzzer, push switch, "
    "inductor) and two-lead parts (Zener diode, diode, resistor, small capacitors).",
    "**Background objects:** in photos with a room or floor behind the parts, the model can draw "
    "low-confidence boxes on furniture or people; the dataset has no component-free images.",
    "**Blur and angle:** out-of-focus parts get overlapping low-confidence boxes, steeply angled "
    "parts are missed, and small parts have the loosest boxes.",
    "**Single-lab data:** all images come from one lab, so accuracy on other desks and lighting "
    "is lower than the numbers below; see the phone-test section.",
]


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        raise SystemExit("[error] no HF token: set HF_TOKEN or attach the Kaggle secret")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_card(owner, m, per_class, export, audit, args_yaml, names):
    fr = m["failure_review"]
    fm = export["formats"]
    ship = export["ship"]
    weakest = sorted(per_class, key=lambda r: float(r["mAP50"]))[:10]
    notes = "\n".join(f"{i}. " + n.format(wrong=fr["wrong_class"], total=fr["true_boxes"],
                                          missed=fr["missed"], fp=fr["false_boxes"])
                      for i, n in enumerate(FAILURE_NOTES, 1))

    def row(r):
        return (f"| {r['class_name']} | {r['boxes']} | {float(r['precision']):.2f} | "
                f"{float(r['recall']):.2f} | {float(r['mAP50']):.3f} | {float(r['mAP50_95']):.3f} |")

    head = "| Class | Test boxes | P | R | mAP50 | mAP50-95 |\n|---|---|---|---|---|---|"
    all_rows = "\n".join(row(r) for r in sorted(per_class, key=lambda r: r["class_name"]))
    weak_rows = "\n".join(row(r) for r in weakest)
    tr = args_yaml

    return f"""---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - object-detection
  - yolo
  - yolo26
  - ultralytics
  - litert
  - tflite
  - electronics
  - computer-vision
---

# CircuitCoach AI — YOLO26n electronic component detector

Detects **{len(names)} kinds of electronic and robotics components** (Arduino boards, sensors,
modules, passive parts, switches, displays…) in a photo and returns a box, a class name and a
confidence for each one. It is the detection model behind **CircuitCoach AI**, a learning
assistant that tells beginners what a part is and what it does.

- **Model:** Ultralytics YOLO26n (2.4 M parameters), fine-tuned from COCO-pretrained weights
- **Data:** ElectroCom61 (CC BY 4.0), cleaned and re-split by the author of this model
- **Formats:** PyTorch `.pt`, LiteRT float32 `.tflite`, LiteRT w8a32 `.tflite` (shipped for Android)

> **Safety:** for low-voltage (5 V or less) learning circuits. Not a safety check. Never use on
> mains wiring. The model names parts; it does not check whether a circuit is correct or safe.

## Results on the held-out test split

{fr['true_boxes']} boxes in {m['images']} images that were never used for training or model selection.

| Metric | Value |
|---|---|
| Precision | {m['precision']:.3f} |
| Recall | {m['recall']:.3f} |
| **mAP50** | **{m['mAP50']:.3f}** |
| mAP50-95 | {m['mAP50_95']:.3f} |
| Inference (NVIDIA T4, PyTorch) | {m['speed_ms_per_image']['inference']:.1f} ms / image |

At confidence {fr['conf']} (IoU 0.5): {fr['correct']} of {fr['true_boxes']} parts found with the
right name, {fr['wrong_class']} wrong class, {fr['missed']} missed, {fr['false_boxes']} extra boxes.

**These are in-lab numbers.** Test images come from the same lab as the training images.
Expect lower accuracy on your own desk.

### Phone test (real-world)

*Not measured yet.* A separate set of phone photos taken outside the original lab will be
reported here.

### Weakest classes

{head}
{weak_rows}

<details>
<summary>All {len(per_class)} classes</summary>

{head}
{all_rows}

</details>

### Failure analysis

{notes}

## Files

| File | Format | Size | Test mAP50 | Use |
|---|---|---|---|---|
| `circuitcoach-yolo26n.pt` | PyTorch | {fm['pytorch']['size_MB']} MB | {fm['pytorch']['mAP50']:.3f} | Python, web demo |
| `circuitcoach-yolo26n.tflite` | LiteRT float32 | {fm['float32']['size_MB']} MB | {fm['float32']['mAP50']:.3f} | Full-precision reference |
| `circuitcoach-yolo26n_w8a32.tflite` | LiteRT int8 weights, float activations | {fm['w8a32']['size_MB']} MB | {fm['w8a32']['mAP50']:.3f} | **Shipped in the Android app** |

**Export check:** at confidence 0.4 the float32 LiteRT model returned the same boxes as PyTorch
on 25 test images (160 of 160 boxes matched, mean IoU 1.000). The validator's LiteRT mAP50-95
({fm['float32']['mAP50_95']:.3f} float32 / {fm['w8a32']['mAP50_95']:.3f} w8a32) is lower than
PyTorch's ({fm['pytorch']['mAP50_95']:.3f}); this comes from how very-low-confidence boxes are
handled during validation, not from different detections at working thresholds.
w8a32 loses {export['w8a32_drop_vs_float32_points']} mAP50 points against float32 and is about
{fm['float32']['size_MB'] / max(fm['w8a32']['size_MB'], 0.01):.1f}x smaller, so it is shipped.

Other files: `data.yaml` (class names), `metrics/` (test metrics, per-class results, confusion
matrix, export report), `training/` (training arguments and curves), `dataset/` (cleaning audit).

## How to use

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

path = hf_hub_download("{owner}/{REPO_NAME}", "circuitcoach-yolo26n.pt")
model = YOLO(path)
results = model.predict("my_parts.jpg", conf=0.4)
results[0].show()
```

A confidence of about **0.4** is recommended: it removes most low-confidence boxes on
background objects while keeping confident detections.

## Training data

**ElectroCom61** — {audit['images_written']} images of 61 component classes photographed in the
Electronic Lab Support Room of United International University, Dhaka, with several
smartphones, varied lighting, backgrounds and angles. Images were resized to 640×640 by the
dataset authors, which stretches some of them. Licence: **CC BY 4.0**.

**Changes made for this model** (dataset not redistributed here; the scripts are in the
project repository):
1. Merged the original train/valid/test splits into one pool.
2. Checked every image and label line ({audit['images_scanned']} images scanned; checks: {audit['checks']}).
3. Grouped exact duplicates, near-duplicates (64-bit difference hash, distance ≤ {audit['near_dup_bits']})
   and copies of the same source photo, giving {audit['groups']} groups.
4. Re-split 75/15/10 **by group** and stratified by class, so near-identical photos never sit
   on both sides of the train/test line: {audit['split_images']['train']} train /
   {audit['split_images']['val']} val / {audit['split_images']['test']} test images.
   Every class appears in all three splits.

Because of this grouped re-split, the numbers here are not directly comparable with the
dataset paper's results on its random split.

## Training

| Setting | Value |
|---|---|
| Base model | `yolo26n.pt` (pretrained) |
| Image size | {tr.get('imgsz')} |
| Epochs | {tr.get('epochs')} (early stopping off; best epoch selected on validation) |
| Batch size | {tr.get('batch')} |
| Optimizer | {tr.get('optimizer')} (Ultralytics automatic choice) |
| Extra augmentation | rotation ±{tr.get('degrees')}°, vertical flip p={tr.get('flipud')}, plus Ultralytics defaults |
| Seed | {tr.get('seed')}, deterministic={tr.get('deterministic')} |
| Hardware | Kaggle, 1× NVIDIA Tesla T4 |
| Ultralytics | 8.4.215 |

## Limitations

- Trained on photos from a single lab; accuracy drops on unfamiliar backgrounds and lighting.
- Cannot read component values (resistor colour bands, capacitor ratings) or exact model numbers.
- Cannot tell genuine boards from clones.
- Look-alike parts (diode vs Zener diode, push vs tact switch, small capacitors) can be confused.
- Does not trace wires or verify circuits.

## Licence and credit

- **Model weights:** AGPL-3.0 (models trained with Ultralytics code fall under AGPL-3.0).
- **Dataset:** ElectroCom61 by Sayeedi et al., CC BY 4.0, used with the changes listed above.
  Original data: https://data.mendeley.com/datasets/6scy6h8sjz/2

```bibtex
{CITATION}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--train-run", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--stage", type=Path, default=Path("/kaggle/working/hf_release"))
    ap.add_argument("--owner", default=None, help="HF username (default: read from the token)")
    ap.add_argument("--no-upload", action="store_true", help="build the folder only")
    args = ap.parse_args()

    token = None if (args.no_upload and args.owner) else get_token()
    owner = args.owner
    if owner is None:
        from huggingface_hub import whoami
        owner = whoami(token=token)["name"]
    repo_id = f"{owner}/{REPO_NAME}"

    if args.stage.exists():
        shutil.rmtree(args.stage)
    for sub in ["metrics", "training", "dataset"]:
        (args.stage / sub).mkdir(parents=True)

    def put(src, dst, required=True):
        src = Path(src)
        if src.exists():
            shutil.copy2(src, args.stage / dst)
        elif required:
            raise SystemExit(f"[error] missing required file: {src}")
        else:
            print(f"[skip] not found: {src}")

    for f in ["circuitcoach-yolo26n.pt", "circuitcoach-yolo26n.tflite",
              "circuitcoach-yolo26n_w8a32.tflite"]:
        put(args.release / f, f)
    put(args.release / "export_report.json", "metrics/export_report.json")
    for f in ["metrics.json", "per_class.csv", "confusions.csv"]:
        put(args.eval / f, f"metrics/{f}")
    put(args.eval / "confusion_matrix_normalized.png", "metrics/confusion_matrix_normalized.png", False)
    put(args.train_run / "args.yaml", "training/args.yaml")
    put(args.train_run / "results.csv", "training/results.csv")
    put(args.train_run / "results.png", "training/results.png", False)
    put(args.dataset_dir / "audit.json", "dataset/audit.json")
    put(args.dataset_dir / "per_class_counts.csv", "dataset/per_class_counts.csv")

    meta = yaml.safe_load((args.dataset_dir / "data.yaml").read_text())
    names = meta["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    (args.stage / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(names), "names": {i: n for i, n in enumerate(names)}},
                       sort_keys=False))

    card = build_card(
        owner,
        json.loads((args.eval / "metrics.json").read_text()),
        read_csv(args.eval / "per_class.csv"),
        json.loads((args.release / "export_report.json").read_text()),
        json.loads((args.dataset_dir / "audit.json").read_text()),
        yaml.safe_load((args.train_run / "args.yaml").read_text()),
        names,
    )
    (args.stage / "README.md").write_text(card)

    print(f"[stage] built {args.stage}:")
    for p in sorted(args.stage.rglob("*")):
        if p.is_file():
            print(f"   {p.relative_to(args.stage)}  ({p.stat().st_size / 1e6:.2f} MB)")

    if args.no_upload:
        print(f"[stage] --no-upload: nothing sent. Review {args.stage / 'README.md'}, then run "
              f"again without --no-upload to publish to {repo_id}")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(args.stage), repo_id=repo_id, repo_type="model",
                      commit_message="CircuitCoach YOLO26n v2: weights, LiteRT exports, metrics, model card")
    print(f"[done] published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
