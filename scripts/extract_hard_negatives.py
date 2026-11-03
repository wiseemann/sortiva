"""
scripts/extract_hard_negatives.py
──────────────────────────────────
Phase 2.2 — Hard-Negative Mining for Paper Class

Runs YOLOv8m inference on validation images and extracts cases where
the model misclassifies paper as medical_waste or e_waste (and vice versa).
Saves these images to a staging folder so they can be re-labeled and
added to the training set to force the model to learn the difference.

Usage (from project root, venv active):
    python scripts/extract_hard_negatives.py
    python scripts/extract_hard_negatives.py --model models/best.pt --conf 0.10
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_DIR   = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"
OUTPUT_DIR  = BASE_DIR / "datasets" / "hard_negatives"

# Target confusions we want to mine
# (true_class, predicted_as) — the biggest error sources from confusion matrix
TARGET_CONFUSIONS = [
    (5, 4),   # paper (5) predicted as medical_waste (4)  — 23 cases
    (5, 3),   # paper (5) predicted as e_waste (3)        — 11 cases
    (2, 1),   # glass (2) predicted as metal (1)          — 2 cases
]

CLASS_NAMES = {
    0: "plastic", 1: "metal", 2: "glass", 3: "e_waste",
    4: "medical_waste", 5: "paper", 6: "cardboard", 7: "organic_waste",
}


def main():
    parser = argparse.ArgumentParser(description="Extract hard-negative samples")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.pt"))
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--conf",  type=float, default=0.10,
                        help="Low conf threshold to catch all predictions")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: {e}\nActivate the venv first.")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    split_dir = DATASET_DIR / args.split
    img_dir   = split_dir / "images"
    lbl_dir   = split_dir / "labels"

    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        print(f"  → Make sure the dataset exists at {DATASET_DIR}")
        print(f"  → If the dataset is on another machine, copy it first.")
        sys.exit(1)

    # Create output directories
    for true_cls, pred_cls in TARGET_CONFUSIONS:
        true_name = CLASS_NAMES[true_cls]
        pred_name = CLASS_NAMES[pred_cls]
        out_dir = OUTPUT_DIR / f"true_{true_name}_pred_{pred_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    net = YOLO(str(model_path))
    print(f"\n{'═'*62}")
    print(f"  Hard-Negative Mining — {args.split} split")
    print(f"{'═'*62}")

    mined = {k: 0 for k in TARGET_CONFUSIONS}
    total_images = 0

    # Process each image
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        # Read ground-truth label
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        true_classes = set()
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    true_classes.add(int(parts[0]))

        # Run inference
        results = net(str(img_path), conf=args.conf, imgsz=640, verbose=False)
        pred_classes = set()
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    pred_classes.add(int(box.cls[0]))

        total_images += 1

        # Check for target confusions
        for true_cls, pred_cls in TARGET_CONFUSIONS:
            if true_cls in true_classes and pred_cls in pred_classes:
                true_name = CLASS_NAMES[true_cls]
                pred_name = CLASS_NAMES[pred_cls]
                out_dir = OUTPUT_DIR / f"true_{true_name}_pred_{pred_name}"

                # Copy image and label
                shutil.copy2(img_path, out_dir / img_path.name)
                shutil.copy2(lbl_path, out_dir / lbl_path.name)
                mined[(true_cls, pred_cls)] += 1

    # Report
    print(f"\n  Processed: {total_images} images")
    print(f"\n  Hard negatives found:")
    for (true_cls, pred_cls), count in mined.items():
        true_name = CLASS_NAMES[true_cls]
        pred_name = CLASS_NAMES[pred_cls]
        print(f"    {true_name} → {pred_name}: {count} samples")

    total_mined = sum(mined.values())
    print(f"\n  Total mined: {total_mined} images → {OUTPUT_DIR}")

    if total_mined > 0:
        print(f"\n  Next steps:")
        print(f"    1. Review images in {OUTPUT_DIR} — verify labels are correct")
        print(f"    2. Copy verified images+labels to balanced_v3/train/")
        print(f"    3. Retrain with: python scripts/train_yolov8m.py --pretrained models/best.pt --epochs 30")
    else:
        print(f"\n  No hard negatives found on this split — model may be performing well")
        print(f"  Try running on 'train' split: python scripts/extract_hard_negatives.py --split train")

    print(f"\n{'═'*62}")


if __name__ == "__main__":
    main()

