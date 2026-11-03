"""
scripts/prepare_kaggle_dataset.py
─────────────────────────────────
Downloads and prepares the Mostafa Abla Garbage Classification dataset
directly on the VPS, mapping 12 classes to SORTIVA's 8 Adiwiyata classes
and generating YOLO bounding boxes (pseudo-labels).

Mapping:
  - biological                     → organic_waste (class 7)
  - battery                        → e_waste (class 3)
  - brown-glass, green-glass, white-glass → glass (class 2)
  - paper                          → paper (class 5)
  - cardboard                      → cardboard (class 6)
  - plastic                        → plastic (class 0)
  - metal                          → metal (class 1)
  - clothes, shoes, trash          → DISCARDED (not Adiwiyata relevant)

Usage (run on VPS):
  python scripts/prepare_kaggle_dataset.py --kaggle-dir /path/to/extracted/garbage_classification
"""

import argparse
import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"

CLASS_MAP = {
    "plastic": 0,
    "metal": 1,
    "glass": 2,
    "brown-glass": 2,
    "green-glass": 2,
    "white-glass": 2,
    "battery": 3,
    "paper": 5,
    "cardboard": 6,
    "biological": 7,
}

IGNORE_CLASSES = {"clothes", "shoes", "trash"}


def prepare_dataset(source_dir: Path, max_per_class: int = 1000, val_ratio: float = 0.15, test_ratio: float = 0.10):
    print(f"\n{'='*60}")
    print(f"  Preparing Dataset from: {source_dir}")
    print(f"  Target directory     : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {v: 0 for v in set(CLASS_MAP.values())}

    # Group images by source folder
    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir():
            continue
        cname = folder.name.lower().strip()
        if cname in IGNORE_CLASSES or cname not in CLASS_MAP:
            print(f"  ⏭ Skipping irrelevant category: {cname}")
            continue

        target_cid = CLASS_MAP[cname]
        images = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.seed(42)
        random.shuffle(images)

        images = images[:max_per_class]
        val_count = int(len(images) * val_ratio)
        test_count = int(len(images) * test_ratio)

        val_imgs = set(images[:val_count])
        test_imgs = set(images[val_count:val_count + test_count])

        for img in images:
            if img in val_imgs:
                split = "valid"
            elif img in test_imgs:
                split = "test"
            else:
                split = "train"

            dst_img = OUTPUT_DIR / split / "images" / f"kaggle_{img.name}"
            dst_lbl = OUTPUT_DIR / split / "labels" / f"kaggle_{img.stem}.txt"

            shutil.copy2(img, dst_img)
            # Generate pseudo-label: object centered in 85% of frame
            dst_lbl.write_text(f"{target_cid} 0.500000 0.500000 0.850000 0.850000\n", encoding="utf-8")
            counts[target_cid] += 1

    print(f"\n{'='*60}")
    print(f"  Dataset preparation complete!")
    print(f"  Images processed per class id: {counts}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", type=str, required=True, help="Path to unzipped Kaggle dataset")
    parser.add_argument("--max-per-class", type=int, default=1000)
    args = parser.parse_args()

    prepare_dataset(Path(args.kaggle_dir), max_per_class=args.max_per_class)

