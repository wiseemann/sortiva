"""
scripts/merge_stembayo_custom.py
─────────────────────────────────
Phase 1.2 — Merge custom Stembayo photos into the training dataset

This script takes a folder of manually collected + labeled photos from
Stembayo (kantin, UKS, bengkel) exported from Roboflow in YOLOv8 format
and merges them into the existing balanced_v3 training dataset.

Expected input structure (Roboflow YOLOv8 export):
    stembayo_custom_v1/
        train/  images/*.jpg   labels/*.txt
        valid/  images/        labels/
        test/   images/        labels/    (optional)

Usage (from project root, venv active):
    python scripts/merge_stembayo_custom.py --source path/to/stembayo_custom_v1
    python scripts/merge_stembayo_custom.py --source path/to/stembayo_custom_v1 --dry-run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"

CLASS_NAMES = {
    0: "plastic", 1: "metal", 2: "glass", 3: "e_waste",
    4: "medical_waste", 5: "paper", 6: "cardboard", 7: "organic_waste",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_label(label_path: Path, num_classes: int = 8) -> tuple[bool, str]:
    """Validate a YOLO label file. Returns (valid, message)."""
    try:
        with open(label_path, "r") as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    return False, f"Line {i}: expected 5+ fields, got {len(parts)}"
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= num_classes:
                    return False, f"Line {i}: class {cls_id} out of range [0, {num_classes-1}]"
                coords = [float(x) for x in parts[1:5]]
                for j, v in enumerate(coords):
                    if v < 0 or v > 1.01:  # small tolerance for rounding
                        return False, f"Line {i}: coordinate {j} = {v} out of [0, 1]"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Merge custom Stembayo photos into balanced_v3 dataset"
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Path to Stembayo custom dataset (Roboflow YOLOv8 export)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied without actually copying")
    parser.add_argument("--prefix", type=str, default="stembayo",
                        help="Filename prefix for merged files (default: stembayo)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        sys.exit(1)

    print(f"\n{'═'*62}")
    print(f"  Merge Stembayo Custom Data → balanced_v3")
    print(f"{'═'*62}")
    print(f"  Source   : {source}")
    print(f"  Target   : {DATASET_DIR}")
    print(f"  Prefix   : {args.prefix}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"{'═'*62}\n")

    if not DATASET_DIR.exists():
        print(f"WARNING: Target dataset dir does not exist: {DATASET_DIR}")
        print(f"  → Create it first or update DATASET_DIR in this script.")
        if not args.dry_run:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)

    stats = {"copied": 0, "skipped": 0, "invalid": 0, "missing_img": 0}
    class_counts = {v: 0 for v in CLASS_NAMES.values()}

    for split in ("train", "valid", "test"):
        src_img_dir = source / split / "images"
        src_lbl_dir = source / split / "labels"

        # Handle 'val' vs 'valid'
        if not src_img_dir.exists():
            alt = source / ("val" if split == "valid" else split)
            if (alt / "images").exists():
                src_img_dir = alt / "images"
                src_lbl_dir = alt / "labels"

        if not src_img_dir.exists():
            print(f"  ⏭  {split}/: not found in source — skipping")
            continue

        dst_img_dir = DATASET_DIR / split / "images"
        dst_lbl_dir = DATASET_DIR / split / "labels"

        if not args.dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in src_img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        print(f"  📂 {split}/: {len(images)} images found")

        for img_path in sorted(images):
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")

            if not lbl_path.exists():
                print(f"    ⚠  Missing label: {img_path.name}")
                stats["skipped"] += 1
                continue

            # Validate label
            valid, msg = validate_label(lbl_path)
            if not valid:
                print(f"    ❌ Invalid label {lbl_path.name}: {msg}")
                stats["invalid"] += 1
                continue

            # Track class distribution
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # Create prefixed filename to avoid collisions
            new_name = f"{args.prefix}_{img_path.name}"
            dst_img = dst_img_dir / new_name
            dst_lbl = dst_lbl_dir / f"{args.prefix}_{img_path.stem}.txt"

            if dst_img.exists():
                stats["skipped"] += 1
                continue

            if not args.dry_run:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

            stats["copied"] += 1

    # Report
    print(f"\n{'─'*62}")
    print(f"  Results {'(DRY RUN)' if args.dry_run else ''}:")
    print(f"    ✓ Copied  : {stats['copied']}")
    print(f"    ⏭ Skipped : {stats['skipped']} (missing label or already exists)")
    print(f"    ❌ Invalid : {stats['invalid']} (bad label format)")

    print(f"\n  Class distribution in source:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls_name:<18}: {count} annotations")

    if not args.dry_run and stats['copied'] > 0:
        print(f"\n  ✅ {stats['copied']} images merged into {DATASET_DIR}")
        print(f"\n  Next steps:")
        print(f"    1. Run: python scripts/RemoveDuplicate.py   (remove any duplicates)")
        print(f"    2. Run: python scripts/verify_dataset.py    (validate integrity)")
        print(f"    3. Run: python scripts/Classcount.py        (check balance)")
        print(f"    4. Retrain: python scripts/train_yolov8m.py --pretrained models/best.pt --epochs 30")

    print(f"\n{'═'*62}")


if __name__ == "__main__":
    main()

