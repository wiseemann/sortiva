"""Validation helpers for SORTIVA YOLO datasets.

The Kaggle Garbage Classification source contains image-level labels, not
object coordinates. A detector trained from those images can only be valid if
every object has a real bounding box annotation. This module catches the
common pseudo-label pattern before it reaches a training run.
"""

from pathlib import Path


def inspect_yolo_dataset(dataset_dir: Path, splits=("train", "valid", "test")) -> dict:
    """Return counts and suspicious full-frame label statistics."""
    result = {}
    for split in splits:
        label_dir = dataset_dir / split / "labels"
        image_dir = dataset_dir / split / "images"
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ] if image_dir.exists() else []

        total_boxes = 0
        full_frame_boxes = 0
        invalid_lines = 0
        for label_path in labels:
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    invalid_lines += 1
                    continue
                try:
                    _, cx, cy, width, height = map(float, parts)
                except ValueError:
                    invalid_lines += 1
                    continue
                total_boxes += 1
                if (
                    abs(cx - 0.5) <= 0.05
                    and abs(cy - 0.5) <= 0.05
                    and width >= 0.80
                    and height >= 0.80
                ):
                    full_frame_boxes += 1

        result[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": total_boxes,
            "full_frame_boxes": full_frame_boxes,
            "invalid_lines": invalid_lines,
        }
    return result


def validate_yolo_dataset(dataset_dir: Path, suspicious_ratio: float = 0.30) -> dict:
    """Raise a useful error when the detector dataset is missing or unsafe."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset_dir}\n"
            "Siapkan dataset YOLO beranotasi sebelum training."
        )

    stats = inspect_yolo_dataset(dataset_dir)
    train = stats["train"]
    if train["images"] == 0 or train["labels"] == 0 or train["boxes"] == 0:
        raise ValueError("Training split kosong atau tidak memiliki bounding box YOLO.")

    ratio = train["full_frame_boxes"] / train["boxes"]
    if ratio >= suspicious_ratio:
        raise ValueError(
            f"{ratio:.1%} label training berbentuk pseudo full-frame. "
            "Ini terlihat seperti label '0.5 0.5 0.85 0.85' dan tidak aman "
            "untuk object detection. Gunakan bounding box nyata dari Roboflow "
            "atau anotator manual."
        )

    if train["invalid_lines"]:
        raise ValueError(f"Ditemukan {train['invalid_lines']} baris label YOLO tidak valid.")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect SORTIVA YOLO labels")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    stats = inspect_yolo_dataset(args.dataset)
    for split, values in stats.items():
        print(f"{split}: {values}")
