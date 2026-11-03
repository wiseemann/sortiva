"""
scripts/finetune_stembayo.py
─────────────────────────────
Phase 3.1 — Fine-tune best.pt with custom Stembayo data

Two-stage fine-tuning strategy:
  Stage 1: Full model fine-tune with low LR on merged dataset (30 epochs)
  Stage 2: Freeze backbone, fine-tune head only on hard classes (20 epochs)

This is NOT training from scratch — it continues from the existing
best.pt checkpoint with a much lower learning rate, preserving all
existing knowledge while adapting to Stembayo-specific visual contexts.

Hardware: NVIDIA RTX 3050 6 GB (Ampere) — same constraints as train_yolov8m.py

Usage (from project root, venv active):
    python scripts/finetune_stembayo.py
    python scripts/finetune_stembayo.py --stage 1 --epochs 30
    python scripts/finetune_stembayo.py --stage 2 --epochs 20
    python scripts/finetune_stembayo.py --stage both
    python scripts/finetune_stembayo.py --stage both --promote
    python scripts/finetune_stembayo.py --pretrained models/best_baseline_v0.pt
"""

import argparse
import os
import shutil
import sys
import yaml
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"
CONFIG_DIR  = BASE_DIR / "config"
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "plastic", "metal", "glass", "e_waste",
    "medical_waste", "paper", "cardboard", "organic_waste",
]

try:
    from dataset_checks import validate_yolo_dataset
except ImportError:
    from scripts.dataset_checks import validate_yolo_dataset


def write_data_yaml() -> Path:
    """Write data.yaml with absolute paths."""
    yaml_path = CONFIG_DIR / "data_yolov8m.yaml"
    cfg = {
        "path":  str(DATASET_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def run_stage1(data_yaml: Path, epochs: int, device, promote: bool = False,
               pretrained_path: Path | None = None):
    """Stage 1: Full model fine-tune with low LR."""
    from ultralytics import YOLO

    print(f"\n{'═'*62}")
    print(f"  Stage 1 — Full Fine-Tune (lr=0.0001)")
    print(f"{'═'*62}")

    model_path = pretrained_path or MODEL_DIR / "best_baseline_v0.pt"
    if not model_path.exists():
        model_path = MODEL_DIR / "best.pt"
    if not model_path.exists():
        print(f"ERROR: pretrained model not found at {model_path}")
        print(f"  → Train a base model first with: python scripts/train_yolov8m.py")
        sys.exit(1)

    net = YOLO(str(model_path))
    net.train(
        data            = str(data_yaml),
        name            = "stembayo_finetune_s1",
        project         = str(BASE_DIR / "runs" / "detect"),
        # ── core ──────────────────────────────────────────────────────────
        imgsz           = 640,
        batch           = 8,
        epochs          = epochs,
        # ── hardware ──────────────────────────────────────────────────────
        device          = device,
        amp             = True,
        workers         = 0,
        cache           = False,
        # ── optimizer — low LR for fine-tuning ────────────────────────────
        optimizer       = "AdamW",
        lr0             = 0.0001,     # 10x lower than base training
        lrf             = 0.01,
        momentum        = 0.937,
        weight_decay    = 0.0005,
        # ── warmup ────────────────────────────────────────────────────────
        warmup_epochs   = 2,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.01,
        # ── augmentation — moderate ──────────────────────────────────────
        mosaic          = 1.0,
        mixup           = 0.1,
        copy_paste      = 0.3,        # paste paper objects onto diverse BGs
        degrees         = 10.0,
        translate       = 0.1,
        scale           = 0.5,
        fliplr          = 0.5,
        flipud          = 0.05,
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,
        # ── training stability ────────────────────────────────────────────
        close_mosaic    = 5,
        label_smoothing = 0.05,
        cls             = 0.7,        # increased class loss weight for paper
        cos_lr          = True,
        patience        = 15,
        # ── output ────────────────────────────────────────────────────────
        val             = True,
        plots           = True,
        save            = True,
        save_period     = -1,
        verbose         = True,
        exist_ok        = True,
    )

    # Copy results
    run_dir = BASE_DIR / "runs" / "detect" / "stembayo_finetune_s1"
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, MODEL_DIR / "best_s1.pt")
        print(f"\n  ✓ Stage 1 candidate → {MODEL_DIR / 'best_s1.pt'}")
        if promote:
            shutil.copy2(best_src, MODEL_DIR / "best.pt")
            print(f"  ✓ Promoted → {MODEL_DIR / 'best.pt'}")

    return run_dir


def run_stage2(data_yaml: Path, epochs: int, device, promote: bool = False):
    """Stage 2: Freeze backbone, fine-tune head only."""
    from ultralytics import YOLO

    print(f"\n{'═'*62}")
    print(f"  Stage 2 — Head-Only Fine-Tune (backbone frozen, lr=0.00005)")
    print(f"{'═'*62}")

    # Prefer the Stage 1 candidate when running both stages. The production
    # model is never replaced unless --promote is explicitly supplied.
    model_path = MODEL_DIR / "best_s1.pt"
    if not model_path.exists():
        model_path = MODEL_DIR / "best.pt"
    if not model_path.exists():
        print(f"ERROR: best.pt not found — run Stage 1 first")
        sys.exit(1)

    net = YOLO(str(model_path))
    net.train(
        data            = str(data_yaml),
        name            = "stembayo_finetune_s2",
        project         = str(BASE_DIR / "runs" / "detect"),
        # ── core ──────────────────────────────────────────────────────────
        imgsz           = 640,
        batch           = 8,
        epochs          = epochs,
        # ── hardware ──────────────────────────────────────────────────────
        device          = device,
        amp             = True,
        workers         = 0,
        cache           = False,
        # ── optimizer — very low LR ──────────────────────────────────────
        optimizer       = "AdamW",
        lr0             = 0.00005,    # 20x lower than base training
        lrf             = 0.01,
        weight_decay    = 0.0005,
        # ── freeze backbone layers ────────────────────────────────────────
        freeze           = 10,        # freeze first 10 layers (backbone)
        # ── minimal augmentation for precision ────────────────────────────
        mosaic          = 0.5,
        mixup           = 0.0,
        copy_paste      = 0.0,
        degrees         = 5.0,
        translate       = 0.05,
        scale           = 0.3,
        fliplr          = 0.5,
        flipud          = 0.0,
        hsv_h           = 0.01,
        hsv_s           = 0.4,
        hsv_v           = 0.3,
        # ── training stability ────────────────────────────────────────────
        close_mosaic    = 3,
        label_smoothing = 0.0,
        cls             = 0.7,
        cos_lr          = True,
        patience        = 10,
        # ── output ────────────────────────────────────────────────────────
        val             = True,
        plots           = True,
        save            = True,
        save_period     = -1,
        verbose         = True,
        exist_ok        = True,
    )

    # Copy results
    run_dir = BASE_DIR / "runs" / "detect" / "stembayo_finetune_s2"
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, MODEL_DIR / "best_s2.pt")
        print(f"\n  ✓ Stage 2 candidate → {MODEL_DIR / 'best_s2.pt'}")
        if promote:
            shutil.copy2(best_src, MODEL_DIR / "best.pt")
            print(f"  ✓ Promoted → {MODEL_DIR / 'best.pt'}")

    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8m for Stembayo")
    parser.add_argument("--stage",  type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Which fine-tuning stage to run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: 30 for s1, 20 for s2)")
    parser.add_argument("--promote", action="store_true",
                        help="Copy the final candidate over models/best.pt after training")
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="Base checkpoint; defaults to models/best_baseline_v0.pt when present")
    args = parser.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: {e}\nActivate the venv first.")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    print(f"\n  GPU: {gpu_name}")
    print(f"  Device: {device}")

    data_yaml = write_data_yaml()

    try:
        stats = validate_yolo_dataset(DATASET_DIR)
        print(f"  Dataset validation : OK ({stats['train']['boxes']} train boxes)")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.stage in ("1", "both"):
        epochs = args.epochs or 30
        run_stage1(data_yaml, epochs, device, args.promote, args.pretrained)

    if args.stage in ("2", "both"):
        epochs = args.epochs or 20
        run_stage2(data_yaml, epochs, device, args.promote)

    # Final evaluation
    print(f"\n{'═'*62}")
    print(f"  Fine-tuning complete!")
    print(f"{'═'*62}")
    print(f"  Run evaluation:")
    print(f"    python scripts/evaluate_yolov8m.py")
    print(f"  Start the app:")
    print(f"    streamlit run app.py")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()

