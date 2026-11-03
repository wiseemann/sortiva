"""
scripts/evaluate_yolov8m.py
───────────────────────────
Generate per-class mAP@0.5 and mAP@0.5:0.95 report for YOLOv8m.

This fills a critical gap — the ResNet50 report exists but YOLOv8m
detection metrics (mAP, per-class AP) have never been explicitly
generated and saved.

Usage (from project root, venv active):
    python scripts/evaluate_yolov8m.py
    python scripts/evaluate_yolov8m.py --split test
    python scripts/evaluate_yolov8m.py --model models/best.pt --split valid
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "plastic", "metal", "glass", "e_waste",
    "medical_waste", "paper", "cardboard", "organic_waste",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8m — mAP Report")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.pt"),
                        help="Path to YOLO .pt weights file")
    parser.add_argument("--data",  type=str, default=str(CONFIG_DIR / "data_yolov8m.yaml"),
                        help="Path to data.yaml config")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid", "val"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    # ── imports ────────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"ERROR: {e}\nActivate the venv first.")
        sys.exit(1)

    # ── validate paths ─────────────────────────────────────────────────────────
    model_path = Path(args.model)
    data_path  = Path(args.data)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"ERROR: Data config not found: {data_path}")
        sys.exit(1)

    # ── GPU info ───────────────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    if device == 0:
        torch.cuda.empty_cache()

    print(f"\n{'═'*62}")
    print(f"  YOLOv8m Waste Detector — mAP Evaluation")
    print(f"{'═'*62}")
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_path}")
    print(f"  Split   : {args.split}")
    print(f"  Device  : {gpu_name}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"{'═'*62}\n")

    # ── load & evaluate ────────────────────────────────────────────────────────
    net = YOLO(str(model_path))
    metrics = net.val(
        data    = str(data_path),
        split   = args.split if args.split != "valid" else "val",
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = device,
        workers = 0,
        verbose = True,
        plots   = True,
    )

    box = metrics.box

    # ── per-class AP ───────────────────────────────────────────────────────────
    # metrics.box.ap50() → array of shape (num_classes,)
    ap50_per_class = box.ap50
    ap_per_class   = box.ap       # mAP@0.5:0.95 per class

    # ── build report ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"YOLOv8m Waste Detector — mAP Evaluation Report")
    lines.append(f"{'='*62}")
    lines.append(f"Generated  : {timestamp}")
    lines.append(f"Model      : {model_path.name}")
    lines.append(f"Split      : {args.split}")
    lines.append(f"Image Size : {args.imgsz}")
    lines.append(f"Device     : {gpu_name}")
    lines.append(f"")
    lines.append(f"{'─'*62}")
    lines.append(f"Overall Metrics:")
    lines.append(f"  mAP@0.5       : {box.map50*100:.2f}%")
    lines.append(f"  mAP@0.5:0.95  : {box.map*100:.2f}%")
    lines.append(f"  Precision (m) : {box.mp*100:.2f}%")
    lines.append(f"  Recall (m)    : {box.mr*100:.2f}%")
    lines.append(f"")
    lines.append(f"{'─'*62}")
    lines.append(f"Per-Class AP:")
    lines.append(f"")
    lines.append(f"{'Class':<18} {'AP@0.5':>10} {'AP@0.5:0.95':>14}")
    lines.append(f"{'-'*18} {'-'*10} {'-'*14}")

    # Get class names from model if available, fallback to ours
    model_names = net.names if hasattr(net, 'names') else {}

    for i in range(len(ap50_per_class)):
        name = model_names.get(i, CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}")
        a50  = ap50_per_class[i] * 100
        a95  = ap_per_class[i] * 100
        flag = " ← WEAK" if a50 < 85.0 else ""
        lines.append(f"{name:<18} {a50:>9.2f}% {a95:>13.2f}%{flag}")

    lines.append(f"")
    lines.append(f"{'='*62}")
    lines.append(f"NOTES:")
    lines.append(f"  - Classes with AP@0.5 < 85% are flagged as WEAK")
    lines.append(f"  - PR curves and confusion matrix saved to runs/detect/")
    lines.append(f"{'='*62}")

    report_text = "\n".join(lines)

    # ── save report ────────────────────────────────────────────────────────────
    report_path = REPORT_DIR / "yolov8m_map_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"\n{report_text}")
    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    main()

