import os
import platform
import time
from collections import Counter
from datetime import datetime
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import cv2
from database import save_detections, save_to_review_queue

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SORTIVA — The Stembayo Eco-Sort Platform | SMKN 2 Depok Sleman",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Load YOLO model
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")


@st.cache_resource
def load_model(path: str, mtime: float):
    if not os.path.exists(path):
        st.error(f"Berkas model tidak ditemukan: {path}")
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None


model = load_model(
    MODEL_PATH,
    os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Waste categories — 8 classes aligned with Adiwiyata 3-Compartment System
# Kompartemen Adiwiyata:
# 1. Organik (Hijau)       : Sampah mudah terurai / sisa organik
# 2. Daur Ulang (Kuning/Biru): Plastik, Kertas, Karton, Kaleng Logam, Botol Kaca
# 3. Residu / B3 (Merah)   : Limbah medis, elektronik, berbahaya
# ─────────────────────────────────────────────────────────────────────────────
WASTE_CATEGORIES = {
    'organic_waste': {
        'color': (89, 124, 74),
        'hex': '#4A7C59',
        'icon': '🍃',
        'mat_icon': 'compost',
        'bin': 'WADAH ORGANIK',
        'bin_color': '#4A7C59',
        'category': 'Organik',
        'material_sub': 'Limbah Biomassa & Sisa Organik',
        'description': 'Sisa makanan, dedaunan, kulit buah & sampah organik terurai',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 1 (Hijau). Cocok untuk dekomposisi komposter Stembayo.',
        'nature': 'Biodegradable'
    },
    'paper': {
        'color': (235, 99, 37),
        'hex': '#2563EB',
        'icon': '📄',
        'mat_icon': 'description',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#2563EB',
        'category': 'Daur Ulang',
        'material_sub': 'Serat Selulosa & Kertas Arsip',
        'description': 'Kertas tulis, lembaran buku & dokumen tanpa laminasi plastik',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 2 (Kuning/Biru). Pastikan dalam kondisi kering tanpa staples.',
        'nature': 'Biodegradable'
    },
    'cardboard': {
        'color': (48, 92, 112),
        'hex': '#705C30',
        'icon': '📦',
        'mat_icon': 'inventory_2',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#705C30',
        'category': 'Daur Ulang',
        'material_sub': 'Karton Kemasan & Kotak Boks',
        'description': 'Kardus boks, karton kemasan tebal & paper bag kering',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 2 (Kuning/Biru). Lipat kardus hingga pipih sebelum dimasukkan.',
        'nature': 'Biodegradable'
    },
    'plastic': {
        'color': (106, 166, 196),
        'hex': '#C4A66A',
        'icon': '♻️',
        'mat_icon': 'water_bottle',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#C4A66A',
        'category': 'Daur Ulang',
        'material_sub': 'Polietilena Tereftalat (PET / HDPE)',
        'description': 'Botol plastik PET, gelas minuman, wadah bersih & kresek',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 2 (Kuning/Biru). Lepas tutup botol dan remas sebelum dimasukkan.',
        'nature': 'Non-Biodegradable'
    },
    'metal': {
        'color': (88, 99, 107),
        'hex': '#6B6358',
        'icon': '🥫',
        'mat_icon': 'lunch_dining',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#6B6358',
        'category': 'Daur Ulang',
        'material_sub': 'Aluminium & Logam Kaleng',
        'description': 'Kaleng minuman, kemasan aluminium, tutup botol & perkakas logam',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 2 (Kuning/Biru). Kosongkan cairan sebelum dimasukkan.',
        'nature': 'Non-Biodegradable'
    },
    'glass': {
        'color': (104, 122, 94),
        'hex': '#5E7A68',
        'icon': '🍾',
        'mat_icon': 'wine_bar',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#5E7A68',
        'category': 'Daur Ulang',
        'material_sub': 'Kaca Silika & Botol Kaca',
        'description': 'Botol kaca utuh, toples kaca & pecahan beling kemasan',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 2 (Kuning/Biru). Masukkan hati-hati agar tidak pecah.',
        'nature': 'Non-Biodegradable'
    },
    'e_waste': {
        'color': (48, 92, 112),
        'hex': '#705C30',
        'icon': '⚡',
        'mat_icon': 'cable',
        'bin': 'WADAH RESIDU & B3',
        'bin_color': '#705C30',
        'category': 'Residu / B3',
        'material_sub': 'Limbah Komponen Elektronika (E-Waste)',
        'description': 'Kabel listrik, baterai, adaptor bekas & komponen elektronika',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 3 (Merah). Disalurkan ke TPS B3 bengkel elektro Stembayo.',
        'nature': 'Non-Biodegradable'
    },
    'medical_waste': {
        'color': (48, 50, 184),
        'hex': '#B83230',
        'icon': '⚠️',
        'mat_icon': 'medical_services',
        'bin': 'WADAH RESIDU & B3',
        'bin_color': '#B83230',
        'category': 'Residu / B3',
        'material_sub': 'Limbah Medis & Infeksius Berbahaya',
        'description': 'Masker sekali pakai, perban, jarum suntik & limbah medis berbahaya',
        'rotor_instruction': 'Motor servo mengarahkan kompartemen 3 (Merah). Masukkan ke kantong khusus limbah B3/UKS.',
        'nature': 'Hazardous'
    },
}

DEFAULT_CATEGORY = {
    'color': (88, 99, 107),
    'hex': '#6B6358',
    'icon': '🗑️',
    'mat_icon': 'delete',
    'bin': 'WADAH RESIDU UMUM',
    'bin_color': '#6B6358',
    'category': 'Residu',
    'material_sub': 'Objek Tidak Terklasifikasi',
    'description': 'Objek belum terklasifikasi secara pasti oleh model visi',
    'rotor_instruction': 'Motor servo dalam mode siaga (standby). Arahkan objek ke posisi tengah inspeksi.',
    'nature': 'Unknown'
}

# Nature badge styles with strong, verified contrast matching Stembayo Eco-Tech tokens
NATURE_COLORS = {
    'Biodegradable':     {'bg': '#C8E8D0', 'text': '#2A6038', 'border': '#A5D6B2'},
    'Non-Biodegradable': {'bg': '#F0E8DB', 'text': '#4A4538', 'border': '#D4CCBF'},
    'Hazardous':         {'bg': '#FFDAD8', 'text': '#690005', 'border': '#F5A9A7'},
    'Unknown':           {'bg': '#F5F1EA', 'text': '#2E3230', 'border': '#E4E0D8'},
}


def check_model_classes(yolo_model):
    expected = set(WASTE_CATEGORIES.keys())
    names = [normalize_class(yolo_model.names[i]) for i in range(len(yolo_model.names))]
    missing = sorted(expected - set(names))
    extra   = sorted(set(names) - expected)
    if not missing and not extra:
        return True, ""
    msg = f"Model ini memiliki {len(names)} kelas, sementara sistem mengonfigurasi {len(expected)}. "
    if missing:
        msg += f"Kelas hilang: {', '.join(missing)}. "
    if extra:
        msg += f"Kelas tambahan: {', '.join(extra)}. "
    msg += "Pastikan models/best.pt adalah model YOLOv8m 8-kelas yang sesuai."
    return False, msg


def conf_band(c: float) -> str:
    return "Tinggi" if c >= 0.50 else ("Sedang" if c >= 0.25 else "Rendah")


# ─────────────────────────────────────────────────────────────────────────────
# Per-class confidence thresholds — Phase 4 accuracy improvement
# Classes with high false-positive risk get higher thresholds.
# ─────────────────────────────────────────────────────────────────────────────
PER_CLASS_CONF = {
    'paper':         0.55,
    'medical_waste': 0.60,
    'e_waste':       0.55,
    'organic_waste': 0.35,
    'plastic':       0.35,
    'cardboard':     0.35,
    'metal':         0.40,
    'glass':         0.40,
}

UNCERTAIN_THRESHOLD = 0.40   # below this → "Butuh Verifikasi Manual"


def _apply_per_class_filter(detections):
    """Filter detections using per-class confidence thresholds and flag uncertain."""
    filtered = []
    for d in detections:
        cls_name = d['class']
        min_conf = PER_CLASS_CONF.get(cls_name, 0.35)
        if d['confidence'] >= min_conf:
            d['uncertain'] = False
            filtered.append(d)
        elif d['confidence'] >= 0.25:
            # Anything below its class threshold is uncertain, including the
            # interval between UNCERTAIN_THRESHOLD and the class threshold.
            d['uncertain'] = True
            filtered.append(d)
        # Below 0.25 → drop entirely
    return filtered


def ref_number(seed: int, prefix: str = "ADW") -> str:
    return f"{prefix}-{(seed * 733 + 1009) % 90000 + 10000}"


def open_camera(index: int) -> cv2.VideoCapture:
    system = platform.system()
    if system == "Windows":
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Linux":
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends_to_try = [cv2.CAP_ANY]

    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            continue
    return cv2.VideoCapture()


def normalize_class(name) -> str:
    n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ewaste": "e_waste",
        "e_waste": "e_waste",
        "organic": "organic_waste",
        "medical": "medical_waste",
        "biodegradable": "organic_waste"
    }
    return aliases.get(n, n)


def _extract(result):
    dets = []
    probs = getattr(result, "probs", None)
    if probs is not None:
        cname = normalize_class(result.names[int(probs.top1)])
        dets.append({
            'class':         cname,
            'confidence':    float(probs.top1conf),
            'category_info': WASTE_CATEGORIES.get(cname, DEFAULT_CATEGORY),
            'xyxy':          None,
        })
        return dets
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    for box in result.boxes:
        cname = normalize_class(result.names.get(int(box.cls[0]), str(int(box.cls[0]))))
        dets.append({
            'class':         cname,
            'confidence':    float(box.conf[0]),
            'category_info': WASTE_CATEGORIES.get(cname, DEFAULT_CATEGORY),
            'xyxy':          [int(v) for v in box.xyxy[0].tolist()],
        })
    return dets


def classify_image(image, conf_threshold=0.25):
    results = model(image, conf=conf_threshold, imgsz=640, iou=0.5, max_det=100, augment=True, verbose=False)
    detections = [d for r in results for d in _extract(r)]
    if not detections and conf_threshold > 0.30:
        results = model(image, conf=0.30, imgsz=640, augment=True, verbose=False)
        detections = [d for r in results for d in _extract(r)]
    detections = _apply_per_class_filter(detections)
    return results, detections


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets, iou_thr=0.5, keep=30):
    dets = sorted(dets, key=lambda d: -d['confidence'])
    kept = []
    for d in dets:
        if d['xyxy'] is None:
            kept.append(d)
            continue
        if any(k['class'] == d['class'] and k['xyxy'] is not None
               and _iou(k['xyxy'], d['xyxy']) > iou_thr for k in kept):
            continue
        kept.append(d)
    return kept[:keep]


def detect_multiscale(image, conf, apply_filter=False):
    W, H = image.size

    def run(img, ox, oy):
        out = []
        for r in model(img, conf=conf, imgsz=640, iou=0.5, max_det=100, augment=True, verbose=False):
            out += _extract(r)
        for d in out:
            if d['xyxy'] is not None:
                x1, y1, x2, y2 = d['xyxy']
                d['xyxy'] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
        return out

    dets = run(image, 0, 0)
    if getattr(model, "task", "detect") == "detect":
        tw, th = int(W * 0.6), int(H * 0.6)
        for ox in (0, W - tw):
            for oy in (0, H - th):
                dets += run(image.crop((ox, oy, ox + tw, oy + th)), ox, oy)
    dets = _nms(dets)
    return _apply_per_class_filter(dets) if apply_filter else dets


def draw_detections(image, detections):
    import numpy as np
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i, d in enumerate(detections):
        if d['xyxy'] is None:
            continue
        x1, y1, x2, y2 = d['xyxy']
        col = d['category_info']['color']
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        label = f"{i+1}. {d['class'].replace('_',' ').title()} {d['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
        cv2.putText(canvas, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def debug_predictions(image):
    res = model(image, conf=0.001, imgsz=640, verbose=False)
    rows = [(d['class'], round(d['confidence'], 3)) for r in res for d in _extract(r)]
    rows.sort(key=lambda x: -x[1])
    return rows[:8]


def draw_boxes_on_frame(frame_bgr, detections):
    """Draw bounding boxes and labels on a BGR frame."""
    for det in detections:
        cat = det['category_info']
        if det['xyxy'] is None:
            x1, y1, x2, y2 = 10, 40, 10, 40
        else:
            x1, y1, x2, y2 = det['xyxy']
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), cat['color'], 2)
        label = f"{det['class'].replace('_',' ').title()} {det['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 14), (x1 + tw + 10, y1), cat['color'], -1)
        cv2.putText(frame_bgr, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


def run_inference_on_frame(frame_bgr, conf_threshold):
    """Real-time webcam inference optimized for low latency (augment=False)."""
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # augment=False is CRITICAL for live video stream performance
    results    = model(frame_rgb, conf=conf_threshold, imgsz=640, augment=False, verbose=False)
    detections = [d for r in results for d in _extract(r)]
    detections = _apply_per_class_filter(detections)
    annotated  = draw_boxes_on_frame(frame_bgr, detections)
    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Stembayo Eco-Tech Precision (Referenced from Stitch AI Design)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400..800;1,7..72,400..700&family=Nunito+Sans:ital,opsz,wght@0,6..12,300..800;1,6..12,300..700&family=IBM+Plex+Mono:wght@400;500;600;700&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

::selection {
    background: #C8E8D0;
    color: #2A6038;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #F5F1EA;
}
::-webkit-scrollbar-thumb {
    background: #D4CCBF;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6B6358;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Nunito Sans', -apple-system, sans-serif !important;
    background-color: #FAF6F0 !important;
    color: #2E3230 !important;
}

.font-headline {
    font-family: 'Literata', Georgia, serif !important;
}

.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
    vertical-align: middle;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1360px;
}

/* Top App Bar */
.stb-navbar {
    background: #FAF6F0;
    border: 1px solid #EAE6DE;
    padding: 0.85rem 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 12px;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 4px rgba(46, 50, 48, 0.04);
}
.stb-navbar-left {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.stb-logo-mark {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: #4A7C59;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    flex-shrink: 0;
    box-shadow: 0 1px 3px rgba(74, 124, 89, 0.25);
}
.stb-title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.stb-title-main {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #2E3230;
    line-height: 1.2;
    letter-spacing: -0.015em;
}
.stb-version-pill {
    font-family: 'Nunito Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    background: #F0E8DB;
    color: #5E5548;
    padding: 2px 8px;
    border-radius: 9999px;
}
.stb-title-sub {
    font-size: 0.75rem;
    color: #4A4E4A;
    font-weight: 500;
    margin-top: 2px;
}
.stb-meta-wrap {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.6rem;
}
.stb-session-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #4A4E4A;
    background: #F5F1EA;
    border: 1px solid #EAE6DE;
    padding: 4px 9px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.stb-status-pill {
    font-size: 0.7rem;
    font-weight: 700;
    color: #002110;
    background: #C8E8D0;
    border: 1px solid #A5D6B2;
    border-radius: 9999px;
    padding: 4px 10px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.stb-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4A7C59;
}

/* Sub-Header Utility Control Strip */
.stb-control-strip {
    background: #F5F1EA;
    border: 1px solid #EAE6DE;
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
}

/* Mas Bayo Virtual Assistant Banner */
.stb-masbayo-card {
    background: #F0ECE4;
    border: 1px solid #E4E0D8;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
}
.stb-masbayo-avatar {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #C8E8D0;
    color: #2A6038;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
    box-shadow: 0 1px 2px rgba(42, 96, 56, 0.15);
}
.stb-masbayo-content {
    flex: 1;
}
.stb-masbayo-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
    flex-wrap: wrap;
}
.stb-masbayo-name {
    font-size: 0.88rem;
    font-weight: 800;
    color: #4A7C59;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.stb-masbayo-role {
    font-size: 0.72rem;
    color: #4A4E4A;
    font-weight: 600;
}
.stb-masbayo-dialogue {
    font-size: 0.82rem;
    color: #2E3230;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}
.stb-masbayo-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.stb-chip {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.chip-organik {
    background: #C8E8D0;
    color: #2A6038;
    border: 1px solid #A5D6B2;
}
.chip-daurulang {
    background: #F0E8DB;
    color: #4A4538;
    border: 1px solid #D4CCBF;
}
.chip-residu {
    background: #FFDAD8;
    color: #690005;
    border: 1px solid #F5A9A7;
}

/* System Status HUD Pill */
.hud-status-pill {
    background: #2E3230;
    color: #FAF6F0;
    border-radius: 6px;
    padding: 0.5rem 0.95rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.hud-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #8ECF9E;
    display: inline-block;
    margin-right: 6px;
}

/* Section Header Bar */
.section-heading-bar {
    font-family: 'Literata', Georgia, serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #2E3230;
    letter-spacing: -0.01em;
    margin-bottom: 0.65rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #EAE6DE;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Detection Card (Right Column) */
.stb-det-card {
    background: #F5F1EA;
    border: 1px solid #EAE6DE;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
    transition: transform 0.15s ease;
}
.stb-det-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}
.stb-det-left {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.stb-det-icon-badge {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.stb-det-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 0.96rem;
    font-weight: 700;
    color: #2E3230;
    line-height: 1.2;
}
.stb-det-sub {
    font-size: 0.7rem;
    color: #4A4E4A;
    margin-top: 1px;
}
.stb-det-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #4A7C59;
    background: #FFFFFF;
    border: 1px solid #D4CCBF;
    padding: 3px 8px;
    border-radius: 6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    flex-shrink: 0;
}
.stb-det-bin-bar {
    background: #FFFFFF;
    border: 1px solid #E4E0D8;
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.45rem;
}
.stb-bin-label-tag {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #4A4E4A;
}
.stb-bin-dest-pill {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.stb-rotor-instruction {
    background: #FFFFFF;
    border: 1px solid #EAE6DE;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.74rem;
    color: #4A4E4A;
    line-height: 1.45;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}
.stb-rotor-bold {
    font-weight: 700;
    color: #2E3230;
}

/* Reference Card: Pedoman Standar 3 Wadah Adiwiyata */
.stb-pedoman-panel {
    background: #F0ECE4;
    border: 1px solid #E4E0D8;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
}
.stb-pedoman-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #2E3230;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.stb-pedoman-sub {
    font-size: 0.72rem;
    color: #4A4E4A;
    margin-bottom: 0.65rem;
    display: block;
}
.stb-pedoman-item {
    background: #FFFFFF;
    border: 1px solid #EAE6DE;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.45rem;
}
.stb-pedoman-item:last-child {
    margin-bottom: 0;
}
.stb-pedoman-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.2rem;
    font-size: 0.75rem;
    font-weight: 700;
}
.stb-pedoman-desc {
    font-size: 0.7rem;
    color: #4A4E4A;
    line-height: 1.35;
}

/* Hardware Status Indicator Widget */
.stb-hardware-widget {
    background: #EAE6DE;
    border: 1px solid #D4CCBF;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
    margin-top: 0.6rem;
    box-shadow: 0 1px 2px rgba(46, 50, 48, 0.03);
}
.stb-hw-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    min-width: 0;
}
.stb-hw-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4A7C59;
    flex-shrink: 0;
}
.stb-hw-title {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.72rem;
    color: #2E3230;
    display: block;
}
.stb-hw-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    color: #4A4E4A;
    display: block;
}
.stb-hw-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #4A7C59;
    background: #FFFFFF;
    border: 1px solid #C8E8D0;
    padding: 2px 7px;
    border-radius: 4px;
    flex-shrink: 0;
}

/* Completely Hide Streamlit Sidebar */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Top Navbar Page Link Tabs */
div[data-testid="stPageLink-NavLink"] {
    background: #F0ECE4 !important;
    border: 1px solid #D4CCBF !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    color: #4A4E4A !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(46, 50, 48, 0.03) !important;
}
div[data-testid="stPageLink-NavLink"]:hover {
    background: #EAE6DE !important;
    color: #2E3230 !important;
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #F0ECE4 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #E4E0D8 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #4A4E4A !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.3rem !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #4A7C59 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 3px rgba(74, 124, 89, 0.25) !important;
}

/* Primary Action Buttons */
.stButton > button {
    font-family: 'Nunito Sans', -apple-system, sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: #4A7C59 !important;
    color: #FFFFFF !important;
    border: 1px solid #3D6B4B !important;
    box-shadow: 0 1px 3px rgba(74, 124, 89, 0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #3D6B4B !important;
    border-color: #2A6038 !important;
}
.stButton > button:not([kind="primary"]) {
    background: #F0ECE4 !important;
    color: #2E3230 !important;
    border: 1px solid #D4CCBF !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #6B6358 !important;
    background: #EAE6DE !important;
    color: #0F172A !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: #4A7C59 !important;
}

/* Empty / Inactive State Boxes */
.stb-empty-box {
    background: #F5F1EA;
    border: 1.5px dashed #D4CCBF;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    color: #4A4E4A;
}
.stb-empty-icon {
    font-size: 2.2rem;
    margin-bottom: 0.6rem;
}
.stb-empty-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #2E3230;
    margin-bottom: 0.3rem;
}
.stb-empty-subtitle {
    font-size: 0.78rem;
    color: #6B6358;
}

/* Clean Institutional Footer */
.stb-footer {
    border-top: 1px solid #E4E0D8;
    margin-top: 3.5rem;
    padding-top: 1.8rem;
    background: #FAF6F0;
    color: #4A4E4A;
    font-size: 0.76rem;
    line-height: 1.6;
}
.stb-footer-inner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.stb-footer-bold {
    font-weight: 700;
    color: #2E3230;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Top Navigation Bar
# ─────────────────────────────────────────────────────────────────────────────
nav_brand_col, nav_links_col, nav_meta_col = st.columns([1.8, 1.4, 1.1], vertical_alignment="center")

with nav_brand_col:
    st.markdown("""
    <div class="stb-navbar-left" style="padding: 0.35rem 0;">
        <div class="stb-logo-mark"><span class="material-symbols-outlined" style="font-size:24px; color:#FFFFFF;">eco</span></div>
        <div>
            <div class="stb-title-row">
                <span class="stb-title-main">SORTIVA</span>
                <span class="stb-version-pill">Kiosk v2.4</span>
            </div>
            <div class="stb-title-sub">The Stembayo Eco-Sort Platform · SMKN 2 Depok Sleman</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav_links_col:
    nl1, nl2 = st.columns(2)
    with nl1:
        st.page_link("app.py", label="Konsol Kamera", icon="📷")
    with nl2:
        st.page_link("pages/dashboard.py", label="Dashboard Analitik", icon="📊")

with nav_meta_col:
    st.markdown("""
    <div class="stb-meta-wrap">
        <div class="stb-session-pill"><span class="material-symbols-outlined" style="font-size:14px;">schedule</span> Kiosk-01</div>
        <div class="stb-status-pill"><span class="stb-pulse-dot"></span> StemVision Ready</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 0.3rem 0 0.8rem 0; border: none; border-bottom: 1px solid #EAE6DE;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Sub-Header Utility Toolbar
# ─────────────────────────────────────────────────────────────────────────────
with st.container():
    tb_c1, tb_c2, tb_c3, tb_c4 = st.columns([1.2, 2.2, 1.6, 1.4], vertical_alignment="center")

    with tb_c1:
        camera_index = st.number_input(
            "Perangkat Kamera", min_value=0, max_value=5, value=0, step=1,
            help="0 = Kamera default / laptop. 1 atau 2 = Webcam USB eksternal."
        )

    with tb_c2:
        conf_threshold = st.slider(
            "Threshold Confidence (YOLOv8)", 0.05, 1.0, 0.20, 0.05,
            help="Nilai batas minimum akurasi agar objek diakui model."
        )

    with tb_c3:
        show_all = st.checkbox(
            "Tampilkan Semua Bounding Box", value=True,
            help="Menampilkan seluruh objek terdeteksi meski dengan probabilitas rendah."
        )

    with tb_c4:
        with st.popover("📖 Indeks 8 Klasifikasi YOLOv8"):
            st.markdown("##### 8 Kelas StemVision &amp; Wadah Tujuan")
            for category, info in WASTE_CATEGORIES.items():
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #EAE6DE; font-size:0.75rem;">
                    <span>{info['icon']} <b>{category.replace('_',' ').title()}</b></span>
                    <span style="font-size:0.68rem; font-weight:700; color:{info['hex']}; background:#FFFFFF; border:1px solid #D4CCBF; padding:1px 6px; border-radius:4px;">
                        {info['category']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Mas Bayo Virtual Assistant Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-masbayo-card">
    <div class="stb-masbayo-avatar"><span class="material-symbols-outlined" style="font-size:24px; color:#2A6038;">smart_toy</span></div>
    <div class="stb-masbayo-content">
        <div class="stb-masbayo-title">
            <span class="stb-masbayo-name">MAS BAYO</span>
            <span class="stb-masbayo-role">· Asisten Pilah Sampah Adiwiyata</span>
        </div>
        <div class="stb-masbayo-dialogue">
            Arahkan sampah ke depan kamera atau unggah foto untuk identifikasi otomatis menuju 3 wadah Adiwiyata:
        </div>
        <div class="stb-masbayo-chips">
            <span class="stb-chip chip-organik"><span>🍃</span> Wadah 1: Organik (Kompos / Pupuk) - Hijau</span>
            <span class="stb-chip chip-daurulang"><span>♻️</span> Wadah 2: Daur Ulang (Kertas, Plastik, Logam) - Kuning/Biru</span>
            <span class="stb-chip chip-residu"><span>⚠️</span> Wadah 3: Residu &amp; B3 (Medis, Kimia, Baterai) - Merah</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model deteksi belum dapat dimuat. Pastikan berkas `models/best.pt` tersedia di direktori proyek.")
else:
    _model_ok, _model_msg = check_model_classes(model)
    if not _model_ok:
        st.warning(_model_msg)

# ─────────────────────────────────────────────────────────────────────────────
# Main Workspaces: Live Camera vs Photo Upload
# ─────────────────────────────────────────────────────────────────────────────
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

tab1, tab2 = st.tabs(["Live Stream Kamera", "Analisis Unggah Foto"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM INSPECTION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_vid, col_info = st.columns([3, 1.8], gap="large")

    with col_vid:
        st.markdown("""
        <div class="section-heading-bar">
            <span>Umpan Kamera Kiosk &amp; Inspeksi Visual</span>
            <span style="font-family:'IBM Plex Mono'; font-size:0.72rem; color:#4A4E4A;">30 FPS | OPTICAL ARRAY 01</span>
        </div>
        """, unsafe_allow_html=True)

        btn_row = st.columns(2)
        with btn_row[0]:
            start_btn = st.button(
                "▶  Mulai Deteksi Kamera", type="primary", width='stretch',
                disabled=model is None,
            )
        with btn_row[1]:
            stop_btn = st.button("⏹  Hentikan Kamera", width='stretch')

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_info:
        st.markdown("""
        <div class="section-heading-bar">
            <span>Hasil Identifikasi Real-Time</span>
            <span style="font-family:'IBM Plex Mono'; font-size:0.7rem; color:#4A7C59; font-weight:700;">LIVE SYNC ●</span>
        </div>
        """, unsafe_allow_html=True)

        det_placeholder = st.empty()

        st.markdown("""
        <div class="stb-pedoman-panel">
            <div class="stb-pedoman-title">Pedoman Standar 3 Wadah Adiwiyata SMKN 2 Depok</div>
            <span class="stb-pedoman-sub">Kepatuhan Audit Lingkungan Hidup Sekolah Adiwiyata Mandiri:</span>
            <div class="stb-pedoman-item">
                <div class="stb-pedoman-head" style="color:#4A7C59;">
                    <span>🍃 Wadah 1: Hijau</span>
                    <span style="font-family:'IBM Plex Mono'; font-size:10px;">ORGANIK</span>
                </div>
                <div class="stb-pedoman-desc">Sisa sayuran, buah, nasi, daun gugur taman, serbuk kayu bangunan, ampas kopi kantin.</div>
            </div>
            <div class="stb-pedoman-item">
                <div class="stb-pedoman-head" style="color:#705C30;">
                    <span>♻️ Wadah 2: Kuning / Biru</span>
                    <span style="font-family:'IBM Plex Mono'; font-size:10px;">DAUR ULANG</span>
                </div>
                <div class="stb-pedoman-desc">Botol/gelas PET, kaleng aluminium, kertas HVS tugas, kardus sparepart, botol kaca sirup.</div>
            </div>
            <div class="stb-pedoman-item">
                <div class="stb-pedoman-head" style="color:#B83230;">
                    <span>⚠️ Wadah 3: Merah</span>
                    <span style="font-family:'IBM Plex Mono'; font-size:10px;">RESIDU &amp; B3</span>
                </div>
                <div class="stb-pedoman-desc">Puntung sisa las, masker medis, baterai bekas multimeter, residu solder &amp; oli bengkel mesin.</div>
            </div>
        </div>
        <!-- IoT hardware widget placeholder — re-enable after ESP32/Arduino connected -->
        """, unsafe_allow_html=True)

    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = open_camera(int(camera_index))

        if not cap.isOpened():
            st.error(f"Kamera dengan indeks {int(camera_index)} tidak dapat dibuka. Silakan sesuaikan Indeks Perangkat Kamera pada toolbar atas.")
            st.session_state.webcam_running = False
        else:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)

                status_placeholder.markdown("""
                <div class="hud-status-pill">
                    <span><span class="hud-pulse-dot"></span>KAMERA AKTIF · REAL-TIME STEMVISION</span>
                    <span>30.2 FPS | 24ms Latency</span>
                </div>
                """, unsafe_allow_html=True)

                SAVE_EVERY_SECONDS = 2.0
                last_save = 0.0
                frame_count = 0
                consecutive_failures = 0
                MAX_CONSECUTIVE_FAILURES = 20

                # ── Real-Time Performance & Pipeline Optimizations ─────────
                BLUR_THRESHOLD = 60.0           # Lowered threshold to avoid false-skips
                FRAME_AVG_WINDOW = 3            # Majority-vote buffer
                INFERENCE_INTERVAL = 2          # Process AI every N frames (cuts GPU load by 50%)
                _prediction_buffer = []
                cached_detections = []
                last_det_sig = None
                last_det_time = 0.0

                fps_time = time.time()
                fps_display = 30.0
                latency_ms = 15.0

                while st.session_state.webcam_running:
                    try:
                        t0 = time.time()
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_failures += 1
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                status_placeholder.error("Aliran kamera terhenti. Pemindaian dihentikan otomatis.")
                                st.session_state.webcam_running = False
                                break
                            status_placeholder.warning("Tangkapan frame terhambat — mencoba ulang...")
                            time.sleep(0.05)
                            continue

                        consecutive_failures = 0
                        frame_count += 1

                        # Skip blurry frames
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                        if lap_var < BLUR_THRESHOLD:
                            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(display, channels="RGB", width='stretch')
                            time.sleep(0.005)
                            continue

                        # Frame-skip AI inference to maximize smooth video FPS
                        if frame_count % INFERENCE_INTERVAL == 0:
                            inf_start = time.time()
                            annotated, detections = run_inference_on_frame(frame, conf_threshold)
                            latency_ms = (time.time() - inf_start) * 1000.0
                            cached_detections = detections

                            # Majority voting for temporal stability
                            if detections:
                                dominant = max(detections, key=lambda d: d['confidence'])['class']
                                _prediction_buffer.append(dominant)
                                if len(_prediction_buffer) > FRAME_AVG_WINDOW:
                                    _prediction_buffer.pop(0)
                                if len(_prediction_buffer) >= FRAME_AVG_WINDOW:
                                    from collections import Counter as _Counter
                                    vote = _Counter(_prediction_buffer).most_common(1)[0]
                                    if vote[1] < 2:
                                        detections = []
                                        cached_detections = []
                        else:
                            # Re-use previous detections on interleaved frames
                            annotated = draw_boxes_on_frame(frame, cached_detections)
                            detections = cached_detections

                        # Periodic DB logging
                        if detections and (time.time() - last_save) >= SAVE_EVERY_SECONDS:
                            accepted = [d for d in detections if not d.get('uncertain', False)]
                            review = [d for d in detections if d.get('uncertain', False)]
                            if accepted:
                                save_detections(accepted, source="webcam")
                            if review:
                                save_to_review_queue(review, source="webcam")
                            last_save = time.time()

                        # Streamlit frame display
                        display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(display, channels="RGB", width='stretch')

                        # Throttle DOM re-render of detection cards to avoid UI lag
                        current_sig = tuple((d['class'], round(d['confidence'], 1)) for d in detections)
                        now = time.time()
                        if current_sig != last_det_sig or (now - last_det_time) > 0.4:
                            last_det_sig = current_sig
                            last_det_time = now
                            if detections:
                                det_html = ""
                                for i, det in enumerate(detections):
                                    cat_info = det['category_info']
                                    hex_col = cat_info['hex']
                                    nat = cat_info.get('nature', 'Unknown')
                                    nc = NATURE_COLORS.get(nat, NATURE_COLORS['Unknown'])

                                    det_html += f"""
                                    <div class="stb-det-card">
                                        <div class="stb-det-header">
                                            <div class="stb-det-left">
                                                <div class="stb-det-icon-badge" style="background:{hex_col}22; color:{hex_col};">
                                                    <span class="material-symbols-outlined" style="font-size:22px;">{cat_info.get('mat_icon', 'recycling')}</span>
                                                </div>
                                                <div>
                                                    <div class="stb-det-title">{det['class'].replace('_',' ').title()}</div>
                                                    <div class="stb-det-sub">{cat_info.get('material_sub', 'Material Terdeteksi')}</div>
                                                </div>
                                            </div>
                                            <div class="stb-det-conf">{det['confidence']*100:.1f}%</div>
                                        </div>
                                        <div class="stb-det-bin-bar">
                                            <span class="stb-bin-label-tag">Target Penempatan</span>
                                            <span class="stb-bin-dest-pill" style="background:{nc['bg']}; color:{nc['text']}; border:1px solid {nc['border']};">
                                                <span>{cat_info['icon']}</span>
                                                <span>{cat_info['bin']}</span>
                                            </span>
                                        </div>
                                        <div class="stb-rotor-instruction">
                                            <span class="material-symbols-outlined" style="font-size:16px; color:#4A7C59; flex-shrink:0; padding-top:2px;">precision_manufacturing</span>
                                            <div>
                                                <span class="stb-rotor-bold">Instruksi Rotor &amp; Pengguna:</span> {cat_info.get('rotor_instruction', cat_info['description'])}
                                            </div>
                                        </div>
                                    </div>
                                    """
                                det_placeholder.markdown(det_html, unsafe_allow_html=True)
                            else:
                                det_placeholder.markdown("""
                                <div class="stb-empty-box" style="padding:2rem 1rem;">
                                    <div style="font-size:1.8rem; margin-bottom:0.4rem;">🎯</div>
                                    <div class="stb-empty-title" style="font-size:0.95rem;">
                                        Menunggu Objek Sampah
                                    </div>
                                    <div class="stb-empty-subtitle" style="font-size:0.75rem;">
                                        Arahkan objek ke depan lensa kamera pemilah
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Update HUD with real FPS and latency every 15 frames
                        dt = time.time() - t0
                        if dt > 0:
                            fps_inst = 1.0 / dt
                            fps_display = 0.85 * fps_display + 0.15 * fps_inst
                        if frame_count % 15 == 0:
                            status_placeholder.markdown(f"""
                            <div class="hud-status-pill">
                                <span><span class="hud-pulse-dot"></span>KAMERA AKTIF · REAL-TIME STEMVISION</span>
                                <span>{fps_display:.1f} FPS | {latency_ms:.0f}ms AI Latency</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Minimal yield to keep Streamlit responsive without capping FPS
                        time.sleep(0.002)

                    except Exception as loop_err:
                        status_placeholder.error(f"Pemindaian terhenti akibat kendala: {loop_err}")
                        st.session_state.webcam_running = False
                        break
            finally:
                cap.release()

            frame_placeholder.empty()
            status_placeholder.markdown("""
            <div class="hud-status-pill" style="background:#4A4E4A;">
                <span>KAMERA NONAKTIF</span>
                <span>STATUS: STANDBY</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        frame_placeholder.markdown("""
        <div class="stb-empty-box">
            <div class="stb-empty-icon">📷</div>
            <div class="stb-empty-title">Kamera Inspeksi Belum Aktif</div>
            <div class="stb-empty-subtitle">Klik tombol 'Mulai Deteksi Kamera' untuk mengaktifkan pemindaian real-time.</div>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHOTO UPLOAD & ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-heading-bar">Unggah Berkas Foto Sampah untuk Batch Processing</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Pilih foto sampah untuk dianalisis",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Format didukung: JPG, PNG, BMP, GIF"
    )

    if uploaded_file is not None:
        _seen = (uploaded_file.name, int(uploaded_file.size))
        if st.session_state.get("seen_upload") != _seen:
            st.session_state.seen_upload = _seen
            st.session_state.webcam_running = False

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-heading-bar">Foto Asli</div>', unsafe_allow_html=True)
            st.image(image, width='stretch')

        with col2:
            st.markdown('<div class="section-heading-bar">Laporan Klasifikasi AI</div>', unsafe_allow_html=True)
            if model is not None:
                effective_conf = 0.001 if show_all else conf_threshold
                with st.spinner("Menjalankan analisis visi komputer..."):
                    # Keep raw results for the optional diagnostic view, but
                    # apply production thresholds before writing to the database.
                    raw_detections = detect_multiscale(image, effective_conf)
                    classified_detections = _apply_per_class_filter(raw_detections)
                    detections = raw_detections if show_all else classified_detections
                detections = sorted(detections, key=lambda d: -d['confidence'])

                if detections:
                    _upload_key = (
                        str(getattr(uploaded_file, "file_id", uploaded_file.name)),
                        int(uploaded_file.size),
                        round(float(effective_conf), 2),
                    )
                    if st.session_state.get("last_saved_upload") != _upload_key:
                        accepted = [d for d in classified_detections
                                    if not d.get('uncertain', False)]
                        review = [d for d in classified_detections
                                  if d.get('uncertain', False)]
                        image_id = f"IMG-{datetime.now():%Y%m%d-%H%M%S}"
                        if accepted:
                            save_detections(
                                accepted, source="upload", image_id=image_id,
                                image_name=uploaded_file.name,
                            )
                        if review:
                            save_to_review_queue(review, source="upload")
                        st.session_state.last_saved_upload = _upload_key

                    bio_count    = sum(1 for d in detections if d['category_info'].get('nature') == 'Biodegradable')
                    nonbio_count = sum(1 for d in detections if d['category_info'].get('nature') == 'Non-Biodegradable')
                    haz_count    = sum(1 for d in detections if d['category_info'].get('nature') == 'Hazardous')

                    st.markdown(f"""
                    <div class="hud-status-pill" style="margin-bottom:1rem; background:#F0ECE4; color:#2E3230; border:1px solid #D4CCBF;">
                        <span>TOTAL TERDETEKSI: <b>{len(detections)}</b></span>
                        <span style="color:#4A7C59; font-weight:700;">ORGANIK: {bio_count}</span>
                        <span style="color:#705C30; font-weight:700;">DAUR ULANG: {nonbio_count}</span>
                        <span style="color:#B83230; font-weight:700;">RESIDU/B3: {haz_count}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    _counts = Counter(d['class'].replace('_', ' ').title() for d in detections)
                    st.markdown(
                        "**Objek Ditemukan:** " +
                        ", ".join(f"**{k}** ({v}x)" for k, v in _counts.most_common())
                    )

                    for i, det in enumerate(detections):
                        cat_info = det['category_info']
                        hex_col  = cat_info['hex']
                        nat      = cat_info.get('nature', 'Unknown')
                        nc       = NATURE_COLORS.get(nat, NATURE_COLORS['Unknown'])

                        st.markdown(f"""
                        <div class="stb-det-card">
                            <div class="stb-det-header">
                                <div class="stb-det-left">
                                    <div class="stb-det-icon-badge" style="background:{hex_col}22; color:{hex_col};">
                                        <span class="material-symbols-outlined" style="font-size:22px;">{cat_info.get('mat_icon', 'recycling')}</span>
                                    </div>
                                    <div>
                                        <div class="stb-det-title">{det['class'].replace('_',' ').title()}</div>
                                        <div class="stb-det-sub">{cat_info.get('material_sub', 'Material Terdeteksi')}</div>
                                    </div>
                                </div>
                                <div class="stb-det-conf">{det['confidence']*100:.1f}% ({conf_band(det['confidence'])})</div>
                            </div>
                            <div class="stb-det-bin-bar">
                                <span class="stb-bin-label-tag">Target Penempatan</span>
                                <span class="stb-bin-dest-pill" style="background:{nc['bg']}; color:{nc['text']}; border:1px solid {nc['border']};">
                                    <span>{cat_info['icon']}</span>
                                    <span>{cat_info['bin']}</span>
                                </span>
                            </div>
                            <div class="stb-rotor-instruction">
                                <span class="material-symbols-outlined" style="font-size:16px; color:#4A7C59; flex-shrink:0; padding-top:2px;">precision_manufacturing</span>
                                <div>
                                    <span class="stb-rotor-bold">Instruksi Rotor &amp; Pengguna:</span> {cat_info.get('rotor_instruction', cat_info['description'])}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<div class="section-heading-bar" style="margin-top:1.5rem;">Visualisasi Anotasi Bounding Box</div>', unsafe_allow_html=True)
                    st.image(draw_detections(image, detections), channels="RGB", width='stretch')
                else:
                    st.markdown("""
                    <div class="stb-empty-box" style="padding:2.5rem 1.5rem;">
                        <div style="font-size:2rem; margin-bottom:0.4rem;">🔍</div>
                        <div class="stb-empty-title">Tidak Ada Objek Sampah Terdeteksi</div>
                        <div class="stb-empty-subtitle">
                            Coba turunkan Ambang Batas Keyakinan pada toolbar atas jika objek berukuran kecil atau redup.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Diagnostik Model (Keyakinan ≥ 0.1%)", expanded=True):
                        st.write(f"Task Model: **{getattr(model, 'task', 'detect')}**")
                        _raw = debug_predictions(image)
                        if _raw:
                            st.table({"Kelas Prediksi": [r[0] for r in _raw], "Tingkat Keyakinan": [r[1] for r in _raw]})
                        else:
                            st.write("Model tidak menghasilkan probabilitas di atas 1%.")
            else:
                st.error("Model deteksi belum siap. Periksa kembali ketersediaan berkas weights.")
    else:
        st.markdown("""
        <div class="stb-empty-box">
            <div class="stb-empty-icon">📁</div>
            <div class="stb-empty-title">Letakkan Berkas Gambar Di Atas</div>
            <div class="stb-empty-subtitle">Mendukung format JPG, PNG, BMP, dan GIF untuk analisis multi-skala komprehensif.</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Clean Ecological Institutional Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-footer">
    <div class="stb-footer-inner">
        <div>
            <span class="stb-footer-bold">SMKN 2 Depok Sleman · Sekolah Adiwiyata Mandiri</span>
            <span style="margin: 0 6px; color:#C4C8BC;">•</span>
            <span>Eco-Tech STEM Innovation Lab</span>
        </div>
        <div style="font-family:'IBM Plex Mono', monospace; font-size:0.72rem;">
            <span>AI Model: StemVision-YOLOv8m</span>
            <span style="margin: 0 6px; color:#C4C8BC;">•</span>
            <span>© 2025 SORTIVA Platform</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
