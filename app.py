import os
import re
import platform
import time
from collections import Counter
from datetime import datetime
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import cv2
from database import save_detections, save_to_review_queue
from sortiva_ui import render_navbar

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SORTIVA — The Stembayo Eco-Sort Platform | SMKN 2 Depok Sleman",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def clean_html(html_str: str) -> str:
    """Strip HTML comments, leading/trailing whitespace from each line, and remove blank lines
    to prevent CommonMark from accidentally triggering 4-space indented code blocks."""
    if not html_str:
        return ""
    no_comments = re.sub(r'<!--.*?-->', '', str(html_str), flags=re.DOTALL)
    return "\n".join(line.strip() for line in no_comments.splitlines() if line.strip())


_ICON_PATHS = {
    "default": '<circle cx="12" cy="12" r="8"/>',
    "eco": '<path d="M20.5 3.5C11.2 3.9 5.2 8.1 4 19.8c11.7-1.2 15.9-7.2 16.5-16.3Z"/><path d="M4.5 19.5c2.8-4.1 6.4-6.9 10.9-9.4"/>',
    "image_search": '<rect x="3" y="4.5" width="14" height="13" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m5.5 15 3.2-3 2.2 2 1.8-1.7"/><circle cx="17.5" cy="17.5" r="3.5"/><path d="m20.2 20.2 2 2"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16.5 9"/>',
    "photo_camera": '<path d="M4 7h3l1.2-2h7.6L17 7h3a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z"/><circle cx="12" cy="13" r="3.5"/>',
    "upload_file": '<path d="M6 3h8l4 4v14H6zM14 3v5h5"/><path d="M12 17V10M9.5 12.5 12 10l2.5 2.5"/>',
    "category": '<path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z"/><path d="m4 12 8 4.5 8-4.5M4 16.5l8 4.5 8-4.5"/>',
    "smart_toy": '<rect x="4" y="7" width="16" height="12" rx="3"/><path d="M12 7V4M9 12h.01M15 12h.01M8 16h8"/>',
    "psychology": '<path d="M9.5 5.5a3 3 0 0 0-5.3 2A3 3 0 0 0 5 13a3 3 0 0 0 2.5 4.5H10v-3M14.5 5.5a3 3 0 0 1 5.3 2A3 3 0 0 1 19 13a3 3 0 0 1-2.5 4.5H14v-3M10 9h4M10 13h4"/>',
    "school": '<path d="m3 9 9-5 9 5-9 5-9-5Z"/><path d="M7 11.5V16c2.8 2.4 7.2 2.4 10 0v-4.5M21 9v7"/>',
    "help_outline": '<circle cx="12" cy="12" r="9"/><path d="M9.7 9a2.4 2.4 0 1 1 4.1 1.7c-1.2 1.1-1.8 1.4-1.8 3M12 17h.01"/>',
    "hourglass_empty": '<path d="M7 3h10M7 21h10M8 3c0 4 4 5 4 9s-4 5-4 9M16 3c0 4-4 5-4 9s4 5 4 9"/>',
    "science": '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 18l-5-9V3"/><path d="M8 15h8"/>',
    "center_focus_strong": '<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"/><circle cx="12" cy="12" r="3"/>',
    "search_off": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5M4 4l16 16"/>',
    "storage": '<path d="M4 5h16v4H4zM4 10h16v4H4zM4 15h16v4H4z"/><path d="M7 7h.01M7 12h.01M7 17h.01"/>',
}


def materialize_icons(html_str: str) -> str:
    """Use local SVG icons so a remote icon font cannot leak ligature names into the UI."""
    icon_pattern = re.compile(
        r'<span class="material-symbols-outlined(?P<classes>[^"]*)"(?P<attrs>[^>]*)>(?P<name>[^<]+)</span>'
    )

    def replace_icon(match: re.Match) -> str:
        name = match.group("name").strip()
        paths = _ICON_PATHS.get(name, _ICON_PATHS["default"])
        return (
            f'<span class="stb-svg-icon material-symbols-outlined{match.group("classes")}"{match.group("attrs")} '
            f'aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false">{paths}</svg></span>'
        )

    return icon_pattern.sub(replace_icon, html_str)


def render_html(html_str: str, container=st):
    """Safely render HTML in Streamlit without CommonMark indented code block artifacts."""
    container.markdown(materialize_icons(clean_html(html_str)), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load YOLO model
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_requested_model = os.environ.get("SORTIVA_MODEL")
_baseline_path = os.path.join(BASE_DIR, "models", "best_baseline_v0.pt")
_default_model = "best_baseline_v0.pt" if os.path.exists(_baseline_path) else "best.pt"
MODEL_PATH = os.path.join(BASE_DIR, "models", _requested_model or _default_model)


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
        'hex': '#006c48',
        'icon': '🍃',
        'mat_icon': 'compost',
        'bin': 'WADAH ORGANIK',
        'bin_color': '#006c48',
        'category': 'Organik',
        'stream_cat': 'Kategori Pilahan A',
        'stream_title': 'ORGANIK / BIODEGRADABLE',
        'stream_banner_bg': '#006c48',
        'stream_badge': 'Wadah Hijau',
        'stream_badge_bg': '#c1ecd4',
        'stream_badge_text': '#002114',
        'material_code': '#Bio Kompos',
        'material_sub': 'Limbah Biomassa & Sisa Organik',
        'description': 'Sisa makanan, dedaunan gugur, kulit buah & biomassa mudah terurai',
        'nature': 'Biodegradable',
        'degradation': '2 - 6 minggu (Alami)',
        'degradation_speed': 'Cepat Terurai',
        'target_bin_title': 'KOMPARTEMEN HIJAU (Kompos Sekolah)',
        'target_bin_box_bg': '#c1ecd4',
        'target_bin_box_text': '#002114',
        'target_bin_icon_bg': '#006c48',
        'target_bin_key': 'organic',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 0° ke kompartemen 1 (Hijau). Cocok untuk dekomposisi komposter Stembayo.',
        'servo_detail': 'Servo Flap Sudut 0° Kiri Aktif — Pintu Organik Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Sampah organik di SMKN 2 Depok dialirkan ke komposter komunal sekolah untuk pupuk tanaman Adiwiyata dan kebun sekolah.',
    },
    'paper': {
        'color': (235, 99, 37),
        'hex': '#003490',
        'icon': '📄',
        'mat_icon': 'description',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#003490',
        'category': 'Daur Ulang',
        'stream_cat': 'Kategori Pilahan B',
        'stream_title': 'DAUR ULANG / RECYCLABLE',
        'stream_banner_bg': '#003490',
        'stream_badge': 'Wadah Biru',
        'stream_badge_bg': '#dbe1ff',
        'stream_badge_text': '#00174b',
        'material_code': '#04 PAP (Kertas Arsip)',
        'material_sub': 'Serat Selulosa & Kertas Arsip',
        'description': 'Kertas tulis, lembaran buku & dokumen tanpa laminasi plastik',
        'nature': 'Biodegradable',
        'degradation': '2 - 5 bulan (Mudah Terurai)',
        'degradation_speed': 'Mudah Terurai',
        'target_bin_title': 'KOMPARTEMEN BIRU (Daur Ulang Kertas)',
        'target_bin_box_bg': '#dbe1ff',
        'target_bin_box_text': '#00174b',
        'target_bin_icon_bg': '#003490',
        'target_bin_key': 'recyclable',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 45° ke kompartemen 2 (Biru). Pastikan dalam kondisi kering tanpa staples.',
        'servo_detail': 'Servo Flap Sudut 45° Kanan Aktif — Pintu Daur Ulang Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Kertas bekas dipadatkan di ruang arsip dan dikirim ke mitra daur ulang kertas pulp Stembayo untuk mengurangi penebangan pohon.',
    },
    'cardboard': {
        'color': (48, 92, 112),
        'hex': '#003490',
        'icon': '📦',
        'mat_icon': 'inventory_2',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#003490',
        'category': 'Daur Ulang',
        'stream_cat': 'Kategori Pilahan B',
        'stream_title': 'DAUR ULANG / RECYCLABLE',
        'stream_banner_bg': '#003490',
        'stream_badge': 'Wadah Biru',
        'stream_badge_bg': '#dbe1ff',
        'stream_badge_text': '#00174b',
        'material_code': '#20 PAP (Karton Kemasan)',
        'material_sub': 'Karton Kemasan & Kotak Boks',
        'description': 'Kardus boks, karton kemasan tebal & paper bag kering',
        'nature': 'Biodegradable',
        'degradation': '2 bulan (Mudah Terurai)',
        'degradation_speed': 'Mudah Terurai',
        'target_bin_title': 'KOMPARTEMEN BIRU (Daur Ulang Karton)',
        'target_bin_box_bg': '#dbe1ff',
        'target_bin_box_text': '#00174b',
        'target_bin_icon_bg': '#003490',
        'target_bin_key': 'recyclable',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 45° ke kompartemen 2 (Biru). Lipat kardus hingga pipih sebelum dimasukkan.',
        'servo_detail': 'Servo Flap Sudut 45° Kanan Aktif — Pintu Daur Ulang Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Karton boks dilipat pipih untuk menghemat volume wadah sebelum diangkut ke fasilitas daur ulang Bank Sampah Stembayo.',
    },
    'plastic': {
        'color': (106, 166, 196),
        'hex': '#003490',
        'icon': '♻️',
        'mat_icon': 'water_bottle',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#003490',
        'category': 'Daur Ulang',
        'stream_cat': 'Kategori Pilahan B',
        'stream_title': 'DAUR ULANG / RECYCLABLE',
        'stream_banner_bg': '#003490',
        'stream_badge': 'Wadah Biru',
        'stream_badge_bg': '#dbe1ff',
        'stream_badge_text': '#00174b',
        'material_code': '#01 PETE / #02 HDPE',
        'material_sub': 'Polietilena Tereftalat (PET / HDPE)',
        'description': 'Botol plastik PET transparan, gelas minuman bersih & wadah HDPE',
        'nature': 'Non-Biodegradable',
        'degradation': '450 tahun (Sangat Lambat)',
        'degradation_speed': 'Sangat Lambat',
        'target_bin_title': 'KOMPARTEMEN BIRU (Daur Ulang Plastik)',
        'target_bin_box_bg': '#dbe1ff',
        'target_bin_box_text': '#00174b',
        'target_bin_icon_bg': '#003490',
        'target_bin_key': 'recyclable',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 45° ke kompartemen 2 (Biru). Lepas tutup dan remas botol.',
        'servo_detail': 'Servo Flap Sudut 45° Kanan Aktif — Pintu Daur Ulang Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Botol plastik PET yang disortir di SMKN 2 Depok akan dicacah di Bengkel Mesin & disalurkan ke Bank Sampah Stembayo untuk poin tabungan Adiwiyata kelas.',
    },
    'metal': {
        'color': (88, 99, 107),
        'hex': '#003490',
        'icon': '🥫',
        'mat_icon': 'lunch_dining',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#003490',
        'category': 'Daur Ulang',
        'stream_cat': 'Kategori Pilahan B',
        'stream_title': 'DAUR ULANG / RECYCLABLE',
        'stream_banner_bg': '#003490',
        'stream_badge': 'Wadah Biru',
        'stream_badge_bg': '#dbe1ff',
        'stream_badge_text': '#00174b',
        'material_code': '#41 ALU (Aluminium Kaleng)',
        'material_sub': 'Aluminium & Logam Kaleng',
        'description': 'Kaleng minuman ringan, kemasan aluminium, tutup botol & perkakas logam',
        'nature': 'Non-Biodegradable',
        'degradation': '50 - 200 tahun (Lambat)',
        'degradation_speed': 'Lambat',
        'target_bin_title': 'KOMPARTEMEN BIRU (Daur Ulang Logam)',
        'target_bin_box_bg': '#dbe1ff',
        'target_bin_box_text': '#00174b',
        'target_bin_icon_bg': '#003490',
        'target_bin_key': 'recyclable',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 45° ke kompartemen 2 (Biru). Kosongkan cairan sebelum dimasukkan.',
        'servo_detail': 'Servo Flap Sudut 45° Kanan Aktif — Pintu Daur Ulang Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Kaleng aluminium dapat didaur ulang 100% tanpa penurunan kualitas fisik logam (closed-loop recycling).',
    },
    'glass': {
        'color': (104, 122, 94),
        'hex': '#003490',
        'icon': '🍾',
        'mat_icon': 'wine_bar',
        'bin': 'WADAH DAUR ULANG',
        'bin_color': '#003490',
        'category': 'Daur Ulang',
        'stream_cat': 'Kategori Pilahan B',
        'stream_title': 'DAUR ULANG / RECYCLABLE',
        'stream_banner_bg': '#003490',
        'stream_badge': 'Wadah Biru',
        'stream_badge_bg': '#dbe1ff',
        'stream_badge_text': '#00174b',
        'material_code': '#70 GL (Kaca Silika)',
        'material_sub': 'Kaca Silika & Botol Kaca',
        'description': 'Botol kaca sirup/minuman, toples kaca & pecahan beling kemasan',
        'nature': 'Non-Biodegradable',
        'degradation': '1.000.000+ tahun (Abadi)',
        'degradation_speed': 'Abadi / Daur Ulang',
        'target_bin_title': 'KOMPARTEMEN BIRU (Daur Ulang Kaca)',
        'target_bin_box_bg': '#dbe1ff',
        'target_bin_box_text': '#00174b',
        'target_bin_icon_bg': '#003490',
        'target_bin_key': 'recyclable',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 45° ke kompartemen 2 (Biru). Masukkan hati-hati agar tidak pecah.',
        'servo_detail': 'Servo Flap Sudut 45° Kanan Aktif — Pintu Daur Ulang Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Kaca dapat dilebur dan dibentuk kembali secara berulang tanpa batas dengan efisiensi energi pengolahan hingga 30%.',
    },
    'e_waste': {
        'color': (48, 92, 112),
        'hex': '#ba1a1a',
        'icon': '⚡',
        'mat_icon': 'cable',
        'bin': 'WADAH RESIDU & B3',
        'bin_color': '#ba1a1a',
        'category': 'Residu / B3',
        'stream_cat': 'Kategori Pilahan C',
        'stream_title': 'RESIDU & B3 / HAZARDOUS',
        'stream_banner_bg': '#ba1a1a',
        'stream_badge': 'Wadah Merah',
        'stream_badge_bg': '#ffdad6',
        'stream_badge_text': '#93000a',
        'material_code': '#Hazardous E-Waste',
        'material_sub': 'Limbah Elektronika & Baterai (B3)',
        'description': 'Kabel listrik, baterai bekas, adaptor sirkuit & komponen elektronika',
        'nature': 'Hazardous',
        'degradation': 'Tidak dapat terurai secara alami',
        'degradation_speed': 'Limbah Berbahaya B3',
        'target_bin_title': 'KOMPARTEMEN MERAH (TPS B3 Bengkel Mesin)',
        'target_bin_box_bg': '#ffdad6',
        'target_bin_box_text': '#93000a',
        'target_bin_icon_bg': '#ba1a1a',
        'target_bin_key': 'residu',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 90° ke kompartemen 3 (Merah). Disalurkan ke TPS B3 bengkel elektro Stembayo.',
        'servo_detail': 'Servo Flap Sudut 90° Belakang Aktif — Pintu Residu & B3 Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Limbah baterai & PCB mengandung logam berat berbahaya (timbal, kadmium, litium). Bengkel Elektro Stembayo mengumpulkan limbah ini untuk pengolahan B3 berizin.',
    },
    'medical_waste': {
        'color': (48, 50, 184),
        'hex': '#ba1a1a',
        'icon': '⚠️',
        'mat_icon': 'medical_services',
        'bin': 'WADAH RESIDU & B3',
        'bin_color': '#ba1a1a',
        'category': 'Residu / B3',
        'stream_cat': 'Kategori Pilahan C',
        'stream_title': 'RESIDU & B3 / HAZARDOUS',
        'stream_banner_bg': '#ba1a1a',
        'stream_badge': 'Wadah Merah',
        'stream_badge_bg': '#ffdad6',
        'stream_badge_text': '#93000a',
        'material_code': '#Hazardous Medis B3',
        'material_sub': 'Limbah Medis & Infeksius Berbahaya',
        'description': 'Masker sekali pakai, perban, jarum suntik & limbah medis UKS',
        'nature': 'Hazardous',
        'degradation': 'Infeksius (Wajib Incinerator)',
        'degradation_speed': 'Limbah Medis B3',
        'target_bin_title': 'KOMPARTEMEN MERAH (TPS B3 / Medis UKS)',
        'target_bin_box_bg': '#ffdad6',
        'target_bin_box_text': '#93000a',
        'target_bin_icon_bg': '#ba1a1a',
        'target_bin_key': 'residu',
        'rotor_instruction': 'Motor servo mengarahkan flap sudut 90° ke kompartemen 3 (Merah). Masukkan ke kantong khusus limbah B3/UKS.',
        'servo_detail': 'Servo Flap Sudut 90° Belakang Aktif — Pintu Residu & B3 Terbuka Otomatis selama 5 detik',
        'educational_tip': 'Limbah medis infeksius berpotensi menyebarkan patogen. SMKN 2 Depok mengelolanya melalui protokol sterilisasi UKS dan insinerator bersertifikat.',
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
    'stream_cat': 'Kategori Residu',
    'stream_title': 'RESIDU UMUM / UNCLASSIFIED',
    'stream_banner_bg': '#414844',
    'stream_badge': 'Wadah Abu-abu',
    'stream_badge_bg': '#e5e2dd',
    'stream_badge_text': '#1c1c19',
    'material_code': '#Residu Umum',
    'material_sub': 'Objek Tidak Terklasifikasi',
    'description': 'Objek belum terklasifikasi secara pasti oleh model visi StemVision',
    'nature': 'Unknown',
    'degradation': 'Tidak diketahui',
    'degradation_speed': 'Butuh Verifikasi',
    'target_bin_title': 'KOMPARTEMEN RESIDU (Inspeksi Ulang)',
    'target_bin_box_bg': '#e5e2dd',
    'target_bin_box_text': '#1c1c19',
    'target_bin_icon_bg': '#414844',
    'target_bin_key': 'residu',
    'rotor_instruction': 'Motor servo dalam mode siaga (standby). Arahkan objek ke posisi tengah inspeksi.',
    'servo_detail': 'Servo Standby Flap Tertutup — Menunggu Posisi Stabil Objek di Baki',
    'educational_tip': 'Pastikan objek diletakkan secara terpisah di bawah pencahayaan optimal agar StemVision dapat mengenali jenis material secara presisi.',
}

# Nature badge styles with strong, verified contrast matching Stembayo Eco-Tech tokens
NATURE_COLORS = {
    'Biodegradable':     {'bg': '#c1ecd4', 'text': '#002114', 'border': '#a5d0b9'},
    'Non-Biodegradable': {'bg': '#dbe1ff', 'text': '#00174b', 'border': '#b4c5ff'},
    'Hazardous':         {'bg': '#ffdad6', 'text': '#93000a', 'border': '#ffb4ab'},
    'Unknown':           {'bg': '#f0ede9', 'text': '#1c1c19', 'border': '#e5e2dd'},
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


def build_detection_card_html(det: dict, latency_ms: float = None) -> str:
    """Build a compact, web-first classification result card."""
    cat_info = det.get('category_info', DEFAULT_CATEGORY)
    cls_name = det.get('class', 'unknown').replace('_', ' ').title()
    conf = float(det.get('confidence', 0.0))
    conf_pct = conf * 100.0
    uncertain = det.get('uncertain', False)

    nature = cat_info.get('nature', 'Unknown')
    nature_style = NATURE_COLORS.get(nature, NATURE_COLORS['Unknown'])
    stream_banner_bg = '#0b3b29'
    stream_cat = 'HASIL KLASIFIKASI'
    stream_title = cls_name.upper()
    stream_badge = nature
    stream_badge_bg = nature_style['bg']
    stream_badge_text = nature_style['text']

    mat_code = cat_info.get('material_code', '#00 MATERIAL')
    mat_icon = cat_info.get('mat_icon', 'recycling')
    icon = cat_info.get('icon', '♻️')
    desc = cat_info.get('description', '')
    degradation = cat_info.get('degradation', 'Tidak diketahui')
    speed = cat_info.get('degradation_speed', 'Normal')

    edu_tip = cat_info.get('educational_tip', '')

    uncertainty_html = ""
    if uncertain:
        uncertainty_html = """
        <div style="background:#fff3cd; border:1px solid #ffeeba; color:#856404; font-size:0.72rem; font-weight:700; padding:6px 10px; border-radius:8px; margin-bottom:0.6rem; display:flex; align-items:center; gap:6px;">
            <span class="material-symbols-outlined" style="font-size:16px;">help_outline</span>
            <span>Butuh Verifikasi Manual (Confidence di bawah threshold kelas — masuk Review Queue)</span>
        </div>
        """

    latency_html = ""
    if latency_ms is not None:
        latency_html = f'<span style="background:rgba(255,255,255,0.2); color:#ffffff; font-size:0.68rem; font-weight:700; font-family:\'IBM Plex Mono\',monospace; padding:2px 8px; border-radius:9999px;">Latency: {latency_ms:.0f}ms</span>'

    return clean_html(f"""
    <div class="stb-kiosk-result-card" style="background:#ffffff; border:1px solid #e5e2dd; border-radius:14px; box-shadow:0 2px 10px rgba(1,45,29,0.06); overflow:hidden; margin-bottom:1rem;">
        <div style="background:{stream_banner_bg}; color:#ffffff; padding:0.85rem 1.15rem; display:flex; align-items:center; justify-content:space-between; gap:0.6rem;">
            <div style="display:flex; align-items:center; gap:0.75rem; min-width:0;">
                <div style="width:40px; height:40px; border-radius:10px; background:rgba(255,255,255,0.18); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                    <span class="material-symbols-outlined" style="font-size:22px; color:#ffffff;">{mat_icon}</span>
                </div>
                <div style="min-width:0;">
                    <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; opacity:0.9;">{stream_cat}</div>
                    <div style="font-family:'Literata',Georgia,serif; font-size:1.02rem; font-weight:800; line-height:1.2; text-shadow:0 1px 2px rgba(0,0,0,0.12);">{stream_title}</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:0.4rem; flex-shrink:0;">
                {latency_html}
                <span style="background:{stream_badge_bg}; color:{stream_badge_text}; font-size:0.72rem; font-weight:800; padding:3px 10px; border-radius:9999px; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                    {stream_badge}
                </span>
            </div>
        </div>

        <div style="padding:1rem 1.15rem; display:flex; flex-direction:column; gap:0.75rem;">
            {uncertainty_html}

            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:3px;">
                    <span style="font-size:0.68rem; text-transform:uppercase; font-weight:700; color:#717973; letter-spacing:0.04em;">Tipe Material Terdeteksi</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; font-weight:700; background:#f0ede9; color:#012d1d; padding:2px 7px; border-radius:4px; border:1px solid #e5e2dd;">{mat_code}</span>
                </div>
                <div style="font-family:'Literata',Georgia,serif; font-size:1.15rem; font-weight:800; color:#012d1d; line-height:1.25;">
                    <span class="material-symbols-outlined" style="font-size:20px; vertical-align:-3px; color:#17633f;">{mat_icon}</span> {cls_name}
                </div>
                <div style="font-size:0.78rem; color:#414844; margin-top:2px; line-height:1.4;">{desc}</div>
            </div>

            <div>
                <div style="display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:3px;">
                    <span style="color:#414844; font-weight:600;">Tingkat Keyakinan Deteksi:</span>
                    <span style="font-weight:700; color:{stream_banner_bg}; font-family:'IBM Plex Mono',monospace;">{conf_pct:.1f}% ({conf_band(conf)})</span>
                </div>
                <div style="width:100%; height:9px; background:#f0ede9; border-radius:9999px; overflow:hidden; border:1px solid #e5e2dd; padding:1px;">
                    <div style="width:{min(100.0, max(4.0, conf_pct)):.1f}%; height:100%; background:{stream_banner_bg}; border-radius:9999px; transition:width 0.4s ease;"></div>
                </div>
            </div>

            <div style="background:#f6f3ee; border:1px solid #e5e2dd; border-radius:10px; padding:0.6rem 0.8rem; display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:0.55rem;">
                    <span class="material-symbols-outlined" style="font-size:18px; color:#006c48;">hourglass_empty</span>
                    <div>
                        <div style="font-size:0.76rem; font-weight:700; color:#012d1d;">{nature}</div>
                        <div style="font-size:0.7rem; color:#717973;">Estimasi penguraian: {degradation}</div>
                    </div>
                </div>
                <span style="font-size:0.68rem; font-weight:700; background:#ffffff; border:1px solid #c1c8c2; color:#012d1d; padding:2px 8px; border-radius:6px; flex-shrink:0;">{speed}</span>
            </div>

            <div style="background:#eaf4ec; border:1px solid #c1e0ca; border-radius:10px; padding:0.7rem 0.85rem; font-size:0.72rem; color:#214f36; line-height:1.45;">
                <div style="font-weight:700; color:#0b3b29; display:flex; align-items:center; gap:4px; margin-bottom:2px;">
                    <span class="material-symbols-outlined" style="font-size:15px; color:#17633f;">school</span>
                    Insight pemilahan
                </div>
                {edu_tip}
            </div>
        </div>
    </div>
    """)


def render_3_compartments_html(active_bin: str = None) -> str:
    """Render the 3-Compartment Kiosk Bin Status Visual Guide from Konsol Kamera.html.
    active_bin can be 'organic', 'recyclable', 'residu', or None."""

    # Styles for Bin 1 (Organic)
    is_org = (active_bin == 'organic')
    org_border = "border: 2px solid #006c48; box-shadow: 0 4px 16px rgba(0,108,72,0.18);" if is_org else "border: 1px solid #e5e2dd; box-shadow: 0 1px 3px rgba(0,0,0,0.03);"
    org_badge = """
    <div style="display:inline-flex; align-items:center; gap:5px; background:#006c48; color:#ffffff; font-size:0.68rem; font-weight:800; padding:2px 8px; border-radius:9999px;">
        <span style="width:6px; height:6px; border-radius:50%; background:#92f7c3; display:inline-block;"></span>
        Target Otomatis
    </div>
    """ if is_org else """<span style="background:#c1ecd4; color:#002114; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:9999px;">Kompos Sekolah</span>"""

    # Styles for Bin 2 (Recyclable)
    is_rec = (active_bin == 'recyclable')
    rec_border = "border: 2px solid #003490; box-shadow: 0 4px 16px rgba(0,52,144,0.18);" if is_rec else "border: 1px solid #e5e2dd; box-shadow: 0 1px 3px rgba(0,0,0,0.03);"
    rec_badge = """
    <div style="display:inline-flex; align-items:center; gap:5px; background:#003490; color:#ffffff; font-size:0.68rem; font-weight:800; padding:2px 8px; border-radius:9999px;">
        <span style="width:6px; height:6px; border-radius:50%; background:#dbe1ff; display:inline-block;"></span>
        Target Otomatis
    </div>
    """ if is_rec else """<span style="background:#dbe1ff; color:#00174b; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:9999px;">Bank Sampah</span>"""

    # Styles for Bin 3 (Residu & B3)
    is_res = (active_bin == 'residu')
    res_border = "border: 2px solid #ba1a1a; box-shadow: 0 4px 16px rgba(186,26,26,0.18);" if is_res else "border: 1px solid #e5e2dd; box-shadow: 0 1px 3px rgba(0,0,0,0.03);"
    res_badge = """
    <div style="display:inline-flex; align-items:center; gap:5px; background:#ba1a1a; color:#ffffff; font-size:0.68rem; font-weight:800; padding:2px 8px; border-radius:9999px;">
        <span style="width:6px; height:6px; border-radius:50%; background:#ffdad6; display:inline-block;"></span>
        Target Otomatis
    </div>
    """ if is_res else """<span style="background:#ffdad6; color:#93000a; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:9999px;">TPS B3 Bengkel</span>"""

    return clean_html(f"""
    <div style="margin-top: 1.25rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.65rem;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span class="material-symbols-outlined" style="font-size:20px; color:#006c48;">storage</span>
                <span style="font-family:'Literata',Georgia,serif; font-size:0.98rem; font-weight:800; color:#012d1d;">Status 3 Kompartemen Wadah Kiosk</span>
            </div>
            <span style="font-size:0.72rem; color:#717973; font-weight:600;">Sinkronisasi Sensor Ultrasonik Realtime</span>
        </div>

        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:0.85rem;">
            <div style="background:#ffffff; border-radius:12px; {org_border} padding:1rem; display:flex; flex-direction:column; justify-content:space-between; gap:0.6rem; transition:all 0.3s ease;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="display:flex; align-items:center; gap:0.65rem;">
                        <div style="width:40px; height:40px; border-radius:10px; background:#c1ecd4; color:#002114; display:flex; align-items:center; justify-content:center; font-size:1.4rem;">🍃</div>
                        <div>
                            <div style="font-family:'Literata',Georgia,serif; font-size:0.95rem; font-weight:800; color:#006c48;">Wadah Hijau</div>
                            <div style="font-size:0.7rem; font-weight:800; color:#006c48; letter-spacing:0.04em;">ORGANIK</div>
                        </div>
                    </div>
                    {org_badge}
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#414844; margin-bottom:3px;">
                        <span>Kapasitas Wadah</span>
                        <span style="font-weight:700; color:#006c48;">42% (Normal)</span>
                    </div>
                    <div style="width:100%; height:6px; background:#f0ede9; border-radius:9999px; overflow:hidden;">
                        <div style="width:42%; height:100%; background:#006c48; border-radius:9999px;"></div>
                    </div>
                </div>
                <div style="font-size:0.7rem; color:#414844; line-height:1.35;">
                    <b style="color:#012d1d;">Contoh:</b> Sisa makanan kantin, daun gugur taman, kulit buah, serbuk kayu bangunan.
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#717973; border-top:1px solid #f0ede9; padding-top:0.35rem;">
                    <span style="color:#006c48; font-weight:700;">Biodegradable</span>
                    <span>Tersisa ~58L</span>
                </div>
            </div>

            <div style="background:#ffffff; border-radius:12px; {rec_border} padding:1rem; display:flex; flex-direction:column; justify-content:space-between; gap:0.6rem; transition:all 0.3s ease;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="display:flex; align-items:center; gap:0.65rem;">
                        <div style="width:40px; height:40px; border-radius:10px; background:#dbe1ff; color:#00174b; display:flex; align-items:center; justify-content:center; font-size:1.4rem;">♻️</div>
                        <div>
                            <div style="font-family:'Literata',Georgia,serif; font-size:0.95rem; font-weight:800; color:#003490;">Wadah Biru</div>
                            <div style="font-size:0.7rem; font-weight:800; color:#003490; letter-spacing:0.04em;">DAUR ULANG</div>
                        </div>
                    </div>
                    {rec_badge}
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#414844; margin-bottom:3px;">
                        <span>Kapasitas Wadah</span>
                        <span style="font-weight:700; color:#003490;">68% (Siap Tampung)</span>
                    </div>
                    <div style="width:100%; height:6px; background:#f0ede9; border-radius:9999px; overflow:hidden;">
                        <div style="width:68%; height:100%; background:#003490; border-radius:9999px;"></div>
                    </div>
                </div>
                <div style="font-size:0.7rem; color:#414844; line-height:1.35;">
                    <b style="color:#012d1d;">Contoh:</b> Botol PET, kaleng soda aluminium, kertas HVS tugas, kardus boks, botol kaca sirup.
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#717973; border-top:1px solid #f0ede9; padding-top:0.35rem;">
                    <span style="color:#003490; font-weight:700;">Recyclable</span>
                    <span>Tersisa ~32L</span>
                </div>
            </div>

            <div style="background:#ffffff; border-radius:12px; {res_border} padding:1rem; display:flex; flex-direction:column; justify-content:space-between; gap:0.6rem; transition:all 0.3s ease;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="display:flex; align-items:center; gap:0.65rem;">
                        <div style="width:40px; height:40px; border-radius:10px; background:#ffdad6; color:#93000a; display:flex; align-items:center; justify-content:center; font-size:1.4rem;">⚠️</div>
                        <div>
                            <div style="font-family:'Literata',Georgia,serif; font-size:0.95rem; font-weight:800; color:#ba1a1a;">Wadah Merah</div>
                            <div style="font-size:0.7rem; font-weight:800; color:#ba1a1a; letter-spacing:0.04em;">RESIDU &amp; B3</div>
                        </div>
                    </div>
                    {res_badge}
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#414844; margin-bottom:3px;">
                        <span>Kapasitas Wadah</span>
                        <span style="font-weight:700; color:#ba1a1a;">15% (Aman)</span>
                    </div>
                    <div style="width:100%; height:6px; background:#f0ede9; border-radius:9999px; overflow:hidden;">
                        <div style="width:15%; height:100%; background:#ba1a1a; border-radius:9999px;"></div>
                    </div>
                </div>
                <div style="font-size:0.7rem; color:#414844; line-height:1.35;">
                    <b style="color:#012d1d;">Contoh:</b> Masker medis, baterai bekas multimeter, kabel terbakar, residu solder, oli bengkel.
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#717973; border-top:1px solid #f0ede9; padding-top:0.35rem;">
                    <span style="color:#ba1a1a; font-weight:700;">Hazardous / B3</span>
                    <span>Tersisa ~85L</span>
                </div>
            </div>
        </div>
    </div>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class confidence thresholds — Phase 4 accuracy improvement
# Classes with high false-positive risk get higher thresholds.
# ─────────────────────────────────────────────────────────────────────────────
def render_model_guide_html(active_class: str = None) -> str:
    """Render supported material classes without physical-device status."""
    class_rows = []
    for class_key, info in WASTE_CATEGORIES.items():
        label = class_key.replace('_', ' ').title()
        nature = info.get('nature', 'Unknown')
        nature_style = NATURE_COLORS.get(nature, NATURE_COLORS['Unknown'])
        is_active = active_class == class_key
        active_style = 'box-shadow:0 0 0 2px #17633f; background:#eef8ef;' if is_active else ''
        class_rows.append(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:0.7rem; padding:0.65rem 0.15rem; border-bottom:1px solid #e5e2dd; {active_style}">
            <span style="display:flex; align-items:center; gap:0.55rem; color:#0b3b29; font-size:0.78rem; font-weight:700;">
                <span class="material-symbols-outlined" style="font-size:17px; color:#17633f;">category</span>
                {label}
            </span>
            <span style="font-size:0.65rem; font-weight:700; color:{nature_style['text']}; background:{nature_style['bg']}; padding:3px 7px; border-radius:9999px; white-space:nowrap;">{nature}</span>
        </div>
        """)

    return clean_html(f"""
    <div class="stb-model-guide">
        <div class="stb-model-guide-heading">
            <div>
                <div class="stb-model-guide-title">Jenis material yang dikenali model</div>
                <div class="stb-model-guide-subtitle">Gunakan hasil ini sebagai panduan awal untuk memahami sampah.</div>
            </div>
            <span class="stb-model-guide-count">{len(WASTE_CATEGORIES)} kelas</span>
        </div>
        <div class="stb-model-guide-grid">{''.join(class_rows)}</div>
    </div>
    """)


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
render_html("""
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

.stb-svg-icon {
    display: inline-flex !important;
    flex: 0 0 auto;
    align-items: center;
    justify-content: center;
    width: 1.1em;
    height: 1.1em;
    line-height: 1;
}

.stb-svg-icon svg {
    display: block;
    width: 100%;
    height: 100%;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1360px;
}

/* Top App Bar */
.stb-navbar {
    background: #1b4332;
    border: 1px solid #012d1d;
    padding: 0.85rem 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 14px;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 20px -4px rgba(1, 45, 29, 0.3);
}
.stb-navbar-left {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.stb-logo-mark {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: #012d1d;
    color: #92f7c3;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(1, 45, 29, 0.25);
    border: 1px solid rgba(146, 247, 195, 0.2);
}
.stb-title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.stb-title-main {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #012d1d;
    line-height: 1.2;
    letter-spacing: -0.015em;
}
.stb-version-pill {
    font-family: 'Nunito Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 800;
    background: #c1ecd4;
    color: #002114;
    padding: 2px 8px;
    border-radius: 9999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.stb-title-sub {
    font-size: 0.74rem;
    color: #414844;
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
    font-family: 'Nunito Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: #002114;
    background: #c1ecd4;
    border: 1px solid #a5d0b9;
    padding: 4px 11px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.stb-status-pill {
    font-size: 0.72rem;
    font-weight: 700;
    color: #ffffff;
    background: #006c48;
    border-radius: 9999px;
    padding: 4px 12px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 1px 3px rgba(0, 108, 72, 0.25);
}
.stb-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #92f7c3;
    box-shadow: 0 0 8px #92f7c3;
}

/* Sub-Header Kiosk Telemetry Bar */
.stb-telemetry-banner {
    background: #ffffff;
    border: 1px solid #e5e2dd;
    border-radius: 14px;
    padding: 0.9rem 1.3rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    box-shadow: 0 1px 4px rgba(1, 45, 29, 0.04);
}
.stb-telemetry-left {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    min-width: 0;
}
.stb-telemetry-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #012d1d;
    color: #92f7c3;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(1, 45, 29, 0.2);
}
.stb-telemetry-title-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.stb-telemetry-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 800;
    color: #012d1d;
}
.stb-telemetry-node {
    font-size: 0.68rem;
    font-weight: 700;
    background: #c1ecd4;
    color: #002114;
    padding: 2px 8px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stb-telemetry-sub {
    font-size: 0.74rem;
    color: #717973;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 2px;
    flex-wrap: wrap;
}
.stb-telemetry-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #006c48;
}
.stb-telemetry-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.stb-tele-chip {
    background: #f0ede9;
    border: 1px solid #e5e2dd;
    padding: 5px 11px;
    border-radius: 8px;
    font-size: 0.74rem;
    color: #1c1c19;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.stb-tele-chip-active {
    background: #1b4332;
    color: #ffffff;
    border: 1px solid #012d1d;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 0.74rem;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    box-shadow: 0 1px 3px rgba(1, 45, 29, 0.2);
}

/* Sub-Header Utility Control Strip */
.stb-control-strip {
    background: #ffffff;
    border: 1px solid #e5e2dd;
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
}

/* Mas Bayo Virtual Assistant Banner */
.stb-masbayo-card {
    background: #ffffff;
    border: 1px solid #e5e2dd;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: flex-start;
    gap: 1.1rem;
    box-shadow: 0 1px 4px rgba(1, 45, 29, 0.04);
}
.stb-masbayo-avatar {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: #c1ecd4;
    color: #006c48;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(0, 108, 72, 0.15);
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
    font-family: 'Literata', Georgia, serif;
    font-size: 0.95rem;
    font-weight: 800;
    color: #012d1d;
    letter-spacing: 0.02em;
}
.stb-masbayo-role {
    font-size: 0.72rem;
    color: #717973;
    font-weight: 600;
}
.stb-masbayo-dialogue {
    font-size: 0.82rem;
    color: #414844;
    line-height: 1.5;
    margin-bottom: 0.55rem;
}
.stb-masbayo-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.stb-chip {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 4px 11px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.chip-organik {
    background: #c1ecd4;
    color: #002114;
    border: 1px solid #a5d0b9;
}
.chip-daurulang {
    background: #dbe1ff;
    color: #00174b;
    border: 1px solid #b4c5ff;
}
.chip-residu {
    background: #ffdad6;
    color: #93000a;
    border: 1px solid #ffb4ab;
}

/* System Status HUD Pill */
.hud-status-pill {
    background: #012d1d;
    color: #FAF6F0;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 6px rgba(1, 45, 29, 0.2);
    border: 1px solid rgba(146, 247, 195, 0.2);
}
.hud-pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #92f7c3;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px #92f7c3;
}

/* Section Header Bar */
.section-heading-bar {
    font-family: 'Literata', Georgia, serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: #012d1d;
    letter-spacing: -0.01em;
    margin-bottom: 0.65rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e5e2dd;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Detection Card (Right Column) */
.stb-det-card {
    background: #ffffff;
    border: 1px solid #e5e2dd;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 6px rgba(1, 45, 29, 0.04);
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
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.stb-det-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1rem;
    font-weight: 800;
    color: #012d1d;
    line-height: 1.2;
}
.stb-det-sub {
    font-size: 0.7rem;
    color: #717973;
    margin-top: 1px;
}
.stb-det-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #006c48;
    background: #c1ecd4;
    border: 1px solid #a5d0b9;
    padding: 3px 8px;
    border-radius: 6px;
    flex-shrink: 0;
}
.stb-det-bin-bar {
    background: #f6f3ee;
    border: 1px solid #e5e2dd;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
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
    color: #717973;
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
    background: #fcf9f4;
    border: 1px solid #e5e2dd;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.74rem;
    color: #414844;
    line-height: 1.45;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}
.stb-rotor-bold {
    font-weight: 700;
    color: #012d1d;
}

/* Reference Card: Pedoman Standar 3 Wadah Adiwiyata */
.stb-pedoman-panel {
    background: #ffffff;
    border: 1px solid #e5e2dd;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
}
.stb-pedoman-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 0.86rem;
    font-weight: 800;
    color: #012d1d;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.stb-pedoman-sub {
    font-size: 0.72rem;
    color: #717973;
    margin-bottom: 0.65rem;
    display: block;
}
.stb-pedoman-item {
    background: #fcf9f4;
    border: 1px solid #e5e2dd;
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
    color: #414844;
    line-height: 1.35;
}

.stb-model-guide {
    background: #ffffff;
    border: 1px solid #e5e2dd;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    box-shadow: 0 8px 22px rgba(16, 38, 28, 0.05);
}
.stb-model-guide-heading {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.stb-model-guide-title {
    color: #0b3b29;
    font-family: 'Literata', Georgia, serif;
    font-size: 0.95rem;
    font-weight: 800;
}
.stb-model-guide-subtitle {
    color: #717973;
    font-size: 0.72rem;
    line-height: 1.4;
    margin-top: 0.2rem;
}
.stb-model-guide-count {
    background: #dcebdc;
    border-radius: 9999px;
    color: #17633f;
    flex-shrink: 0;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 8px;
}
.stb-model-guide-grid {
    display: grid;
    gap: 0.55rem;
    grid-template-columns: repeat(4, minmax(0, 1fr));
}
@media (max-width: 780px) {
    .stb-model-guide-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 480px) {
    .stb-model-guide-grid {
        grid-template-columns: minmax(0, 1fr);
    }
}

/* Hardware Status Indicator Widget */
.stb-hardware-widget {
    background: #f0ede9;
    border: 1px solid #e5e2dd;
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
    background: #006c48;
    flex-shrink: 0;
}
.stb-hw-title {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 0.72rem;
    color: #012d1d;
    display: block;
}
.stb-hw-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    color: #717973;
    display: block;
}
.stb-hw-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #002114;
    background: #c1ecd4;
    border: 1px solid #a5d0b9;
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
    background: #ffffff !important;
    border: 1px solid #e5e2dd !important;
    border-radius: 9999px !important;
    padding: 6px 16px !important;
    font-weight: 700 !important;
    color: #012d1d !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(1, 45, 29, 0.04) !important;
}
div[data-testid="stPageLink-NavLink"]:hover {
    background: #006c48 !important;
    color: #ffffff !important;
    border-color: #006c48 !important;
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #f0ede9 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #e5e2dd !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: #414844 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.4rem !important;
    border: none !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #1b4332 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(1, 45, 29, 0.25) !important;
}

/* Primary Action Buttons */
.stButton > button {
    font-family: 'Nunito Sans', -apple-system, sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: #006c48 !important;
    color: #FFFFFF !important;
    border: 1px solid #005235 !important;
    box-shadow: 0 2px 6px rgba(0, 108, 72, 0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #005235 !important;
    border-color: #003a25 !important;
}
.stButton > button:not([kind="primary"]) {
    background: #ffffff !important;
    color: #012d1d !important;
    border: 1px solid #e5e2dd !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #717973 !important;
    background: #f0ede9 !important;
    color: #012d1d !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: #006c48 !important;
}

/* Empty / Inactive State Boxes */
.stb-empty-box {
    background: #ffffff;
    border: 1.5px dashed #c1c8c2;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    color: #414844;
}
.stb-empty-icon {
    font-size: 2.2rem;
    margin-bottom: 0.6rem;
}
.stb-empty-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #012d1d;
    margin-bottom: 0.3rem;
}
.stb-empty-subtitle {
    font-size: 0.78rem;
    color: #717973;
}

/* Clean Institutional Footer */
.stb-footer {
    border-top: 1px solid #e5e2dd;
    margin-top: 3.5rem;
    padding-top: 1.8rem;
    background: #FAF6F0;
    color: #717973;
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
    color: #012d1d;
}
</style>
""")
# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Top Navigation Bar (Identical to Konsol Kamera.html & dashboard.html)
# ─────────────────────────────────────────────────────────────────────────────
render_navbar("camera", "Model Siap" if model is not None else "Model Belum Siap")


# ─────────────────────────────────────────────────────────────────────────────
# Model context bar
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<div class="stb-telemetry-banner">
    <div class="stb-telemetry-left">
        <div class="stb-telemetry-icon"><span class="material-symbols-outlined" style="font-size:24px;">image_search</span></div>
        <div>
            <div class="stb-telemetry-title-row">
                <span class="stb-telemetry-title">Klasifikasi Sampah SORTIVA</span>
                <span class="stb-telemetry-node">APLIKASI WEB</span>
            </div>
            <div class="stb-telemetry-sub">
                <span class="stb-telemetry-dot"></span>
                <span>Model machine learning untuk 8 jenis material</span>
                <span style="opacity:0.4;">•</span>
                <span>Input kamera atau unggah gambar</span>
            </div>
        </div>
    </div>
    <div class="stb-telemetry-right">
        <div class="stb-tele-chip"><span class="material-symbols-outlined" style="font-size:16px; color:#006c48;">photo_camera</span> Input: <b>Kamera</b></div>
        <div class="stb-tele-chip"><span class="material-symbols-outlined" style="font-size:16px; color:#006c48;">upload_file</span> Input: <b>Unggah</b></div>
        <div class="stb-tele-chip-active"><span class="material-symbols-outlined" style="font-size:16px; color:#92f7c3;">check_circle</span> Model: <b>Siap</b></div>
    </div>
</div>
""")

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
            "Threshold Confidence (YOLOv8)", 0.05, 1.0, 0.35, 0.05,
            help="Nilai batas minimum akurasi agar objek diakui model."
        )

    with tb_c3:
        show_all = st.checkbox(
            "Tampilkan Semua Bounding Box", value=False,
            help="Mode diagnostik: menampilkan prediksi mentah meski confidence rendah."
        )

    with tb_c4:
        with st.popover("📖 Indeks 8 Klasifikasi YOLOv8"):
            st.markdown("##### 8 Kelas Model SORTIVA")
            for category, info in WASTE_CATEGORIES.items():
                render_html(f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #e5e2dd; font-size:0.75rem;">
                    <span><span class="material-symbols-outlined" style="font-size:16px; vertical-align:-3px; color:#17633f;">category</span> <b>{category.replace('_',' ').title()}</b></span>
                    <span style="font-size:0.68rem; font-weight:700; color:{info['hex']}; background:#f0ede9; border:1px solid #e5e2dd; padding:1px 6px; border-radius:4px;">
                        {info['category']}
                    </span>
                </div>
                """)

# ─────────────────────────────────────────────────────────────────────────────
# Mas Bayo Virtual Assistant Banner
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<div class="stb-masbayo-card">
    <div class="stb-masbayo-avatar"><span class="material-symbols-outlined" style="font-size:26px; color:#006c48;">smart_toy</span></div>
    <div class="stb-masbayo-content">
        <div class="stb-masbayo-title">
            <span class="stb-masbayo-name">MAS BAYO</span>
            <span class="stb-masbayo-role">· Panduan belajar memilah sampah</span>
        </div>
        <div class="stb-masbayo-dialogue">
            Masukkan gambar melalui kamera atau unggah foto. SORTIVA akan menampilkan jenis material, tingkat keyakinan, dan insight singkat untuk membantu proses belajar:
        </div>
        <div class="stb-masbayo-chips">
            <span class="stb-chip chip-organik"><span class="material-symbols-outlined">photo_camera</span> Ambil gambar</span>
            <span class="stb-chip chip-daurulang"><span class="material-symbols-outlined">psychology</span> Model membaca objek</span>
            <span class="stb-chip chip-residu"><span class="material-symbols-outlined">school</span> Pahami hasilnya</span>
        </div>
    </div>
</div>
""")

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
        render_html("""
        <div class="section-heading-bar">
            <span>Umpan Kamera &amp; Inspeksi Visual</span>
            <span style="font-family:'IBM Plex Mono'; font-size:0.72rem; color:#4A4E4A;">INPUT KAMERA</span>
        </div>
        """)

        btn_row = st.columns(2)
        with btn_row[0]:
            start_btn = st.button(
                "Mulai Deteksi Kamera", type="primary", width='stretch',
                disabled=model is None,
            )
        with btn_row[1]:
            stop_btn = st.button("Hentikan Kamera", width='stretch')

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_info:
        render_html("""
        <div class="section-heading-bar">
            <span>Hasil Identifikasi</span>
            <span style="font-family:'IBM Plex Mono'; font-size:0.7rem; color:#4A7C59; font-weight:700;">MODEL AKTIF</span>
        </div>
        """)

        det_placeholder = st.empty()

        render_html("""
        <div class="stb-pedoman-panel">
            <div class="stb-pedoman-title">Tips mendapatkan hasil yang lebih jelas</div>
            <span class="stb-pedoman-sub">Gunakan panduan ini sebelum menjalankan klasifikasi:</span>
            <div class="stb-pedoman-item">
                <div class="stb-pedoman-head" style="color:#006c48;">
                    <span>01. Pastikan objek terlihat utuh</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:10px; background:#c1ecd4; color:#002114; padding:1px 6px; border-radius:4px;">GAMBAR</span>
                </div>
                <div class="stb-pedoman-desc">Hindari objek terlalu kecil, terpotong, atau tertutup objek lain.</div>
            </div>
            <div class="stb-pedoman-item">
                <div class="stb-pedoman-head" style="color:#003490;">
                    <span>02. Gunakan pencahayaan yang cukup</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:10px; background:#dbe1ff; color:#00174b; padding:1px 6px; border-radius:4px;">VISUAL</span>
                </div>
                <div class="stb-pedoman-desc">Foto yang terang dan tidak buram membantu model membaca ciri material.</div>
            </div>
            <div class="stb-pedoman-item">
                <div class="stb-pedoman-head" style="color:#ba1a1a;">
                    <span>03. Periksa tingkat keyakinan</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:10px; background:#ffdad6; color:#93000a; padding:1px 6px; border-radius:4px;">REVIEW</span>
                </div>
                <div class="stb-pedoman-desc">Gunakan confidence dan insight sebagai panduan, bukan pengganti penilaian manusia.</div>
            </div>
        </div>
        """)

    bins_placeholder = st.empty()
    render_html(render_model_guide_html(), container=bins_placeholder)

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

                render_html("""
                <div class="hud-status-pill">
                                <span><span class="hud-pulse-dot"></span>PEMINDAIAN AKTIF · MODEL SORTIVA</span>
                                <span>30.2 FPS | 24ms</span>
                </div>
                """, container=status_placeholder)

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
                                det_html = "".join(build_detection_card_html(d, latency_ms) for d in detections)
                                render_html(det_html, container=det_placeholder)
                                render_html(render_model_guide_html(active_class=detections[0].get('class')), container=bins_placeholder)
                            else:
                                render_html("""
                                <div class="stb-empty-box" style="padding:2.5rem 1rem;">
                                    <span class="material-symbols-outlined stb-empty-icon">center_focus_strong</span>
                                    <div class="stb-empty-title" style="font-size:1rem;">
                                        Menunggu Objek Sampah
                                    </div>
                                    <div class="stb-empty-subtitle" style="font-size:0.75rem;">
                                        Pastikan objek terlihat jelas di depan kamera
                                    </div>
                                </div>
                                """, container=det_placeholder)
                                render_html(render_model_guide_html(), container=bins_placeholder)

                        # Update HUD with real FPS and latency every 15 frames
                        dt = time.time() - t0
                        if dt > 0:
                            fps_inst = 1.0 / dt
                            fps_display = 0.85 * fps_display + 0.15 * fps_inst
                        if frame_count % 15 == 0:
                            render_html(f"""
                            <div class="hud-status-pill">
                <span><span class="hud-pulse-dot"></span>PEMINDAIAN AKTIF · MODEL SORTIVA</span>
                <span>{fps_display:.1f} FPS | {latency_ms:.0f}ms</span>
                            </div>
                            """, container=status_placeholder)

                        # Minimal yield to keep Streamlit responsive without capping FPS
                        time.sleep(0.002)

                    except Exception as loop_err:
                        status_placeholder.error(f"Pemindaian terhenti akibat kendala: {loop_err}")
                        st.session_state.webcam_running = False
                        break
            finally:
                cap.release()
                render_html(render_model_guide_html(), container=bins_placeholder)

            frame_placeholder.empty()
            render_html("""
            <div class="hud-status-pill" style="background:#4A4E4A;">
                <span>KAMERA NONAKTIF</span>
                <span>STATUS: STANDBY</span>
            </div>
            """, container=status_placeholder)
    else:
        render_html("""
        <div class="stb-empty-box">
            <span class="material-symbols-outlined stb-empty-icon">photo_camera</span>
            <div class="stb-empty-title">Kamera Belum Aktif</div>
            <div class="stb-empty-subtitle">Klik tombol 'Mulai Deteksi Kamera' untuk memulai pemindaian.</div>
        </div>
        """, container=frame_placeholder)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PHOTO UPLOAD & ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    render_html('<div class="section-heading-bar">Unggah Berkas Foto Sampah untuk Analisis StemVision</div>')

    # Quick Test Samples Guidance (Referenced from Konsol Kamera.html)
    render_html("""
    <div style="background:#ffffff; border:1px solid #e5e2dd; border-radius:12px; padding:0.75rem 1.1rem; margin-bottom:1rem; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;">
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span class="material-symbols-outlined" style="font-size:18px; color:#006c48;">science</span>
            <span style="font-size:0.78rem; font-weight:700; color:#012d1d;">Contoh Sampel Gambar:</span>
        </div>
        <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
            <span style="background:#f0ede9; border:1px solid #e5e2dd; padding:3px 9px; border-radius:6px; font-size:0.72rem; color:#1c1c19;">Botol PET</span>
            <span style="background:#f0ede9; border:1px solid #e5e2dd; padding:3px 9px; border-radius:6px; font-size:0.72rem; color:#1c1c19;">Daun organik</span>
            <span style="background:#f0ede9; border:1px solid #e5e2dd; padding:3px 9px; border-radius:6px; font-size:0.72rem; color:#1c1c19;">Dus kemasan</span>
            <span style="background:#f0ede9; border:1px solid #e5e2dd; padding:3px 9px; border-radius:6px; font-size:0.72rem; color:#1c1c19;">Kaleng soda</span>
            <span style="background:#f0ede9; border:1px solid #e5e2dd; padding:3px 9px; border-radius:6px; font-size:0.72rem; color:#1c1c19;">Baterai</span>
        </div>
    </div>
    """)

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
            render_html('<div class="section-heading-bar">Gambar Asli</div>')
            st.image(image, width='stretch')

        with col2:
            render_html('<div class="section-heading-bar">Hasil Klasifikasi SORTIVA</div>')
            if model is not None:
                effective_conf = 0.001 if show_all else conf_threshold
                with st.spinner("Menjalankan analisis visi komputer multi-skala..."):
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

                    render_html(f"""
                    <div class="hud-status-pill" style="margin-bottom:1rem; background:#ffffff; color:#012d1d; border:1px solid #e5e2dd; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                        <span>TOTAL TERDETEKSI: <b>{len(detections)}</b></span>
                        <span style="color:#006c48; font-weight:800;">ORGANIK: {bio_count}</span>
                        <span style="color:#003490; font-weight:800;">DAUR ULANG: {nonbio_count}</span>
                        <span style="color:#ba1a1a; font-weight:800;">RESIDU/B3: {haz_count}</span>
                    </div>
                    """)

                    _counts = Counter(d['class'].replace('_', ' ').title() for d in detections)
                    st.markdown(
                        "**Objek Teridentifikasi:** " +
                        ", ".join(f"**{k}** ({v}x)" for k, v in _counts.most_common())
                    )

                    for i, det in enumerate(detections):
                        render_html(build_detection_card_html(det))

                    render_html('<div class="section-heading-bar" style="margin-top:1.5rem;">Visualisasi Anotasi Bounding Box</div>')
                    st.image(draw_detections(image, detections), channels="RGB", width='stretch')
                else:
                    render_html("""
                    <div class="stb-empty-box" style="padding:2.5rem 1.5rem;">
                        <span class="material-symbols-outlined stb-empty-icon">search_off</span>
                        <div class="stb-empty-title">Tidak Ada Objek Sampah Terdeteksi</div>
                        <div class="stb-empty-subtitle">
                            Coba sesuaikan Ambang Batas Keyakinan pada toolbar atas jika objek berukuran kecil atau redup.
                        </div>
                    </div>
                    """)
                    with st.expander("Diagnostik Model (Keyakinan ≥ 0.1%)", expanded=True):
                        st.write(f"Task Model: **{getattr(model, 'task', 'detect')}**")
                        _raw = debug_predictions(image)
                        if _raw:
                            st.table({"Kelas Prediksi": [r[0] for r in _raw], "Tingkat Keyakinan": [r[1] for r in _raw]})
                        else:
                            st.write("Model tidak menghasilkan probabilitas di atas 1%.")
            else:
                st.error("Model deteksi belum siap. Periksa kembali ketersediaan berkas weights.")

        # Keep the supported model classes visible below the upload results.
        active_class = detections[0].get('class') if (model is not None and detections) else None
        render_html(render_model_guide_html(active_class=active_class))
    else:
        render_html("""
        <div class="stb-empty-box">
            <span class="material-symbols-outlined stb-empty-icon">upload_file</span>
            <div class="stb-empty-title">Letakkan Berkas Gambar Di Atas</div>
            <div class="stb-empty-subtitle">Mendukung format JPG, PNG, BMP, dan GIF untuk analisis visi komputer multi-skala komprehensif.</div>
        </div>
        """)
        render_html(render_model_guide_html())

# ─────────────────────────────────────────────────────────────────────────────
# Clean Ecological Institutional Footer
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
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
""")
