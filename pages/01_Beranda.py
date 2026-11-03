"""SORTIVA Streamlit home page: web-first waste classification experience."""

import re
import streamlit as st
from sortiva_ui import render_navbar


st.set_page_config(
    page_title="SORTIVA — Klasifikasi Sampah Berbasis Machine Learning",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def clean_html(html_str: str) -> str:
    no_comments = re.sub(r"<!--.*?-->", "", str(html_str), flags=re.DOTALL)
    return "\n".join(line.strip() for line in no_comments.splitlines() if line.strip())


_ICON_PATHS = {
    "default": '<circle cx="12" cy="12" r="8"/>',
    "eco": '<path d="M20.5 3.5C11.2 3.9 5.2 8.1 4 19.8c11.7-1.2 15.9-7.2 16.5-16.3Z"/><path d="M4.5 19.5c2.8-4.1 6.4-6.9 10.9-9.4"/>',
    "image_search": '<rect x="3" y="4.5" width="14" height="13" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m5.5 15 3.2-3 2.2 2 1.8-1.7"/><circle cx="17.5" cy="17.5" r="3.5"/><path d="m20.2 20.2 2 2"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16.5 9"/>',
    "auto_awesome": '<path d="m12 3 1.3 5.7L19 10l-5.7 1.3L12 17l-1.3-5.7L5 10l5.7-1.3L12 3Z"/><path d="m19 16 .6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z"/>',
    "water_bottle": '<path d="M9 3h6M10 3v3l-1.5 2v10.5A2.5 2.5 0 0 0 11 21h2a2.5 2.5 0 0 0 2.5-2.5V8L14 6V3"/><path d="M8.5 11h7"/>',
    "lightbulb": '<path d="M9 18h6M10 21h4"/><path d="M8.5 14.5A6 6 0 1 1 15.8 14c-.8.6-1.3 1.3-1.5 2H9.7c-.2-.6-.6-1.1-1.2-1.5Z"/>',
    "lunch_dining": '<path d="M4 10h16M5 10c.5-3.2 3.2-5 7-5s6.5 1.8 7 5M5 10l1 9h12l1-9"/>',
    "wine_bar": '<path d="M6 4h12l-1 5a5 5 0 0 1-10 0L6 4Z"/><path d="M12 14v6M8 20h8"/>',
    "description": '<path d="M6 3.5h8l4 4V20.5H6z"/><path d="M14 3.5v4h4M9 12h6M9 15.5h6"/>',
    "inventory_2": '<path d="M4 7h16v13H4zM3 4h18v3H3z"/><path d="M9 11h6"/>',
    "compost": '<path d="M12 20V8M12 13c-4.8 0-7-2.7-7-6 4.6 0 7 1.7 7 6ZM12 11c0-3.8 2.4-6 7-6 0 3.5-2.2 6-7 6Z"/>',
    "cable": '<path d="M8 4v5a4 4 0 0 0 4 4h1a4 4 0 0 1 4 4v3M5 3v3M8 3v3M16 17v4M19 17v4"/>',
    "medical_services": '<path d="M5 7h14v13H5zM9 7V5h6v2"/><path d="M12 10v7M8.5 13.5h7"/>',
    "school": '<path d="m3 9 9-5 9 5-9 5-9-5Z"/><path d="M7 11.5V16c2.8 2.4 7.2 2.4 10 0v-4.5M21 9v7"/>',
    "recycling": '<path d="m8 5 2-3 2 3M10 2v5M6 8l-3 5 2 3h5M3 13l4 2M17 8l3 5-2 3h-5M20 13l-4 2M12 18v4M12 22l-3-3M12 22l3-3"/>',
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


def render_html(html_str: str, container=st) -> None:
    container.markdown(materialize_icons(clean_html(html_str)), unsafe_allow_html=True)


render_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

:root {
    --ink: #10261c;
    --forest: #0b3b29;
    --fern: #17633f;
    --mint: #b8e8c8;
    --paper: #f7f8f3;
    --sand: #ecefe7;
    --moss: #dcebdc;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--paper);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none;
}

.block-container {
    max-width: 1240px;
    padding: 1.35rem 2rem 3rem;
}

.material-symbols-outlined {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 1.1rem;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
    font-feature-settings: 'liga';
}

.stb-svg-icon {
    display: inline-flex;
    flex: 0 0 auto;
    align-items: center;
    justify-content: center;
    width: 1.1em;
    height: 1.1em;
    line-height: 1;
    vertical-align: middle;
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

.stb-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.25rem 0 0.9rem;
    border-bottom: 1px solid rgba(16, 38, 28, 0.1);
    margin-bottom: 1.4rem;
}

.stb-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    color: var(--forest);
}

.stb-brand-mark {
    width: 2.65rem;
    height: 2.65rem;
    display: grid;
    place-items: center;
    border-radius: 0.8rem;
    color: var(--mint);
    background: var(--forest);
}

.stb-brand-name {
    font: 800 1.2rem/1 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.03em;
}

.stb-brand-sub {
    margin-top: 0.25rem;
    color: rgba(16, 38, 28, 0.6);
    font: 500 0.72rem/1.2 'DM Sans', sans-serif;
}

.stb-meta {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.45rem;
    flex-wrap: wrap;
}

.stb-meta span {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    font: 700 0.7rem/1 'DM Sans', sans-serif;
}

.stb-meta-input {
    color: var(--fern);
    background: var(--moss);
}

.stb-meta-ready {
    color: #fff;
    background: var(--forest);
}

.stb-hero {
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: minmax(0, 1.08fr) minmax(290px, 0.92fr);
    align-items: center;
    gap: 3.2rem;
    padding: 4.2rem 4.5rem;
    color: #fff;
    background:
        radial-gradient(circle at 88% 12%, rgba(110, 199, 143, 0.25), transparent 28%),
        radial-gradient(circle at 8% 90%, rgba(62, 143, 95, 0.2), transparent 30%),
        linear-gradient(135deg, #092c1d 0%, #0b3b29 55%, #145237 100%);
}

.stb-hero::after {
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
    opacity: 0.16;
    background-image: radial-gradient(rgba(217, 249, 226, 0.45) 0.8px, transparent 0.8px);
    background-size: 26px 26px;
    mask-image: linear-gradient(to bottom right, black, transparent 70%);
}

.stb-hero > * {
    position: relative;
    z-index: 1;
}

.stb-hero h1 {
    max-width: 680px;
    margin: 0;
    color: #fff;
    font: 800 clamp(2.6rem, 5.4vw, 5rem)/1.04 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.055em;
}

.stb-hero h1 span {
    color: var(--mint);
}

.stb-hero-copy {
    max-width: 590px;
    margin: 1.3rem 0 0;
    color: rgba(255, 255, 255, 0.75);
    font: 400 1.05rem/1.65 'DM Sans', sans-serif;
}

.stb-hero-points {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem 1.2rem;
    margin-top: 1.8rem;
    color: rgba(255, 255, 255, 0.72);
    font: 600 0.78rem/1.3 'DM Sans', sans-serif;
}

.stb-hero-points span {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.stb-hero-points .material-symbols-outlined {
    color: var(--mint);
    font-size: 1rem;
}

.stb-preview {
    position: relative;
    overflow: hidden;
    padding: 1.15rem;
    border-radius: 1rem;
    color: var(--ink);
    background: #f6faf2;
    border: 1px solid rgba(16, 38, 28, 0.08);
    box-shadow: 0 20px 45px -25px rgba(0, 0, 0, 0.65);
}

.stb-preview::before {
    content: '';
    position: absolute;
    width: 13rem;
    height: 13rem;
    top: -7rem;
    right: -5rem;
    border: 1px solid rgba(23, 99, 63, 0.12);
    border-radius: 50%;
    pointer-events: none;
}

.stb-preview-top, .stb-preview-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.7rem;
}

.stb-preview-top {
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(16, 38, 28, 0.1);
}

.stb-preview-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font: 700 0.78rem/1.2 'DM Sans', sans-serif;
}

.stb-preview-title .material-symbols-outlined {
    display: grid;
    place-items: center;
    width: 2rem;
    height: 2rem;
    border-radius: 0.55rem;
    color: var(--fern);
    background: var(--moss);
}

.stb-preview-note {
    margin-top: 0.15rem;
    color: rgba(16, 38, 28, 0.5);
    font: 400 0.67rem/1.2 'DM Sans', sans-serif;
}

.stb-preview-status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.55rem;
    border-radius: 999px;
    color: var(--fern);
    background: var(--moss);
    font: 700 0.64rem/1 'DM Sans', sans-serif;
}

.stb-preview-status::before {
    content: '';
    width: 0.38rem;
    height: 0.38rem;
    border-radius: 50%;
    background: #4caa70;
}

.stb-preview-image {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    display: grid;
    place-items: center;
    height: 9.5rem;
    margin: 1rem 0;
    border-radius: 0.75rem;
    color: var(--fern);
    background:
        radial-gradient(circle at 50% 42%, rgba(255, 255, 255, 0.7), transparent 28%),
        linear-gradient(135deg, #dcebdc, #f5f6ec);
}

.stb-preview-image::before,
.stb-preview-image::after {
    content: '';
    position: absolute;
    z-index: 0;
    border: 1px solid rgba(23, 99, 63, 0.13);
    border-radius: 50%;
}

.stb-preview-image::before {
    width: 12rem;
    height: 12rem;
}

.stb-preview-image::after {
    width: 17rem;
    height: 17rem;
    opacity: 0.55;
}

.stb-preview-sample-label {
    position: absolute;
    z-index: 2;
    top: 0.75rem;
    left: 0.8rem;
    color: rgba(16, 38, 28, 0.45);
    font: 700 0.58rem/1 'DM Sans', sans-serif;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.stb-preview-object {
    position: relative;
    z-index: 1;
    display: grid;
    place-items: center;
    width: 5.5rem;
    height: 5.5rem;
    border: 0.65rem solid rgba(255, 255, 255, 0.92);
    border-radius: 50%;
    color: var(--fern);
    background: #e8f4e8;
    box-shadow: 0 10px 22px -16px rgba(11, 59, 41, 0.75);
}

.stb-preview-image .material-symbols-outlined {
    display: grid;
    place-items: center;
    font-size: 2.2rem;
}

.stb-preview-sample-state {
    position: absolute;
    z-index: 2;
    right: 0.8rem;
    bottom: 0.75rem;
    color: rgba(16, 38, 28, 0.5);
    font: 700 0.58rem/1 'DM Sans', sans-serif;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stb-preview-label {
    color: rgba(16, 38, 28, 0.5);
    font: 700 0.62rem/1.2 'DM Sans', sans-serif;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.stb-preview-result {
    color: var(--forest);
    font: 800 1.45rem/1.1 'Plus Jakarta Sans', sans-serif;
}

.stb-preview-confidence {
    display: flex;
    align-items: baseline;
    gap: 0.25rem;
    color: var(--fern);
    font: 800 1.35rem/1 'Plus Jakarta Sans', sans-serif;
}

.stb-preview-confidence small {
    color: rgba(16, 38, 28, 0.5);
    font: 700 0.62rem/1 'DM Sans', sans-serif;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.stb-progress {
    height: 0.45rem;
    margin-top: 0.55rem;
    overflow: hidden;
    border-radius: 999px;
    background: var(--sand);
}

.stb-progress span {
    display: block;
    width: 94%;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #68c78b, #b6e9c4);
}

.stb-preview-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.7rem;
    margin-top: 0.75rem;
    color: rgba(16, 38, 28, 0.58);
    font: 600 0.65rem/1.3 'DM Sans', sans-serif;
}

.stb-preview-nature {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    color: var(--fern);
}

.stb-preview-nature .material-symbols-outlined {
    font-size: 0.9rem;
}

.stb-insight {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.9rem;
    padding: 0.8rem 0.85rem;
    border: 1px solid rgba(23, 99, 63, 0.12);
    border-radius: 0.75rem;
    background: #edf8ef;
    color: var(--fern);
    font: 400 0.72rem/1.5 'DM Sans', sans-serif;
}

.stb-insight strong {
    font-weight: 700;
}

.stb-section {
    padding: 4rem 0 0;
}

.stb-section h2 {
    margin: 0;
    color: var(--forest);
    font: 800 clamp(1.8rem, 3.3vw, 3rem)/1.1 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.04em;
}

.stb-section-intro {
    max-width: 650px;
    margin: 0.9rem 0 0;
    color: rgba(16, 38, 28, 0.65);
    font: 400 1rem/1.65 'DM Sans', sans-serif;
}

.stb-step-grid, .stb-class-grid {
    display: grid;
    gap: 1.4rem;
}

.stb-step-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 2.2rem;
}

.stb-step {
    padding-top: 1.1rem;
    border-top: 2px solid var(--fern);
}

.stb-step-number {
    color: var(--moss);
    font: 800 2.8rem/1 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.06em;
}

.stb-step h3 {
    margin: 1rem 0 0;
    color: var(--forest);
    font: 700 1.1rem/1.3 'Plus Jakarta Sans', sans-serif;
}

.stb-step p {
    margin: 0.55rem 0 0;
    color: rgba(16, 38, 28, 0.65);
    font: 400 0.87rem/1.6 'DM Sans', sans-serif;
}

.stb-class-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-top: 1.7rem;
}

.stb-class-item {
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(16, 38, 28, 0.12);
}

.stb-class-item .material-symbols-outlined {
    color: var(--fern);
}

.stb-class-name {
    margin-top: 0.45rem;
    color: var(--forest);
    font: 700 0.84rem/1.25 'DM Sans', sans-serif;
}

.stb-class-sub {
    margin-top: 0.2rem;
    color: rgba(16, 38, 28, 0.5);
    font: 400 0.72rem/1.3 'DM Sans', sans-serif;
}

.stb-about {
    display: block;
    width: 100%;
    box-sizing: border-box;
    margin-top: 4rem;
    padding: 2.5rem;
    color: #fff;
    background: var(--forest);
}

.stb-about h2 {
    color: #fff;
    max-width: 100%;
    font-size: clamp(1.75rem, 3vw, 2.6rem);
}

.stb-about p {
    margin: 0.9rem 0 0;
    color: rgba(255, 255, 255, 0.7);
    font: 400 0.9rem/1.65 'DM Sans', sans-serif;
}

.stb-about-copy h3 {
    margin: 0;
    color: var(--forest);
    font: 800 clamp(1.8rem, 3.3vw, 3rem)/1.1 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.04em;
}

.stb-about-copy p {
    color: rgba(16, 38, 28, 0.65);
}

.stb-footer {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-top: 4rem;
    padding-top: 1.3rem;
    border-top: 1px solid rgba(16, 38, 28, 0.1);
    color: rgba(16, 38, 28, 0.55);
    font: 400 0.75rem/1.5 'DM Sans', sans-serif;
}

@media (max-width: 900px) {
    .block-container { padding: 1rem 1.25rem 2.5rem; }
    .stb-hero { grid-template-columns: 1fr; padding: 3rem 2rem; }
    .stb-preview { max-width: 30rem; }
    .stb-class-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 640px) {
    .stb-nav { align-items: flex-start; }
    .stb-brand-sub { display: none; }
    .stb-meta { justify-content: flex-end; }
    .stb-meta-input { display: none !important; }
    .stb-hero { padding: 2.5rem 1.25rem; }
    .stb-hero-copy { font-size: 0.95rem; }
    .stb-step-grid { grid-template-columns: 1fr; }
    .stb-about { padding: 1.5rem; }
    .stb-footer { flex-direction: column; }
}
</style>
"""
)


render_navbar("home")


render_html(
    """
<section class="stb-hero">
    <div>
        <h1>Kenali sampah.<br><span>Pilah lebih tepat.</span></h1>
        <p class="stb-hero-copy">SORTIVA membantu kamu mengenali jenis sampah melalui kamera atau unggahan gambar menggunakan model machine learning buatan Stembayo.</p>
        <div class="stb-hero-points">
            <span><span class="material-symbols-outlined">check_circle</span>Berbasis gambar</span>
            <span><span class="material-symbols-outlined">check_circle</span>Hasil mudah dipahami</span>
            <span><span class="material-symbols-outlined">check_circle</span>Untuk pembelajaran Adiwiyata</span>
        </div>
    </div>
    <div class="stb-preview">
        <div class="stb-preview-top">
            <div class="stb-preview-title"><span class="material-symbols-outlined">auto_awesome</span><div><div>Hasil klasifikasi</div><div class="stb-preview-note">Contoh tampilan SORTIVA</div></div></div>
            <span class="stb-preview-status">Selesai</span>
        </div>
        <div class="stb-preview-image">
            <span class="stb-preview-sample-label">Visual sample / 01</span>
            <div class="stb-preview-object"><span class="material-symbols-outlined">water_bottle</span></div>
            <span class="stb-preview-sample-state">Visual match</span>
        </div>
        <div class="stb-preview-row"><div><div class="stb-preview-label">Jenis terdeteksi</div><div class="stb-preview-result">Plastik</div></div><div class="stb-preview-confidence">94% <small>Tinggi</small></div></div>
        <div class="stb-progress" aria-label="Tingkat keyakinan 94 persen"><span></span></div>
        <div class="stb-preview-meta"><span class="stb-preview-nature"><span class="material-symbols-outlined">recycling</span> Non-Biodegradable</span><span>8 kelas material</span></div>
        <div class="stb-insight"><span class="material-symbols-outlined">lightbulb</span><span><strong>Insight:</strong> pastikan botol kosong sebelum dipilah agar lebih mudah diproses.</span></div>
    </div>
</section>
"""
)


render_html(
    """
<section class="stb-section">
    <h2>Dari gambar menjadi pemahaman.</h2>
    <p class="stb-section-intro">Masukkan gambar, lihat hasil prediksi model, lalu gunakan informasinya untuk membiasakan pemilahan yang lebih baik.</p>
    <div class="stb-step-grid">
        <article class="stb-step"><div class="stb-step-number">01</div><h3>Ambil atau unggah gambar</h3><p>Gunakan konsol kamera atau unggah foto sampah dari perangkatmu.</p></article>
        <article class="stb-step"><div class="stb-step-number">02</div><h3>Model membaca objek</h3><p>Model machine learning SORTIVA menganalisis ciri visual untuk memperkirakan jenis material.</p></article>
        <article class="stb-step"><div class="stb-step-number">03</div><h3>Pahami hasilnya</h3><p>Hasil klasifikasi, confidence, dan insight ditampilkan secara ringkas.</p></article>
    </div>
</section>
"""
)


render_html(
    """
<section class="stb-section">
    <h2>Kenali lebih banyak jenis sampah.</h2>
    <p class="stb-section-intro">Model SORTIVA dilatih untuk mengenali material yang sering ditemui di lingkungan sekolah.</p>
    <div class="stb-class-grid">
        <div class="stb-class-item"><span class="material-symbols-outlined">water_bottle</span><div class="stb-class-name">Plastik</div><div class="stb-class-sub">Plastic</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">lunch_dining</span><div class="stb-class-name">Logam</div><div class="stb-class-sub">Metal</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">wine_bar</span><div class="stb-class-name">Kaca</div><div class="stb-class-sub">Glass</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">description</span><div class="stb-class-name">Kertas</div><div class="stb-class-sub">Paper</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">inventory_2</span><div class="stb-class-name">Kardus</div><div class="stb-class-sub">Cardboard</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">compost</span><div class="stb-class-name">Organik</div><div class="stb-class-sub">Organic waste</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">cable</span><div class="stb-class-name">E-waste</div><div class="stb-class-sub">Electronic waste</div></div>
        <div class="stb-class-item"><span class="material-symbols-outlined">medical_services</span><div class="stb-class-name">Medis</div><div class="stb-class-sub">Medical waste</div></div>
    </div>
</section>
"""
)


about_col, action_col = st.columns([1, 1.15], gap="large", vertical_alignment="center")
with about_col:
    render_html(
        """
<div class="stb-about"><div><span class="material-symbols-outlined" style="font-size:3rem; color:#b8e8c8;">school</span><h2>Belajar memilah dimulai dari mengenali.</h2><p>Setiap klasifikasi menjadi kesempatan untuk belajar mengambil keputusan yang lebih bertanggung jawab.</p></div></div>
"""
    )
with action_col:
    render_html(
        """
<div class="stb-about-copy"><h3>Teknologi yang dekat dengan keseharian sekolah.</h3><p>SORTIVA menggabungkan antarmuka yang sederhana dengan model computer vision agar proses mengenali sampah terasa jelas dan mudah dipahami.</p></div>
"""
    )
    cta1, cta2 = st.columns(2, gap="small")
    with cta1:
        st.page_link("app.py", label="Buka Konsol Kamera", icon=":material/photo_camera:")
    with cta2:
        st.page_link("pages/dashboard.py", label="Lihat Dashboard", icon=":material/analytics:")


render_html(
    """
<footer class="stb-footer"><span>© 2026 SORTIVA · SMK Negeri 2 Depok Sleman</span><span>Aplikasi klasifikasi sampah berbasis machine learning</span></footer>
"""
)
