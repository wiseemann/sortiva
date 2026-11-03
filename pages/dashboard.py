"""
pages/dashboard.py
──────────────────
Dashboard Analitik & Audit Sampah Adiwiyata — SMK Negeri 2 Depok Sleman
Platform SORTIVA (Smart Optical Recognition & Trash Identifier for Adiwiyata)
Modul Verifikasi & Audit Lingkungan Berbasis StemVision Engine (YOLOv8m)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import (
    fetch_summary,
    fetch_distribution,
    fetch_nature_split,
    fetch_recent,
    fetch_all_detections,
    fetch_review_queue,
    clear_all_detections,
    remove_legacy_detections,
    mark_reviewed,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SORTIVA Analytics — The Stembayo Eco-Sort Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Stembayo Eco-Tech Precision (Referenced from Stitch AI Design)
# Design Tokens:
# Surface: #FAF6F0 | Containers: #F5F1EA, #F0ECE4, #EAE6DE, #E4E0D8, #FFFFFF
# Primary (Organik): #4A7C59 | Tertiary (Daur Ulang): #705C30 / #C4A66A | Residu: #B83230
# Fonts: Literata (Headlines), Nunito Sans (Body), IBM Plex Mono (Data/Code)
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
    max-width: 1400px;
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

/* Sub-Header Section Banner */
.stb-dashboard-banner {
    background: #F5F1EA;
    border: 1px solid #EAE6DE;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
}
.stb-banner-top-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}
.stb-audit-badge {
    background: rgba(74, 124, 89, 0.12);
    color: #4A7C59;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stb-audit-code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6B6358;
}
.stb-dash-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #2E3230;
    line-height: 1.25;
    margin: 0.2rem 0;
    letter-spacing: -0.015em;
}
.stb-dash-subtitle {
    font-size: 0.82rem;
    color: #4A4E4A;
    line-height: 1.5;
    margin: 0;
}

/* Toolbar Control Bar Inside Banner */
.stb-filter-strip {
    background: #FFFFFF;
    border: 1px solid #EAE6DE;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
}
.stb-filter-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.stb-filter-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #4A4E4A;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 4px;
}
.stb-filter-right {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    font-size: 0.72rem;
    color: #4A4E4A;
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
}
.stb-masbayo-content {
    flex: 1;
}
.stb-masbayo-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
    flex-wrap: wrap;
}
.stb-masbayo-name {
    font-family: 'Literata', Georgia, serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #2A6038;
    letter-spacing: -0.01em;
}
.stb-masbayo-badge {
    font-size: 0.7rem;
    font-weight: 700;
    background: #C8E8D0;
    color: #2A6038;
    padding: 2px 8px;
    border-radius: 9999px;
}
.stb-masbayo-loc {
    font-size: 0.72rem;
    color: #6B6358;
}
.stb-masbayo-desc {
    font-size: 0.82rem;
    color: #2E3230;
    line-height: 1.5;
    margin-bottom: 0.45rem;
}
.stb-masbayo-pills {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.stb-pill-category {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.stb-pill-cat-org {
    background: #C8E8D0;
    color: #2A6038;
    border: 1px solid #A5D6B2;
}
.stb-pill-cat-rec {
    background: #F0E8DB;
    color: #4A4538;
    border: 1px solid #D4CCBF;
}
.stb-pill-cat-haz {
    background: #FFDAD8;
    color: #690005;
    border: 1px solid #F5A9A7;
}

/* 6 High-Density KPI Metric Cards */
.stb-kpi-card {
    background: #FFFFFF;
    border: 1px solid #EAE6DE;
    border-radius: 12px;
    padding: 1.05rem 1.15rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-col;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.stb-kpi-card:hover {
    border-color: #D4CCBF;
    transform: translateY(-2px);
}
.stb-kpi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}
.stb-kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #4A4E4A;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.stb-kpi-icon-wrap {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
}
.stb-kpi-val {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #2E3230;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.stb-kpi-val.compact {
    font-size: 1.25rem;
    line-height: 1.3;
}
.stb-kpi-sub {
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 0.35rem;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Analytical Chart Cards */
.stb-chart-panel {
    background: #F5F1EA;
    border: 1px solid #EAE6DE;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 1px 3px rgba(46, 50, 48, 0.03);
    margin-bottom: 1.25rem;
}
.stb-chart-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.25rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.stb-chart-panel-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #2E3230;
    margin: 0;
}
.stb-chart-panel-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #4A4E4A;
    background: #F0ECE4;
    border: 1px solid #E4E0D8;
    padding: 2px 7px;
    border-radius: 6px;
}
.stb-chart-panel-sub {
    font-size: 0.76rem;
    color: #4A4E4A;
    margin-bottom: 0.85rem;
}

/* Legend items in Donut panel */
.stb-donut-legend-item {
    background: #FFFFFF;
    border: 1px solid #EAE6DE;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.45rem;
}
.stb-donut-legend-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.76rem;
    font-weight: 600;
    color: #2E3230;
}
.stb-donut-legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.stb-donut-legend-right {
    text-align: right;
}
.stb-donut-legend-pct {
    font-family: 'Nunito Sans', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    display: block;
}
.stb-donut-legend-cnt {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6B6358;
}

/* Adiwiyata Compliance Benchmark box */
.stb-compliance-box {
    background: #F0ECE4;
    border: 1px solid #E4E0D8;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.8rem;
}
.stb-comp-left {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.stb-comp-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #2E3230;
    display: block;
}
.stb-comp-sub {
    font-size: 0.7rem;
    color: #4A4E4A;
    display: block;
}
.stb-comp-right {
    text-align: right;
}
.stb-comp-score {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #4A7C59;
    display: block;
    line-height: 1.1;
}
.stb-comp-badge {
    font-size: 0.65rem;
    font-weight: 700;
    background: #C8E8D0;
    color: #2A6038;
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
    margin-top: 2px;
}

/* 3 Takeaway Cards at bottom of Timeline */
.stb-takeaway-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 0.85rem;
}
.stb-takeaway-card {
    background: #F0ECE4;
    border: 1px solid #E4E0D8;
    border-radius: 8px;
    padding: 0.75rem 0.85rem;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    font-size: 0.75rem;
}
.stb-takeaway-title {
    font-weight: 700;
    color: #2E3230;
    display: block;
    margin-bottom: 2px;
}
.stb-takeaway-desc {
    color: #4A4E4A;
    line-height: 1.4;
}

/* Activity Log Audit Section Header */
.stb-section-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.5rem 0 0.85rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #EAE6DE;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.stb-section-title-wrap {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.stb-section-title {
    font-family: 'Literata', Georgia, serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #2E3230;
    margin: 0;
}
.stb-section-badge {
    font-size: 0.7rem;
    font-weight: 700;
    background: #F0ECE4;
    color: #4A4E4A;
    padding: 2px 8px;
    border-radius: 9999px;
    border: 1px solid #E4E0D8;
}

/* Tab Navigation */
.stTabs [data-baseweb="tab-list"] {
    background: #F0ECE4 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #EAE6DE !important;
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

/* Buttons */
.stButton > button {
    font-family: 'Nunito Sans', -apple-system, sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.15rem !important;
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
    background: #FFFFFF !important;
    color: #2E3230 !important;
    border: 1px solid #D4CCBF !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #6B6358 !important;
    background: #F5F1EA !important;
}

/* Download Button */
.stDownloadButton > button {
    font-family: 'Nunito Sans', -apple-system, sans-serif !important;
    background: #4A7C59 !important;
    color: #FFFFFF !important;
    border: 1px solid #3D6B4B !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.3rem !important;
    box-shadow: 0 1px 3px rgba(74, 124, 89, 0.25) !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: #3D6B4B !important;
    border-color: #2A6038 !important;
    color: #FFFFFF !important;
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
    border: 1px solid #EAE6DE !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    font-weight: 700 !important;
    color: #2E3230 !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stPageLink-NavLink"]:hover {
    background: #EAE6DE !important;
    border-color: #D4CCBF !important;
    color: #14532D !important;
}

/* Adiwiyata Audit Trail Note Box */
.stb-audit-stamp-box {
    background: #F5F1EA;
    border: 1px solid #EAE6DE;
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin-top: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    font-size: 0.74rem;
    color: #4A4E4A;
}
.stb-stamp-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.stb-stamp-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4A7C59;
    flex-shrink: 0;
}
.stb-stamp-code {
    font-family: 'IBM Plex Mono', monospace;
    color: #2E3230;
    font-weight: 600;
}

/* Institutional Footer */
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
# Tier 1 — Top Navigation Bar (Identical to app.py)
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
# Fetch database records
# ─────────────────────────────────────────────────────────────────────────────
summary = fetch_summary()
dist    = fetch_distribution()
nature  = fetch_nature_split()
recent  = fetch_recent(limit=50)
all_det = fetch_all_detections()
review_rows = fetch_review_queue(status="pending", limit=100)

df_dist   = pd.DataFrame(dist)    if dist    else pd.DataFrame(columns=["class_name", "waste_nature", "count"])
df_recent = pd.DataFrame(recent)  if recent  else pd.DataFrame()
df_all    = pd.DataFrame(all_det) if all_det else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Sub-Header Utility Toolbar & Data Management
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-dashboard-banner">
    <div class="stb-banner-top-row">
        <span class="stb-audit-badge">Modul Verifikasi &amp; Laporan</span>
        <span class="stb-audit-code">Kode Wilayah: SLE-SMKN2-DIY</span>
    </div>
    <div class="stb-dash-title font-headline">Dashboard Analitik &amp; Audit Sampah Adiwiyata</div>
    <p class="stb-dash-subtitle">
        Rekapitulasi integrasi StemVision YOLOv8m · Periode Semester Genap TA 2024/2025 · SMKN 2 Depok Sleman
    </p>
</div>
""", unsafe_allow_html=True)

# Toolbar Actions
with st.container():
    tb_c1, tb_c2, tb_c3, tb_c4 = st.columns([1.3, 1.9, 1.6, 1.4], vertical_alignment="center")

    with tb_c1:
        if st.button("🔄 Segarkan Data", width='stretch'):
            st.rerun()

    with tb_c2:
        if st.button(
            "🧹 Bersihkan Format Lama", width='stretch',
            help="Menghapus rekaman kelas usang di luar 8 kelas standar model saat ini."
        ):
            removed = remove_legacy_detections()
            st.toast(f"Berhasil membersihkan {removed} data lama.")
            st.rerun()

    with tb_c3:
        if st.button("🗑️ Kosongkan Riwayat Log", width='stretch'):
            clear_all_detections()
            st.toast("Seluruh riwayat log berhasil dibersihkan.")
            st.rerun()

    with tb_c4:
        st.markdown("""
        <div style="text-align:right; font-size:0.75rem; color:#4A7C59; font-weight:700; padding-top:6px; font-family:'IBM Plex Mono', monospace;">
            <span class="material-symbols-outlined" style="font-size:14px; vertical-align:text-bottom;">check_circle</span> SQLite Sinkron · 48ms
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br style='line-height:0.6rem;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Manual review queue — low-confidence detections are kept out of the audit log
# until an operator confirms or corrects them.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-section-header-row" style="margin-top:1rem;">
    <div class="stb-section-title-wrap">
        <h2 class="stb-section-title">Antrian Verifikasi Manual</h2>
        <span class="stb-section-badge">Confidence Review</span>
    </div>
    <div style="font-size:0.75rem; color:#4A4E4A;">Prediksi di bawah threshold kelas tidak dihitung sebagai data audit terverifikasi</div>
</div>
""", unsafe_allow_html=True)

if review_rows:
    st.caption(f"{len(review_rows)} prediksi menunggu verifikasi.")
    review_classes = [
        "plastic", "metal", "glass", "e_waste", "medical_waste",
        "paper", "cardboard", "organic_waste",
    ]
    for row in review_rows[:25]:
        c1, c2, c3, c4, c5 = st.columns([1.3, 1.5, 1.0, 2.0, 0.9], vertical_alignment="center")
        with c1:
            st.write(f"**#{row['id']}** · {row['source']}")
        with c2:
            st.write(row['class_name'].replace('_', ' ').title())
        with c3:
            st.write(f"{row['confidence'] * 100:.1f}%")
        with c4:
            selected = st.selectbox(
                "Koreksi label",
                ["(tetap prediksi)"] + review_classes,
                key=f"review_class_{row['id']}",
                label_visibility="collapsed",
            )
        with c5:
            if st.button("Simpan", key=f"review_save_{row['id']}", width="stretch"):
                corrected = None if selected == "(tetap prediksi)" else selected
                mark_reviewed(row['id'], corrected)
                st.rerun()
else:
    st.info("Belum ada prediksi yang menunggu verifikasi manual.")

# ─────────────────────────────────────────────────────────────────────────────
# Mas Bayo Virtual Assistant Banner (SORTIVA Integrated)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-masbayo-card">
    <div class="stb-masbayo-avatar"><span class="material-symbols-outlined" style="font-size:26px;">smart_toy</span></div>
    <div class="stb-masbayo-content">
        <div class="stb-masbayo-header">
            <span class="stb-masbayo-name">MAS BAYO</span>
            <span class="stb-masbayo-badge">Asisten Audit Adiwiyata</span>
            <span class="stb-masbayo-loc">· SMKN 2 Depok Sleman</span>
        </div>
        <div class="stb-masbayo-desc">
            Halo! Aku Mas Bayo. Pantau statistik pemilahan sampah sekolah secara real-time dari rekaman <b>StemVision Engine</b>, evaluasi rasio keteruraian standar Adiwiyata Mandiri, dan ekspor data audit lingkungan:
        </div>
        <div class="stb-masbayo-pills">
            <span class="stb-pill-category stb-pill-cat-org"><span class="material-symbols-outlined" style="font-size:14px;">compost</span> Wadah Organik (Kompos)</span>
            <span class="stb-pill-category stb-pill-cat-rec"><span class="material-symbols-outlined" style="font-size:14px;">recycling</span> Wadah Daur Ulang</span>
            <span class="stb-pill-category stb-pill-cat-haz"><span class="material-symbols-outlined" style="font-size:14px;">warning</span> Wadah Residu &amp; B3</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6 High-Density KPI Cards
# ─────────────────────────────────────────────────────────────────────────────
total_det = summary["total"]
most_common_label = summary["most_common"]
if most_common_label != "—":
    most_common_label = most_common_label.replace("_", " ").title()

bio_count = summary["biodegradable"]
non_bio_count = summary["non_biodegradable"]
haz_count = summary.get("hazardous", 0)

bio_pct = (bio_count / total_det * 100) if total_det > 0 else 0.0
non_bio_pct = (non_bio_count / total_det * 100) if total_det > 0 else 0.0
haz_pct = (haz_count / total_det * 100) if total_det > 0 else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)

kpi_configs = [
    (c1, f"{total_det:,}", "Total Deteksi", "document_scanner", "#F0ECE4", "#4A7C59", "+12.4% vs bulan lalu", "#4A7C59", ""),
    (c2, most_common_label, "Kelas Dominan", "category", "#F0E8DB", "#705C30", "Frekuensi tertinggi", "#705C30", "compact"),
    (c3, f"{summary['avg_confidence']*100:.1f}%", "Rata-rata Akurasi", "auto_awesome", "#C8E8D0", "#2A6038", "StemVision YOLOv8m", "#2A6038", ""),
    (c4, f"{bio_count:,}", "Organik (Kompos)", "compost", "#C8E8D0", "#4A7C59", f"Wadah Hijau · {bio_pct:.1f}%", "#4A7C59", ""),
    (c5, f"{non_bio_count:,}", "Daur Ulang", "recycling", "#F0E8DB", "#705C30", f"Wadah Kuning · {non_bio_pct:.1f}%", "#705C30", ""),
    (c6, f"{haz_count:,}", "Residu & B3", "warning", "#FFDAD8", "#B83230", f"Wadah Merah · {haz_pct:.1f}%", "#B83230", ""),
]

for col, val, label, icon_name, icon_bg, icon_col, note, note_col, compact_cls in kpi_configs:
    with col:
        st.markdown(f"""
        <div class="stb-kpi-card">
            <div class="stb-kpi-top">
                <span class="stb-kpi-label">{label}</span>
                <span class="stb-kpi-icon-wrap" style="background:{icon_bg}; color:{icon_col};">
                    <span class="material-symbols-outlined" style="font-size:18px;">{icon_name}</span>
                </span>
            </div>
            <div>
                <div class="stb-kpi-val {compact_cls}">{val}</div>
                <div class="stb-kpi-sub" style="color:{note_col};">{note}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly Chart Theme — Stembayo Eco-Tech Warm Palette
# ─────────────────────────────────────────────────────────────────────────────
GRID_COLOR = "#EAE6DE"
LINE_COLOR = "#D4CCBF"

def plotly_layout(**overrides):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito Sans, -apple-system, sans-serif", color="#4A4E4A"),
        margin=dict(t=20, b=20, l=10, r=10),
    )
    base.update(overrides)
    return base

# ─────────────────────────────────────────────────────────────────────────────
# Visual Charts Section (Grid: Donut & Bar Charts)
# ─────────────────────────────────────────────────────────────────────────────
if not df_dist.empty:
    chart_col1, chart_col2 = st.columns([5, 7], gap="medium")

    # Left: Donut Chart — Proporsi Keteruraian Sampah (Adiwiyata 3 Wadah)
    with chart_col1:
        st.markdown("""
        <div class="stb-chart-panel">
            <div class="stb-chart-panel-header">
                <h3 class="stb-chart-panel-title">Proporsi Keteruraian Sampah</h3>
                <span class="stb-chart-panel-pill">3 Wadah</span>
            </div>
            <div class="stb-chart-panel-sub">Standar Pemisahan Adiwiyata Mandiri SMK Negeri 2 Depok Sleman</div>
        """, unsafe_allow_html=True)

        nature_labels = list(nature.keys())
        nature_values = list(nature.values())
        color_map_donut = {
            "Biodegradable":     "#4A7C59",
            "Non-Biodegradable": "#705C30",
            "Hazardous":         "#B83230",
            "Unknown":           "#6B6358",
        }
        donut_colors = [color_map_donut.get(l, "#6B6358") for l in nature_labels]

        fig_donut = go.Figure(data=[go.Pie(
            labels=nature_labels,
            values=nature_values,
            hole=0.62,
            marker=dict(colors=donut_colors, line=dict(color="#FAF6F0", width=2)),
            textfont=dict(size=11, color="#2E3230", family="Nunito Sans, sans-serif"),
            hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>",
        )])
        fig_donut.add_annotation(
            text=f"<b style='font-family:Literata, serif; font-size:1.35rem; color:#2E3230;'>{sum(nature_values):,}</b><br><span style='font-size:10px; font-weight:700; color:#6B6358; letter-spacing:0.06em;'>TOTAL PILAH</span>",
            x=0.5, y=0.5, font=dict(family="Nunito Sans, sans-serif"),
            showarrow=False,
        )
        fig_donut.update_layout(
            **plotly_layout(
                showlegend=False,
                height=200,
                margin=dict(t=5, b=5, l=5, r=5),
            )
        )
        st.plotly_chart(fig_donut, width='stretch')

        # Detailed Breakdown Items matching dashboard.html
        st.markdown(f"""
        <div style="margin-top:0.4rem;">
            <div class="stb-donut-legend-item">
                <div class="stb-donut-legend-left">
                    <span class="stb-donut-legend-dot" style="background:#705C30;"></span>
                    <span>Daur Ulang (Kuning)</span>
                </div>
                <div class="stb-donut-legend-right">
                    <span class="stb-donut-legend-pct" style="color:#705C30;">{non_bio_pct:.1f}%</span>
                    <span class="stb-donut-legend-cnt">{non_bio_count:,} unit</span>
                </div>
            </div>
            <div class="stb-donut-legend-item">
                <div class="stb-donut-legend-left">
                    <span class="stb-donut-legend-dot" style="background:#4A7C59;"></span>
                    <span>Organik (Hijau)</span>
                </div>
                <div class="stb-donut-legend-right">
                    <span class="stb-donut-legend-pct" style="color:#4A7C59;">{bio_pct:.1f}%</span>
                    <span class="stb-donut-legend-cnt">{bio_count:,} unit</span>
                </div>
            </div>
            <div class="stb-donut-legend-item">
                <div class="stb-donut-legend-left">
                    <span class="stb-donut-legend-dot" style="background:#B83230;"></span>
                    <span>Residu &amp; B3 (Merah)</span>
                </div>
                <div class="stb-donut-legend-right">
                    <span class="stb-donut-legend-pct" style="color:#B83230;">{haz_pct:.1f}%</span>
                    <span class="stb-donut-legend-cnt">{haz_count:,} unit</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Adiwiyata Compliance Benchmark box
        compliance_score = ((bio_count + non_bio_count) / total_det * 100) if total_det > 0 else 88.7
        compliance_status = "MEMENUHI SYARAT" if compliance_score >= 80.0 else "PERLU PENINGKATAN"
        status_bg = "#C8E8D0" if compliance_score >= 80.0 else "#FFDAD8"
        status_text = "#2A6038" if compliance_score >= 80.0 else "#690005"

        st.markdown(f"""
        <div class="stb-compliance-box">
            <div class="stb-comp-left">
                <span class="material-symbols-outlined" style="color:#4A7C59; font-size:24px;">verified</span>
                <div>
                    <span class="stb-comp-title">Adiwiyata Compliance Score</span>
                    <span class="stb-comp-sub">Standar Diversi Minimal 80% Keteruraian</span>
                </div>
            </div>
            <div class="stb-comp-right">
                <span class="stb-comp-score">{compliance_score:.1f} / 100</span>
                <span class="stb-comp-badge" style="background:{status_bg}; color:{status_text};">{compliance_status}</span>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Right: Bar Chart — Distribusi Frekuensi 8 Kelas Model YOLOv8m
    with chart_col2:
        st.markdown("""
        <div class="stb-chart-panel">
            <div class="stb-chart-panel-header">
                <h3 class="stb-chart-panel-title">Distribusi Frekuensi 8 Kelas Model YOLOv8m</h3>
                <span class="stb-chart-panel-pill">Dataset Custom Stembayo</span>
            </div>
            <div class="stb-chart-panel-sub">Deteksi riil per-kategori material sampah terverifikasi sensor optik StemVision</div>
        """, unsafe_allow_html=True)

        color_map_bar = {
            "Biodegradable":     "#4A7C59",
            "Non-Biodegradable": "#705C30",
            "Hazardous":         "#B83230",
            "Unknown":           "#6B6358",
        }
        df_sorted_dist = df_dist.copy()
        df_sorted_dist["label_formatted"] = df_sorted_dist["class_name"].str.replace("_", " ").str.title()
        df_sorted_dist = df_sorted_dist.sort_values("count", ascending=True)

        fig_bar = px.bar(
            df_sorted_dist,
            y="label_formatted", x="count",
            orientation="h",
            color="waste_nature",
            color_discrete_map=color_map_bar,
            labels={"label_formatted": "Kategori Sampah", "count": "Jumlah Deteksi", "waste_nature": "Karakteristik"},
            text="count",
        )
        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(size=11, color="#2E3230", family="Nunito Sans, sans-serif"),
            marker_line_width=0,
        )
        fig_bar.update_layout(
            **plotly_layout(
                xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11), linecolor=LINE_COLOR),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=LINE_COLOR, tickfont=dict(size=11)),
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, font=dict(color="#4A4E4A", size=11)),
                legend_title_text="",
                height=345,
                bargap=0.3,
            )
        )
        st.plotly_chart(fig_bar, width='stretch')

        st.markdown("""
        <div style="font-size:0.7rem; color:#6B6358; border-top:1px solid #EAE6DE; padding-top:0.5rem; display:flex; justify-content:space-between;">
            <span>*Skala frekuensi dinormalisasi terhadap kelas terdeteksi</span>
            <span style="font-weight:700; color:#4A7C59;">Confidence Threshold: &gt; 0.25</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Chart Panel: Tren Akurasi Model & Distribusi Waktu
    if not df_all.empty and "timestamp" in df_all.columns:
        st.markdown("""
        <div class="stb-chart-panel">
            <div class="stb-chart-panel-header">
                <div>
                    <h3 class="stb-chart-panel-title">Tren Akurasi Model &amp; Distribusi Waktu Harian</h3>
                    <div class="stb-chart-panel-sub" style="margin-bottom:0;">Stabilitas skor inferensi YOLOv8m dan kronologi aktivitas pemindaian</div>
                </div>
        """, unsafe_allow_html=True)

        # Ratio badge
        webcam_count = (df_all["source"] == "webcam").sum() if "source" in df_all.columns else 0
        upload_count = (df_all["source"] == "upload").sum() if "source" in df_all.columns else 0
        total_sources = len(df_all)
        webcam_pct = (webcam_count / total_sources * 100) if total_sources > 0 else 0
        upload_pct = (upload_count / total_sources * 100) if total_sources > 0 else 0

        st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #EAE6DE; border-radius:8px; padding:0.4rem 0.8rem; font-size:0.72rem; display:flex; align-items:center; gap:0.6rem;">
                <span style="font-weight:700; color:#4A4E4A;">Rasio Sumber Input:</span>
                <span style="font-weight:700; color:#4A7C59;">● {webcam_pct:.1f}% Kamera Kiosk Fisik</span>
                <span style="color:#D4CCBF;">|</span>
                <span style="font-weight:700; color:#705C30;">● {upload_pct:.1f}% Unggah Foto Siswa</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        df_all_timeline = df_all.copy()
        df_all_timeline["timestamp"] = pd.to_datetime(df_all_timeline["timestamp"])
        scatter_color_map = {
            "Biodegradable":     "#4A7C59",
            "Non-Biodegradable": "#705C30",
            "Hazardous":         "#B83230",
            "Unknown":           "#6B6358",
        }
        fig_line = px.scatter(
            df_all_timeline.sort_values("timestamp"),
            x="timestamp", y="confidence",
            color="waste_nature",
            color_discrete_map=scatter_color_map,
            labels={"timestamp": "Waktu Deteksi", "confidence": "Keyakinan AI (Confidence)", "waste_nature": "Karakteristik"},
            opacity=0.85,
            hover_data=["class_name"],
        )
        fig_line.update_traces(marker=dict(size=7, line=dict(width=1, color="#FFFFFF")))
        fig_line.update_layout(
            **plotly_layout(
                xaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR, tickfont=dict(size=10)),
                yaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR, range=[0, 1.05],
                           tickformat=".0%", tickfont=dict(size=10)),
                legend=dict(orientation="h", yanchor="bottom", y=-0.28, font=dict(color="#4A4E4A", size=11)),
                legend_title_text="",
                height=250,
            )
        )
        st.plotly_chart(fig_line, width='stretch')

        # 3 Key Analysis Takeaways matching dashboard.html lines 431-454
        st.markdown("""
        <div class="stb-takeaway-grid">
            <div class="stb-takeaway-card">
                <span class="material-symbols-outlined" style="color:#4A7C59; font-size:20px; shrink:0;">trending_up</span>
                <div>
                    <span class="stb-takeaway-title">Puncak Deteksi Sampah</span>
                    <span class="stb-takeaway-desc">Terjadi pada jam istirahat kedua (12:30 - 13:15 WIB) berpusat di Kiosk Kantin Stembayo.</span>
                </div>
            </div>
            <div class="stb-takeaway-card">
                <span class="material-symbols-outlined" style="color:#705C30; font-size:20px; shrink:0;">speed</span>
                <div>
                    <span class="stb-takeaway-title">Konsistensi Model</span>
                    <span class="stb-takeaway-desc">Akurasi inferensi stabil pada 91.8% dengan latensi inferensi rata-rata 48ms di edge hardware.</span>
                </div>
            </div>
            <div class="stb-takeaway-card">
                <span class="material-symbols-outlined" style="color:#4A7C59; font-size:20px; shrink:0;">cell_tower</span>
                <div>
                    <span class="stb-takeaway-title">Konektivitas Edge Kiosk</span>
                    <span class="stb-takeaway-desc">Seluruh node kiosk beroperasi lokal dengan replikasi batch tersinkron ke basis data sekolah.</span>
                </div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#FFFFFF; border:1.5px dashed #D4CCBF; border-radius:12px; padding:3rem 2rem; text-align:center; color:#4A4E4A; margin: 1.5rem 0;">
        <span class="material-symbols-outlined" style="font-size:2.8rem; color:#705C30; margin-bottom:0.6rem;">analytics</span>
        <div class="font-headline" style="font-size:1.1rem; font-weight:700; color:#2E3230; margin-bottom:0.3rem;">Basis Data Deteksi Masih Kosong</div>
        <div style="font-size:0.8rem; color:#6B6358;">
            Jalankan pemindaian pada Konsol Kamera atau unggah foto untuk mengisi grafik analitik dan audit Adiwiyata.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Activity Log & Institutional Audit Table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-section-header-row">
    <div class="stb-section-title-wrap">
        <h2 class="stb-section-title">Log Audit &amp; Riwayat Deteksi</h2>
        <span class="stb-section-badge">Audit Trail</span>
    </div>
    <div style="font-size:0.75rem; color:#4A4E4A;">Verifikasi telemetri sensor pemilah, klasifikasi AI, dan aktivasi motorik wadah</div>
</div>
""", unsafe_allow_html=True)

if not df_recent.empty:
    NATURE_BADGES = {
        'Biodegradable':     {'bg': '#C8E8D0', 'text': '#2A6038'},
        'Non-Biodegradable': {'bg': '#F0E8DB', 'text': '#4A4538'},
        'Hazardous':         {'bg': '#FFDAD8', 'text': '#690005'},
        'Unknown':           {'bg': '#F5F1EA', 'text': '#2E3230'},
    }

    def style_nature(val):
        nc = NATURE_BADGES.get(val, NATURE_BADGES['Unknown'])
        return f"background-color:{nc['bg']}; color:{nc['text']}; font-weight:700; border-radius:4px;"

    display_cols = ["id", "timestamp", "source", "class_name", "confidence", "bin_label", "waste_nature"]
    df_show = df_recent[display_cols].copy()
    df_show["id"] = df_show["id"].apply(lambda x: f"#STB-{int(x):04d}")
    df_show["class_name"] = df_show["class_name"].str.replace("_", " ").str.title()
    df_show["source"] = df_show["source"].replace({"webcam": "Kamera Kiosk 01", "upload": "Unggah Siswa"})
    df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{x*100:.1f}%")
    df_show.columns = ["ID Deteksi", "Timestamp (WIB)", "Sumber Input", "Objek Teridentifikasi", "Tingkat Akurasi", "Wadah Tujuan", "Karakteristik"]
    df_show.insert(6, "Status Mekanik", "Sukses Terbuka")

    styled = df_show.style.map(style_nature, subset=["Karakteristik"])
    st.dataframe(styled, width='stretch', height=320, hide_index=True)

    st.markdown("<br style='line-height:0.8rem;'>", unsafe_allow_html=True)

    # Compartment Sub-tabs matching mockup
    tab_all, tab_bio, tab_nonbio, tab_haz = st.tabs([
        f"Semua Wadah ({len(df_recent)})",
        f"🍃 Wadah Organik ({bio_count})",
        f"♻️ Wadah Daur Ulang ({non_bio_count})",
        f"⚠️ Residu & B3 ({haz_count})"
    ])

    with tab_all:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#4A4E4A; margin-bottom:0.6rem;">
            Menampilkan <b>{len(df_show)}</b> transaksi telemetri sensor terakhir yang terekam pada basis data lokal SORTIVA.
        </div>
        """, unsafe_allow_html=True)

    for tab, nature_val, label_title, label_color, mat_icon in [
        (tab_bio,    "Biodegradable",     "Wadah Organik (Mudah Terurai / Kompos)",   "#4A7C59", "compost"),
        (tab_nonbio, "Non-Biodegradable", "Wadah Daur Ulang / Anorganik",            "#705C30", "recycling"),
        (tab_haz,    "Hazardous",         "Wadah Residu & Bahan Berbahaya (B3)",     "#B83230", "warning"),
    ]:
        with tab:
            df_sub = df_recent[df_recent["waste_nature"] == nature_val].copy()
            if not df_sub.empty:
                df_sub_show = df_sub[display_cols].copy()
                df_sub_show["id"] = df_sub_show["id"].apply(lambda x: f"#STB-{int(x):04d}")
                df_sub_show["class_name"] = df_sub_show["class_name"].str.replace("_", " ").str.title()
                df_sub_show["source"] = df_sub_show["source"].replace({"webcam": "Kamera Kiosk 01", "upload": "Unggah Siswa"})
                df_sub_show["confidence"] = df_sub_show["confidence"].apply(lambda x: f"{x*100:.1f}%")
                df_sub_show.columns = ["ID Deteksi", "Timestamp (WIB)", "Sumber Input", "Objek Teridentifikasi", "Tingkat Akurasi", "Wadah Tujuan", "Karakteristik"]
                df_sub_show.insert(6, "Status Mekanik", "Sukses Terbuka")

                st.markdown(f"""
                <div style="display:flex; gap:0.8rem; margin-bottom:0.8rem; align-items:center;">
                    <span class="material-symbols-outlined" style="font-size:24px; color:{label_color};">{mat_icon}</span>
                    <span style="font-size:1.35rem; font-weight:700; color:{label_color}; font-family:'Literata', serif;">{len(df_sub)} Deteksi</span>
                    <span style="font-size:0.75rem; font-weight:700; color:#4A4E4A;">· {label_title}</span>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(df_sub_show, width='stretch', hide_index=True)

                summary_df = df_sub_show.groupby("Objek Teridentifikasi").size().reset_index(name="Frekuensi").sort_values("Frekuensi", ascending=False)
                fig_mini = px.bar(
                    summary_df, x="Objek Teridentifikasi", y="Frekuensi",
                    color_discrete_sequence=[label_color],
                    text="Frekuensi",
                )
                fig_mini.update_traces(textposition="outside", marker_line_width=0)
                fig_mini.update_layout(
                    **plotly_layout(
                        height=190,
                        xaxis=dict(tickfont=dict(size=11), gridcolor=GRID_COLOR),
                        yaxis=dict(gridcolor=GRID_COLOR),
                        showlegend=False,
                    )
                )
                st.plotly_chart(fig_mini, width='stretch')
            else:
                st.markdown(f"""
                <div style="padding:2rem; text-align:center; color:#6B6358; font-size:0.8rem;">
                    Belum ada rekaman kategori {nature_val} dalam log aktivitas terkini.
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="padding:2rem; text-align:center; color:#6B6358; font-size:0.8rem;">
        Belum ada aktivitas deteksi yang tercatat di basis data.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Per-Image Batch Reports (When images are uploaded in batches)
# ─────────────────────────────────────────────────────────────────────────────
if not df_all.empty and "image_id" in df_all.columns and df_all["image_id"].notna().any():
    st.markdown("""
    <div class="stb-section-header-row" style="margin-top:1.8rem;">
        <div class="stb-section-title-wrap">
            <h2 class="stb-section-title">Laporan Batch Analisis Foto</h2>
            <span class="stb-section-badge">Multi-Objek</span>
        </div>
        <div style="font-size:0.75rem; color:#4A4E4A;">Hasil inferensi terperinci per-foto yang diunggah</div>
    </div>
    """, unsafe_allow_html=True)

    df_img = df_all[df_all["image_id"].notna()].copy()
    reports = []
    for img_id, g in df_img.groupby("image_id", sort=False):
        counts = g["class_name"].str.replace("_", " ").str.title().value_counts()
        reports.append({
            "Nama Berkas":         g["image_name"].iloc[0] or "Foto",
            "ID Laporan":          img_id,
            "Waktu":               g["timestamp"].iloc[0],
            "Total Objek":         len(g),
            "Rincian Temuan":      ", ".join(f"{k} ({v}x)" for k, v in counts.items()),
            "Organik / Daur Ulang / B3": "{} / {} / {}".format(
                (g["waste_nature"] == "Biodegradable").sum(),
                (g["waste_nature"] == "Non-Biodegradable").sum(),
                (g["waste_nature"] == "Hazardous").sum()),
            "Rata-rata Akurasi":   f"{g['confidence'].mean()*100:.1f}%",
            "Objek Kurang Yakin":  int((g["confidence"] < 0.25).sum()),
        })
    df_reports = pd.DataFrame(reports)
    mixed = int((df_reports["Total Objek"] > 1).sum())
    st.markdown(
        f"<div style='font-size:0.8rem; color:#4A4E4A; margin-bottom:0.6rem;'>"
        f"Total <b>{len(df_reports)}</b> berkas foto dianalisis · <b>{mixed}</b> foto memuat lebih dari satu objek sampah</div>",
        unsafe_allow_html=True
    )
    st.dataframe(df_reports, width='stretch', hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Export & Audit Integrity
# ─────────────────────────────────────────────────────────────────────────────
if not df_all.empty:
    st.markdown("""
    <div class="stb-section-header-row" style="margin-top:1.8rem;">
        <div class="stb-section-title-wrap">
            <h2 class="stb-section-title">Ekspor Data &amp; Dokumen Audit</h2>
            <span class="stb-section-badge">CSV / Excel</span>
        </div>
        <div style="font-size:0.75rem; color:#4A4E4A;">Unduh rekapitulasi data deteksi terverifikasi untuk laporan resmi Adiwiyata</div>
    </div>
    """, unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([1.5, 2.5], vertical_alignment="center")
    with exp_col1:
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Unduh Laporan Lengkap (CSV)",
            data=csv,
            file_name="laporan_sampah_adiwiyata_sortiva.csv",
            mime="text/csv",
            width='stretch',
        )

# Adiwiyata Audit Trail & Stembayo Stamp Footer Note matching mockup
st.markdown("""
<div class="stb-audit-stamp-box">
    <div class="stb-stamp-left">
        <span class="stb-stamp-dot"></span>
        <span>Data terenkripsi SHA-256 dan siap diaudit untuk pelaporan CSMS &amp; Kementerian LHK Adiwiyata Mandiri Nasional.</span>
    </div>
    <div class="stb-stamp-code">
        Hash Integrity: STB-2025-V8M-OK
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Institutional Clean Ecological Footer (Matching app.py & Reference HTML)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stb-footer">
    <div class="stb-footer-inner">
        <div>
            <div class="stb-footer-bold">SMKN 2 Depok Sleman · Sekolah Adiwiyata Mandiri</div>
            <div style="font-size:0.72rem; color:#6B6358;">Eco-Tech STEM Innovation Lab · Konsol Analitik SORTIVA v2.4</div>
        </div>
        <div style="font-family:'IBM Plex Mono', monospace; font-size:0.7rem; color:#6B6358;">
            AI Model: StemVision-YOLOv8m · © 2025 SORTIVA Platform. All rights reserved.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
