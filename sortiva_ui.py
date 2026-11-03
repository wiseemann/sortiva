"""Shared Streamlit UI primitives for the SORTIVA web application."""

import re

import streamlit as st


_ICON_PATHS = {
    "default": '<circle cx="12" cy="12" r="8"/>',
    "eco": '<path d="M20.5 3.5C11.2 3.9 5.2 8.1 4 19.8c11.7-1.2 15.9-7.2 16.5-16.3Z"/><path d="M4.5 19.5c2.8-4.1 6.4-6.9 10.9-9.4"/>',
    "image_search": '<rect x="3" y="4.5" width="14" height="13" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m5.5 15 3.2-3 2.2 2 1.8-1.7"/><circle cx="17.5" cy="17.5" r="3.5"/><path d="m20.2 20.2 2 2"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16.5 9"/>',
    "photo_camera": '<path d="M4 7h3l1.2-2h7.6L17 7h3a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z"/><circle cx="12" cy="13" r="3.5"/>',
    "analytics": '<path d="M4 19V5M4 19h17"/><path d="M8 16v-4M12 16V8M16 16v-7M20 16v-3"/>',
    "upload_file": '<path d="M6 3h8l4 4v14H6zM14 3v5h5"/><path d="M12 17V10M9.5 12.5 12 10l2.5 2.5"/>',
    "category": '<path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z"/><path d="m4 12 8 4.5 8-4.5M4 16.5l8 4.5 8-4.5"/>',
    "smart_toy": '<rect x="4" y="7" width="16" height="12" rx="3"/><path d="M12 7V4M9 12h.01M15 12h.01M8 16h8"/>',
    "psychology": '<path d="M9.5 5.5a3 3 0 0 0-5.3 2A3 3 0 0 0 5 13a3 3 0 0 0 2.5 4.5H10v-3M14.5 5.5a3 3 0 0 1 5.3 2A3 3 0 0 1 19 13a3 3 0 0 1-2.5 4.5H14v-3M10 9h4M10 13h4"/>',
    "school": '<path d="m3 9 9-5 9 5-9 5-9-5Z"/><path d="M7 11.5V16c2.8 2.4 7.2 2.4 10 0v-4.5M21 9v7"/>',
    "delete_sweep": '<path d="M5 7h14M10 4h4M7 7l1 14h8l1-14M10 11v6M14 11v6"/><path d="m18 13 2 2 3-4"/>',
    "arrow_upward": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "recycling": '<path d="m8 5 2-3 2 3M10 2v5M6 8l-3 5 2 3h5M3 13l4 2M17 8l3 5-2 3h-5M20 13l-4 2M12 18v4M12 22l-3-3M12 22l3-3"/>',
    "verified": '<path d="m12 3 2 1 2.3-.2 1.2 2 2 1.2-.2 2.3 1 2-1 2 .2 2.3-2 1.2-1.2 2-2.3-.2-2 1-2-1-2.3.2-1.2-2-2-1.2.2-2.3-1-2 1-2-.2-2.3 2-1.2 1.2-2 2.3.2 2-1Z"/><path d="m8 12 2.5 2.5L16 9"/>',
    "pie_chart": '<path d="M12 3a9 9 0 1 0 9 9h-9V3Z"/><path d="M15 3.5A9 9 0 0 1 20.5 9H15V3.5Z"/>',
    "trending_up": '<path d="m4 16 5-5 4 3 7-8M15 6h5v5"/>',
    "speed": '<path d="M4 16a8 8 0 1 1 16 0"/><path d="m12 16 4-5M7 16h.01M17 16h.01M12 8v.01"/>',
    "cell_tower": '<path d="M12 4v16M8 20h8M9 9a5 5 0 0 0 0 6M15 9a5 5 0 0 1 0 6M6 6a9 9 0 0 0 0 12M18 6a9 9 0 0 1 0 12"/>',
    "sensors": '<circle cx="12" cy="12" r="2"/><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.9 4.9a10 10 0 0 0 0 14.2M19.1 4.9a10 10 0 0 1 0 14.2"/>',
}


def clean_html(html_str: str) -> str:
    if not html_str:
        return ""
    no_comments = re.sub(r"<!--.*?-->", "", str(html_str), flags=re.DOTALL)
    return "\n".join(line.strip() for line in no_comments.splitlines() if line.strip())


def materialize_icons(html_str: str) -> str:
    """Replace Material Symbols ligatures with local SVGs before Streamlit renders HTML."""
    icon_pattern = re.compile(
        r'<span class="material-symbols-outlined(?P<classes>[^"]*)"(?P<attrs>[^>]*)>(?P<name>[^<]+)</span>'
    )

    def replace_icon(match: re.Match) -> str:
        name = match.group("name").strip()
        paths = _ICON_PATHS.get(name, _ICON_PATHS["default"])
        return (
            f'<span class="sortiva-svg-icon material-symbols-outlined{match.group("classes")}"'
            f'{match.group("attrs")} aria-hidden="true">'
            f'<svg viewBox="0 0 24 24" focusable="false">{paths}</svg></span>'
        )

    return icon_pattern.sub(replace_icon, html_str)


def render_html(html_str: str, container=st) -> None:
    container.markdown(materialize_icons(clean_html(html_str)), unsafe_allow_html=True)


def render_navbar(active_page: str, status_label: str = "Model Siap") -> None:
    """Render the same brand/navigation/status bar on every Streamlit page."""
    render_html(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

.sortiva-navbar-wrap {
    margin: 0;
    padding: 0.35rem 0;
}
.sortiva-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    min-width: 0;
}
.sortiva-brand-mark {
    display: inline-flex;
    width: 2.6rem;
    height: 2.6rem;
    flex: 0 0 auto;
    align-items: center;
    justify-content: center;
    border-radius: 0.8rem;
    color: #b8e8c8;
    background: #0b3b29;
    box-shadow: 0 8px 18px -12px rgba(11, 59, 41, 0.65);
}
.sortiva-brand-mark svg {
    width: 1.5rem;
    height: 1.5rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.sortiva-brand-copy { min-width: 0; }
.sortiva-brand-title {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: #0b3b29;
    font: 800 1.12rem/1 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.03em;
}
.sortiva-brand-pill {
    display: inline-flex;
    padding: 0.25rem 0.55rem;
    border-radius: 999px;
    color: #17633f;
    background: #dcebdc;
    font: 800 0.62rem/1 'DM Sans', sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sortiva-brand-sub {
    max-width: 30rem;
    margin-top: 0.25rem;
    overflow: hidden;
    color: rgba(16, 38, 28, 0.58);
    font: 500 0.7rem/1.2 'DM Sans', sans-serif;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.sortiva-meta {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.sortiva-meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.5rem 0.75rem;
    border-radius: 999px;
    font: 700 0.7rem/1 'DM Sans', sans-serif;
    white-space: nowrap;
}
.sortiva-meta-input { color: #17633f; background: #dcebdc; }
.sortiva-meta-ready { color: #fff; background: #0b3b29; }
.sortiva-navbar-line {
    height: 1px;
    margin: 0.35rem 0 1rem;
    background: rgba(16, 38, 28, 0.1);
}
.sortiva-status-dot {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 50%;
    background: #b8e8c8;
    box-shadow: 0 0 0.5rem rgba(184, 232, 200, 0.75);
}
div[data-testid="stHorizontalBlock"]:has(.sortiva-navbar-wrap) div[data-testid="stPageLink-NavLink"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-height: 2.15rem !important;
    padding: 0.45rem 0.7rem !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    color: rgba(16, 38, 28, 0.68) !important;
    background: transparent !important;
    font: 700 0.72rem/1 'DM Sans', sans-serif !important;
    transition: background 140ms ease, color 140ms ease, border-color 140ms ease !important;
}
div[data-testid="stHorizontalBlock"]:has(.sortiva-navbar-wrap) div[data-testid="stPageLink-NavLink"]:hover,
div[data-testid="stHorizontalBlock"]:has(.sortiva-navbar-wrap) div[data-testid="stPageLink-NavLink"][aria-current="page"] {
    color: #0b3b29 !important;
    background: #dcebdc !important;
    border-color: rgba(23, 99, 63, 0.12) !important;
}
.sortiva-svg-icon {
    display: inline-flex !important;
    flex: 0 0 auto;
    align-items: center;
    justify-content: center;
    width: 1.1em;
    height: 1.1em;
    line-height: 1;
    vertical-align: middle;
}
.sortiva-svg-icon svg {
    display: block;
    width: 100%;
    height: 100%;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
@media (max-width: 900px) {
    .sortiva-brand-sub { display: none; }
}
@media (max-width: 640px) {
    .sortiva-navbar-wrap { padding-bottom: 0.7rem; }
    .sortiva-brand-pill, .sortiva-meta-input { display: none; }
    .sortiva-brand-title { font-size: 1rem; }
    .sortiva-meta-pill { padding: 0.45rem 0.6rem; font-size: 0.64rem; }
}
</style>
"""
    )

    brand_col, links_col, meta_col = st.columns([1.8, 2.2, 1.25], vertical_alignment="center")
    with brand_col:
        render_html(
            """
<div class="sortiva-navbar-wrap">
    <div class="sortiva-brand">
        <div class="sortiva-brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false"><path d="M20.5 3.5C11.2 3.9 5.2 8.1 4 19.8c11.7-1.2 15.9-7.2 16.5-16.3Z"/><path d="M4.5 19.5c2.8-4.1 6.4-6.9 10.9-9.4"/></svg>
        </div>
        <div class="sortiva-brand-copy">
            <div class="sortiva-brand-title">SORTIVA <span class="sortiva-brand-pill">Aplikasi Web</span></div>
            <div class="sortiva-brand-sub">The Stembayo Eco-Sort Platform &middot; SMKN 2 Depok Sleman</div>
        </div>
    </div>
</div>
""",
            container=st,
        )

    with links_col:
        link_cols = st.columns(3, gap="small")
        links = (
            ("pages/01_Beranda.py", "Beranda", ":material/home:"),
            ("app.py", "Konsol Kamera", ":material/photo_camera:"),
            ("pages/dashboard.py", "Dashboard Analitik", ":material/analytics:"),
        )
        for column, (page, label, icon) in zip(link_cols, links):
            with column:
                st.page_link(page, label=label, icon=icon)

    with meta_col:
        render_html(
            f"""
<div class="sortiva-meta">
    <span class="sortiva-meta-pill sortiva-meta-input"><span class="material-symbols-outlined">image_search</span> Aplikasi Web</span>
    <span class="sortiva-meta-pill sortiva-meta-ready"><span class="sortiva-status-dot" aria-hidden="true"></span> {status_label}</span>
</div>
""",
            container=st,
        )

    render_html('<div class="sortiva-navbar-line" aria-hidden="true"></div>')
