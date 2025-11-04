# components/ui_theme.py
from __future__ import annotations
import streamlit as st
import plotly.io as pio

# ---- Color tokens (light/dark aware) ----
COLORS = {
    "bg_glass": "rgba(255,255,255,0.6)",
    "text": "#0f172a",
    "muted": "#475569",
    "primary": "#5B8DEF",        # actions
    "success": "#22c55e",        # healthy
    "warning": "#f59e0b",        # moderate
    "danger": "#ef4444",         # high stress
    "accent": "#8b5cf6",         # highlights
    "grid": "#E2E8F0",
}

# Plotly template tuned for Streamlit (light/dark)
_PLOTLY_BASE = dict(
    font=dict(family="Inter, Segoe UI, Roboto, system-ui, sans-serif", size=14, color=COLORS["text"]),
    margin=dict(l=16, r=16, t=28, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
)

def _build_template(name: str, is_dark: bool):
    t = pio.templates[name]
    t.layout.update(_PLOTLY_BASE)
    if is_dark:
        t.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    else:
        t.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)")
    return t

def apply_theme():
    """Inject minimal CSS + set Plotly template (no page_config here)."""
    # Pick template based on Streamlit theme
    is_dark = st.get_option("theme.base") == "dark"
    base = "plotly_dark" if is_dark else "plotly_white"
    pio.templates["plant_theme"] = _build_template(base, is_dark)
    pio.templates.default = "plant_theme"

    # Load our CSS
    css_path = "assets/theme.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Safe no-op if CSS not present
        pass

# ---- Simple UI primitives ----
from contextlib import contextmanager

@contextmanager
def card(title: str | None = None, icon: str | None = None):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if title:
        ico = f"<span class='badge-icon'>{icon}</span>" if icon else ""
        st.markdown(f"<div class='card-title'>{ico}{title}</div>", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

def metric_badge(label: str, value: str, tone: str = "neutral", help_text: str | None = None):
    tone_map = {
        "neutral": "",
        "success": "badge--success",
        "warning": "badge--warning",
        "danger": "badge--danger",
        "accent": "badge--accent",
    }
    cls = f"badge {tone_map.get(tone, '')}"
    h = f" title='{help_text}'" if help_text else ""
    st.markdown(f"<span class='{cls}'{h}>{label}: <strong>{value}</strong></span>", unsafe_allow_html=True)

def toast_success(msg: str): st.toast(msg, icon="✅")
def toast_warning(msg: str): st.toast(msg, icon="⚠️")
def toast_info(msg: str):    st.toast(msg, icon="ℹ️")
