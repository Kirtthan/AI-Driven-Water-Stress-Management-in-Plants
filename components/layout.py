# components/layout.py
from __future__ import annotations
import streamlit as st

def header(title: str, subtitle: str | None = None):
    st.markdown(f"""
    <div class="app-header">
      <div class="glow"></div>
      <div class="header-content">
        <div class="title-row">
          <span class="logo">🌿</span>
          <h1>{title}</h1>
        </div>
        {f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)

def sticky_sidebar_tips():
    with st.sidebar:
        st.markdown("<div class='sticky-tips'><div class='tips-title'>Tips</div>", unsafe_allow_html=True)
        st.write("• Use ↑/↓ on sliders for fine steps.")
        st.write("• Switch models to see animated comparisons.")
        st.write("• Hover charts for precise values.")
        st.write("• Use the camera icon to download PNG.")
        st.markdown("</div>", unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class="app-footer">
      <div>BUILT BY KIRTTHAN,AKSHAR,KUSHAL</div>
    </div>
    """, unsafe_allow_html=True)

def responsive_cols(ratios=(2,1)):
    return st.columns(ratios)
