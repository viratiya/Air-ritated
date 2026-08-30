from __future__ import annotations

import streamlit as st


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(150deg, #07141b 0%, #0a1e25 48%, #07141b 100%); }
[data-testid="stSidebar"] { background: #081820; border-right: 1px solid rgba(104, 211, 173, .14); }
.block-container { max-width: 1420px; padding-top: 1.8rem; padding-bottom: 3rem; }
.hero {
  padding: 1.55rem 1.8rem; border-radius: 22px;
  background: linear-gradient(115deg, rgba(19, 78, 74, .78), rgba(8, 25, 33, .94));
  border: 1px solid rgba(94, 234, 212, .22); margin-bottom: 1rem;
  box-shadow: 0 18px 55px rgba(0,0,0,.22);
}
.eyebrow { color: #79e7cd; letter-spacing: .14em; font-size: .72rem; font-weight: 700; text-transform: uppercase; }
.hero h1 { color: #f0fdfa; margin: .35rem 0 .25rem; font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1; }
.hero p { color: #b5ced1; margin: 0; max-width: 800px; }
.kpi-card {
  background: rgba(11, 33, 41, .86); border: 1px solid rgba(148, 210, 200, .15);
  border-radius: 18px; padding: 1rem 1.1rem; min-height: 126px;
}
.kpi-label { color: #8fb0b4; font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }
.kpi-value { color: #ecfdf5; font-size: 2rem; font-weight: 700; margin: .3rem 0 .15rem; }
.kpi-note { color: #8fb0b4; font-size: .78rem; }
.callout { padding: .85rem 1rem; border-radius: 13px; background: rgba(13, 148, 136, .10); border-left: 3px solid #2dd4bf; color: #c8e5e3; }
.warning-callout { padding: .85rem 1rem; border-radius: 13px; background: rgba(245, 158, 11, .10); border-left: 3px solid #f59e0b; color: #fde7b0; }
.small-muted { color: #8aa6aa; font-size: .82rem; }
div[data-testid="stMetric"] { background: rgba(11, 33, 41, .75); border: 1px solid rgba(148, 210, 200, .13); padding: .8rem; border-radius: 14px; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; background: rgba(7,20,27,.55); padding: .35rem; border-radius: 14px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .5rem .8rem; }
.stTabs [aria-selected="true"] { background: rgba(45,212,191,.15); }
footer, #MainMenu { visibility: hidden; }
</style>
"""


def apply_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Environmental Intelligence · Explainable ML</div>
          <h1>Air-ritated</h1>
          <p>Carbon monoxide estimation, sensor diagnostics and transparent model intelligence—built from real hourly environmental observations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text: str, warning: bool = False) -> None:
    class_name = "warning-callout" if warning else "callout"
    st.markdown(f'<div class="{class_name}">{text}</div>', unsafe_allow_html=True)
