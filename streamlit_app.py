"""
Streamlit entry at **repository root** so `ml`, `utils`, `models`, and `app` all import correctly
(local + Streamlit Cloud).

Run (from this repo’s root directory):

    streamlit run streamlit_app.py

Cloud: set **Main file path** to `streamlit_app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

try:
    from app.main import main
except Exception as e:
    st.set_page_config(page_title="Baseball Predictor — startup error", layout="centered")
    st.error("The app failed while loading. Common fix on Streamlit Cloud: ensure **tzdata** is in requirements (see repo).")
    st.exception(e)
    st.stop()

main()
