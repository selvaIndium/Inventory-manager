"""Streamlit entrypoint for Inventory Optimization AI Agent."""

from __future__ import annotations

import streamlit as st

from main import DISCLAIMER
from tools.load_data import load_threshold_config
from ui.session import init_session_state
from ui.sidebar import render_sidebar
from ui.styles import inject_css, render_hero
from ui.tabs import render_tabs


def main() -> None:
    """Run the tab-first Streamlit application."""
    st.set_page_config(
        page_title="Inventory Optimization Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_css()
    render_hero()
    st.warning(DISCLAIMER)

    base_cfg = load_threshold_config("config/thresholds.yaml")
    render_sidebar(base_cfg)
    render_tabs(base_cfg)


if __name__ == "__main__":
    main()
