"""Minimal sidebar renderer for Streamlit app."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import streamlit as st

from ui.config import get_ollama_models
from ui.styles import status_pill


def render_sidebar(base_cfg: Dict[str, Any]) -> None:
    """Render lightweight status sidebar without heavy controls."""
    with st.sidebar:
        st.markdown("### Inventory Agent")
        st.caption("Local-first advisory workflow")

        base_url = base_cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
        ok, models, _ = get_ollama_models(base_url)

        st.markdown("#### System Health")
        if ok:
            status_pill("Ollama online", "ok")
        else:
            status_pill("Ollama offline", "warn")

        status_pill("Graph: NetworkX/cache", "ok")
        if models:
            status_pill(f"Models available: {len(models)}", "ok")
        else:
            status_pill("Models available: 0", "warn")

        if models:
            st.caption("Installed models:")
            for model in models[:8]:
                st.caption(f"- {model}")

        payload = st.session_state.get("last_payload")
        st.markdown("#### Last Run")
        if payload is None:
            st.caption("No run yet")
        else:
            st.caption(f"Run ID: {payload.get('run_id', 'unknown')}")
            st.caption(f"Generated: {payload.get('generated_at', 'unknown')}")

        st.markdown("#### Notes")
        st.caption(
            "Use the Run tab for configuration.\n"
            "Use Scenario Lab for what-if comparisons."
        )
        st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
