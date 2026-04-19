"""Styling and visual header helpers for Streamlit UI."""

from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    """Inject app-level style overrides."""
    st.markdown(
        """
        <style>
            .hero-wrap {
                padding: 1.2rem 1.3rem;
                border-radius: 16px;
                border: 1px solid var(--st-secondary-background-color, #dcdcdc);
                background: linear-gradient(135deg,
                    color-mix(in srgb, var(--st-background-color, #fff) 87%, #4c8bf5 13%),
                    color-mix(in srgb, var(--st-background-color, #fff) 94%, #20c997 6%));
                margin-bottom: 0.75rem;
            }
            .hero-title {
                margin: 0;
                font-size: 1.78rem;
                font-weight: 760;
                color: var(--st-text-color, #111);
            }
            .hero-sub {
                margin-top: 0.38rem;
                font-size: 0.97rem;
                color: var(--st-text-color, #444);
                opacity: 0.88;
            }
            .stat-card {
                border-radius: 12px;
                border: 1px solid var(--st-secondary-background-color, #dcdcdc);
                padding: 0.65rem 0.8rem;
                background: var(--st-background-color, #fff);
            }
            .stat-label {
                font-size: 0.78rem;
                opacity: 0.72;
            }
            .stat-value {
                margin-top: 0.18rem;
                font-size: 1.3rem;
                font-weight: 740;
                line-height: 1.1;
            }
            .mini-pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.24rem 0.62rem;
                font-size: 0.75rem;
                font-weight: 600;
                margin-right: 0.35rem;
                margin-bottom: 0.3rem;
            }
            .mini-pill.ok {
                background: color-mix(in srgb, #27ae60 16%, var(--st-background-color, #fff) 84%);
                color: #1f7a43;
            }
            .mini-pill.warn {
                background: color-mix(in srgb, #f39c12 18%, var(--st-background-color, #fff) 82%);
                color: #8a5c00;
            }
            .section-kicker {
                font-size: 0.83rem;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                opacity: 0.62;
                margin-bottom: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render top hero block."""
    st.markdown(
        """
        <div class="hero-wrap">
            <p class="hero-title">Inventory Optimization AI Agent</p>
            <p class="hero-sub">Local-first decision support powered by LangGraph, FastMCP, NetworkX context, and Ollama explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str) -> None:
    """Render compact metric card."""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text: str, kind: str = "ok") -> None:
    """Render compact status pill."""
    kind = "warn" if kind == "warn" else "ok"
    st.markdown(f"<span class='mini-pill {kind}'>{text}</span>", unsafe_allow_html=True)
