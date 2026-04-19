"""Session-state helpers for Streamlit UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import streamlit as st


DEFAULT_STATE: Dict[str, Any] = {
    "last_payload": None,
    "last_elapsed_ms": None,
    "last_run_settings": None,
    "input_snapshot": None,
    "run_history": [],
    "scenario_baseline_payload": None,
    "scenario_payload": None,
    "scenario_elapsed_ms": None,
    "last_report_markdown": None,
    "last_preflight": None,
    "live_trace_events": [],
    "live_trace_tool_logs": [],
    "live_trace_flow_events": [],
    "live_trace_llm_events": [],
    "live_trace_refresh_ms": 400,
    "live_trace_enabled": False,
    "diag_caller": [],
    "diag_tool": [],
    "diag_status": [],
}


def init_session_state() -> None:
    """Initialize missing session-state keys with defaults."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_history_entry(payload: Dict[str, Any], elapsed_ms: float, label: str) -> None:
    """Append compact run metadata entry to in-memory history."""
    summary = payload.get("summary", {})
    metadata = payload.get("metadata", {})
    warnings = metadata.get("warnings", [])
    st.session_state["run_history"] = (
        st.session_state.get("run_history", [])
        + [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "label": label,
                "run_id": payload.get("run_id", "unknown"),
                "runtime_ms": int(elapsed_ms),
                "total_skus": int(summary.get("total_skus_analyzed", 0)),
                "critical": int(summary.get("critical_count", 0)),
                "watch": int(summary.get("watch_count", 0)),
                "graph_source": metadata.get("graph_source", "unknown"),
                "fallback_used": any("fallback" in str(item).lower() for item in warnings),
            }
        ]
    )[-50:]
