"""Dataframe preparation and formatting helpers for Streamlit tabs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd


def payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert recommendations payload to analysis dataframe."""
    rows = payload.get("recommendations", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["priority_score"] = (
        (-frame["reorder_urgency_days"].fillna(0)).clip(lower=0)
        + frame["reorder_qty"].fillna(0) / 10
    )
    frame["llm_source"] = frame["recommended_action"].apply(
        lambda value: "llm"
        if str(value).strip().lower()
        not in {
            "consider immediate reorder review to reduce stockout risk.",
            "consider scheduling a reorder in the next planning cycle.",
            "consider slowing replenishment and monitoring demand movement.",
            "maintain current policy and continue routine monitoring.",
            "consider reviewing this sku with your planning team.",
            "consider planning a reorder in the upcoming cycle.",
            "consider prioritizing reorder review immediately due to low coverage.",
            "maintain current replenishment approach and continue routine monitoring.",
            "consider reducing replenishment frequency and monitoring demand movement.",
        }
        else "template"
    )

    keep_cols = [
        "sku_id",
        "name",
        "category",
        "status",
        "status_emoji",
        "days_of_stock",
        "reorder_qty",
        "reorder_urgency_days",
        "priority_score",
        "velocity_trend",
        "seasonal_factor",
        "category_avg_dos",
        "context_source",
        "confidence",
        "llm_source",
        "recommended_action",
        "plain_english_explanation",
    ]
    ordered = [col for col in keep_cols if col in frame.columns]
    return frame[ordered]


def summarize_flow(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact summary stats from flow/tool logs."""
    tool_logs = metadata.get("tool_call_logs", [])
    llm_batches = metadata.get("llm_batch_events", [])
    deterministic_calls = sum(1 for item in tool_logs if item.get("caller") == "deterministic_system")
    planner_calls = sum(1 for item in tool_logs if item.get("caller") == "planner_model")
    failed_calls = sum(1 for item in tool_logs if item.get("status") == "error")
    llm_failed_batches = sum(1 for item in llm_batches if not bool(item.get("batch_success", False)))
    duplicate_suppressed = 1 if str(metadata.get("agent_fallback_reason", "")).startswith("duplicate_action_suppressed") else 0

    return {
        "tool_calls_total": len(tool_logs),
        "tool_calls_deterministic": deterministic_calls,
        "tool_calls_planner": planner_calls,
        "tool_calls_failed": failed_calls,
        "llm_batches_total": len(llm_batches),
        "llm_batches_failed": llm_failed_batches,
        "duplicate_suppressed": duplicate_suppressed,
    }


def filter_df(
    frame: pd.DataFrame,
    status_filter: list[str],
    category_filter: list[str],
    context_filter: list[str],
    source_filter: list[str],
    search: str,
) -> pd.DataFrame:
    """Apply explorer filters and return sorted result."""
    filtered = frame.copy()
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if context_filter:
        filtered = filtered[filtered["context_source"].isin(context_filter)]
    if source_filter:
        filtered = filtered[filtered["llm_source"].isin(source_filter)]
    if search:
        search_val = search.strip().lower()
        filtered = filtered[
            filtered["sku_id"].str.lower().str.contains(search_val)
            | filtered["name"].str.lower().str.contains(search_val)
        ]

    return filtered.sort_values(by=["priority_score", "reorder_urgency_days"], ascending=[False, True])


def now_file_suffix() -> str:
    """Return timestamp suffix for exports."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
