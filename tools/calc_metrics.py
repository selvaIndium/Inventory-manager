"""Inventory metric calculations for advisory analysis."""

from __future__ import annotations

import math
from typing import Any, Dict, List


def calculate_metrics(records: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calculate inventory metrics per SKU record.

    Args:
        records: Validated SKU records.
        config: Configuration dictionary loaded from thresholds YAML.

    Returns:
        List of SKU metric dictionaries.
    """
    output: List[Dict[str, Any]] = []
    for row in records:
        output.append(calculate_metrics_for_sku(row, config))

    return output


def calculate_metrics_for_sku(sku: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate inventory metrics for a single SKU dictionary."""
    thresholds = config.get("thresholds", {})
    healthy_dos_min = float(thresholds.get("healthy_dos_min", 14))
    watch_dos_min = float(thresholds.get("watch_dos_min", 7))
    overstock_dos_min = float(thresholds.get("overstock_dos_min", 60))

    avg_daily_sales = float(sku["avg_daily_sales"])
    current_stock = float(sku["current_stock"])
    lead_time_days = int(sku["lead_time_days"])
    safety_stock = float(sku["safety_stock"])

    if avg_daily_sales <= 0:
        days_of_stock = math.inf
    else:
        days_of_stock = current_stock / avg_daily_sales

    reorder_qty = max(
        0.0,
        (avg_daily_sales * lead_time_days) + safety_stock - current_stock,
    )
    reorder_urgency_days = days_of_stock - lead_time_days

    velocity_trend = _detect_velocity_trend(sku)
    status, status_emoji = _classify_status(
        days_of_stock,
        healthy_dos_min,
        watch_dos_min,
        overstock_dos_min,
    )

    return {
        "sku_id": sku["sku_id"],
        "name": sku.get("name", "Unknown"),
        "category": sku.get("category", "unknown"),
        "days_of_stock": days_of_stock,
        "reorder_qty": reorder_qty,
        "reorder_urgency_days": reorder_urgency_days,
        "velocity_trend": velocity_trend,
        "status": status,
        "status_emoji": status_emoji,
        "formula_used": (
            f"reorder_qty = ({avg_daily_sales:.2f} x {lead_time_days}) + "
            f"{safety_stock:.2f} - {current_stock:.2f}"
        ),
    }


def _detect_velocity_trend(record: Dict[str, Any]) -> str:
    """Infer velocity trend from 7-day and 30-day average sales."""
    avg_7d = record.get("avg_daily_sales_7d")
    avg_30d = record.get("avg_daily_sales_30d")

    if avg_7d is None or avg_30d is None:
        return "unknown"

    avg_7d_val = float(avg_7d)
    avg_30d_val = float(avg_30d)

    if avg_30d_val <= 0:
        return "unknown"

    delta_ratio = (avg_7d_val - avg_30d_val) / avg_30d_val
    if delta_ratio > 0.1:
        return "rising"
    if delta_ratio < -0.1:
        return "falling"
    return "stable"


def _classify_status(
    days_of_stock: float,
    healthy_dos_min: float,
    watch_dos_min: float,
    overstock_dos_min: float,
) -> tuple[str, str]:
    """Classify a SKU status based on days-of-stock thresholds."""
    if days_of_stock > overstock_dos_min:
        return "overstock", "⚠️"
    if days_of_stock >= healthy_dos_min:
        return "healthy", "🟢"
    if days_of_stock >= watch_dos_min:
        return "watch", "🟡"
    return "critical", "🔴"
