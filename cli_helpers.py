"""CLI helper utilities for main.py output and scenario handling."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_scenario_overrides(scenarios: List[str]) -> Dict[str, float]:
    """Parse --scenario key=value overrides."""
    parsed: Dict[str, float] = {}
    for item in scenarios:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            parsed[key] = float(value)
        except ValueError:
            continue
    return parsed


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, float]) -> Dict[str, Any]:
    """Apply threshold/default scenario overrides to config copy."""
    updated = json.loads(json.dumps(config))
    thresholds = updated.setdefault("thresholds", {})
    defaults = updated.setdefault("defaults", {})

    for key, value in overrides.items():
        if key in {"healthy_dos_min", "watch_dos_min", "critical_dos_max", "overstock_dos_min"}:
            thresholds[key] = value
        elif key in {"lead_time", "lead_time_days"}:
            defaults["lead_time_days"] = int(value)
        elif key in {"safety_stock", "default_safety_stock"}:
            defaults["safety_stock"] = value

    return updated


def safe_number(value: float) -> str:
    """Render finite and non-finite numbers safely for display."""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def print_table(metrics: List[Dict[str, Any]]) -> None:
    """Print a compact table to stdout."""
    headers = ["sku_id", "status", "dos", "reorder_qty", "urgency_days", "trend"]
    widths = [10, 10, 10, 12, 14, 9]

    def row(values: List[str]) -> str:
        return " ".join(value.ljust(width) for value, width in zip(values, widths))

    print(row(headers))
    print("-" * (sum(widths) + len(widths) - 1))

    for item in metrics:
        print(
            row(
                [
                    str(item["sku_id"]),
                    f"{item['status_emoji']} {item['status']}",
                    safe_number(float(item["days_of_stock"])),
                    safe_number(float(item["reorder_qty"])),
                    safe_number(float(item["reorder_urgency_days"])),
                    item["velocity_trend"],
                ]
            )
        )


def print_comparison(base_payload: Dict[str, Any], scenario_payload: Dict[str, Any]) -> None:
    """Print side-by-side summary comparison baseline vs scenario."""
    base = base_payload.get("summary", {})
    scenario = scenario_payload.get("summary", {})
    fields = ["critical_count", "watch_count", "healthy_count", "overstock_count"]

    print("Scenario Comparison (baseline -> scenario)")
    for field in fields:
        left = int(base.get(field, 0))
        right = int(scenario.get(field, 0))
        print(f"- {field}: {left} -> {right} (delta {right - left:+d})")


def generate_report(payload: Dict[str, Any], output_path: Path, disclaimer: str) -> Path:
    """Generate markdown report card with executive summary and top actions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path.parent / f"report_{timestamp}.md"

    summary = payload.get("summary", {})
    recommendations = payload.get("recommendations", [])
    top_actions = [rec for rec in recommendations if rec.get("status") in {"critical", "watch"}][:5]

    lines = [
        "# Inventory Optimization Report Card",
        "",
        f"- Run ID: {payload.get('run_id', 'unknown')}",
        f"- Generated At: {payload.get('generated_at', 'unknown')}",
        "",
        "## Executive Summary",
        f"- Total SKUs analyzed: {summary.get('total_skus_analyzed', 0)}",
        f"- Critical: {summary.get('critical_count', 0)}",
        f"- Watch: {summary.get('watch_count', 0)}",
        f"- Healthy: {summary.get('healthy_count', 0)}",
        f"- Overstock: {summary.get('overstock_count', 0)}",
        f"- Overall Health: {summary.get('overall_health', 'unknown')}",
        "",
        "## Top Actions",
    ]

    if top_actions:
        for action in top_actions:
            lines.append(f"- {action.get('sku_id')}: {action.get('recommended_action', 'No action provided')}")
    else:
        lines.append("- No high-priority actions identified in this run.")

    lines.extend(["", "## Advisory Disclaimer", payload.get("disclaimer", disclaimer)])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
