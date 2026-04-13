"""Rule retrieval utilities for threshold-based status logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.load_data import load_threshold_config


def fetch_rules(config_path: str, category: str | None = None) -> Dict[str, Any]:
    """Load active threshold rules from config or return defaults.

    Args:
        config_path: Path to the YAML thresholds config.
        category: Optional category for future category-level overrides.

    Returns:
        Rule payload containing rules and source marker.
    """
    try:
        config = load_threshold_config(Path(config_path))
        source = "file"
    except (FileNotFoundError, ValueError):
        config = {}
        source = "default"

    thresholds = config.get("thresholds", {}) if isinstance(config, dict) else {}
    healthy_dos_min = float(thresholds.get("healthy_dos_min", 14))
    watch_dos_min = float(thresholds.get("watch_dos_min", 7))
    overstock_dos_min = float(thresholds.get("overstock_dos_min", 60))

    rules: List[Dict[str, Any]] = [
        {
            "rule_id": "R-OVERSTOCK",
            "condition": f"days_of_stock > {overstock_dos_min}",
            "action": "Review purchasing cadence and consider reducing reorder frequency.",
            "priority": 1,
        },
        {
            "rule_id": "R-HEALTHY",
            "condition": f"days_of_stock >= {healthy_dos_min}",
            "action": "Maintain current replenishment policy and monitor trend weekly.",
            "priority": 2,
        },
        {
            "rule_id": "R-WATCH",
            "condition": f"days_of_stock >= {watch_dos_min}",
            "action": "Consider planning a reorder soon to avoid slipping into critical status.",
            "priority": 3,
        },
        {
            "rule_id": "R-CRITICAL",
            "condition": f"days_of_stock < {watch_dos_min}",
            "action": "Consider prioritizing this SKU for immediate replenishment review.",
            "priority": 4,
        },
    ]

    _ = category
    return {"rules": rules, "source": source}
