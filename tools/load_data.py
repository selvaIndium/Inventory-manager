"""Data and configuration loading utilities for inventory analysis."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

LOGGER = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "sku_id",
    "name",
    "category",
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "safety_stock",
}


def load_threshold_config(config_path: str | Path) -> Dict[str, Any]:
    """Load threshold and default configuration from YAML.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the loaded config is empty or invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict) or not config:
        raise ValueError(f"Invalid config payload in: {path}")

    LOGGER.info("Loaded configuration from %s", path)
    return config


def load_inventory_data(file_path: str | Path) -> Dict[str, Any]:
    """Load inventory rows from CSV or JSON and validate required fields.

    Args:
        file_path: Path to an inventory CSV or JSON file.

    Returns:
        Dictionary containing valid records, row count, invalid rows, and warnings.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If file format is unsupported or JSON root is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv_rows(path)
    elif suffix == ".json":
        rows = _load_json_rows(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    valid_records: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for row_index, row in enumerate(rows, start=1):
        is_valid, normalized_row, reason = _normalize_and_validate_row(row)
        if is_valid:
            valid_records.append(normalized_row)
        else:
            invalid_rows.append({"row_index": row_index, "reason": reason})

    if invalid_rows:
        warnings.append(
            f"Skipped {len(invalid_rows)} invalid rows during data loading."
        )

    LOGGER.info(
        "Loaded %s rows from %s (%s valid, %s invalid)",
        len(rows),
        path,
        len(valid_records),
        len(invalid_rows),
    )

    return {
        "records": valid_records,
        "row_count": len(rows),
        "invalid_rows": invalid_rows,
        "warnings": warnings,
    }


def _load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Load raw row dictionaries from a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _load_json_rows(path: Path) -> List[Dict[str, Any]]:
    """Load raw row dictionaries from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Inventory JSON must be a list of row objects.")

    return payload


def _normalize_and_validate_row(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """Normalize row values and validate required fields.

    Args:
        row: Raw row dictionary from CSV or JSON.

    Returns:
        A tuple of (is_valid, normalized_row, reason_if_invalid).
    """
    missing_fields = [field for field in REQUIRED_FIELDS if not _has_value(row.get(field))]
    if missing_fields:
        return False, row, f"Missing required fields: {', '.join(sorted(missing_fields))}"

    normalized = dict(row)

    try:
        normalized["current_stock"] = float(row["current_stock"])
        normalized["avg_daily_sales"] = float(row["avg_daily_sales"])
        normalized["lead_time_days"] = int(float(row["lead_time_days"]))
        normalized["safety_stock"] = float(row["safety_stock"])

        if _has_value(row.get("avg_daily_sales_7d")):
            normalized["avg_daily_sales_7d"] = float(row["avg_daily_sales_7d"])
        if _has_value(row.get("avg_daily_sales_30d")):
            normalized["avg_daily_sales_30d"] = float(row["avg_daily_sales_30d"])
    except (TypeError, ValueError) as exc:
        return False, row, f"Numeric conversion failed: {exc}"

    return True, normalized, ""


def _has_value(value: Any) -> bool:
    """Return True when a value is not empty."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True
