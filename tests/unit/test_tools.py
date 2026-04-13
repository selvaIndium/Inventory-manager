"""Unit tests for core MCP-backed tool functions."""

from __future__ import annotations

import json
from pathlib import Path

from tools.calc_metrics import calculate_metrics_for_sku
from tools.fetch_rules import fetch_rules
from tools.load_data import load_inventory_data
from tools.query_graph import query_graph


def test_calc_metrics_valid_sku() -> None:
    config = {"thresholds": {"healthy_dos_min": 14, "watch_dos_min": 7, "overstock_dos_min": 60}}
    sku = {
        "sku_id": "SKU-T1",
        "name": "Test",
        "category": "electronics",
        "current_stock": 70,
        "avg_daily_sales": 10,
        "lead_time_days": 7,
        "safety_stock": 10,
    }
    result = calculate_metrics_for_sku(sku, config)
    assert result["days_of_stock"] == 7
    assert result["reorder_qty"] == 10
    assert result["status"] == "watch"


def test_calc_metrics_zero_sales() -> None:
    config = {"thresholds": {"healthy_dos_min": 14, "watch_dos_min": 7, "overstock_dos_min": 60}}
    sku = {
        "sku_id": "SKU-T2",
        "name": "Test",
        "category": "electronics",
        "current_stock": 70,
        "avg_daily_sales": 0,
        "lead_time_days": 7,
        "safety_stock": 0,
    }
    result = calculate_metrics_for_sku(sku, config)
    assert result["status"] == "overstock"


def test_load_inventory_skips_invalid_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "sku_id,name,category,current_stock,avg_daily_sales,lead_time_days,safety_stock\n"
        "SKU-1,Valid,electronics,10,2,3,1\n"
        "SKU-2,Bad,electronics,abc,2,3,1\n",
        encoding="utf-8",
    )
    result = load_inventory_data(csv_path)
    assert len(result["records"]) == 1
    assert len(result["invalid_rows"]) == 1


def test_fetch_rules_missing_file_returns_default(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    result = fetch_rules(str(missing))
    assert result["source"] == "default"
    assert result["rules"]


def test_query_graph_networkx_fallback() -> None:
    config = {
        "neo4j": {"enabled": False},
        "cache": {"ttl_graph_seconds": 1},
        "kg_seed_path": "data/kg_seed.json",
    }
    result = query_graph("SKU-001", "electronics", "all", config)
    assert result["source"] in {"networkx", "cache", "default"}
    assert "seasonal_factor" in result
