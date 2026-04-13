"""Integration tests for major fallback scenarios."""

from __future__ import annotations

from pathlib import Path

from main import run_analysis
from tools.load_data import load_threshold_config


def _base_config(data_path: str) -> dict:
    cfg = load_threshold_config("config/thresholds.yaml")
    cfg["data_path"] = data_path
    cfg["config_path"] = "config/thresholds.yaml"
    cfg["kg_seed_path"] = "data/kg_seed.json"
    return cfg


def test_full_run_neo4j_unreachable_networkx(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    payload = run_analysis(_base_config("data/inventory_mock.csv"))
    assert payload["recommendations"]
    assert payload["metadata"]["graph_source"] in {"networkx", "cache"}


def test_full_run_ollama_unavailable_uses_template() -> None:
    cfg = _base_config("data/inventory_mock.csv")
    cfg.setdefault("ollama", {})["base_url"] = "http://127.0.0.1:65534"
    payload = run_analysis(cfg)
    explanations = [rec["plain_english_explanation"] for rec in payload["recommendations"]]
    assert explanations
    assert all(explanations)


def test_malformed_input_partial_data(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "sku_id,name,category,current_stock,avg_daily_sales,lead_time_days,safety_stock\n"
        "SKU-X,Good,electronics,100,10,7,20\n"
        "SKU-Y,Bad,electronics,bad,5,7,5\n",
        encoding="utf-8",
    )
    payload = run_analysis(_base_config(str(bad)))
    assert payload["summary"]["total_skus_analyzed"] == 1
