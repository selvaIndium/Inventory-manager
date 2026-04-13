"""Focused fallback behavior tests."""

from __future__ import annotations

from main import run_analysis
from tools.load_data import load_threshold_config


def _config() -> dict:
    cfg = load_threshold_config("config/thresholds.yaml")
    cfg["data_path"] = "data/inventory_mock.csv"
    cfg["config_path"] = "config/thresholds.yaml"
    cfg["kg_seed_path"] = "data/kg_seed.json"
    return cfg


def test_neo4j_disabled_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    payload = run_analysis(_config())
    assert payload["metadata"]["graph_source"] in {"networkx", "cache"}


def test_llm_timeout_falls_back_template() -> None:
    cfg = _config()
    cfg.setdefault("ollama", {})["base_url"] = "http://127.0.0.1:65534"
    cfg.setdefault("ollama", {})["timeout_ms"] = 10
    payload = run_analysis(cfg)
    assert all(rec["plain_english_explanation"] for rec in payload["recommendations"])
