"""Basic performance guard for local execution."""

from __future__ import annotations

import time

from main import run_analysis
from tools.load_data import load_threshold_config


def test_local_run_under_reasonable_budget(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    cfg = load_threshold_config("config/thresholds.yaml")
    cfg["data_path"] = "data/inventory_mock.csv"
    cfg["config_path"] = "config/thresholds.yaml"
    cfg["kg_seed_path"] = "data/kg_seed.json"
    cfg.setdefault("ollama", {})["base_url"] = "http://127.0.0.1:65534"
    start = time.perf_counter()
    payload = run_analysis(cfg)
    elapsed = (time.perf_counter() - start) * 1000
    assert payload["recommendations"]
    assert elapsed < 5000
