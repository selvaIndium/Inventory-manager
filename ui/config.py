"""Configuration and health utilities for Streamlit runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import streamlit as st


def profile_defaults(profile: str) -> Dict[str, Any]:
    """Return default LLM settings for a profile."""
    if profile == "Strict latency":
        return {
            "temperature": 0.1,
        }
    return {
        "temperature": 0.1,
    }


@st.cache_data(ttl=12)
def get_ollama_models(base_url: str) -> Tuple[bool, List[str], str]:
    """Fetch installed Ollama model tags from local runtime."""
    try:
        with httpx.Client(timeout=2.5) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        return True, models, "Ollama reachable"
    except Exception as exc:
        return False, [], f"Ollama unreachable: {exc}"


def make_input_snapshot(data_mode: str, uploaded_file) -> Dict[str, Any]:
    """Serialize user-selected data source for rerunnable scenarios."""
    if data_mode == "Use mock dataset":
        return {"type": "mock"}
    if uploaded_file is None:
        raise ValueError("Upload a CSV or JSON file first.")
    return {
        "type": "upload",
        "filename": uploaded_file.name,
        "bytes": uploaded_file.getvalue(),
    }


def materialize_input_file(snapshot: Dict[str, Any], temp_dir: str) -> str:
    """Materialize selected input source to a concrete data path."""
    if snapshot.get("type") == "mock":
        return "data/inventory_mock.csv"

    filename = str(snapshot.get("filename", "uploaded.csv"))
    data = snapshot.get("bytes", b"")
    path = Path(temp_dir) / filename
    path.write_bytes(data)
    return str(path)


def build_run_config(
    base_cfg: Dict[str, Any],
    data_path: str,
    settings: Dict[str, Any],
    scenario_overrides: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Build run config from UI-selected settings."""
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data_path"] = data_path
    cfg["config_path"] = "config/thresholds.yaml"
    cfg["kg_seed_path"] = "data/kg_seed.json"

    ollama_cfg = cfg.setdefault("ollama", {})
    ollama_cfg["base_url"] = settings["base_url"]
    ollama_cfg["model"] = settings["model"]
    ollama_cfg["timeout_ms"] = int(settings.get("timeout_ms", 0))
    ollama_cfg["temperature"] = float(settings["temperature"])
    if "num_predict" in settings:
        ollama_cfg["num_predict"] = int(settings["num_predict"])
    else:
        ollama_cfg.pop("num_predict", None)
    ollama_cfg["planner_timeout_ms"] = int(settings.get("planner_timeout_ms", 0))

    cfg["mode"] = settings.get("mode", "thinking")
    cfg["agent_mode"] = settings.get("agent_mode", "deterministic")
    cfg["fast_template_only"] = bool(settings.get("fast_template_only", False))
    cfg["agent_max_steps"] = int(settings.get("agent_max_steps", cfg.get("agent_max_steps", 3)))

    selected_skus = settings.get("analysis_sku_ids", [])
    if isinstance(selected_skus, list):
        clean_ids = sorted({str(item).strip() for item in selected_skus if str(item).strip()})
        if clean_ids:
            cfg["analysis_sku_ids"] = clean_ids
        elif "analysis_sku_ids" in cfg:
            cfg.pop("analysis_sku_ids", None)

    if scenario_overrides:
        cfg["scenario_overrides"] = scenario_overrides
    return cfg


def new_tempdir() -> tempfile.TemporaryDirectory:
    """Return new temporary directory context helper."""
    return tempfile.TemporaryDirectory()
