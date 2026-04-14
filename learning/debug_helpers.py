"""Reusable helpers for MCP and LangGraph testing in notebooks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from main import run_analysis
from tools.load_data import load_threshold_config
from tools.server import call_mcp_tool_sync


def base_config(data_path: str = "data/inventory_mock.csv") -> Dict[str, Any]:
    """Build a standard config payload used by tests."""
    cfg = load_threshold_config("config/thresholds.yaml")
    cfg["data_path"] = data_path
    cfg["config_path"] = "config/thresholds.yaml"
    cfg["kg_seed_path"] = "data/kg_seed.json"
    return cfg


def mcp_call(tool_name: str, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Call one MCP tool and return payload plus elapsed milliseconds."""
    start = time.perf_counter()
    result = call_mcp_tool_sync(tool_name, arguments)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def run_graph_with_timing(config: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Run LangGraph pipeline and return output payload plus elapsed milliseconds."""
    start = time.perf_counter()
    payload = run_analysis(config)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return payload, elapsed_ms


def set_neo4j(enabled: bool) -> None:
    """Enable or disable Neo4j lookup for deterministic fallback tests."""
    os.environ["NEO4J_ENABLED"] = "true" if enabled else "false"


def quick_output_checks(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact diagnostics snapshot for a graph run."""
    recs = payload.get("recommendations", [])
    missing_explanations = [
        rec.get("sku_id", "unknown")
        for rec in recs
        if not str(rec.get("plain_english_explanation", "")).strip()
    ]
    return {
        "run_id": payload.get("run_id"),
        "total_recommendations": len(recs),
        "graph_source": payload.get("metadata", {}).get("graph_source"),
        "warning_count": len(payload.get("metadata", {}).get("warnings", [])),
        "error_count": len(payload.get("metadata", {}).get("errors", [])),
        "missing_explanations": missing_explanations,
    }


def ensure_repo_root() -> Path:
    """Return and validate repository root based on this file location."""
    root = Path(__file__).resolve().parent.parent
    if not (root / "main.py").exists():
        raise FileNotFoundError("Could not resolve repository root from learning/debug_helpers.py")
    return root
