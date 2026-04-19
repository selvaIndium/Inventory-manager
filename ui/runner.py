"""Execution helpers for standard and scenario analysis runs."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Tuple

from agent.graph import build_graph
from agent.state import AgentState
from cli_helpers import apply_overrides
from main import run_analysis


def run_once(config: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Run one analysis and return payload plus elapsed milliseconds."""
    start = time.perf_counter()
    payload = run_analysis(config)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return payload, elapsed_ms


def run_with_scenario(
    base_config: Dict[str, Any],
    scenario_overrides: Dict[str, float],
) -> Tuple[Dict[str, Any], float, Dict[str, Any], float]:
    """Run baseline and scenario analysis for side-by-side comparison."""
    baseline_payload, baseline_ms = run_once(base_config)
    scenario_config = apply_overrides(base_config, scenario_overrides)
    scenario_config["scenario_overrides"] = scenario_overrides
    scenario_payload, scenario_ms = run_once(scenario_config)
    return baseline_payload, baseline_ms, scenario_payload, scenario_ms


def _initial_state(config: Dict[str, Any]) -> AgentState:
    """Build initial state payload for streamed graph execution."""
    return {
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "raw_records": [],
        "sku_records": [],
        "sku_metrics": [],
        "sku_contexts": [],
        "rule_results": {},
        "recommendations": [],
        "llm_prompts": {},
        "llm_responses": {},
        "llm_retries": {},
        "flow_events": [],
        "tool_call_logs": [],
        "llm_batch_events": [],
        "agent_step_count": 0,
        "agent_max_steps": int(config.get("agent_max_steps", 3)),
        "agent_scratchpad": [],
        "agent_tool_history": [],
        "agent_seen_action_fingerprints": [],
        "agent_done": False,
        "agent_pending_action": None,
        "agent_fallback_reason": "",
        "current_node": "",
        "errors": [],
        "warnings": [],
        "partial_data": False,
        "graph_source": "default",
        "output_valid": False,
        "final_output": None,
    }


def run_analysis_stream(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Stream LangGraph node updates and emit structured events."""
    app = build_graph()
    state = _initial_state(config)
    start = time.perf_counter()

    last_tool_count = 0
    last_flow_count = 0
    last_llm_count = 0
    last_state_payload: Dict[str, Any] = {}

    for chunk in app.stream(state, stream_mode="updates"):
        if not isinstance(chunk, dict) or not chunk:
            continue

        node = next(iter(chunk.keys()))
        payload = chunk[node]
        if not isinstance(payload, dict):
            continue
        last_state_payload = payload

        flow_events = payload.get("flow_events", [])
        tool_logs = payload.get("tool_call_logs", [])
        llm_events = payload.get("llm_batch_events", [])

        new_flow = flow_events[last_flow_count:] if isinstance(flow_events, list) else []
        new_tools = tool_logs[last_tool_count:] if isinstance(tool_logs, list) else []
        new_llm = llm_events[last_llm_count:] if isinstance(llm_events, list) else []

        if isinstance(flow_events, list):
            last_flow_count = len(flow_events)
        if isinstance(tool_logs, list):
            last_tool_count = len(tool_logs)
        if isinstance(llm_events, list):
            last_llm_count = len(llm_events)

        yield {
            "event_type": "node_update",
            "node": node,
            "new_flow_events": new_flow,
            "new_tool_logs": new_tools,
            "new_llm_events": new_llm,
            "agent_step_count": payload.get("agent_step_count", 0),
            "agent_fallback_reason": payload.get("agent_fallback_reason", ""),
            "warnings": payload.get("warnings", []),
        }

    elapsed_ms = (time.perf_counter() - start) * 1000
    final_payload = last_state_payload.get("final_output") if isinstance(last_state_payload, dict) else None
    if not isinstance(final_payload, dict):
        final_payload = run_analysis(config)
    yield {
        "event_type": "run_complete",
        "elapsed_ms": elapsed_ms,
        "final_payload": final_payload,
    }
