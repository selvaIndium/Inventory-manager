"""LangGraph planner node for hybrid agentic tool orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import httpx

from agent.state import AgentState

ALLOWED_TOOLS = {"load_csv", "load_json", "calc_metrics", "fetch_rules", "query_graph"}


def _parse_content_json(content: str) -> Dict[str, Any]:
    """Parse model content into JSON, tolerating fenced code blocks."""
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            raise

    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError("Planner returned non-object JSON payload.")
    return parsed


def _default_done_action(reason: str) -> Dict[str, Any]:
    """Return a safe action payload that ends the loop."""
    return {
        "thought": reason,
        "tool_name": "",
        "arguments": {},
        "done": True,
    }


def _validate_action(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    """Validate strict planner action schema and tool allowlist."""
    required = {"thought", "tool_name", "arguments", "done"}
    if not required.issubset(set(raw.keys())):
        return None

    thought = str(raw.get("thought", "")).strip()
    tool_name = str(raw.get("tool_name", "")).strip()
    arguments = raw.get("arguments", {})
    done = bool(raw.get("done", False))

    if not isinstance(arguments, dict):
        return None

    if done:
        return {
            "thought": thought or "Planner signaled completion.",
            "tool_name": "",
            "arguments": {},
            "done": True,
        }

    if tool_name not in ALLOWED_TOOLS:
        return None

    return {
        "thought": thought or "Planner selected next tool action.",
        "tool_name": tool_name,
        "arguments": arguments,
        "done": False,
    }


def planner_action_node(state: AgentState) -> AgentState:
    """Plan one MCP tool action (or done) for hybrid mode."""
    state["current_node"] = "planner_action"

    mode = str(state["config"].get("agent_mode", "deterministic"))
    if mode not in {"hybrid", "full"}:
        state["agent_done"] = True
        state["agent_pending_action"] = _default_done_action("Deterministic mode bypasses planner.")
        return state

    if state["agent_done"]:
        state["agent_pending_action"] = _default_done_action("Planner loop already marked done.")
        return state

    if state["agent_step_count"] >= state["agent_max_steps"]:
        state["agent_done"] = True
        state["agent_fallback_reason"] = "agent_max_steps_reached"
        state["agent_pending_action"] = _default_done_action("Reached step cap; exiting planner loop.")
        return state

    ollama_cfg = state["config"].get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434")).rstrip("/")
    model = str(ollama_cfg.get("model", "llama3.2:1b"))
    planner_timeout_ms = int(ollama_cfg.get("planner_timeout_ms", 2500))
    timeout_seconds: float | None = None if planner_timeout_ms <= 0 else max(1.0, float(planner_timeout_ms) / 1000)
    api_key = str(ollama_cfg.get("api_key", "")).strip()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    sample_metric = state["sku_metrics"][0].sku_id if state["sku_metrics"] else ""
    prompt_payload = {
        "instruction": "Return strict JSON only with one action.",
        "schema": {
            "thought": "string",
            "tool_name": "one of load_csv, load_json, calc_metrics, fetch_rules, query_graph or empty when done=true",
            "arguments": "object",
            "done": "boolean",
        },
        "constraints": {
            "allowed_tools": sorted(ALLOWED_TOOLS),
            "current_step": state["agent_step_count"],
            "max_steps": state["agent_max_steps"],
        },
        "context": {
            "records": len(state["sku_records"]),
            "metrics": len(state["sku_metrics"]),
            "sample_sku": sample_metric,
            "tool_history_count": len(state["agent_tool_history"]),
        },
    }

    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": Path("prompts/system_prompt.txt").read_text(encoding="utf-8")
                + "\nYou are now a planner node. Output strict JSON action only.",
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 160},
    }

    action = None
    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            response = client.post(f"{base_url}/api/chat", json=request)
            response.raise_for_status()
            raw = response.json()
        content = raw.get("message", {}).get("content", "{}")
        parsed = _parse_content_json(content)
        if isinstance(parsed, dict):
            action = _validate_action(parsed)
    except Exception as exc:
        state["warnings"].append(f"Planner action generation failed; using safe fallback. Detail: {exc}")

    if action is None:
        fallback_tool = "query_graph" if state["sku_records"] else ""
        if fallback_tool:
            sku = state["sku_records"][0]
            action = {
                "thought": "Fallback planner action: probe context via query_graph.",
                "tool_name": "query_graph",
                "arguments": {
                    "sku_id": sku.sku_id,
                    "category": sku.category,
                    "query_type": "all",
                    "config": state["config"],
                },
                "done": False,
            }
        else:
            action = _default_done_action("No records available for planner actions.")

    state["agent_pending_action"] = action
    if bool(action.get("done", False)):
        state["agent_done"] = True

    return state
