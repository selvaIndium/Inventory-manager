"""LangGraph planner node for hybrid agentic tool orchestration."""

from __future__ import annotations

import json
from typing import Any, Dict
import hashlib

import httpx

from agent.logging_utils import add_flow_event, timer_ms, timer_start
from agent.state import AgentState

ALLOWED_TOOLS = {
    "load_inventory",
    "calc_metrics_batch",
    "query_graph_batch",
    "apply_rules_batch",
    "query_graph",
    "fetch_rules",
}

PLANNER_SYSTEM_PROMPT = """
You are a planner node in an inventory LangGraph workflow.
You must return strict JSON only with this schema:
{
  "thought": "string",
  "tool_name": "string",
  "arguments": {"any": "object"},
  "done": boolean
}

Rules:
- Choose exactly one action per step.
- If data is not loaded, call `load_inventory` with file_path.
- If records exist but metrics missing, call `calc_metrics_batch`.
- If metrics exist and mode is thinking and contexts missing, call `query_graph_batch`.
- If metrics exist and rule_results missing, call `apply_rules_batch`.
- If all needed state is present, return done=true.
- Never repeat the same successful tool action with the same arguments.
- `load_inventory` is only valid when records == 0.
- `calc_metrics_batch` is only valid when records > 0 and metrics == 0.
- In thinking mode, `query_graph_batch` is only valid when metrics > 0 and contexts == 0.
- `apply_rules_batch` is only valid when metrics > 0 and rule_results == 0.
- Never return markdown, prose, or fenced blocks.
""".strip()


def _parse_content_json(content: str) -> Dict[str, Any]:
    """Parse model content into JSON, tolerating fenced code blocks."""
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    if not text:
        raise ValueError("empty_output")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            raise ValueError("malformed_json")

    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError("malformed_json")
    return parsed


def _default_done_action(reason: str) -> Dict[str, Any]:
    """Return a safe action payload that ends the loop."""
    return {
        "thought": reason,
        "tool_name": "",
        "arguments": {},
        "done": True,
    }


def _tool_allowed_for_stage(state: AgentState, tool_name: str) -> bool:
    """Return True when a planner tool is valid for current state stage."""
    records = len(state.get("sku_records", []))
    metrics = len(state.get("sku_metrics", []))
    contexts = len(state.get("sku_contexts", []))
    rules = len(state.get("rule_results", {}))
    mode = str(state["config"].get("mode", "thinking")).lower()

    if records == 0:
        return tool_name == "load_inventory"
    if metrics == 0:
        return tool_name == "calc_metrics_batch"
    if mode == "thinking" and contexts == 0:
        return tool_name in {"query_graph_batch", "query_graph"}
    if rules == 0:
        return tool_name in {"apply_rules_batch", "fetch_rules"}
    return False


def _validate_action(raw: Dict[str, Any], state: AgentState) -> Dict[str, Any] | None:
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
        stage_complete = not any(
            [
                len(state.get("sku_records", [])) == 0,
                len(state.get("sku_metrics", [])) == 0,
                str(state["config"].get("mode", "thinking")).lower() == "thinking"
                and len(state.get("sku_contexts", [])) == 0,
                len(state.get("rule_results", {})) == 0,
            ]
        )
        if not stage_complete:
            return None
        return {
            "thought": thought or "Planner signaled completion.",
            "tool_name": "",
            "arguments": {},
            "done": True,
        }

    if tool_name not in ALLOWED_TOOLS or not _tool_allowed_for_stage(state, tool_name):
        return None

    return {
        "thought": thought or "Planner selected next tool action.",
        "tool_name": tool_name,
        "arguments": arguments,
        "done": False,
    }


def _classify_planner_error(exc: Exception) -> str:
    """Map planner exceptions to stable diagnostics."""
    detail = str(exc).strip() or "planner_unknown_error"
    lowered = detail.lower()
    if "timed out" in lowered:
        return "timeout"
    if "404" in detail and "not found" in lowered:
        if "model" in lowered:
            return "model_not_found"
        return "http_404"
    if "500" in detail:
        return "http_500"
    if detail in {"empty_output", "malformed_json", "invalid_action_schema"}:
        return detail
    if "expecting value" in lowered or "expecting ','" in lowered:
        return "malformed_json"
    return detail


def planner_action_node(state: AgentState) -> AgentState:
    """Plan one MCP tool action (or done) for hybrid mode."""
    node = "planner_action"
    add_flow_event(state, node=node, event="start")
    state["current_node"] = "planner_action"

    mode = str(state["config"].get("agent_mode", "deterministic"))
    if mode not in {"hybrid", "full"}:
        state["agent_done"] = True
        state["agent_pending_action"] = _default_done_action("Deterministic mode bypasses planner.")
        add_flow_event(state, node=node, event="end", detail="bypassed_deterministic")
        return state

    if state["agent_done"]:
        state["agent_pending_action"] = _default_done_action("Planner loop already marked done.")
        return state

    if state["agent_step_count"] >= state["agent_max_steps"]:
        state["agent_done"] = True
        state["agent_fallback_reason"] = "agent_max_steps_reached"
        state["agent_pending_action"] = _default_done_action("Reached step cap; exiting planner loop.")
        add_flow_event(state, node=node, event="end", detail="step_cap_reached")
        return state

    ollama_cfg = state["config"].get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434")).rstrip("/")
    model = str(ollama_cfg.get("model", "llama3.2:1b"))
    timeout_seconds: float | None = None
    api_key = str(ollama_cfg.get("api_key", "")).strip()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    sample_metric = state["sku_metrics"][0].sku_id if state["sku_metrics"] else ""
    contexts_count = len(state.get("sku_contexts", []))
    prompt_payload = {
        "instruction": "Return strict JSON only with one action.",
        "schema": {
            "thought": "string",
            "tool_name": "one of load_inventory, calc_metrics_batch, query_graph_batch, apply_rules_batch, fetch_rules, query_graph or empty when done=true",
            "arguments": "object",
            "done": "boolean",
        },
        "constraints": {
            "allowed_tools": sorted(ALLOWED_TOOLS),
            "current_step": state["agent_step_count"],
            "max_steps": state["agent_max_steps"],
            "mode": str(state["config"].get("mode", "thinking")),
        },
        "context": {
            "data_path": str(state["config"].get("data_path", "data/inventory_mock.csv")),
            "records": len(state["sku_records"]),
            "metrics": len(state["sku_metrics"]),
            "contexts": contexts_count,
            "rule_results": len(state.get("rule_results", {})),
            "sample_sku": sample_metric,
            "tool_history_count": len(state["agent_tool_history"]),
            "last_action": state["agent_tool_history"][-1] if state["agent_tool_history"] else None,
            "stage_flags": {
                "needs_load": len(state["sku_records"]) == 0,
                "needs_metrics": len(state["sku_records"]) > 0 and len(state["sku_metrics"]) == 0,
                "needs_contexts": (
                    str(state["config"].get("mode", "thinking")).lower() == "thinking"
                    and len(state["sku_metrics"]) > 0
                    and contexts_count == 0
                ),
                "needs_rules": len(state["sku_metrics"]) > 0 and len(state.get("rule_results", {})) == 0,
            },
        },
    }

    response_schema = {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool_name": {"type": "string"},
            "arguments": {"type": "object"},
            "done": {"type": "boolean"},
        },
        "required": ["thought", "tool_name", "arguments", "done"],
    }

    strict_request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
        "stream": False,
        "think": False,
        "format": response_schema,
        "options": {"temperature": 0.0},
    }

    relaxed_request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    action = None
    started = timer_start()
    failures: list[str] = []
    for attempt_name, request in (("strict", strict_request), ("relaxed", relaxed_request)):
        try:
            with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
                response = client.post(f"{base_url}/api/chat", json=request)
                response.raise_for_status()
                raw = response.json()
            content = raw.get("message", {}).get("content", "{}")
            parsed = _parse_content_json(content)
            if isinstance(parsed, dict):
                action = _validate_action(parsed, state)
            if action is None:
                raise ValueError("invalid_action_schema")
            break
        except Exception as exc:
            failures.append(f"{attempt_name}:{_classify_planner_error(exc)}")

    if action is None:
        detail = " | ".join(failures) if failures else "planner_unknown_error"
        state["warnings"].append(
            f"Planner action generation failed; using safe fallback. Detail: {detail}"
        )
        state["agent_fallback_reason"] = f"planner_unavailable:{detail}"

    if action is None:
        if not state.get("agent_fallback_reason"):
            state["agent_fallback_reason"] = "planner_unavailable"
        action = _default_done_action("Planner unavailable; exiting planner loop safely.")

    if not bool(action.get("done", False)):
        normalized = {
            "tool_name": action.get("tool_name", ""),
            "arguments": action.get("arguments", {}),
        }
        #this fingerprinting approach is intentionally simple and not cryptographically secure, just a quick way to detect exact duplicates in tool calls which is a common failure mode for LLMs in this context. 
        # It normalizes the tool name and arguments to ensure that semantically identical actions produce the same fingerprint even if there are minor formatting differences in the JSON output. 
        # The use of sort_keys and ensure_ascii in json.dumps helps achieve this normalization.
        fingerprint = hashlib.sha1(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        seen = state.setdefault("agent_seen_action_fingerprints", [])
        if fingerprint in seen:
            state["agent_fallback_reason"] = "duplicate_action_suppressed"
            action = _default_done_action("Duplicate planner action suppressed; exiting planner loop.")
        else:
            seen.append(fingerprint)

    state["agent_pending_action"] = action
    if bool(action.get("done", False)):
        state["agent_done"] = True

    add_flow_event(
        state,
        node=node,
        event="end",
        duration_ms=timer_ms(started),
        extra={
            "planner_done": bool(action.get("done", False)),
            "planner_tool_name": action.get("tool_name", ""),
        },
    )

    return state
