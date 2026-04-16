"""LangGraph executor node for hybrid planner-selected MCP actions."""

from __future__ import annotations

from typing import Any, Dict

from agent.state import AgentState
from tools.server import call_mcp_tool_sync


def _sanitize_arguments(tool_name: str, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    """Normalize arguments for known tool contracts."""
    if tool_name in {"load_csv", "load_json"}:
        return {"file_path": str(arguments.get("file_path", state["config"].get("data_path", "")))}

    if tool_name == "fetch_rules":
        return {
            "config_path": str(arguments.get("config_path", state["config"].get("config_path", "config/thresholds.yaml"))),
            "category": arguments.get("category"),
        }

    if tool_name == "query_graph":
        if state["sku_records"]:
            sample = state["sku_records"][0]
            return {
                "sku_id": str(arguments.get("sku_id", sample.sku_id)),
                "category": str(arguments.get("category", sample.category)),
                "query_type": str(arguments.get("query_type", "all")),
                "config": arguments.get("config", state["config"]),
            }
        return {
            "sku_id": str(arguments.get("sku_id", "")),
            "category": str(arguments.get("category", "unknown")),
            "query_type": str(arguments.get("query_type", "all")),
            "config": arguments.get("config", state["config"]),
        }

    if tool_name == "calc_metrics":
        sku_payload = arguments.get("sku")
        if isinstance(sku_payload, dict):
            return {"sku": sku_payload, "config": arguments.get("config", state["config"])}
        if state["raw_records"]:
            return {"sku": state["raw_records"][0], "config": state["config"]}
        return {"sku": {}, "config": state["config"]}

    return arguments


def execute_action_node(state: AgentState) -> AgentState:
    """Execute planner-selected MCP action and append observation history."""
    state["current_node"] = "execute_action"

    pending = state.get("agent_pending_action") or {}
    done = bool(pending.get("done", False))
    tool_name = str(pending.get("tool_name", "")).strip()
    arguments = pending.get("arguments", {})
    thought = str(pending.get("thought", "")).strip()

    if done or not tool_name:
        state["agent_done"] = True
        state["agent_tool_history"].append(
            {
                "step": state["agent_step_count"],
                "thought": thought or "Planner marked done.",
                "tool_name": tool_name,
                "arguments": {},
                "status": "done",
                "observation": "Planner loop ended without execution.",
            }
        )
        return state

    safe_args = _sanitize_arguments(tool_name, arguments if isinstance(arguments, dict) else {}, state)
    status = "ok"
    observation: Any
    try:
        observation = call_mcp_tool_sync(tool_name, safe_args)
    except Exception as exc:
        observation = {"error": str(exc)}
        status = "error"
        state["warnings"].append(f"Hybrid executor failed for tool '{tool_name}'. Detail: {exc}")

    state["agent_step_count"] += 1
    state["agent_scratchpad"].append(f"step={state['agent_step_count']} tool={tool_name} status={status}")
    state["agent_tool_history"].append(
        {
            "step": state["agent_step_count"],
            "thought": thought,
            "tool_name": tool_name,
            "arguments": safe_args,
            "status": status,
            "observation": observation,
        }
    )

    if state["agent_step_count"] >= state["agent_max_steps"]:
        state["agent_done"] = True
        state["agent_fallback_reason"] = "agent_max_steps_reached"

    return state
