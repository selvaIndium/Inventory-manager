"""LangGraph node to apply threshold rules and map rule matches."""

from __future__ import annotations

from agent.logging_utils import add_flow_event, add_tool_call_log, timer_ms, timer_start
from agent.state import AgentState
from tools.server import call_mcp_tool_sync

STATUS_TO_RULE = {
    "overstock": "R-OVERSTOCK",
    "healthy": "R-HEALTHY",
    "watch": "R-WATCH",
    "critical": "R-CRITICAL",
}


def apply_rules_node(state: AgentState) -> AgentState:
    """Fetch active rules and assign matched rules by SKU status."""
    node = "apply_rules"
    add_flow_event(state, node=node, event="start")
    state["current_node"] = "apply_rules"
    config_path = str(state["config"].get("config_path", "config/thresholds.yaml"))

    args = {"config_path": config_path, "category": None}
    started = timer_start()
    try:
        result = call_mcp_tool_sync("fetch_rules", args)
        add_tool_call_log(
            state,
            node=node,
            tool_name="fetch_rules",
            caller="deterministic_system",
            arguments=args,
            status="ok",
            duration_ms=timer_ms(started),
            output_count=len(result.get("rules", [])),
        )
    except Exception as exc:
        add_tool_call_log(
            state,
            node=node,
            tool_name="fetch_rules",
            caller="deterministic_system",
            arguments=args,
            status="error",
            duration_ms=timer_ms(started),
            error=str(exc),
        )
        state["errors"].append({"node": "apply_rules", "sku_id": "", "message": str(exc)})
        state["rule_results"] = {}
        add_flow_event(state, node=node, event="end", detail="error")
        return state

    rule_ids = {rule["rule_id"]: rule for rule in result.get("rules", [])}

    mapped: dict[str, list[str]] = {}
    for metric in state["sku_metrics"]:
        matched = STATUS_TO_RULE.get(metric.status)
        if matched and matched in rule_ids:
            mapped[metric.sku_id] = [matched]
        else:
            mapped[metric.sku_id] = []

    state["rule_results"] = mapped
    add_flow_event(state, node=node, event="end", extra={"rule_results": len(mapped)})
    return state
