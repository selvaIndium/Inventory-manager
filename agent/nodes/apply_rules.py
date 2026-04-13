"""LangGraph node to apply threshold rules and map rule matches."""

from __future__ import annotations

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
    state["current_node"] = "apply_rules"
    config_path = str(state["config"].get("config_path", "config/thresholds.yaml"))

    try:
        result = call_mcp_tool_sync("fetch_rules", {"config_path": config_path, "category": None})
    except Exception as exc:
        state["errors"].append({"node": "apply_rules", "sku_id": "", "message": str(exc)})
        state["rule_results"] = {}
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
    return state
