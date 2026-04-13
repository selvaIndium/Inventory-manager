"""LangGraph node to calculate SKU metrics via MCP tools."""

from __future__ import annotations

from agent.state import AgentState, SKUMetrics
from tools.server import call_mcp_tool_sync


def calculate_metrics_node(state: AgentState) -> AgentState:
    """Calculate inventory metrics for all validated SKU records."""
    state["current_node"] = "calculate_metrics"
    metrics: list[SKUMetrics] = []
    scenario_overrides = state["config"].get("scenario_overrides", {})

    raw_by_sku = {str(row.get("sku_id", "")): row for row in state["raw_records"]}

    for record in state["sku_records"]:
        try:
            sku_payload = dict(raw_by_sku.get(record.sku_id, {}))
            sku_payload.update(
                {
                    "sku_id": record.sku_id,
                    "name": record.name,
                    "category": record.category,
                    "current_stock": record.current_stock,
                    "avg_daily_sales": record.avg_daily_sales,
                    "lead_time_days": record.lead_time_days,
                    "safety_stock": record.safety_stock,
                }
            )

            if "lead_time" in scenario_overrides or "lead_time_days" in scenario_overrides:
                sku_payload["lead_time_days"] = int(
                    scenario_overrides.get("lead_time_days", scenario_overrides.get("lead_time", record.lead_time_days))
                )
            if "safety_stock" in scenario_overrides or "default_safety_stock" in scenario_overrides:
                sku_payload["safety_stock"] = float(
                    scenario_overrides.get("safety_stock", scenario_overrides.get("default_safety_stock", record.safety_stock))
                )
            payload = call_mcp_tool_sync(
                "calc_metrics",
                {"sku": sku_payload, "config": state["config"]},
            )
            metrics.append(
                SKUMetrics(
                    sku_id=payload["sku_id"],
                    days_of_stock=float(payload["days_of_stock"]),
                    reorder_qty=float(payload["reorder_qty"]),
                    reorder_urgency_days=float(payload["reorder_urgency_days"]),
                    velocity_trend=payload["velocity_trend"],
                    status=payload["status"],
                    status_emoji=payload["status_emoji"],
                )
            )
        except Exception as exc:
            state["errors"].append(
                {"node": "calculate_metrics", "sku_id": record.sku_id, "message": str(exc)}
            )

    state["sku_metrics"] = metrics
    return state
