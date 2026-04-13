"""LangGraph node to assemble batched LLM payloads per SKU."""

from __future__ import annotations

import json

from agent.state import AgentState


def generate_recs_node(state: AgentState) -> AgentState:
    """Prepare per-SKU payloads for single-shot batched LLM explanation generation."""
    state["current_node"] = "generate_recs"

    contexts = {context.sku_id: context for context in state["sku_contexts"]}
    records = {record.sku_id: record for record in state["sku_records"]}

    state["llm_prompts"] = {}
    for metric in state["sku_metrics"]:
        record = records.get(metric.sku_id)
        context = contexts.get(metric.sku_id)
        payload = {
            "sku_id": metric.sku_id,
            "name": record.name if record else "Unknown",
            "category": record.category if record else "unknown",
            "status": metric.status,
            "status_emoji": metric.status_emoji,
            "days_of_stock": metric.days_of_stock,
            "reorder_qty": metric.reorder_qty,
            "reorder_urgency_days": metric.reorder_urgency_days,
            "velocity_trend": metric.velocity_trend,
            "seasonal_factor": context.seasonal_factor if context else 1.0,
            "risk_tags": context.risk_tags if context else [],
            "formula_used": (
                "reorder_qty = (avg_daily_sales x lead_time_days) + safety_stock - current_stock"
            ),
            "context_source": context.context_source if context else "default",
            "rule_results": state["rule_results"].get(metric.sku_id, []),
        }
        state["llm_prompts"][metric.sku_id] = json.dumps(payload, ensure_ascii=False)

    return state
