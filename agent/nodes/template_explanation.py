"""LangGraph fallback node for deterministic rule-based explanations."""

from __future__ import annotations

import json

from agent.nodes.format_output import DISCLAIMER
from agent.state import AgentState

MANDATORY_ENDING = "This recommendation is advisory only and should be reviewed by your supply chain team."


def template_explanation_node(state: AgentState) -> AgentState:
    """Fill missing explanations using deterministic templates by SKU status."""
    state["current_node"] = "template_explanation"

    for metric in state["sku_metrics"]:
        if metric.sku_id in state["llm_responses"]:
            continue

        if metric.status == "critical":
            action = "Consider prioritizing reorder review immediately due to low coverage."
            explanation = (
                f"{metric.sku_id} is in critical status with about {metric.days_of_stock:.1f} days of stock and "
                f"a reorder urgency of {metric.reorder_urgency_days:.1f} days. "
                "Based on current demand and lead time, delayed action may increase stockout risk. "
                f"{MANDATORY_ENDING}"
            )
            confidence = "high"
        elif metric.status == "watch":
            action = "Consider planning a reorder in the upcoming cycle."
            explanation = (
                f"{metric.sku_id} is in watch status with approximately {metric.days_of_stock:.1f} days of stock. "
                "Coverage remains above critical levels but may tighten if demand rises. "
                f"{MANDATORY_ENDING}"
            )
            confidence = "medium"
        elif metric.status == "overstock":
            action = "Consider reducing replenishment frequency and monitoring demand movement."
            explanation = (
                f"{metric.sku_id} is currently overstocked at around {metric.days_of_stock:.1f} days of stock. "
                "This level may tie up working capital and increase carrying costs if demand remains stable. "
                f"{MANDATORY_ENDING}"
            )
            confidence = "medium"
        else:
            action = "Maintain current replenishment approach and continue routine monitoring."
            explanation = (
                f"{metric.sku_id} is in healthy status with approximately {metric.days_of_stock:.1f} days of stock. "
                "Current coverage appears aligned with standard planning thresholds based on available data. "
                f"{MANDATORY_ENDING}"
            )
            confidence = "high"

        state["llm_responses"][metric.sku_id] = json.dumps(
            {
                "explanation": explanation,
                "action": action,
                "confidence": confidence,
            },
            ensure_ascii=False,
        )

    if "llm_fallback_used" not in state["warnings"]:
        if any("LLM" in warning for warning in state["warnings"]):
            state["warnings"].append("llm_fallback_used")

    _ = DISCLAIMER
    return state
