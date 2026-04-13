"""LangGraph node to enrich SKUs with hybrid knowledge graph context."""

from __future__ import annotations

from agent.state import AgentState, SKUContext
from tools.server import call_mcp_tool_sync


def enrich_context_node(state: AgentState) -> AgentState:
    """Attach contextual graph metadata to each SKU metric."""
    state["current_node"] = "enrich_context"

    contexts: list[SKUContext] = []
    source_counts: dict[str, int] = {}
    category_by_sku = {record.sku_id: record.category for record in state["sku_records"]}

    for metric in state["sku_metrics"]:
        category = category_by_sku.get(metric.sku_id, "unknown")
        try:
            payload = call_mcp_tool_sync(
                "query_graph",
                {
                    "sku_id": metric.sku_id,
                    "category": category,
                    "query_type": "all",
                    "config": state["config"],
                },
            )
        except Exception as exc:
            state["errors"].append(
                {"node": "enrich_context", "sku_id": metric.sku_id, "message": str(exc)}
            )
            payload = {
                "sku_id": metric.sku_id,
                "seasonal_factor": 1.0,
                "category_avg_dos": 30.0,
                "risk_tags": [],
                "source": "default",
            }

        source = str(payload.get("source", "default"))
        source_counts[source] = source_counts.get(source, 0) + 1

        contexts.append(
            SKUContext(
                sku_id=metric.sku_id,
                seasonal_factor=float(payload.get("seasonal_factor", 1.0)),
                category_avg_dos=float(payload.get("category_avg_dos", 30.0)),
                risk_tags=list(payload.get("risk_tags", [])),
                context_source=source if source in {"neo4j", "networkx", "cache", "default"} else "default",
            )
        )

    state["sku_contexts"] = contexts
    if source_counts:
        state["graph_source"] = max(source_counts.items(), key=lambda item: item[1])[0]

    return state
