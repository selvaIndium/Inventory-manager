"""FastMCP server and local tool wrappers for the inventory agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastmcp import FastMCP

from tools.calc_metrics import calculate_metrics_for_sku
from tools.fetch_rules import fetch_rules as fetch_rules_impl
from tools.load_data import load_inventory_data
from tools.query_graph import query_graph as query_graph_impl


STATUS_TO_RULE = {
    "overstock": "R-OVERSTOCK",
    "healthy": "R-HEALTHY",
    "watch": "R-WATCH",
    "critical": "R-CRITICAL",
}


def _calc_metrics_batch(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Calculate metrics for a batch of records."""
    metrics = [calculate_metrics_for_sku(record, config) for record in records]
    return {"metrics": metrics, "count": len(metrics)}


def _query_graph_batch(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Query graph context for all records in one call."""
    contexts: list[dict[str, Any]] = []
    for record in records:
        sku_id = str(record.get("sku_id", "")).strip()
        category = str(record.get("category", "unknown")).strip()
        if not sku_id:
            continue
        contexts.append(
            query_graph_impl(
                sku_id=sku_id,
                category=category,
                query_type="all",
                config=config,
            )
        )
    return {"contexts": contexts, "count": len(contexts)}


def _apply_rules_batch(metrics: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Apply status-based rules to all metrics in one call."""
    config_path = str(config.get("config_path", "config/thresholds.yaml"))
    rules_payload = fetch_rules_impl(config_path=config_path, category=None)
    rules = rules_payload.get("rules", [])
    rule_ids = {
        str(rule.get("rule_id", "")): rule for rule in rules if isinstance(rule, dict)
    }

    mapped: dict[str, list[str]] = {}
    for metric in metrics:
        sku_id = str(metric.get("sku_id", "")).strip()
        status = str(metric.get("status", "")).strip()
        match = STATUS_TO_RULE.get(status)
        mapped[sku_id] = [match] if match and match in rule_ids else []

    return {
        "rule_results": mapped,
        "rules": rules,
        "count": len(mapped),
    }


def create_mcp_server() -> FastMCP:
    """Create and register FastMCP tools for local execution."""
    mcp = FastMCP("inventory-agent")

    @mcp.tool(name="load_csv")
    def load_csv(file_path: str) -> Dict[str, Any]:
        """Load and validate CSV inventory data."""
        return load_inventory_data(file_path)

    @mcp.tool(name="load_json")
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load and validate JSON inventory data."""
        return load_inventory_data(file_path)

    @mcp.tool(name="load_inventory")
    def load_inventory(file_path: str) -> Dict[str, Any]:
        """Load and validate CSV/JSON inventory data."""
        return load_inventory_data(file_path)

    @mcp.tool(name="calc_metrics")
    def calc_metrics(sku: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for one SKU payload."""
        return calculate_metrics_for_sku(sku, config)

    @mcp.tool(name="calc_metrics_batch")
    def calc_metrics_batch(records: list[dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a record batch."""
        return _calc_metrics_batch(records=records, config=config)

    @mcp.tool(name="fetch_rules")
    def fetch_rules(config_path: str, category: str | None = None) -> Dict[str, Any]:
        """Fetch active status and action rules."""
        return fetch_rules_impl(config_path=config_path, category=category)

    @mcp.tool(name="query_graph")
    def query_graph(
        sku_id: str,
        category: str,
        query_type: str = "all",
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Query cache + graph layers for contextual SKU data."""
        return query_graph_impl(
            sku_id=sku_id,
            category=category,
            query_type=query_type,
            config=config or {},
        )

    @mcp.tool(name="query_graph_batch")
    def query_graph_batch(records: list[dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Query graph context for all records in one call."""
        return _query_graph_batch(records=records, config=config)

    @mcp.tool(name="apply_rules_batch")
    def apply_rules_batch(metrics: list[dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Map rules for all metrics in one call."""
        return _apply_rules_batch(metrics=metrics, config=config)

    return mcp


MCP_SERVER = create_mcp_server()


#this is how the tools are being called, u pass the tool name and the args inside.
async def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a registered MCP tool and return structured content."""
    result = await MCP_SERVER.call_tool(name, arguments)
    if isinstance(result.structured_content, dict):
        return result.structured_content

    if result.content:
        first = result.content[0]
        text = getattr(first, "text", "{}")
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload

    raise ValueError(f"Unexpected MCP result format for tool: {name}")


def call_mcp_tool_sync(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronously invoke an MCP tool from LangGraph nodes."""
    
    try:
        return asyncio.run(call_mcp_tool(name, arguments))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(call_mcp_tool(name, arguments))
        finally:
            loop.close()

if __name__ == "__main__":
    MCP_SERVER.run()  # stdio transport (default)
