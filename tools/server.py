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

    @mcp.tool(name="calc_metrics")
    def calc_metrics(sku: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for one SKU payload."""
        return calculate_metrics_for_sku(sku, config)

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

    return mcp


MCP_SERVER = create_mcp_server()


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