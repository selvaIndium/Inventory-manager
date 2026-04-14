"""LangGraph node to load inventory records via MCP tools."""

from __future__ import annotations

from pathlib import Path

from agent.state import AgentState, SKURecord
from tools.server import call_mcp_tool_sync


def load_data_node(state: AgentState) -> AgentState:
    """Load CSV/JSON data using MCP and build validated SKURecord objects."""
    
    state["current_node"] = "load_data"
    data_path = Path(state["config"].get("data_path", "data/inventory_mock.csv"))
    tool_name = "load_json" if data_path.suffix.lower() == ".json" else "load_csv"

    try:
        result = call_mcp_tool_sync(tool_name, {"file_path": str(data_path)})
    except Exception as exc:
        state["errors"].append({"node": "load_data", "sku_id": "", "message": str(exc)})
        state["partial_data"] = True
        return state

    records = result.get("records", [])
    invalid_rows = result.get("invalid_rows", [])
    warnings = result.get("warnings", [])

    state["raw_records"] = records
    state["warnings"].extend(warnings)
    if invalid_rows:
        state["partial_data"] = True
        state["warnings"].append(f"Invalid rows encountered: {len(invalid_rows)}")

    state["sku_records"] = [
        SKURecord(
            sku_id=str(row["sku_id"]),
            name=str(row["name"]),
            category=str(row["category"]),
            current_stock=float(row["current_stock"]),
            avg_daily_sales=float(row["avg_daily_sales"]),
            lead_time_days=int(row["lead_time_days"]),
            safety_stock=float(row["safety_stock"]),
            reorder_point=row.get("reorder_point"),
            last_sale_date=row.get("last_sale_date"),
            supplier_id=row.get("supplier_id"),
        )
        for row in records
    ]

    return state
