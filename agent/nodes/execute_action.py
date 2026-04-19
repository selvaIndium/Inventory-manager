"""LangGraph executor node for hybrid planner-selected MCP actions."""

from __future__ import annotations

from typing import Any, Dict, List

from agent.logging_utils import add_flow_event, add_tool_call_log, timer_ms, timer_start
from agent.nodes.apply_rules import STATUS_TO_RULE
from agent.state import AgentState, SKUContext, SKUMetrics, SKURecord
from tools.server import call_mcp_tool_sync


def _sanitize_arguments(tool_name: str, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    """Normalize arguments for known tool contracts."""
    if tool_name in {"load_csv", "load_json", "load_inventory"}:
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

    if tool_name == "calc_metrics_batch":
        records = arguments.get("records")
        if isinstance(records, list) and records:
            return {"records": records, "config": arguments.get("config", state["config"])}
        return {"records": state.get("raw_records", []), "config": arguments.get("config", state["config"])}

    if tool_name == "query_graph_batch":
        records = arguments.get("records")
        if isinstance(records, list) and records:
            return {"records": records, "config": arguments.get("config", state["config"])}
        fallback = [
            {"sku_id": record.sku_id, "category": record.category}
            for record in state.get("sku_records", [])
        ]
        return {"records": fallback, "config": arguments.get("config", state["config"])}

    if tool_name == "apply_rules_batch":
        metrics = arguments.get("metrics")
        if isinstance(metrics, list) and metrics:
            return {"metrics": metrics, "config": arguments.get("config", state["config"])}
        fallback_metrics = [
            {
                "sku_id": metric.sku_id,
                "status": metric.status,
                "days_of_stock": metric.days_of_stock,
                "reorder_qty": metric.reorder_qty,
                "reorder_urgency_days": metric.reorder_urgency_days,
                "velocity_trend": metric.velocity_trend,
                "status_emoji": metric.status_emoji,
            }
            for metric in state.get("sku_metrics", [])
        ]
        return {"metrics": fallback_metrics, "config": arguments.get("config", state["config"])}

    return arguments


def _records_to_sku_records(records: List[Dict[str, Any]]) -> List[SKURecord]:
    """Convert normalized dictionaries into SKURecord dataclasses."""
    converted: List[SKURecord] = []
    for row in records:
        try:
            converted.append(
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
            )
        except Exception:
            continue
    return converted


def _upsert_metric(state: AgentState, payload: Dict[str, Any]) -> None:
    """Insert or replace one SKUMetrics entry by sku_id."""
    try:
        metric = SKUMetrics(
            sku_id=str(payload["sku_id"]),
            days_of_stock=float(payload["days_of_stock"]),
            reorder_qty=float(payload["reorder_qty"]),
            reorder_urgency_days=float(payload["reorder_urgency_days"]),
            velocity_trend=str(payload["velocity_trend"]),
            status=str(payload["status"]),
            status_emoji=str(payload["status_emoji"]),
        )
    except Exception:
        return

    existing = {item.sku_id: item for item in state["sku_metrics"]}
    existing[metric.sku_id] = metric
    state["sku_metrics"] = list(existing.values())


def _upsert_context(state: AgentState, payload: Dict[str, Any]) -> None:
    """Insert or replace one SKUContext entry by sku_id."""
    try:
        source = str(payload.get("source", "default"))
        context = SKUContext(
            sku_id=str(payload["sku_id"]),
            seasonal_factor=float(payload.get("seasonal_factor", 1.0)),
            category_avg_dos=float(payload.get("category_avg_dos", 30.0)),
            risk_tags=list(payload.get("risk_tags", [])),
            context_source=source if source in {"networkx", "cache", "default"} else "default",
        )
    except Exception:
        return

    existing = {item.sku_id: item for item in state["sku_contexts"]}
    existing[context.sku_id] = context
    state["sku_contexts"] = list(existing.values())
    state["graph_source"] = context.context_source


def _apply_tool_observation(state: AgentState, tool_name: str, observation: Any) -> None:
    """Apply successful planner tool outputs into shared state."""
    if not isinstance(observation, dict):
        return

    if tool_name in {"load_csv", "load_json", "load_inventory"}:
        records = observation.get("records", [])
        if isinstance(records, list):
            state["raw_records"] = records
            state["sku_records"] = _records_to_sku_records(records)
        warnings = observation.get("warnings", [])
        if isinstance(warnings, list):
            state["warnings"].extend(str(item) for item in warnings)
        invalid_rows = observation.get("invalid_rows", [])
        if isinstance(invalid_rows, list) and invalid_rows:
            state["partial_data"] = True
            state["warnings"].append(f"Invalid rows encountered: {len(invalid_rows)}")
        return

    if tool_name == "calc_metrics":
        _upsert_metric(state, observation)
        return

    if tool_name == "calc_metrics_batch":
        metrics = observation.get("metrics", [])
        if isinstance(metrics, list):
            for item in metrics:
                if isinstance(item, dict):
                    _upsert_metric(state, item)
        return

    if tool_name == "query_graph":
        _upsert_context(state, observation)
        return

    if tool_name == "query_graph_batch":
        contexts = observation.get("contexts", [])
        if isinstance(contexts, list):
            for item in contexts:
                if isinstance(item, dict):
                    _upsert_context(state, item)
        return

    if tool_name == "fetch_rules":
        rules = observation.get("rules", [])
        if not isinstance(rules, list):
            return
        rule_ids = {
            str(rule.get("rule_id", "")): rule for rule in rules if isinstance(rule, dict)
        }
        mapped: Dict[str, List[str]] = {}
        for metric in state["sku_metrics"]:
            match = STATUS_TO_RULE.get(metric.status)
            mapped[metric.sku_id] = [match] if match and match in rule_ids else []
        state["rule_results"] = mapped
        return

    if tool_name == "apply_rules_batch":
        mapped = observation.get("rule_results", {})
        if isinstance(mapped, dict):
            normalized: Dict[str, List[str]] = {}
            for key, value in mapped.items():
                if isinstance(value, list):
                    normalized[str(key)] = [str(item) for item in value]
            state["rule_results"] = normalized
        return


def execute_action_node(state: AgentState) -> AgentState:
    """Execute planner-selected MCP action and append observation history."""
    node = "execute_action"
    add_flow_event(state, node=node, event="start")
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
        add_flow_event(state, node=node, event="end", detail="planner_done")
        return state

    safe_args = _sanitize_arguments(tool_name, arguments if isinstance(arguments, dict) else {}, state)
    status = "ok"
    observation: Any
    started = timer_start()
    try:
        observation = call_mcp_tool_sync(tool_name, safe_args)
        add_tool_call_log(
            state,
            node=node,
            tool_name=tool_name,
            caller="planner_model",
            arguments=safe_args,
            status="ok",
            duration_ms=timer_ms(started),
            output_count=1,
        )
    except Exception as exc:
        observation = {"error": str(exc)}
        status = "error"
        state["warnings"].append(f"Hybrid executor failed for tool '{tool_name}'. Detail: {exc}")
        add_tool_call_log(
            state,
            node=node,
            tool_name=tool_name,
            caller="planner_model",
            arguments=safe_args,
            status="error",
            duration_ms=timer_ms(started),
            error=str(exc),
        )

    if status == "ok":
        _apply_tool_observation(state, tool_name, observation)

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

    add_flow_event(
        state,
        node=node,
        event="end",
        extra={"tool_name": tool_name, "status": status, "agent_step_count": state["agent_step_count"]},
    )

    return state
