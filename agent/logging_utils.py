"""Structured logging helpers for LangGraph execution metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict


def now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def add_flow_event(
    state: Dict[str, Any],
    *,
    node: str,
    event: str,
    duration_ms: float | None = None,
    detail: str = "",
    extra: Dict[str, Any] | None = None,
) -> None:
    """Append a flow timeline event to runtime state."""
    payload: Dict[str, Any] = {
        "ts": now_iso(),
        "node": node,
        "event": event,
    }
    if duration_ms is not None:
        payload["duration_ms"] = round(float(duration_ms), 2)
    if detail:
        payload["detail"] = detail
    if extra:
        payload.update(extra)

    state.setdefault("flow_events", []).append(payload)


def _sanitize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Remove heavy nested values from logged tool arguments."""
    safe = dict(arguments)
    if "config" in safe:
        safe["config"] = "<omitted>"
    if "sku" in safe and isinstance(safe["sku"], dict):
        sku = safe["sku"]
        safe["sku"] = {
            "sku_id": sku.get("sku_id", ""),
            "status": sku.get("status", ""),
            "category": sku.get("category", ""),
        }
    return safe


def add_tool_call_log(
    state: Dict[str, Any],
    *,
    node: str,
    tool_name: str,
    caller: str,
    arguments: Dict[str, Any],
    status: str,
    duration_ms: float,
    error: str = "",
    output_count: int | None = None,
) -> None:
    """Append a structured tool call log entry."""
    entry: Dict[str, Any] = {
        "ts": now_iso(),
        "node": node,
        "caller": caller,
        "tool_name": tool_name,
        "status": status,
        "duration_ms": round(float(duration_ms), 2),
        "arguments": _sanitize_arguments(arguments),
    }
    if error:
        entry["error"] = error
    if output_count is not None:
        entry["output_count"] = int(output_count)

    state.setdefault("tool_call_logs", []).append(entry)


def add_llm_batch_event(state: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Append one LLM batch event for diagnostics."""
    payload = dict(event)
    payload["ts"] = now_iso()
    state.setdefault("llm_batch_events", []).append(payload)


def timer_start() -> float:
    """Return high-resolution timer start."""
    return perf_counter()


def timer_ms(start: float) -> float:
    """Return elapsed milliseconds from timer start."""
    return (perf_counter() - start) * 1000
