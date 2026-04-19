"""Preflight checks for full-agent executions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import httpx

from tools.server import MCP_SERVER


def _parse_probe_content(content: str) -> Dict[str, Any]:
    """Parse planner probe content with tolerant JSON extraction."""
    text = str(content or "").strip()
    if not text:
        raise ValueError("empty_output")

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            raise ValueError("malformed_json")

    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError("malformed_json")
    return parsed


def _classify_probe_error(exc: Exception) -> str:
    """Normalize probe exceptions into stable error classes."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"

    detail = str(exc).strip() or "probe_unknown_error"
    lowered = detail.lower()
    if "timed out" in lowered:
        return "timeout"
    if "404" in detail and "not found" in lowered:
        if "model" in lowered:
            return "model_not_found"
        return "http_404"
    if "500" in detail:
        return "http_500"
    if "empty_output" in lowered:
        return "empty_output"
    if "malformed_json" in lowered or "expecting value" in lowered:
        return "malformed_json"
    return detail


def _check_registered_tools() -> Dict[str, Any]:
    """Verify required planner tools are registered in MCP server."""
    required = {"load_inventory", "calc_metrics_batch", "query_graph_batch", "apply_rules_batch"}
    try:
        import asyncio

        tools = asyncio.run(MCP_SERVER.list_tools())
        names = {str(getattr(item, "name", "")) for item in tools}
        missing = sorted(required - names)
        return {
            "name": "required_tools",
            "ok": len(missing) == 0,
            "detail": "ok" if not missing else f"Missing tools: {', '.join(missing)}",
        }
    except Exception as exc:
        return {"name": "required_tools", "ok": False, "detail": f"Tool registry check failed: {exc}"}


def _check_ollama_tags(base_url: str, model: str) -> Dict[str, Any]:
    """Check Ollama reachability and model presence."""
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        if not models:
            return {"name": "ollama_model", "ok": False, "detail": "No models found in Ollama /api/tags"}
        if model not in models:
            return {"name": "ollama_model", "ok": False, "detail": f"Model '{model}' not installed"}
        return {"name": "ollama_model", "ok": True, "detail": f"Model '{model}' available"}
    except Exception as exc:
        return {"name": "ollama_model", "ok": False, "detail": f"Ollama unreachable: {exc}"}


def _parse_param_size_billions(value: str) -> float | None:
    """Parse model parameter size strings like '1.2B' into float billions."""
    text = str(value or "").strip().upper()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*B$", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _check_planner_model_gate(base_url: str, model: str, min_params_b: float = 3.0) -> Dict[str, Any]:
    """Gate full-agent mode to planner-capable models."""
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()

        models = payload.get("models", []) if isinstance(payload, dict) else []
        selected = None
        for item in models:
            if isinstance(item, dict) and str(item.get("name", "")) == model:
                selected = item
                break

        if not isinstance(selected, dict):
            return {
                "name": "planner_model_gate",
                "ok": False,
                "detail": f"Model gate failed: '{model}' not present in Ollama /api/tags",
            }

        details = selected.get("details", {}) if isinstance(selected.get("details"), dict) else {}
        param_size = _parse_param_size_billions(str(details.get("parameter_size", "")))
        if param_size is None:
            return {
                "name": "planner_model_gate",
                "ok": False,
                "detail": (
                    f"Model gate failed: cannot verify parameter size for '{model}'. "
                    f"Thinking+full requires >= {min_params_b:.1f}B parameters."
                ),
            }

        if param_size < min_params_b:
            return {
                "name": "planner_model_gate",
                "ok": False,
                "detail": (
                    f"Model gate failed: '{model}' is {param_size:.1f}B (< {min_params_b:.1f}B). "
                    "Use a stronger model for thinking+full planning."
                ),
            }

        return {
            "name": "planner_model_gate",
            "ok": True,
            "detail": f"Planner model gate passed: '{model}' ({param_size:.1f}B)",
        }
    except Exception as exc:
        return {
            "name": "planner_model_gate",
            "ok": False,
            "detail": f"Model gate failed: could not inspect Ollama tags ({exc})",
        }


def _check_planner_probe(
    base_url: str,
    model: str,
    api_key: str = "",
    planner_timeout_ms: int = 0,
) -> Dict[str, Any]:
    """Run a lightweight planner probe and validate strict JSON shape."""
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    timeout_seconds: float | None = None if int(planner_timeout_ms) <= 0 else max(5.0, float(planner_timeout_ms) / 1000.0)
    timeout_label = "none" if timeout_seconds is None else f"{timeout_seconds:.1f}s"

    probe_prompt = {
        "instruction": "Return one action in strict JSON.",
        "context": {"records": 0, "metrics": 0, "contexts": 0, "rule_results": 0},
        "constraints": {"allowed_tools": ["load_inventory"], "current_step": 0, "max_steps": 3},
    }
    strict_schema = {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool_name": {"type": "string"},
            "arguments": {"type": "object"},
            "done": {"type": "boolean"},
        },
        "required": ["thought", "tool_name", "arguments", "done"],
    }
    strict_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a planner. Return strict JSON action only."},
            {"role": "user", "content": json.dumps(probe_prompt, ensure_ascii=False)},
        ],
        "stream": False,
        "think": False,
        "format": strict_schema,
        "options": {"temperature": 0.0},
    }

    relaxed_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only one JSON object action. No markdown."},
            {"role": "user", "content": json.dumps(probe_prompt, ensure_ascii=False)},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    required = {"thought", "tool_name", "arguments", "done"}
    failures: List[str] = []
    attempts = [("strict", strict_request), ("relaxed", relaxed_request)]

    for attempt_name, request in attempts:
        try:
            with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
                response = client.post(f"{base_url.rstrip('/')}/api/chat", json=request)
                response.raise_for_status()
                payload = response.json()

            content = payload.get("message", {}).get("content", "")
            parsed = _parse_probe_content(str(content))
            if not required.issubset(set(parsed.keys())):
                failures.append(f"{attempt_name}:invalid_action_schema")
                continue

            return {
                "name": "planner_probe",
                "ok": True,
                "detail": f"Planner probe succeeded ({attempt_name}, model={model}, timeout={timeout_label}).",
            }
        except Exception as exc:
            reason = _classify_probe_error(exc)
            if reason == "timeout":
                reason = f"timeout({timeout_label})"
            failures.append(f"{attempt_name}:{reason}")

    return {
        "name": "planner_probe",
        "ok": False,
        "detail": f"Planner probe failed (model={model}, timeout={timeout_label}): " + " | ".join(failures),
    }


def run_preflight_checks(
    *,
    mode: str,
    agent_mode: str,
    base_url: str,
    model: str,
    records_count: int,
    api_key: str = "",
    planner_timeout_ms: int = 0,
) -> Dict[str, Any]:
    """Run E2E preflight checks for full-agent runs."""
    checks: List[Dict[str, Any]] = []

    checks.append(
        {
            "name": "dataset_ready",
            "ok": records_count > 0,
            "detail": f"Records available: {records_count}",
        }
    )

    if str(mode).lower() == "thinking" and str(agent_mode).lower() == "full":
        checks.append(_check_registered_tools())
        checks.append(_check_ollama_tags(base_url, model))
        if checks[-1]["ok"]:
            checks.append(_check_planner_model_gate(base_url, model, min_params_b=3.0))
        if checks[-1]["ok"]:
            checks.append(
                _check_planner_probe(
                    base_url,
                    model,
                    api_key=api_key,
                    planner_timeout_ms=planner_timeout_ms,
                )
            )

    blocking = [item for item in checks if not bool(item.get("ok", False))]
    return {
        "ok": len(blocking) == 0,
        "checks": checks,
        "blocking_reason": blocking[0]["detail"] if blocking else "",
    }
