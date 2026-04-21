"""LangGraph node to call Ollama in fixed-size SKU batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import httpx

from agent.logging_utils import add_flow_event, add_llm_batch_event
from agent.state import AgentState


BATCH_SIZE = 5
SMALL_MODELS = {"llama3.2:1b"}


def _normalize_action(status: str, reorder_qty: float, action: str) -> tuple[str, bool]:
    """Enforce deterministic action-policy guardrails over LLM output."""
    raw = str(action or "").strip()
    lowered = raw.lower()

    healthy_action = "Maintain current replenishment approach and continue routine monitoring."
    critical_action = "Consider prioritizing reorder review immediately due to low coverage."
    watch_action = "Consider planning a reorder in the upcoming cycle."
    overstock_action = "Consider reducing replenishment frequency and monitoring demand movement."

    if status == "healthy":
        if reorder_qty <= 0 and any(token in lowered for token in ["order", "reorder", "restock", "additional"]):
            return healthy_action, True
        if not raw:
            return healthy_action, True
        return raw, False
    if status == "critical":
        if any(token in lowered for token in ["maintain", "hold", "delay", "reduce"]):
            return critical_action, True
        return (raw or critical_action), not bool(raw)
    if status == "watch":
        if any(token in lowered for token in ["maintain", "hold", "delay", "reduce"]):
            return watch_action, True
        return (raw or watch_action), not bool(raw)
    if status == "overstock":
        if any(token in lowered for token in ["order", "reorder", "restock", "additional"]):
            return overstock_action, True
        return (raw or overstock_action), not bool(raw)
    return (raw or "Consider reviewing this SKU with your planning team."), not bool(raw)


def _deterministic_reasoning_summary(entry: Dict[str, Any]) -> str:
    """Create concise deterministic rationale summary from input metrics."""
    status = str(entry.get("status", "unknown"))
    dos = float(entry.get("days_of_stock", 0.0))
    rq = float(entry.get("reorder_qty", 0.0))
    urgency = float(entry.get("reorder_urgency_days", 0.0))
    seasonal = float(entry.get("seasonal_factor", 1.0))
    return (
        f"status={status}; days_of_stock={dos:.1f}; reorder_qty={rq:.1f}; "
        f"urgency_days={urgency:.1f}; seasonal_factor={seasonal:.2f}"
    )


def _stream_reasoning_text(
    *,
    base_url: str,
    model: str,
    headers: Dict[str, str],
    system_prompt: str,
    batch_ids: List[str],
    compact_inputs: List[Dict[str, Any]],
) -> str:
    """Stream raw model thinking text (experimental) for transparency."""
    payload = {
        "task": "Provide raw reasoning only for this SKU batch.",
        "instruction": "Think through inventory action selection for each SKU and provide internal reasoning.",
        "expected_sku_ids": batch_ids,
        "sku_inputs": compact_inputs,
    }
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "stream": True,
        "think": True,
        "options": {"temperature": 0.0},
    }

    parts: List[str] = []
    with httpx.Client(timeout=None, headers=headers) as client:
        with client.stream("POST", f"{base_url}/api/chat", json=request_payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                message = chunk.get("message", {}) if isinstance(chunk, dict) else {}
                thinking = str(message.get("thinking", ""))
                content = str(message.get("content", ""))
                if thinking:
                    parts.append(thinking)
                elif content:
                    parts.append(content)

    return "".join(parts).strip()



#helper scrubs out markdown backticks if the LLM output fenced code blocks.
def _parse_content_json(content: str) -> Dict[str, Any]:
    """Parse model content into JSON, tolerating fenced code blocks."""
    text = str(content or "").strip()
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
            raise

    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned non-object JSON payload.")
    return parsed


def _chunked(values: List[str], size: int) -> List[List[str]]:
    """Split a list into fixed-size chunks."""
    return [values[index : index + size] for index in range(0, len(values), size)]


def _compact_input(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce token-heavy fields before sending to the LLM."""
    return {
        "sku_id": raw.get("sku_id"),
        "status": raw.get("status"),
        "days_of_stock": round(float(raw.get("days_of_stock", 0.0)), 1),
        "reorder_qty": round(float(raw.get("reorder_qty", 0.0)), 1),
        "reorder_urgency_days": round(float(raw.get("reorder_urgency_days", 0.0)), 1),
        "velocity_trend": raw.get("velocity_trend"),
        "seasonal_factor": raw.get("seasonal_factor", 1.0),
        "risk_tags": raw.get("risk_tags", []),
    }


def _call_batch(
    *,
    base_url: str,
    model: str,
    timeout_seconds: float | None,
    headers: Dict[str, str],
    temperature: float,
    system_prompt: str,
    batch_ids: List[str],
    compact_inputs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Invoke Ollama for one SKU batch and parse JSON output."""
    user_payload = {
        "task": "Generate inventory recommendations for this SKU batch.",
        "instruction": (
            "Return JSON only. Include exactly one entry for every sku_id in expected_sku_ids. "
            "Use concise advisory language. "
            "Choose action consistent with status and reorder_qty. "
            "healthy with reorder_qty<=0 must not request ordering."
        ),
        "expected_sku_ids": batch_ids,
        "sku_inputs": compact_inputs,
    }

    response_schema = {
        "type": "object",
        "properties": {
            "sku_recommendations": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string"},
                        "action": {"type": "string"},
                        "confidence": {"type": "string"},
                        "reasoning_summary": {"type": "string"},
                    },
                    "required": ["explanation", "action", "confidence"],
                },
            }
        },
        "required": ["sku_recommendations"],
    }

    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "stream": False,
        "think": False,
        "format": response_schema,
        "options": {"temperature": temperature},
    }

    with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
        response = client.post(f"{base_url}/api/chat", json=request_payload)
        response.raise_for_status()
        raw = response.json()

    content = raw.get("message", {}).get("content", "{}")
    payload = _parse_content_json(content)

    if isinstance(payload, dict) and "sku_recommendations" in payload:
        batched = payload.get("sku_recommendations", {})
    elif isinstance(payload, dict):
        batched = payload
    else:
        batched = {}

    if not isinstance(batched, dict):
        raise ValueError("Invalid batched recommendation payload shape.")
    return batched


def stream_explain_llm_batches(state: AgentState) -> Iterator[Dict[str, Any]]:
    """Process fixed-size batches and yield progress events after each batch."""
    node = "explain_llm"
    add_flow_event(state, node=node, event="start")
    state["current_node"] = "explain_llm"

    if not state["llm_prompts"]:
        reason = str(state.get("agent_fallback_reason", ""))
        if reason.startswith("planner_unavailable"):
            state["warnings"].append(
                "No LLM prompts available because planner stopped before building metrics/prompts; template fallback will be used."
            )
        else:
            state["warnings"].append("No LLM prompts available; template fallback will be used.")
        add_flow_event(state, node=node, event="end", detail="no_prompts")
        return

    mode = str(state["config"].get("mode", "thinking")).lower()
    fast_template_only = bool(state["config"].get("fast_template_only", False))
    if mode == "fast" and fast_template_only:
        state["warnings"].append("Fast mode template-only enabled; skipping LLM explanation step.")
        add_flow_event(state, node=node, event="end", detail="fast_template_only")
        return

    ollama_cfg = state["config"].get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434"))
    model = str(ollama_cfg.get("model", "llama3.2:1b"))
    timeout_seconds: float | None = None
    temperature = float(ollama_cfg.get("temperature", 0.1))
    api_key = str(ollama_cfg.get("api_key", "")).strip()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    system_prompt = Path("prompts/system_prompt.txt").read_text(encoding="utf-8")
    sku_ids = list(state["llm_prompts"].keys())
    batches = _chunked(sku_ids, BATCH_SIZE)
    failed_batches = 0
    processed_skus = 0

    for batch_index, batch_ids in enumerate(batches, start=1):
        raw_inputs = [json.loads(state["llm_prompts"][sku_id]) for sku_id in batch_ids]
        compact_inputs = [_compact_input(item) for item in raw_inputs]

        batch_success = False
        filled_skus: List[str] = []
        batch_detail = ""
        try:
            batched = _call_batch(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                headers=headers,
                temperature=temperature,
                system_prompt=system_prompt,
                batch_ids=batch_ids,
                compact_inputs=compact_inputs,
            )

            filled_count = 0
            corrected_count = 0
            for sku_id in batch_ids:
                rec = batched.get(sku_id)
                if not isinstance(rec, dict):
                    continue

                explanation = str(rec.get("explanation", "")).strip()
                action = str(rec.get("action", "")).strip()
                confidence = str(rec.get("confidence", "medium")).strip().lower() or "medium"
                reasoning_summary = str(rec.get("reasoning_summary", "")).strip()
                if not explanation:
                    continue

                entry = next((item for item in compact_inputs if str(item.get("sku_id", "")) == sku_id), {})
                normalized_action, corrected = _normalize_action(
                    str(entry.get("status", "")),
                    float(entry.get("reorder_qty", 0.0)),
                    action,
                )
                if corrected:
                    corrected_count += 1
                    state["warnings"].append(f"llm_action_corrected:{sku_id}")
                if not reasoning_summary:
                    reasoning_summary = _deterministic_reasoning_summary(entry)

                state["llm_responses"][sku_id] = json.dumps(
                    {
                        "explanation": explanation,
                        "action": normalized_action,
                        "confidence": confidence if confidence in {"high", "medium", "low"} else "medium",
                        "reasoning_summary": reasoning_summary,
                    },
                    ensure_ascii=False,
                )
                filled_count += 1
                filled_skus.append(sku_id)

            batch_success = filled_count > 0
            if filled_count < len(batch_ids):
                state["warnings"].append(
                    f"LLM returned partial recommendations for batch {batch_index}; template fallback will complete missing SKUs."
                )
            batch_detail = (
                f"{filled_count}/{len(batch_ids)} SKU(s) produced LLM explanations; "
                f"corrected_actions={corrected_count}."
            )

        except Exception as exc:
            failed_batches += 1
            batch_detail = str(exc)
            state["warnings"].append(
                f"LLM batch {batch_index} failed or timed out; template fallback will be used. Detail: {exc}"
            )

        processed_skus += len(batch_ids)
        event = {
            "batch_index": batch_index,
            "batch_total": len(batches),
            "batch_size": len(batch_ids),
            "batch_skus": list(batch_ids),
            "filled_skus": filled_skus,
            "processed_skus": processed_skus,
            "total_skus": len(sku_ids),
            "responses_count": len(state["llm_responses"]),
            "batch_success": batch_success,
            "detail": batch_detail,
        }
        add_llm_batch_event(state, event)
        yield event

    state["llm_retries"]["__batch__"] = failed_batches
    if not state["llm_responses"]:
        state["warnings"].append("LLM batch call failed or timed out; using template fallback. Detail: no usable responses")
    add_flow_event(
        state,
        node=node,
        event="end",
        extra={"batches": len(batches), "failed_batches": failed_batches, "llm_responses": len(state["llm_responses"])},
    )


def explain_llm_node(state: AgentState) -> AgentState:
    """Call local Ollama in fixed-size batches and update LLM responses."""
    for _event in stream_explain_llm_batches(state):
        pass

    ollama_cfg = state["config"].get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434"))
    model = str(ollama_cfg.get("model", "llama3.2:1b"))
    api_key = str(ollama_cfg.get("api_key", "")).strip()
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    reasoning_enabled = bool(state["config"].get("reasoning_enabled", False)) and model not in SMALL_MODELS
    if reasoning_enabled and state.get("llm_prompts"):
        try:
            sample_ids = list(state["llm_prompts"].keys())[:1]
            raw_inputs = [json.loads(state["llm_prompts"][sku_id]) for sku_id in sample_ids]
            compact_inputs = [_compact_input(item) for item in raw_inputs]
            system_prompt = Path("prompts/system_prompt.txt").read_text(encoding="utf-8")
            reasoning_text = _stream_reasoning_text(
                base_url=base_url,
                model=model,
                headers=headers,
                system_prompt=system_prompt,
                batch_ids=sample_ids,
                compact_inputs=compact_inputs,
            )
            if reasoning_text:
                state["llm_reasoning"]["sample_batch"] = reasoning_text
                for sku_id in state["llm_prompts"].keys():
                    state["llm_reasoning_by_sku"][sku_id] = reasoning_text
        except Exception as exc:
            state["warnings"].append(f"raw_cot_capture_failed:{exc}")
    return state
