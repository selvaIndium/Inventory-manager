"""LangGraph node to call Ollama in fixed-size SKU batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import httpx

from agent.state import AgentState


BATCH_SIZE = 5


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
    num_predict: int,
    system_prompt: str,
    batch_ids: List[str],
    compact_inputs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Invoke Ollama for one SKU batch and parse JSON output."""
    user_payload = {
        "task": "Generate inventory recommendations for this SKU batch.",
        "instruction": (
            "Return JSON only. Include exactly one entry for every sku_id in expected_sku_ids. "
            "Use concise advisory language. Each explanation must be one sentence (max 22 words). "
            "Each action must be <= 8 words."
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
        "format": response_schema,
        "options": {"temperature": temperature, "num_predict": num_predict},
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
    state["current_node"] = "explain_llm"

    if not state["llm_prompts"]:
        state["warnings"].append("No LLM prompts available; template fallback will be used.")
        return

    ollama_cfg = state["config"].get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434"))
    model = str(ollama_cfg.get("model", "llama3.2:1b"))
    timeout_ms = int(ollama_cfg.get("timeout_ms", 4000))
    timeout_seconds: float | None = None if timeout_ms <= 0 else (float(timeout_ms) / 1000)
    temperature = float(ollama_cfg.get("temperature", 0.1))
    num_predict = int(ollama_cfg.get("num_predict", 1800))
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
                num_predict=num_predict,
                system_prompt=system_prompt,
                batch_ids=batch_ids,
                compact_inputs=compact_inputs,
            )

            filled_count = 0
            for sku_id in batch_ids:
                rec = batched.get(sku_id)
                if not isinstance(rec, dict):
                    continue

                explanation = str(rec.get("explanation", "")).strip()
                action = str(rec.get("action", "")).strip()
                confidence = str(rec.get("confidence", "medium")).strip().lower() or "medium"
                if not explanation:
                    continue

                state["llm_responses"][sku_id] = json.dumps(
                    {
                        "explanation": explanation,
                        "action": action or "Consider reviewing this SKU with your planning team.",
                        "confidence": confidence if confidence in {"high", "medium", "low"} else "medium",
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
            batch_detail = f"{filled_count}/{len(batch_ids)} SKU(s) produced LLM explanations."
        except Exception as exc:
            failed_batches += 1
            batch_detail = str(exc)
            state["warnings"].append(
                f"LLM batch {batch_index} failed or timed out; template fallback will be used. Detail: {exc}"
            )

        processed_skus += len(batch_ids)
        yield {
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

    state["llm_retries"]["__batch__"] = failed_batches
    if not state["llm_responses"]:
        state["warnings"].append("LLM batch call failed or timed out; using template fallback. Detail: no usable responses")


def explain_llm_node(state: AgentState) -> AgentState:
    """Call local Ollama in fixed-size batches and update LLM responses."""
    for _event in stream_explain_llm_batches(state):
        pass
    return state
