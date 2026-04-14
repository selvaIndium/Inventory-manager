"""LangGraph node to call Ollama once with batched SKU prompts."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from agent.state import AgentState


def explain_llm_node(state: AgentState) -> AgentState:
    """Call local Ollama in a single batch; capture per-SKU explanations."""
    state["current_node"] = "explain_llm"

    if not state["llm_prompts"]:
        state["warnings"].append("No LLM prompts available; template fallback will be used.")
        return state

    ollama_cfg = state["config"].get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434"))
    model = str(ollama_cfg.get("model", "llama3.2:1b"))
    timeout_seconds = float(ollama_cfg.get("timeout_ms", 4000)) / 1000
    temperature = float(ollama_cfg.get("temperature", 0.1))

    system_prompt = Path("prompts/system_prompt.txt").read_text(encoding="utf-8")
    expected_sku_ids = list(state["llm_prompts"].keys())
    sku_inputs = [json.loads(item) for item in state["llm_prompts"].values()]
    compact_inputs = [
        {
            "sku_id": item.get("sku_id"),
            "status": item.get("status"),
            "days_of_stock": round(float(item.get("days_of_stock", 0.0)), 1),
            "reorder_qty": round(float(item.get("reorder_qty", 0.0)), 1),
            "reorder_urgency_days": round(float(item.get("reorder_urgency_days", 0.0)), 1),
            "velocity_trend": item.get("velocity_trend"),
            "seasonal_factor": item.get("seasonal_factor", 1.0),
            "risk_tags": item.get("risk_tags", []),
        }
        for item in sku_inputs
    ]

    user_payload = {
        "task": "Generate inventory recommendations for all SKUs in one response.",
        "instruction": (
            "Return JSON only. Include exactly one entry for every sku_id in expected_sku_ids. "
            "Use concise explanations with advisory language. "
            "Each explanation must be one sentence (max 22 words). "
            "Each action must be <= 8 words."
        ),
        "expected_sku_ids": expected_sku_ids,
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

    num_predict = int(ollama_cfg.get("num_predict", 1800))

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

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(f"{base_url}/api/chat", json=request_payload)
            response.raise_for_status()
            raw = response.json()

        content = raw.get("message", {}).get("content", "{}")
        payload = json.loads(content)
        if isinstance(payload, str):
            payload = json.loads(payload)

        if isinstance(payload, dict) and "sku_recommendations" in payload:
            batched = payload.get("sku_recommendations", {})
        elif isinstance(payload, dict):
            batched = payload
        else:
            batched = {}

        if not isinstance(batched, dict):
            raise ValueError("Invalid batched recommendation payload shape.")
        if not batched:
            raise ValueError("LLM returned empty recommendation map.")

        filled_count = 0
        for sku_id in expected_sku_ids:
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

        if filled_count == 0:
            raise ValueError("LLM response parsed but no usable SKU recommendations found.")
        if filled_count < len(expected_sku_ids):
            state["warnings"].append(
                "LLM returned partial recommendations; template fallback will complete missing SKUs."
            )

        state["llm_retries"]["__batch__"] = 0
    except Exception as exc:
        state["warnings"].append(f"LLM batch call failed or timed out; using template fallback. Detail: {exc}")
        state["llm_retries"]["__batch__"] = state["llm_retries"].get("__batch__", 0) + 1

    return state
