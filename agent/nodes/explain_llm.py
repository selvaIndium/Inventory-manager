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
    user_payload = {
        "task": "Generate recommendations for all SKUs in one response.",
        "required_format": {
            "sku_recommendations": {
                "<sku_id>": {
                    "explanation": "string",
                    "action": "string",
                    "confidence": "high | medium | low",
                }
            }
        },
        "sku_inputs": [json.loads(item) for item in state["llm_prompts"].values()],
    }

    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(f"{base_url}/api/chat", json=request_payload)
            response.raise_for_status()
            raw = response.json()

        content = raw.get("message", {}).get("content", "{}")
        payload = json.loads(content)
        batched = payload.get("sku_recommendations", {}) if isinstance(payload, dict) else {}
        if not isinstance(batched, dict):
            raise ValueError("Invalid batched recommendation payload shape.")

        for sku_id, rec in batched.items():
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

        state["llm_retries"]["__batch__"] = 0
    except Exception as exc:
        state["warnings"].append(f"LLM batch call failed or timed out; using template fallback. Detail: {exc}")
        state["llm_retries"]["__batch__"] = state["llm_retries"].get("__batch__", 0) + 1

    return state
