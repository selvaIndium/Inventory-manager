"""Debug helper to inspect raw Ollama batched response shape."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calc_metrics import calculate_metrics
from tools.load_data import load_inventory_data, load_threshold_config


def main() -> None:
    cfg = load_threshold_config("config/thresholds.yaml")
    rows = load_inventory_data("data/inventory_mock.csv")["records"]
    metrics = calculate_metrics(rows, cfg)

    sku_inputs = [
        {
            "sku_id": item["sku_id"],
            "name": item["name"],
            "category": item["category"],
            "status": item["status"],
            "days_of_stock": item["days_of_stock"],
            "reorder_qty": item["reorder_qty"],
            "reorder_urgency_days": item["reorder_urgency_days"],
            "velocity_trend": item["velocity_trend"],
            "seasonal_factor": 1.0,
            "risk_tags": [],
        }
        for item in metrics
    ]

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
        "sku_inputs": sku_inputs,
    }

    request_payload = {
        "model": cfg["ollama"]["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": cfg["ollama"]["temperature"]},
    }

    response = httpx.post(
        f"{cfg['ollama']['base_url']}/api/chat",
        json=request_payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    content = data.get("message", {}).get("content", "")
    print("status:", response.status_code)
    print("raw_content_preview:")
    print(content[:1500])

    try:
        parsed = json.loads(content)
        print("top_level_keys:", list(parsed.keys()))
        recs = parsed.get("sku_recommendations", {}) if isinstance(parsed, dict) else {}
        print("sku_recommendations_type:", type(recs).__name__)
        if isinstance(recs, dict):
            print("sku_recommendations_count:", len(recs))
            if recs:
                first_key = next(iter(recs))
                print("first_sku_key:", first_key)
                print("first_sku_payload:", recs[first_key])
    except Exception as exc:  # pragma: no cover - debug script
        print("parse_error:", exc)


if __name__ == "__main__":
    main()
