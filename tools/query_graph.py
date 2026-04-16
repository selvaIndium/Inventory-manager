"""Knowledge graph query logic using cache and NetworkX fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from knowledge.cache_layer import CACHE
from knowledge.networkx_graph import build_fallback_graph, query_networkx

_NETWORKX_GRAPH = None

def query_graph(
    sku_id: str,
    category: str,
    query_type: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Query context using cache, then NetworkX fallback."""
    _ = category
    _ = query_type

    cache_ttl = int(config.get("cache", {}).get("ttl_graph_seconds", 86400))
    now_utc = datetime.now(timezone.utc)
    current_month = now_utc.month
    date_key = now_utc.strftime("%Y%m%d")
    cache_key = f"graph:{sku_id}:all:{date_key}"

    hit, value, _ttl = CACHE.get(cache_key)
    if hit and isinstance(value, dict):
        cached = dict(value)
        cached["source"] = "cache"
        return cached

    graph = _get_networkx_graph(config)
    context = query_networkx(graph, sku_id, current_month)

    CACHE.set(cache_key, context, ttl_seconds=cache_ttl)
    return context


def _get_networkx_graph(config: Dict[str, Any]):
    """Lazily initialize and return fallback NetworkX graph."""
    global _NETWORKX_GRAPH
    if _NETWORKX_GRAPH is None:
        seed_path = str(config.get("kg_seed_path", "data/kg_seed.json"))
        _NETWORKX_GRAPH = build_fallback_graph(seed_path=seed_path)
    return _NETWORKX_GRAPH
