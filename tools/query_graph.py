"""Hybrid knowledge graph query logic with cache and fallback chain."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

from knowledge.cache_layer import CACHE
from knowledge.neo4j_client import Neo4jClient
from knowledge.networkx_graph import build_fallback_graph, query_networkx

_NETWORKX_GRAPH = None


def query_graph(
    sku_id: str,
    category: str,
    query_type: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Query context using cache, then Neo4j, then NetworkX fallback."""
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

    context = _query_neo4j_if_available(sku_id, current_month, config)
    if context is None:
        graph = _get_networkx_graph(config)
        context = query_networkx(graph, sku_id, current_month)

    CACHE.set(cache_key, context, ttl_seconds=cache_ttl)
    return context


def _query_neo4j_if_available(
    sku_id: str,
    current_month: int,
    config: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Attempt Neo4j query when enabled and reachable."""
    neo4j_cfg = config.get("neo4j", {})
    enabled_from_cfg = bool(neo4j_cfg.get("enabled", True))
    enabled_from_env = os.environ.get("NEO4J_ENABLED", "true").lower() != "false"
    if not (enabled_from_cfg and enabled_from_env):
        return None

    uri = str(neo4j_cfg.get("uri", "bolt://localhost:7687"))
    user = str(neo4j_cfg.get("user", "neo4j"))
    password = str(neo4j_cfg.get("password", "inventory123"))
    timeout_ms = int(neo4j_cfg.get("timeout_ms", 800))

    client = None
    try:
        client = Neo4jClient(uri=uri, user=user, password=password, timeout_ms=timeout_ms)
        return client.query_context(sku_id=sku_id, current_month=current_month)
    except Exception:
        return None
    finally:
        if client is not None:
            client.close()


def _get_networkx_graph(config: Dict[str, Any]):
    """Lazily initialize and return fallback NetworkX graph."""
    global _NETWORKX_GRAPH
    if _NETWORKX_GRAPH is None:
        seed_path = str(config.get("kg_seed_path", "data/kg_seed.json"))
        _NETWORKX_GRAPH = build_fallback_graph(seed_path=seed_path)
    return _NETWORKX_GRAPH
