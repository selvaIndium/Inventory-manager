"""NetworkX-based fallback graph builder and query interface."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import networkx as nx


def build_fallback_graph(seed_path: str = "data/kg_seed.json") -> nx.DiGraph:
    """Build an in-memory fallback graph from seeded JSON data."""
    graph = nx.DiGraph()
    with Path(seed_path).open("r", encoding="utf-8") as handle:
        seed = json.load(handle)

    for category in seed.get("categories", []):
        graph.add_node(category["id"], **category, node_type="category")

    for season in seed.get("seasons", []):
        graph.add_node(season["id"], **season, node_type="season")

    for sku in seed.get("skus", []):
        graph.add_node(sku["sku_id"], **sku, node_type="sku")
        graph.add_edge(sku["sku_id"], sku["category_id"], rel="BELONGS_TO")
        for season_id in sku.get("affected_seasons", []):
            graph.add_edge(
                sku["sku_id"],
                season_id,
                rel="AFFECTED_BY",
                weight=float(sku.get("season_weight", 1.0)),
            )

    return graph


def query_networkx(graph: nx.DiGraph, sku_id: str, current_month: int | None = None) -> Dict[str, Any]:
    """Query fallback graph and return context in the PRD contract shape."""
    month = current_month or datetime.utcnow().month
    seasonal_factor = 1.0
    category_avg_dos = 30.0
    risk_tags: list[str] = []

    if sku_id not in graph:
        return {
            "sku_id": sku_id,
            "seasonal_factor": seasonal_factor,
            "category_avg_dos": category_avg_dos,
            "risk_tags": risk_tags,
            "source": "default",
        }

    for neighbor in graph.successors(sku_id):
        node = graph.nodes[neighbor]
        node_type = node.get("node_type")

        if node_type == "season":
            start_month = int(node.get("start_month", 1))
            end_month = int(node.get("end_month", 12))
            in_range = _month_in_range(month, start_month, end_month)
            if in_range:
                seasonal_factor = max(seasonal_factor, float(node.get("demand_multiplier", 1.0)))
                if "seasonal_peak" not in risk_tags:
                    risk_tags.append("seasonal_peak")

        if node_type == "category":
            category_avg_dos = float(node.get("avg_dos_target", 30.0))

    return {
        "sku_id": sku_id,
        "seasonal_factor": seasonal_factor,
        "category_avg_dos": category_avg_dos,
        "risk_tags": risk_tags,
        "source": "networkx",
    }


def _month_in_range(month: int, start_month: int, end_month: int) -> bool:
    """Determine whether month is inside an inclusive month range."""
    if start_month <= end_month:
        return start_month <= month <= end_month
    return month >= start_month or month <= end_month
