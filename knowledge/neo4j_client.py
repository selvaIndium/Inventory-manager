"""Neo4j client wrapper with timeout-safe query helpers."""

from __future__ import annotations

from typing import Any, Dict

from neo4j import GraphDatabase


class Neo4jClient:
    """Thin client around neo4j driver for context enrichment queries."""

    def __init__(self, uri: str, user: str, password: str, timeout_ms: int = 800) -> None:
        """Initialize Neo4j driver with configured timeout."""
        self.timeout_s = max(0.1, timeout_ms / 1000)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close driver connections."""
        self._driver.close()

    def query_context(self, sku_id: str, current_month: int) -> Dict[str, Any]:
        """Query seasonal factor and category target for a SKU."""
        query = """
        MATCH (s:SKU {sku_id: $sku_id})
        OPTIONAL MATCH (s)-[a:AFFECTED_BY]->(sn:Season)
        OPTIONAL MATCH (s)-[:BELONGS_TO]->(c:Category)
        RETURN
          c.avg_dos_target AS category_avg_dos,
          max(CASE
                WHEN sn IS NOT NULL AND (
                    (sn.start_month <= sn.end_month AND $current_month >= sn.start_month AND $current_month <= sn.end_month)
                    OR
                    (sn.start_month > sn.end_month AND ($current_month >= sn.start_month OR $current_month <= sn.end_month))
                )
                THEN sn.demand_multiplier
                ELSE 1.0
              END) AS seasonal_factor
        """
        with self._driver.session() as session:
            record = session.run(
                query,
                sku_id=sku_id,
                current_month=current_month,
                timeout=self.timeout_s,
            ).single()

        if record is None:
            return {
                "sku_id": sku_id,
                "seasonal_factor": 1.0,
                "category_avg_dos": 30.0,
                "risk_tags": [],
                "source": "default",
            }

        seasonal_factor = float(record.get("seasonal_factor") or 1.0)
        category_avg_dos = float(record.get("category_avg_dos") or 30.0)
        risk_tags = ["seasonal_peak"] if seasonal_factor > 1.1 else []

        return {
            "sku_id": sku_id,
            "seasonal_factor": seasonal_factor,
            "category_avg_dos": category_avg_dos,
            "risk_tags": risk_tags,
            "source": "neo4j",
        }
