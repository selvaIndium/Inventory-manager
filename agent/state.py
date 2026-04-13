"""Agent state schema and dataclasses for LangGraph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict


@dataclass
class SKURecord:
    """Validated input SKU row."""

    sku_id: str
    name: str
    category: str
    current_stock: float
    avg_daily_sales: float
    lead_time_days: int
    safety_stock: float
    reorder_point: Optional[float] = None
    last_sale_date: Optional[str] = None
    supplier_id: Optional[str] = None


@dataclass
class SKUMetrics:
    """Calculated health metrics for one SKU."""

    sku_id: str
    days_of_stock: float
    reorder_qty: float
    reorder_urgency_days: float
    velocity_trend: Literal["rising", "stable", "falling", "unknown"]
    status: Literal["healthy", "watch", "critical", "overstock"]
    status_emoji: str


@dataclass
class SKUContext:
    """Contextual enrichment payload for one SKU."""

    sku_id: str
    seasonal_factor: float
    category_avg_dos: float
    risk_tags: List[str]
    context_source: Literal["neo4j", "networkx", "cache", "default"]


@dataclass
class SKURecommendation:
    """Final recommendation for one SKU."""

    sku_id: str
    name: str
    status: str
    status_emoji: str
    days_of_stock: float
    reorder_qty: float
    reorder_urgency_days: float
    recommended_action: str
    plain_english_explanation: str
    risk_tags: List[str]
    confidence: Literal["high", "medium", "low"]
    data_quality_flag: Optional[str]


class AgentState(TypedDict):
    """LangGraph runtime state contract."""

    run_id: str
    started_at: str
    config: Dict[str, Any]

    raw_records: List[Dict[str, Any]]
    sku_records: List[SKURecord]
    sku_metrics: List[SKUMetrics]
    sku_contexts: List[SKUContext]

    rule_results: Dict[str, List[str]]
    recommendations: List[SKURecommendation]

    llm_prompts: Dict[str, str]
    llm_responses: Dict[str, str]
    llm_retries: Dict[str, int]

    current_node: str
    errors: List[Dict[str, str]]
    warnings: List[str]
    partial_data: bool
    graph_source: str
    output_valid: bool

    final_output: Optional[Dict[str, Any]]
