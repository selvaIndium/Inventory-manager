# Product Requirements Document
# Inventory Optimization AI Agent (Decision Support)

---

## 1. Document Header & Versioning

| Field | Value |
|---|---|
| **Document Title** | Inventory Optimization AI Agent — PRD |
| **Version** | 1.0.0 |
| **Status** | Draft — Ready for Engineering Review |
| **Author(s)** | Senior PM / AI Systems Architect |
| **Assessment Reference** | RA8 |
| **Created** | 2025-07-01 |
| **Last Updated** | 2025-07-01 |
| **Target Release** | MVP (Local / Offline) |
| **Review Cycle** | Per sprint, major version on architecture change |

### Revision History

| Version | Date | Author | Change Summary |
|---|---|---|---|
| 0.1 | 2025-06-15 | Architect | Initial outline |
| 0.9 | 2025-06-28 | PM | Added LangGraph + MCP architecture |
| 1.0 | 2025-07-01 | PM / Architect | Final PRD — production-ready draft |

---

## 2. Problem Statement & Product Vision

### 2.1 Problem Statement

Inventory teams at small-to-mid-sized retail and supply chain operations continue to rely on manual, spreadsheet-driven processes to make reorder decisions. This reactive approach creates two persistent, expensive failure modes:

- **Stockouts**: High-velocity SKUs run dry before a reorder is placed, leading to lost sales, customer dissatisfaction, and emergency procurement costs.
- **Overstocking**: Slow-moving or seasonal SKUs accumulate excess inventory that ties up capital, consumes warehouse space, and risks write-offs due to spoilage or obsolescence.

The root cause is not a lack of data — it is a lack of timely, explainable, context-aware decision support that a non-technical planner can trust and act on.

### 2.2 Product Vision

> **"Give every inventory planner an always-available AI analyst that reads their stock data, spots risks before they become crises, and explains exactly what to do and why — in plain business language."**

The Inventory Optimization AI Agent is a **locally-executable, offline-first decision-support tool** that:

1. Ingests mock inventory data (CSV/JSON, 20–50 synthetic SKUs).
2. Calculates key inventory health metrics using configurable rule thresholds.
3. Reasons over those metrics using a stateful AI agent (LangGraph + local LLM via Ollama).
4. Enriches decisions with seasonal/category context from a hybrid knowledge graph (Neo4j or NetworkX fallback).
5. Produces structured, explainable recommendations with a mandatory human-review disclaimer.

The agent is a **co-pilot, not an autopilot**. It never places orders or modifies systems. Every recommendation is framed as advisory input for a human decision-maker.

---

## 3. Target Users & Personas

### Persona 1 — Maya, Inventory Planner (Primary)

| Attribute | Detail |
|---|---|
| **Role** | Inventory / Replenishment Planner |
| **Tech Comfort** | Low-to-medium; comfortable with Excel, basic dashboards |
| **Pain Points** | Spends 3–4 hours/week manually reviewing stock levels; misses reorder windows; no visibility into seasonal patterns |
| **Goal** | Get a prioritized, plain-English list of what to reorder, when, and how much — without building formulas |
| **Success Metric** | Reduces stockout incidents by ≥30%; reclaims 2+ hours/week |

### Persona 2 — Raj, Retail Operations Manager (Secondary)

| Attribute | Detail |
|---|---|
| **Role** | Ops Manager overseeing 3–5 planners |
| **Tech Comfort** | Medium |
| **Pain Points** | No consolidated view of inventory risk across SKUs; reactive firefighting |
| **Goal** | Weekly health summary with risk-flagged SKUs and recommended actions per category |
| **Success Metric** | Reduction in emergency reorders; improved cash flow from reduced overstocking |

### Persona 3 — Priya, AI/IT Learner (Tertiary)

| Attribute | Detail |
|---|---|
| **Role** | Junior ML Engineer or CS student studying agentic AI systems |
| **Tech Comfort** | High |
| **Pain Points** | Limited access to real-world agentic AI codebases with MCP + LangGraph integration |
| **Goal** | Understand how LangGraph, MCP tools, LLMs, and knowledge graphs compose into a production-pattern system |
| **Success Metric** | Can extend a new MCP tool or LangGraph node within 30 minutes of reading the codebase |

---

## 4. Scope

### 4.1 In Scope (MVP)

| ID | Feature |
|---|---|
| S-01 | Ingest mock inventory data from CSV or JSON (20–50 synthetic SKUs) |
| S-02 | Calculate Days of Stock, reorder quantity, and reorder timing per SKU |
| S-03 | Apply configurable status thresholds (🟢 Healthy / 🟡 Watch / 🔴 Critical / ⚠️ Overstock) |
| S-04 | LangGraph orchestration with stateful, cyclic agent workflow |
| S-05 | MCP tool layer (FastMCP) wrapping data loading, metric calculation, rule fetching, graph queries |
| S-06 | Hybrid knowledge graph: Neo4j (EC2, primary) + NetworkX (in-memory fallback) |
| S-07 | Latency guard via diskcache/SQLite for frequent lookups |
| S-08 | Local Ollama LLM (qwen2.5:7b or llama3.1:8b) for explanation generation |
| S-09 | Structured JSON output with plain-English narrative per SKU |
| S-10 | CLI-first execution; optional Streamlit UI |
| S-11 | Graceful handling of missing/malformed data fields |
| S-12 | Mandatory disclaimer on all outputs (advisory only, human review required) |
| S-13 | Basic demand trend detection (e.g., 7-day rolling average vs. 30-day baseline) |
| S-14 | Offline-first execution — no internet dependency for core logic |

### 4.2 Out of Scope (MVP)

| ID | Exclusion | Reason |
|---|---|---|
| OS-01 | Real-time POS integration | Requires live data feeds; beyond RA8 boundaries |
| OS-02 | ML/statistical forecasting models | Adds complexity; mock data insufficient for training |
| OS-03 | ERP/WMS integration (SAP, Oracle, etc.) | Out of scope per RA8; compliance risk |
| OS-04 | Supplier contract management | Domain outside RA8; requires external APIs |
| OS-05 | Real commercial inventory data | Privacy/compliance risk; mock data only |
| OS-06 | Automated order placement | Violates advisory-only mandate; human-in-the-loop required |
| OS-07 | Multi-tenant SaaS deployment | Architectural scope beyond MVP |
| OS-08 | Mobile application | UI scope limited to CLI + optional Streamlit |

---

## 5. System Architecture & Data Flow

### 5.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                             │
│                                                                         │
│   ┌──────────────┐              ┌──────────────────────────────────┐   │
│   │   CLI Runner  │              │   Streamlit UI (Optional)        │   │
│   │  (main.py)   │              │   (app.py)                       │   │
│   └──────┬───────┘              └───────────────┬──────────────────┘   │
└──────────┼───────────────────────────────────────┼───────────────────────┘
           │                                       │
           ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER (LangGraph)                     │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    LangGraph State Machine                       │  │
│   │                                                                  │  │
│   │  [load_data] → [calculate_metrics] → [apply_rules]              │  │
│   │       ↑               ↓                    ↓                    │  │
│   │  [retry/fix]   [enrich_context]    [generate_recs]              │  │
│   │                       ↓                    ↓                    │  │
│   │               [query_knowledge]    [explain_llm]                │  │
│   │                       ↓                    ↓                    │  │
│   │               [cache_lookup]       [format_output]              │  │
│   │                                         ↓                      │  │
│   │                                  [validate_output]             │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
           ┌───────────────────┼──────────────────────┐
           ▼                   ▼                      ▼
┌──────────────────┐ ┌─────────────────┐  ┌──────────────────────────────┐
│   MCP TOOL LAYER │ │   LLM LAYER     │  │   KNOWLEDGE LAYER            │
│   (FastMCP)      │ │   (Ollama)      │  │                              │
│                  │ │                 │  │  ┌────────────────────────┐  │
│  ┌────────────┐  │ │  qwen2.5:7b or  │  │  │  Neo4j (EC2 Docker)    │  │
│  │load_csv    │  │ │  llama3.1:8b    │  │  │  Primary KG            │  │
│  │load_json   │  │ │  (quantized)    │  │  │  (Seasonal / Category) │  │
│  │calc_metrics│  │ │                 │  │  └────────────┬───────────┘  │
│  │fetch_rules │  │ │  localhost:11434│  │               │ unreachable? │
│  │query_graph │  │ │                 │  │               ▼              │
│  │cache_get   │  │ └─────────────────┘  │  ┌────────────────────────┐  │
│  │cache_set   │  │                      │  │  NetworkX (In-Memory)  │  │
│  └────────────┘  │                      │  │  Offline Fallback KG   │  │
└──────────────────┘                      │  └────────────────────────┘  │
                                          │               │              │
                                          │  ┌────────────▼───────────┐  │
                                          │  │  diskcache / SQLite    │  │
                                          │  │  Latency Guard Cache   │  │
                                          │  └────────────────────────┘  │
                                          └──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                     │
│                                                                         │
│   ┌──────────────────┐         ┌──────────────────────────────────┐   │
│   │  Mock CSV / JSON  │         │  Threshold Config (YAML/JSON)    │   │
│   │  (20–50 SKUs)    │         │  (configurable per environment)   │   │
│   └──────────────────┘         └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow (End-to-End)

```
1. User invokes CLI or Streamlit UI
       │
       ▼
2. LangGraph initializes AgentState with run_id, config, empty results
       │
       ▼
3. [load_data] node calls MCP tool load_csv / load_json
   → Returns validated list of SKURecord objects
   → On error: retry once, then set partial_data flag
       │
       ▼
4. [calculate_metrics] node calls MCP tool calc_metrics per SKU
   → Computes: days_of_stock, reorder_qty, reorder_date, velocity
       │
       ▼
5. [enrich_context] node calls MCP tool query_graph
   → Checks diskcache first (< 50ms target)
   → On cache miss: queries Neo4j (< 800ms timeout)
   → On Neo4j timeout/fail: queries NetworkX fallback
   → Result: seasonal_factor, category_norms, risk_tags
       │
       ▼
6. [apply_rules] node calls MCP tool fetch_rules
   → Applies threshold logic → assigns status (🟢🟡🔴⚠️)
   → Produces rule_match list per SKU
       │
       ▼
7. [generate_recs] node assembles per-SKU recommendation payload
       │
       ▼
8. [explain_llm] node sends structured prompt to Ollama
   → System prompt enforces: neutral tone, advisory language, disclaimer
   → Returns: plain_english_explanation per SKU
   → Timeout: 4s max; fallback to template string if exceeded
       │
       ▼
9. [format_output] node assembles final JSON response
       │
       ▼
10. [validate_output] node checks schema compliance
    → On validation fail: re-prompt once (loop back to step 8)
    → On second fail: emit structured error with partial results
       │
       ▼
11. Output rendered to CLI stdout (JSON + human-readable summary)
    or Streamlit dashboard cards
```

---

## 6. LangGraph State Machine & Node Definitions

### 6.1 AgentState Schema (TypedDict)

```python
from typing import TypedDict, List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime


# ── Sub-types ────────────────────────────────────────────────────────────────

@dataclass
class SKURecord:
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
    sku_id: str
    days_of_stock: float                  # current_stock / avg_daily_sales
    reorder_qty: float                    # (avg_daily_sales × lead_time) + safety_stock - current_stock
    reorder_urgency_days: float           # days until stock hits reorder point
    velocity_trend: Literal["rising", "stable", "falling", "unknown"]
    status: Literal["healthy", "watch", "critical", "overstock"]
    status_emoji: str                     # 🟢 🟡 🔴 ⚠️


@dataclass
class SKUContext:
    sku_id: str
    seasonal_factor: float                # multiplier from KG (1.0 = neutral)
    category_avg_dos: float               # avg days-of-stock for category
    risk_tags: List[str]                  # e.g. ["peak_season", "long_lead_time"]
    context_source: Literal["neo4j", "networkx", "cache", "default"]


@dataclass
class SKURecommendation:
    sku_id: str
    name: str
    status: str
    status_emoji: str
    days_of_stock: float
    reorder_qty: float
    reorder_urgency_days: float
    recommended_action: str               # e.g. "Place reorder within 3 days"
    plain_english_explanation: str
    risk_tags: List[str]
    confidence: Literal["high", "medium", "low"]
    data_quality_flag: Optional[str]      # e.g. "missing_lead_time_used_default"


# ── Primary State Schema ─────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Run metadata
    run_id: str
    started_at: str                               # ISO 8601
    config: Dict[str, Any]                        # thresholds, paths, flags

    # Data pipeline
    raw_records: List[Dict[str, Any]]             # unparsed rows from CSV/JSON
    sku_records: List[SKURecord]                  # validated SKU objects
    sku_metrics: List[SKUMetrics]                 # calculated metrics
    sku_contexts: List[SKUContext]                # KG-enriched context

    # Rule & recommendation state
    rule_results: Dict[str, List[str]]            # sku_id → matched rule IDs
    recommendations: List[SKURecommendation]      # final recs

    # LLM interaction
    llm_prompts: Dict[str, str]                   # sku_id → prompt sent
    llm_responses: Dict[str, str]                 # sku_id → raw LLM response
    llm_retries: Dict[str, int]                   # sku_id → retry count

    # Control flow
    current_node: str
    errors: List[Dict[str, str]]                  # {node, sku_id, message}
    warnings: List[str]
    partial_data: bool                            # True if some SKUs had missing fields
    graph_source: str                             # "neo4j" | "networkx" | "cache"
    output_valid: bool

    # Final output
    final_output: Optional[Dict[str, Any]]        # serialized JSON response
```

### 6.2 Node Definitions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NODE GRAPH (LangGraph)                                                     │
│                                                                             │
│   START                                                                     │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────┐    error     ┌───────────────┐                            │
│  │  load_data  │ ──────────▶  │ handle_error  │ ──▶ END (partial/fail)     │
│  └──────┬──────┘              └───────────────┘                            │
│         │ success                                                           │
│         ▼                                                                   │
│  ┌──────────────────┐                                                       │
│  │ calculate_metrics│                                                       │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  enrich_context  │ ◀──────────────────────────────────┐                 │
│  └────────┬─────────┘                                    │                 │
│           │                                    cache miss + KG fail        │
│           ▼                                              │                 │
│  ┌──────────────────┐                         ┌──────────────────┐        │
│  │   apply_rules    │                         │  fallback_graph  │        │
│  └────────┬─────────┘                         └──────────────────┘        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  generate_recs   │                                                       │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐   timeout / fail   ┌──────────────────────┐         │
│  │   explain_llm    │ ─────────────────▶ │ template_explanation │         │
│  └────────┬─────────┘                    └──────────┬───────────┘         │
│           │                                         │                      │
│           └─────────────────┬───────────────────────┘                      │
│                             ▼                                               │
│                   ┌──────────────────┐                                      │
│                   │  format_output   │                                      │
│                   └────────┬─────────┘                                      │
│                            │                                                │
│                            ▼                                                │
│                   ┌──────────────────┐  invalid (retry ≤1)                 │
│                   │ validate_output  │ ────────────────────▶ [explain_llm] │
│                   └────────┬─────────┘                                      │
│                            │ valid                                           │
│                            ▼                                                │
│                          END ✓                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Node Specifications

| Node | Responsibility | MCP Tools Called | Max Duration | On Failure |
|---|---|---|---|---|
| `load_data` | Parse CSV/JSON, validate schema, build SKURecord list | `load_csv`, `load_json` | 500ms | Retry once; set `partial_data=True` for bad rows |
| `calculate_metrics` | Compute days_of_stock, reorder_qty, urgency, velocity | `calc_metrics` | 300ms | Use defaults; log warning |
| `enrich_context` | Fetch seasonal/category context from KG or cache | `query_graph`, `cache_get` | 900ms | Activate `fallback_graph` sub-node |
| `apply_rules` | Match thresholds → assign status, rule_results | `fetch_rules` | 200ms | Use default rule set |
| `generate_recs` | Assemble per-SKU recommendation payload | None (internal) | 100ms | Emit error record |
| `explain_llm` | Generate plain-English explanation via Ollama | None (direct HTTP) | 4000ms | Route to `template_explanation` |
| `template_explanation` | Return rule-based fallback explanation string | None (internal) | 50ms | Log warning; continue |
| `format_output` | Serialize final JSON output with metadata | `cache_set` | 200ms | Emit partial JSON |
| `validate_output` | Assert JSON schema compliance | None (internal) | 100ms | Re-prompt once; else emit structured error |
| `handle_error` | Log errors, emit failure response | None | 100ms | Always succeeds |

---

## 7. MCP Tool Layer & Contracts

### 7.1 Overview

All external capabilities are exposed as **FastMCP tools**. LangGraph nodes call these tools via the MCP client — they never directly import data loaders, DB clients, or formula modules. This enforces clean separation and testability.

```python
# Example tool registration (server.py)
from fastmcp import FastMCP

mcp = FastMCP("inventory-agent")

@mcp.tool()
def load_csv(file_path: str) -> dict: ...

@mcp.tool()
def calc_metrics(sku: dict, config: dict) -> dict: ...
```

### 7.2 Tool Contracts

---

#### Tool: `load_csv`

**Purpose**: Load and validate inventory data from a CSV file.

```json
{
  "tool": "load_csv",
  "input": {
    "file_path": "string (absolute or relative path to .csv file)"
  },
  "output": {
    "records": "[array of raw row dicts]",
    "row_count": "integer",
    "invalid_rows": "[array of {row_index, reason}]",
    "warnings": "[array of strings]"
  },
  "timeout_ms": 500,
  "fallback": "Return empty records list with error message; set partial_data=true",
  "errors": ["FileNotFoundError", "CSVParseError", "SchemaValidationError"]
}
```

---

#### Tool: `load_json`

**Purpose**: Load and validate inventory data from a JSON file.

```json
{
  "tool": "load_json",
  "input": {
    "file_path": "string"
  },
  "output": {
    "records": "[array of raw row dicts]",
    "row_count": "integer",
    "invalid_rows": "[array of {row_index, reason}]",
    "warnings": "[array of strings]"
  },
  "timeout_ms": 500,
  "fallback": "Same as load_csv fallback",
  "errors": ["FileNotFoundError", "JSONDecodeError", "SchemaValidationError"]
}
```

---

#### Tool: `calc_metrics`

**Purpose**: Compute all inventory health metrics for a single SKU.

```json
{
  "tool": "calc_metrics",
  "input": {
    "sku": {
      "sku_id": "string",
      "current_stock": "float",
      "avg_daily_sales": "float",
      "lead_time_days": "integer",
      "safety_stock": "float"
    },
    "config": {
      "healthy_dos_min": "float (default: 14)",
      "watch_dos_min": "float (default: 7)",
      "critical_dos_max": "float (default: 7)",
      "overstock_dos_min": "float (default: 60)"
    }
  },
  "output": {
    "sku_id": "string",
    "days_of_stock": "float",
    "reorder_qty": "float",
    "reorder_urgency_days": "float",
    "velocity_trend": "rising | stable | falling | unknown",
    "status": "healthy | watch | critical | overstock",
    "status_emoji": "🟢 | 🟡 | 🔴 | ⚠️",
    "formula_used": "string (human-readable formula trace)"
  },
  "timeout_ms": 100,
  "fallback": "Use default config values for missing fields; flag in data_quality_flag",
  "formula_note": "reorder_qty = (avg_daily_sales × lead_time_days) + safety_stock - current_stock"
}
```

**Core Formulas**:
```
days_of_stock       = current_stock ÷ avg_daily_sales
reorder_qty         = (avg_daily_sales × lead_time_days) + safety_stock − current_stock
reorder_urgency     = days_of_stock − lead_time_days
```

**Status Threshold Logic**:
```
IF days_of_stock > overstock_dos_min          → ⚠️  overstock
ELIF days_of_stock >= healthy_dos_min         → 🟢  healthy
ELIF days_of_stock >= watch_dos_min           → 🟡  watch
ELSE                                          → 🔴  critical
```

---

#### Tool: `fetch_rules`

**Purpose**: Return the active rule set for threshold evaluation.

```json
{
  "tool": "fetch_rules",
  "input": {
    "config_path": "string (path to thresholds YAML/JSON)",
    "category": "string (optional — for category-specific overrides)"
  },
  "output": {
    "rules": [
      {
        "rule_id": "string",
        "condition": "string (human-readable)",
        "action": "string",
        "priority": "integer"
      }
    ],
    "source": "file | default"
  },
  "timeout_ms": 200,
  "fallback": "Return hardcoded default rule set"
}
```

---

#### Tool: `query_graph`

**Purpose**: Query the knowledge graph for seasonal/category context.

```json
{
  "tool": "query_graph",
  "input": {
    "sku_id": "string",
    "category": "string",
    "query_type": "seasonal_factor | category_norms | risk_tags | all"
  },
  "output": {
    "sku_id": "string",
    "seasonal_factor": "float (1.0 = neutral)",
    "category_avg_dos": "float",
    "risk_tags": "[array of strings]",
    "source": "neo4j | networkx | cache | default"
  },
  "timeout_ms": 800,
  "fallback": "If Neo4j unreachable or >800ms: try NetworkX; if unavailable: return defaults (seasonal_factor=1.0, risk_tags=[])",
  "cache_key_pattern": "graph:{sku_id}:{query_type}:{date_yyyymmdd}"
}
```

---

#### Tool: `cache_get`

**Purpose**: Retrieve a cached value from diskcache/SQLite.

```json
{
  "tool": "cache_get",
  "input": {
    "key": "string"
  },
  "output": {
    "hit": "boolean",
    "value": "any | null",
    "ttl_remaining_seconds": "integer | null"
  },
  "timeout_ms": 50,
  "fallback": "Return hit=false; proceed to live query"
}
```

---

#### Tool: `cache_set`

**Purpose**: Store a value in the diskcache/SQLite cache.

```json
{
  "tool": "cache_set",
  "input": {
    "key": "string",
    "value": "any",
    "ttl_seconds": "integer (default: 3600)"
  },
  "output": {
    "success": "boolean"
  },
  "timeout_ms": 50,
  "fallback": "Log warning; continue without caching"
}
```

---

## 8. Knowledge Graph Strategy

### 8.1 Neo4j Schema (EC2 Community Edition via Docker)

#### Node Labels

```cypher
// Category node — groups SKUs by type
(:Category {
  id: STRING,           // e.g. "electronics"
  name: STRING,
  avg_dos_target: FLOAT,
  reorder_sensitivity: STRING  // "low" | "medium" | "high"
})

// SKU node — one per product
(:SKU {
  sku_id: STRING,
  name: STRING,
  category_id: STRING,
  supplier_lead_time_days: INTEGER,
  abc_class: STRING     // "A" | "B" | "C" (velocity classification)
})

// Season node — represents a named seasonal period
(:Season {
  id: STRING,           // e.g. "diwali_2025"
  name: STRING,
  start_month: INTEGER,
  end_month: INTEGER,
  demand_multiplier: FLOAT
})

// RuleTemplate node — reusable rule definitions
(:RuleTemplate {
  rule_id: STRING,
  condition_type: STRING,
  threshold_value: FLOAT,
  action_template: STRING,
  priority: INTEGER
})
```

#### Relationships

```cypher
(:SKU)-[:BELONGS_TO]->(:Category)
(:SKU)-[:AFFECTED_BY {weight: FLOAT}]->(:Season)
(:Category)-[:HAS_RULE]->(:RuleTemplate)
(:Season)-[:OVERLAPS_WITH]->(:Season)
```

#### Example Cypher Queries

```cypher
// Get seasonal factor for a SKU in current month
MATCH (s:SKU {sku_id: $sku_id})-[r:AFFECTED_BY]->(sn:Season)
WHERE sn.start_month <= $current_month <= sn.end_month
RETURN sn.demand_multiplier AS seasonal_factor, r.weight AS relevance

// Get category norms
MATCH (s:SKU {sku_id: $sku_id})-[:BELONGS_TO]->(c:Category)
RETURN c.avg_dos_target AS category_avg_dos, c.reorder_sensitivity

// Get applicable rule templates
MATCH (s:SKU {sku_id: $sku_id})-[:BELONGS_TO]->(c:Category)-[:HAS_RULE]->(r:RuleTemplate)
RETURN r ORDER BY r.priority ASC
```

#### Docker Setup

```bash
docker run \
  --name neo4j-inventory \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/inventory123 \
  -v $HOME/neo4j/data:/data \
  -d neo4j:5-community
```

### 8.2 NetworkX Fallback (In-Memory)

**Trigger Conditions**:
- Neo4j TCP connection refused (EC2 unreachable)
- Neo4j query response time > 800ms
- `NEO4J_ENABLED=false` environment flag

**Initialization** (at agent startup):

```python
import networkx as nx
import json

def build_fallback_graph(seed_path: str = "data/kg_seed.json") -> nx.DiGraph:
    """
    Loads a pre-seeded JSON file describing categories, seasons, and rules.
    Returns a directed NetworkX graph usable as offline KG substitute.
    """
    G = nx.DiGraph()
    with open(seed_path) as f:
        seed = json.load(f)

    for category in seed["categories"]:
        G.add_node(category["id"], **category, node_type="category")

    for season in seed["seasons"]:
        G.add_node(season["id"], **season, node_type="season")

    for sku in seed["skus"]:
        G.add_node(sku["sku_id"], **sku, node_type="sku")
        G.add_edge(sku["sku_id"], sku["category_id"], rel="BELONGS_TO")
        for s_id in sku.get("affected_seasons", []):
            G.add_edge(sku["sku_id"], s_id, rel="AFFECTED_BY",
                       weight=sku.get("season_weight", 1.0))
    return G
```

**Query Interface** (mirrors Neo4j output contract):

```python
def query_networkx(G: nx.DiGraph, sku_id: str, current_month: int) -> dict:
    neighbors = list(G.successors(sku_id))
    seasonal_factor = 1.0
    risk_tags = []
    category_avg_dos = 30.0  # global default

    for n in neighbors:
        node = G.nodes[n]
        if node.get("node_type") == "season":
            if node["start_month"] <= current_month <= node["end_month"]:
                seasonal_factor = node.get("demand_multiplier", 1.0)
                risk_tags.append("seasonal_peak")
        if node.get("node_type") == "category":
            category_avg_dos = node.get("avg_dos_target", 30.0)

    return {
        "seasonal_factor": seasonal_factor,
        "category_avg_dos": category_avg_dos,
        "risk_tags": risk_tags,
        "source": "networkx"
    }
```

### 8.3 Caching Rules (diskcache / SQLite)

| Cache Key Pattern | TTL | Invalidation |
|---|---|---|
| `graph:{sku_id}:seasonal_factor:{YYYYMM}` | 24 hours | Month rollover |
| `graph:{category_id}:norms` | 6 hours | Config change |
| `rules:{category_id}` | 1 hour | Config file mtime change |
| `metrics:{sku_id}:{date}` | 30 minutes | New data load |

**Latency Guarantees**:

```
cache_get:        < 50ms   (target: < 10ms)
networkx_query:   < 200ms
neo4j_query:      < 800ms  (hard timeout; fallback triggered at 800ms)
llm_call:         < 4000ms (hard timeout; template fallback at 4000ms)
total_e2e:        < 5000ms (RA8 SLA)
```

---

## 9. Functional Requirements

> All requirements are mapped directly to RA8 specification goals.

### FR-01 — Data Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-01.1 | System SHALL load inventory data from CSV or JSON files | P0 |
| FR-01.2 | System SHALL support 20–50 synthetic SKU records per run | P0 |
| FR-01.3 | System SHALL validate required fields: sku_id, current_stock, avg_daily_sales, lead_time_days, safety_stock | P0 |
| FR-01.4 | System SHALL handle missing optional fields gracefully using defaults, with a data_quality_flag in output | P0 |
| FR-01.5 | System SHALL report invalid rows without halting the full analysis run | P1 |

### FR-02 — Metric Calculation

| ID | Requirement | Priority |
|---|---|---|
| FR-02.1 | System SHALL calculate `days_of_stock = current_stock ÷ avg_daily_sales` for each SKU | P0 |
| FR-02.2 | System SHALL calculate `reorder_qty = (avg_daily_sales × lead_time_days) + safety_stock − current_stock` | P0 |
| FR-02.3 | System SHALL calculate `reorder_urgency_days = days_of_stock − lead_time_days` | P0 |
| FR-02.4 | System SHALL detect velocity trend (rising / stable / falling) using 7-day rolling vs. 30-day baseline where data available | P1 |

### FR-03 — Status Classification

| ID | Requirement | Priority |
|---|---|---|
| FR-03.1 | System SHALL assign one of four statuses per SKU: 🟢 Healthy, 🟡 Watch, 🔴 Critical, ⚠️ Overstock | P0 |
| FR-03.2 | System SHALL apply configurable thresholds loaded from a YAML/JSON config file | P0 |
| FR-03.3 | System SHALL support category-level threshold overrides | P1 |

### FR-04 — Context Enrichment

| ID | Requirement | Priority |
|---|---|---|
| FR-04.1 | System SHALL query the knowledge graph for seasonal demand factors per SKU | P1 |
| FR-04.2 | System SHALL fall back to NetworkX in-memory graph if Neo4j is unreachable or slow | P1 |
| FR-04.3 | System SHALL use diskcache for repeated KG queries within the same session | P1 |

### FR-05 — Recommendation Generation

| ID | Requirement | Priority |
|---|---|---|
| FR-05.1 | System SHALL generate a recommended_action string per SKU | P0 |
| FR-05.2 | System SHALL generate a plain_english_explanation per SKU using the local LLM | P0 |
| FR-05.3 | System SHALL fall back to a rule-based template explanation if LLM times out | P0 |
| FR-05.4 | System SHALL include proactive actions for 🟡 and 🔴 SKUs | P0 |
| FR-05.5 | System SHALL include context-aware notes when seasonal_factor > 1.1 | P1 |

### FR-06 — Output & Explainability

| ID | Requirement | Priority |
|---|---|---|
| FR-06.1 | System SHALL produce output as a valid JSON document conforming to the Output Schema (Section 11) | P0 |
| FR-06.2 | System SHALL include a mandatory advisory disclaimer on all outputs | P0 |
| FR-06.3 | System SHALL include a human-readable summary in CLI stdout | P0 |
| FR-06.4 | System SHALL report the source of KG context (neo4j / networkx / cache / default) | P1 |
| FR-06.5 | System SHALL include formula_used trace in per-SKU output | P1 |

### FR-07 — UI

| ID | Requirement | Priority |
|---|---|---|
| FR-07.1 | System SHALL execute fully via CLI as primary interface | P0 |
| FR-07.2 | System SHOULD provide an optional Streamlit dashboard UI | P2 |
| FR-07.3 | Streamlit UI SHALL display SKU cards with status emoji, metrics, and recommendation | P2 |

---

## 10. Non-Functional Requirements

### NFR-01 — Performance & Latency

| ID | Requirement | Budget |
|---|---|---|
| NFR-01.1 | End-to-end response time for 50 SKUs | ≤ 5,000ms |
| NFR-01.2 | Data load + metric calculation | ≤ 800ms |
| NFR-01.3 | KG context enrichment (cache hit) | ≤ 50ms |
| NFR-01.4 | KG context enrichment (Neo4j) | ≤ 800ms |
| NFR-01.5 | KG context enrichment (NetworkX) | ≤ 200ms |
| NFR-01.6 | LLM explanation generation (all SKUs, batched) | ≤ 4,000ms |
| NFR-01.7 | Output formatting + validation | ≤ 200ms |

### NFR-02 — Offline Guarantee

| ID | Requirement |
|---|---|
| NFR-02.1 | System SHALL operate fully without internet connectivity |
| NFR-02.2 | LLM model SHALL be pre-pulled locally via `ollama pull` before first run |
| NFR-02.3 | NetworkX fallback graph SHALL be pre-loaded from `data/kg_seed.json` at startup |
| NFR-02.4 | System SHALL NOT make any external HTTP calls except to `localhost:11434` (Ollama) and optional EC2 Neo4j |

### NFR-03 — Resource Limits

| Resource | Limit |
|---|---|
| RAM (agent process) | ≤ 1 GB |
| RAM (Ollama model) | ≤ 6 GB (quantized 7B/8B) |
| Disk (cache) | ≤ 500 MB |
| CPU (peak, single run) | ≤ 4 cores |
| SKU batch size (MVP) | 20–50 SKUs |

### NFR-04 — Reliability & Fallback

| ID | Requirement |
|---|---|
| NFR-04.1 | System SHALL complete a run (with partial output) even if Neo4j and Ollama both fail |
| NFR-04.2 | Each LangGraph node SHALL have explicit error handling with logged error records |
| NFR-04.3 | LLM timeout SHALL trigger template_explanation — never a blank explanation field |
| NFR-04.4 | System SHALL not crash on malformed CSV rows; invalid rows are skipped with warnings |

### NFR-05 — Maintainability

| ID | Requirement |
|---|---|
| NFR-05.1 | All thresholds SHALL be externalized to a YAML/JSON config file — no magic numbers in code |
| NFR-05.2 | Each MCP tool SHALL be independently unit-testable |
| NFR-05.3 | LangGraph nodes SHALL be pure functions accepting and returning AgentState |
| NFR-05.4 | Code complexity per module SHALL not exceed 200 lines (excluding tests) |

---

## 11. Output Schema & Prompt Guidelines

### 11.1 Final JSON Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InventoryAgentOutput",
  "type": "object",
  "required": ["run_id", "generated_at", "summary", "recommendations", "metadata", "disclaimer"],
  "properties": {
    "run_id": { "type": "string", "description": "UUID for this analysis run" },
    "generated_at": { "type": "string", "format": "date-time" },
    "summary": {
      "type": "object",
      "properties": {
        "total_skus_analyzed": { "type": "integer" },
        "critical_count": { "type": "integer" },
        "watch_count": { "type": "integer" },
        "healthy_count": { "type": "integer" },
        "overstock_count": { "type": "integer" },
        "skus_skipped": { "type": "integer" },
        "overall_health": { "type": "string", "enum": ["good", "fair", "poor"] },
        "top_priority_skus": {
          "type": "array",
          "items": { "type": "string" },
          "maxItems": 5
        }
      }
    },
    "recommendations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["sku_id", "name", "status", "status_emoji", "days_of_stock",
                     "reorder_qty", "reorder_urgency_days", "recommended_action",
                     "plain_english_explanation", "risk_tags", "confidence"],
        "properties": {
          "sku_id": { "type": "string" },
          "name": { "type": "string" },
          "category": { "type": "string" },
          "status": { "type": "string", "enum": ["healthy", "watch", "critical", "overstock"] },
          "status_emoji": { "type": "string", "enum": ["🟢", "🟡", "🔴", "⚠️"] },
          "days_of_stock": { "type": "number" },
          "reorder_qty": { "type": "number" },
          "reorder_urgency_days": { "type": "number" },
          "recommended_action": { "type": "string" },
          "plain_english_explanation": { "type": "string", "minLength": 30 },
          "risk_tags": { "type": "array", "items": { "type": "string" } },
          "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
          "seasonal_factor": { "type": "number" },
          "formula_used": { "type": "string" },
          "data_quality_flag": { "type": ["string", "null"] },
          "context_source": { "type": "string", "enum": ["neo4j", "networkx", "cache", "default"] },
          "velocity_trend": { "type": "string", "enum": ["rising", "stable", "falling", "unknown"] }
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "llm_model": { "type": "string" },
        "graph_source": { "type": "string" },
        "config_version": { "type": "string" },
        "execution_time_ms": { "type": "integer" },
        "partial_data": { "type": "boolean" },
        "errors": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "node": { "type": "string" },
              "sku_id": { "type": ["string", "null"] },
              "message": { "type": "string" }
            }
          }
        },
        "warnings": { "type": "array", "items": { "type": "string" } }
      }
    },
    "disclaimer": {
      "type": "string",
      "const": "⚠️ ADVISORY ONLY: This analysis is generated by an AI decision-support tool using synthetic mock data. All recommendations require human review and validation before any action is taken. This tool does not place orders, modify systems, or reflect real commercial inventory. Always consult your supply chain team before acting on these suggestions."
    }
  }
}
```

### 11.2 Example Output Record

```json
{
  "sku_id": "SKU-042",
  "name": "USB-C Charging Cable 2m",
  "category": "electronics",
  "status": "critical",
  "status_emoji": "🔴",
  "days_of_stock": 4.2,
  "reorder_qty": 320,
  "reorder_urgency_days": -2.8,
  "recommended_action": "Place reorder immediately. Stock will be exhausted before lead time completes.",
  "plain_english_explanation": "USB-C Charging Cable 2m is critically low with only 4.2 days of stock remaining, but your supplier needs 7 days to deliver. This means you are already 2.8 days behind the safe reorder window. Based on your average daily sales of 45 units, you should reorder approximately 320 units as soon as possible to avoid a stockout. This is a high-velocity product in the electronics category and current demand appears to be rising.",
  "risk_tags": ["stockout_imminent", "past_reorder_point", "rising_demand"],
  "confidence": "high",
  "seasonal_factor": 1.15,
  "formula_used": "reorder_qty = (45 × 7) + 50 − 165 = 200; adjusted for seasonal_factor 1.15 → 320",
  "data_quality_flag": null,
  "context_source": "neo4j",
  "velocity_trend": "rising"
}
```

### 11.3 System Prompt Template (Ollama)

```text
SYSTEM PROMPT — INVENTORY OPTIMIZATION AI AGENT
================================================

You are an Inventory Decision Support Analyst. Your role is to help inventory
planners understand their stock health and decide what actions to consider.

TONE & LANGUAGE RULES:
- Use plain, professional business English. Avoid jargon.
- Always use advisory language: "consider", "may want to", "based on available data".
- Never use commanding language: "you must", "you should immediately" (unless urgency is critical).
- Be concise: explanations should be 2–4 sentences. No filler phrases.
- Never claim certainty about future demand or supplier behavior.

OUTPUT FORMAT:
- Respond ONLY with a valid JSON object. No markdown, no backticks, no preamble.
- Use this exact structure:
  {
    "plain_english_explanation": "string (2-4 sentences)",
    "recommended_action": "string (1 sentence, action-oriented)",
    "confidence": "high | medium | low"
  }

MANDATORY DISCLAIMER:
- Every explanation must end with this exact phrase:
  "This recommendation is advisory only and should be reviewed by your supply chain team."

DATA PROVIDED TO YOU:
- sku_id, name, category, days_of_stock, reorder_qty, reorder_urgency_days,
  status, velocity_trend, seasonal_factor, risk_tags, formula_used

CONSTRAINTS:
- Do NOT invent data. If a field is null or unknown, say "data unavailable".
- Do NOT reference real suppliers, real prices, or real order systems.
- Do NOT place or suggest automated orders.
- This analysis is based on synthetic mock data for decision-support purposes only.

USER PROMPT TEMPLATE:
Analyze the following SKU and generate a plain-English explanation and recommended action.

SKU Data:
{sku_json}

Respond ONLY with the JSON object described above.
```

---

## 12. Ethics, Safety & Compliance

### 12.1 Advisory-Only Mandate

| Principle | Implementation |
|---|---|
| **No Automated Action** | System never calls any API that places orders or modifies inventory systems |
| **Human-in-the-Loop** | Mandatory disclaimer on every output; Streamlit UI includes prominent warning banner |
| **Decision Transparency** | Every recommendation includes `formula_used` trace and `context_source` |
| **Uncertainty Acknowledgment** | LLM prompt requires advisory language; `confidence` field reflects data completeness |

### 12.2 Data Privacy & Handling

| Principle | Implementation |
|---|---|
| **Synthetic Data Only** | System is validated exclusively on mock/synthetic SKU data; no real PII or commercial data |
| **No Data Egress** | LLM runs locally via Ollama; no data is sent to external APIs |
| **Cache Isolation** | diskcache stored locally; no shared storage in MVP |
| **Audit Trail** | All runs produce a `run_id`; logs include node execution trace |

### 12.3 Fairness & Bias

| Risk | Mitigation |
|---|---|
| **Category Bias in KG** | Seasonal factors in KG are seeded from neutral synthetic data; category norms are adjustable |
| **LLM Hallucination** | LLM prompt instructs model to say "data unavailable" rather than invent; output is validated against schema |
| **Threshold Bias** | All thresholds are externalized and configurable; no hardcoded assumptions about "normal" stock levels |

### 12.4 Compliance Boundaries

- This system does **not** constitute financial, procurement, or legal advice.
- The system does **not** integrate with or replace any ERP, WMS, or purchasing system.
- All recommendations are advisory. No output from this system creates binding obligations.
- The system is scoped to educational and analytical use cases under RA8.

---

## 13. Testing & Validation Plan

### 13.1 Unit Tests

| Test ID | Target | Test Case | Expected Result |
|---|---|---|---|
| UT-01 | `calc_metrics` tool | Valid SKU with all fields | Correct dos, reorder_qty, status |
| UT-02 | `calc_metrics` tool | avg_daily_sales = 0 | Returns dos=∞, status=overstock, quality_flag set |
| UT-03 | `calc_metrics` tool | Missing safety_stock | Uses default=0, quality_flag="missing_safety_stock_used_default" |
| UT-04 | `load_csv` tool | Malformed row (non-numeric stock) | Skips row, reports invalid_rows |
| UT-05 | `fetch_rules` tool | Missing config file | Returns hardcoded default rule set |
| UT-06 | `cache_get` tool | Non-existent key | Returns hit=false |
| UT-07 | Threshold logic | dos=5, watch_min=7 | Status=🔴 critical |
| UT-08 | Threshold logic | dos=90, overstock_min=60 | Status=⚠️ overstock |

### 13.2 Integration Tests

| Test ID | Scenario | Expected Result |
|---|---|---|
| IT-01 | Full run with 20 valid SKUs, Neo4j available | All 20 recs generated in <5s, source=neo4j |
| IT-02 | Full run with Neo4j unreachable | All recs generated, source=networkx, no crash |
| IT-03 | Full run with Ollama unavailable | All recs generated with template_explanation, no crash |
| IT-04 | Mix of valid and invalid rows (5 invalid of 25) | 20 valid recs + 5 warnings in metadata |
| IT-05 | Repeat run within cache TTL | KG queries served from cache, context_source=cache |

### 13.3 Fallback Tests

| Test ID | Fallback Triggered | Validation |
|---|---|---|
| FT-01 | Neo4j timeout (mock 900ms delay) | NetworkX query completes; source=networkx |
| FT-02 | LLM timeout (mock 5s delay) | template_explanation used; no blank explanation field |
| FT-03 | Both Neo4j and Ollama down | Run completes with defaults + template; partial_data=true where applicable |
| FT-04 | Empty input file | Graceful error; run_id returned; error logged in metadata.errors |

### 13.4 Performance Tests

| Test ID | Scenario | SLA | Pass Criteria |
|---|---|---|---|
| PT-01 | 50 SKUs, Neo4j cache cold | ≤5,000ms | p95 < 5000ms over 10 runs |
| PT-02 | 50 SKUs, Neo4j cache warm | ≤5,000ms | p95 < 3000ms over 10 runs |
| PT-03 | 50 SKUs, NetworkX fallback | ≤5,000ms | p95 < 4000ms over 10 runs |
| PT-04 | Memory usage during run | ≤1 GB | Measured via `memory_profiler` |

### 13.5 Ethics & Output Quality Tests

| Test ID | Concern | Test Method | Pass Criteria |
|---|---|---|---|
| ET-01 | Disclaimer present | Assert disclaimer field == expected constant | 100% of runs |
| ET-02 | No commanding language | Scan plain_english_explanation for banned phrases ("you must", "you will") | 0 violations |
| ET-03 | No hallucinated data | Compare LLM output fields against input SKU data | No invented sku_ids or figures |
| ET-04 | Schema compliance | JSON Schema validation on every output | 100% valid or retry triggered |
| ET-05 | Advisory confidence | confidence="low" when ≥2 fields have data_quality_flag | Assert in test fixture |

---

## 14. Local Development & Setup Guide

### 14.1 Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | Runtime |
| Ollama | Latest | Local LLM inference |
| Docker | ≥ 24 | Neo4j (optional) |
| Git | Any | Source control |

### 14.2 Project Structure

```
inventory-agent/
├── main.py                     # CLI entrypoint
├── app.py                      # Streamlit UI (optional)
├── config/
│   └── thresholds.yaml         # Configurable thresholds
├── data/
│   ├── inventory_mock.csv      # Synthetic SKU data (20–50 rows)
│   ├── inventory_mock.json     # JSON alternative
│   └── kg_seed.json            # NetworkX fallback graph seed
├── agent/
│   ├── state.py                # AgentState TypedDict + dataclasses
│   ├── graph.py                # LangGraph state machine definition
│   └── nodes/
│       ├── load_data.py
│       ├── calculate_metrics.py
│       ├── enrich_context.py
│       ├── apply_rules.py
│       ├── generate_recs.py
│       ├── explain_llm.py
│       ├── format_output.py
│       └── validate_output.py
├── tools/
│   ├── server.py               # FastMCP server registration
│   ├── load_csv.py
│   ├── load_json.py
│   ├── calc_metrics.py
│   ├── fetch_rules.py
│   ├── query_graph.py
│   └── cache.py
├── knowledge/
│   ├── neo4j_client.py         # Neo4j driver wrapper with timeout
│   ├── networkx_graph.py       # Fallback graph builder + query
│   └── cache_layer.py          # diskcache / SQLite wrapper
├── prompts/
│   └── system_prompt.txt       # LLM system prompt template
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fallback/
│   └── performance/
├── requirements.txt
├── docker-compose.yml          # Neo4j (optional)
└── README.md
```

### 14.3 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/inventory-agent.git
cd inventory-agent

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM model (requires Ollama installed)
ollama pull qwen2.5:7b
# OR
ollama pull llama3.1:8b

# 5. Verify Ollama is running
ollama serve &
curl http://localhost:11434/api/tags
```

### 14.4 `requirements.txt`

```text
# Core agent
langgraph>=0.2.0
langchain-core>=0.3.0

# MCP
fastmcp>=0.9.0

# Knowledge graph
neo4j>=5.0.0
networkx>=3.2

# LLM client
httpx>=0.27.0

# Caching
diskcache>=5.6.3

# Data handling
pandas>=2.0.0
pydantic>=2.0.0

# UI (optional)
streamlit>=1.35.0

# Configuration
pyyaml>=6.0.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
memory-profiler>=0.61.0
jsonschema>=4.22.0
```

### 14.5 Configuration (`config/thresholds.yaml`)

```yaml
thresholds:
  healthy_dos_min: 14        # days
  watch_dos_min: 7           # days
  critical_dos_max: 7        # days (exclusive)
  overstock_dos_min: 60      # days

defaults:
  safety_stock: 0
  lead_time_days: 7

cache:
  ttl_graph_seconds: 86400   # 24 hours
  ttl_rules_seconds: 3600    # 1 hour
  ttl_metrics_seconds: 1800  # 30 minutes
  max_size_mb: 500

neo4j:
  enabled: true
  uri: "bolt://your-ec2-ip:7687"
  user: "neo4j"
  password: "inventory123"
  timeout_ms: 800

ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:7b"
  timeout_ms: 4000

output:
  format: "json"
  pretty_print: true
```

### 14.6 Running the Agent (CLI)

```bash
# Run with default CSV data
python main.py --data data/inventory_mock.csv

# Run with JSON data
python main.py --data data/inventory_mock.json --format json

# Run with Neo4j disabled (force NetworkX fallback)
NEO4J_ENABLED=false python main.py --data data/inventory_mock.csv

# Run with custom config
python main.py --data data/inventory_mock.csv --config config/thresholds.yaml

# Output to file
python main.py --data data/inventory_mock.csv --output results/run_001.json
```

### 14.7 Starting Neo4j (Optional, EC2 or Local)

```bash
# Local Docker (for development)
docker-compose up -d neo4j

# docker-compose.yml snippet:
# services:
#   neo4j:
#     image: neo4j:5-community
#     ports:
#       - "7474:7474"
#       - "7687:7687"
#     environment:
#       - NEO4J_AUTH=neo4j/inventory123
#     volumes:
#       - ./neo4j_data:/data

# Seed the knowledge graph
python knowledge/neo4j_seed.py --uri bolt://localhost:7687

# Verify connectivity
python -c "from knowledge.neo4j_client import test_connection; test_connection()"
```

### 14.8 Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires Ollama running)
pytest tests/integration/ -v

# Fallback tests (Neo4j intentionally down)
NEO4J_ENABLED=false pytest tests/fallback/ -v

# Performance tests
pytest tests/performance/ -v --benchmark-min-rounds=10

# With coverage
pytest tests/ --cov=agent --cov=tools --cov-report=html
```

### 14.9 Launching Streamlit UI (Optional)

```bash
streamlit run app.py
# Open: http://localhost:8501
```

---

## 15. Future Roadmap (Post-MVP)

### Phase 2 — Analytical Depth

| Feature | Description | Priority |
|---|---|---|
| **Statistical Forecasting** | Integrate lightweight statsmodels (ARIMA / Exponential Smoothing) for trend-based demand forecasting on real historical data | High |
| **ABC-XYZ Classification** | Automate SKU classification (velocity × variability) to refine safety stock recommendations | High |
| **Multi-Location Support** | Extend data schema and KG to handle inventory across warehouses or store locations | Medium |
| **Historical Run Comparison** | Track recommendation history per run_id; surface drift and repeated stockout patterns | Medium |

### Phase 3 — Ecosystem Integration

| Feature | Description | Priority |
|---|---|---|
| **ERP Read-Only Connector** | One-way adapter to pull live inventory snapshots from SAP/Oracle (no write-back; advisory only) | Medium |
| **Webhook Notifications** | Push critical-status SKU alerts to Slack or email (human-reviewed, never automated orders) | Medium |
| **Supplier Lead Time API** | Query supplier portals for current lead time estimates to improve reorder_urgency accuracy | Low |

### Phase 4 — Intelligence Upgrades

| Feature | Description | Priority |
|---|---|---|
| **Fine-Tuned LLM** | Fine-tune a small model on domain-specific inventory commentary to improve explanation quality | Medium |
| **Feedback Loop** | Allow planners to rate explanations; log feedback to improve prompt templates | Medium |
| **Anomaly Detection** | Flag SKUs with unusual velocity spikes or unexplained inventory drops | High |
| **Multi-Agent Collaboration** | Separate specialized agents for demand sensing, supplier risk, and reorder optimization, orchestrated by a supervisor agent | Low |

### Phase 5 — Production Hardening

| Feature | Description | Priority |
|---|---|---|
| **Role-Based Access** | Per-user configuration profiles for planners vs. managers | Medium |
| **Audit Log Compliance** | Immutable run history for audit trail and regulatory review | Medium |
| **Cloud Deployment Option** | Containerized deployment on AWS ECS or GCP Cloud Run with optional managed Neo4j AuraDB | Low |
| **Real Data Certification** | Security review and data classification process to onboard real (non-mock) inventory data | High |

---

*End of Document — PRD v1.0.0*

---

> **⚠️ DOCUMENT DISCLAIMER**: This PRD describes a decision-support tool operating exclusively on synthetic mock data. It does not constitute a specification for any system that handles real commercial inventory, financial decisions, or procurement actions. All architectural decisions are subject to engineering review before implementation.
