# Product Requirements Document
# Inventory Optimization AI Agent (Decision Support)

---

## 1. Document Header & Versioning

| Field | Value |
|---|---|
| **Document Title** | Inventory Optimization AI Agent — PRD |
| **Version** | 1.2.0 |
| **Status** | Active Draft — Updated to current implemented architecture |
| **Author(s)** | Senior PM / AI Systems Architect |
| **Assessment Reference** | RA8 |
| **Created** | 2025-07-01 |
| **Last Updated** | 2026-04-21 |
| **Target Release** | MVP (Local / Offline) |
| **Review Cycle** | Per sprint, major version on architecture change |

### Revision History

| Version | Date | Author | Change Summary |
|---|---|---|---|
| 0.1 | 2025-06-15 | Architect | Initial outline |
| 0.9 | 2025-06-28 | PM | Added LangGraph + MCP architecture |
| 1.0 | 2025-07-01 | PM / Architect | Final PRD — production-ready draft |
| 1.1 | 2026-04-16 | PM / Architect | Added phased hybrid-agentic plan; removed Neo4j from MVP |
| 1.2 | 2026-04-21 | PM / Architect | Synced PRD to implemented system: mode router flow, runtime NetworkX graph from uploaded data, in-memory cache only, full/hybrid planner loop behavior, action guardrails, raw CoT (experimental), expanded diagnostics metadata |

---

## 2. Problem Statement & Product Vision

### 2.1 Problem Statement

Inventory teams still depend on manual spreadsheet reviews for reorder decisions. This causes:

- **Stockouts** from delayed reorder action on high-velocity SKUs.
- **Overstocking** from weak visibility on low-velocity and seasonal demand patterns.

The problem is not data availability; it is lack of explainable, trustworthy, context-aware decision support for non-technical planners.

### 2.2 Product Vision

> Give planners a local, always-available AI analyst that reads inventory data, surfaces risk early, and explains what to do in plain language.

The system is a **decision-support co-pilot**:

1. Ingests inventory CSV/JSON.
2. Computes inventory health metrics.
3. Enriches thinking-mode decisions using a runtime NetworkX graph built from uploaded data.
4. Generates recommendations via deterministic and/or agentic execution.
5. Provides explainable output with mandatory advisory disclaimer.

It never auto-places orders and never modifies external systems.

---

## 3. Target Users & Personas

### Persona 1 — Maya, Inventory Planner (Primary)

| Attribute | Detail |
|---|---|
| **Role** | Inventory / Replenishment Planner |
| **Tech Comfort** | Low-to-medium |
| **Pain Points** | Manual review load, missed reorder windows, weak context visibility |
| **Goal** | Prioritized, plain-English guidance by SKU |

### Persona 2 — Raj, Operations Manager (Secondary)

| Attribute | Detail |
|---|---|
| **Role** | Retail / Supply Operations Manager |
| **Goal** | Consistent weekly risk visibility and auditability |

### Persona 3 — Priya, AI/IT Learner (Tertiary)

| Attribute | Detail |
|---|---|
| **Role** | ML Engineer / Student |
| **Goal** | Learn production-pattern LangGraph + MCP + local LLM architecture |

---

## 4. Scope

### 4.1 In Scope (MVP)

| ID | Feature |
|---|---|
| S-01 | CSV/JSON ingestion with row validation |
| S-02 | Metric calculation: days_of_stock, reorder_qty, urgency |
| S-03 | Status assignment: healthy/watch/critical/overstock |
| S-04 | LangGraph orchestration with mode routing |
| S-05 | MCP tool layer via FastMCP |
| S-06 | Runtime NetworkX graph built from uploaded records (thinking mode) |
| S-07 | In-memory cache for graph artifact reuse |
| S-08 | Local Ollama LLM explanations |
| S-09 | Deterministic and agentic execution modes |
| S-10 | Streamlit UI + CLI entrypoint |
| S-11 | Action correctness guardrails over LLM output |
| S-12 | Raw CoT capture (experimental, optional) |
| S-13 | Structured diagnostics: flow/tool/llm events |
| S-14 | Advisory-only disclaimer on all outputs |

### 4.2 Out of Scope (MVP)

- ERP/WMS integration.
- Real-time POS ingestion.
- Automated order placement.
- SaaS multi-tenant deployment.
- Forecasting ML training pipelines.

---

## 5. System Architecture & Data Flow

### 5.1 Implemented Architecture Overview

```
UI Layer
  - Streamlit: ui/tabs.py, ui/preflight.py, ui/runner.py
  - CLI: main.py

Orchestration Layer
  - LangGraph: agent/graph.py
  - State contract: agent/state.py
  - Node logic: agent/nodes/*

Tool Layer (MCP)
  - tools/server.py
  - tools/load_data.py, tools/calc_metrics.py, tools/query_graph.py, tools/fetch_rules.py

Knowledge Layer
  - knowledge/networkx_graph.py
  - knowledge/cache_layer.py (in-memory only)

Model Layer
  - Ollama local endpoint (localhost)

Output Layer
  - Structured JSON + Streamlit views + CSV/trace exports
```

### 5.2 Current End-to-End Flow (Canonical)

#### Mode Router

```
START -> mode_router
if mode=thinking and agent_mode=full: planner_action
else: load_data
```

#### Deterministic Backbone

```
load_data -> calculate_metrics ->
  if mode=fast: apply_rules
  else: enrich_context -> apply_rules
-> generate_recs -> explain_llm -> template_explanation -> format_output -> validate_output -> END
```

#### Agentic Loop (Hybrid / Full)

```
generate_recs -> planner_action
planner_action:
  if done=true:
    if mode=thinking and agent_mode=full: generate_recs
    else: explain_llm
  else: execute_action -> planner_action (loop)

... -> explain_llm -> template_explanation -> format_output -> validate_output -> END
```

### 5.3 Mode Behavior Summary

- **fast**: deterministic only, no graph enrichment.
- **thinking + deterministic**: deterministic flow plus graph enrichment.
- **thinking + hybrid**: deterministic backbone plus planner loop where applicable.
- **thinking + full**: planner-first route with strict full-mode contract checks.

---

## 6. LangGraph State Machine & Node Definitions

### 6.1 State Contract (Implemented)

`agent/state.py` defines runtime state including:

- data pipeline: `raw_records`, `sku_records`, `sku_metrics`, `sku_contexts`
- tool/rule outputs: `rule_results`, `recommendations`
- llm fields: `llm_prompts`, `llm_responses`, `llm_reasoning`, `llm_reasoning_by_sku`, `llm_retries`
- observability: `flow_events`, `tool_call_logs`, `llm_batch_events`
- agent loop control: `agent_step_count`, `agent_max_steps`, `agent_tool_history`, `agent_done`, `agent_pending_action`, `agent_fallback_reason`
- run diagnostics: `errors`, `warnings`, `partial_data`, `graph_source`, `graph_runtime_stats`, `output_valid`

### 6.2 Node Inventory

- Routing/core: `mode_router`, `load_data`, `calculate_metrics`, `enrich_context`, `apply_rules`, `generate_recs`
- Agentic: `planner_action`, `execute_action`
- Explanation/output: `explain_llm`, `template_explanation`, `format_output`, `validate_output`

### 6.3 Reliability Logic Highlights

- Planner schema validation and stage-aware tool checks.
- Duplicate planner action suppression.
- Explicit stop reason tracking in `agent_fallback_reason`.
- Full-mode contract validation (`full_mode_contract_ok`) in output metadata.

---

## 7. MCP Tool Layer & Contracts

### 7.1 Tool Surface

From `tools/server.py`:

- `load_inventory`
- `calc_metrics_batch`
- `query_graph_batch`
- `apply_rules_batch`
- plus core helper tools (`fetch_rules`, etc.)

### 7.2 Runtime Contract Notes

- Tools are consumed by both deterministic nodes and planner executor paths.
- `execute_action` maps planner-selected tool observations back into shared state.
- Batch tools reduce loop overhead for planner-driven runs.

### 7.3 Query Graph Contract (Current)

- `tools/query_graph.py` requires `config.runtime_records`.
- Graph artifact key = dataset fingerprint + graph schema version.
- Cache = in-memory artifact reuse only.
- Thinking mode expects runtime graph context; no default graph-context fallback in graph query path.

---

## 8. Knowledge Graph Strategy

### 8.1 Runtime NetworkX (Current)

The system builds a runtime directed graph from uploaded records (`knowledge/networkx_graph.py`):

- node types: SKU, category, optional supplier
- edges: SKU->category, optional SKU->supplier
- derived context: category DOS baselines, risk tags, relative factor signals

### 8.2 Caching Strategy

- cache implementation: `knowledge/cache_layer.py`
- type: in-memory TTL store
- purpose: performance optimization only
- non-goal: fallback decision logic

### 8.3 Thinking-Mode Policy

- thinking runs depend on runtime graph availability.
- graph build/query failures are explicit errors, not silent default-context substitution.

---

## 9. Functional Requirements

### FR-01 Data and Metrics

- System loads CSV/JSON, validates rows, and computes inventory metrics.
- Invalid rows are skipped with warnings.

### FR-02 Mode-Aware Orchestration

- System supports `fast` and `thinking` modes.
- In `fast`, graph enrichment is skipped.
- In `thinking`, graph enrichment is enabled.

### FR-03 Agent Modes

- System supports `deterministic`, `hybrid`, and `full` agent modes in thinking mode.
- In fast mode, effective execution remains deterministic.

### FR-04 Recommendation Quality Guardrails

- System post-validates LLM action recommendations against status/metrics policy.
- Contradictory actions are corrected and logged (`llm_action_corrected:<sku>`).

### FR-05 Explainability

- System returns `plain_english_explanation` per SKU.
- System returns `reasoning_summary` per SKU.
- Optional raw CoT is available in experimental mode and never used for final action policy.

### FR-06 Observability

- System emits flow/tool/llm event logs in metadata.
- Streamlit displays live trace and diagnostics views.

---

## 10. Non-Functional Requirements

### NFR-01 Offline and Local-First

- Core pipeline runs locally.
- LLM calls are local to configured Ollama endpoint.

### NFR-02 Reliability

- Deterministic fallback path remains available when LLM/planner behavior degrades.
- Runs produce structured errors/warnings instead of crashing UI pathways.

### NFR-03 Transparency

- Recommendations include rationale fields.
- Metadata includes graph runtime stats, event traces, and stop reasons.

### NFR-04 Performance

- Latency targets are mode/model dependent and not defined as a single hard timeout for all paths.
- Cache reuse should reduce repeated graph-build cost for same dataset fingerprint.

---

## 11. Output Schema & Prompt Guidelines

### 11.1 Output Structure (Current)

Top-level output:

- `run_id`, `generated_at`, `summary`, `recommendations`, `metadata`, `disclaimer`

Per recommendation includes (core):

- `sku_id`, `name`, `category`, `status`, `status_emoji`
- `days_of_stock`, `reorder_qty`, `reorder_urgency_days`
- `recommended_action`, `plain_english_explanation`, `confidence`
- `risk_tags`, `seasonal_factor`, `category_avg_dos`, `context_source`, `velocity_trend`
- `reasoning_summary` (new)
- `raw_cot` (experimental, optional)

Metadata includes (core):

- `mode`, `agent_mode`, `graph_source`, `graph_runtime_stats`
- `flow_events`, `tool_call_logs`, `llm_batch_events`
- `llm_reasoning` (experimental)
- `agent_fallback_reason`, `full_mode_contract_ok`
- `errors`, `warnings`

### 11.2 Prompt/Response Policy

- Prompting requires advisory language and action-policy consistency.
- Action-policy enforcement is implemented as post-validation in `agent/nodes/explain_llm.py`.
- Raw CoT capture is separate, optional, and excluded from action decision logic.

---

## 12. Ethics, Safety & Compliance

- Advisory-only mandate remains mandatory.
- No automatic procurement actions.
- Human review required for all recommendations.
- Synthetic/mock-first handling remains preferred for demos and validation.

---

## 13. Testing & Validation Plan

### 13.1 Unit Coverage Focus

- metric calculations and status assignment
- runtime graph query behavior
- tool contract behavior

### 13.2 Integration Coverage Focus

- deterministic and agentic route execution
- full-mode planner failure handling and contract checks
- thinking-mode graph dependency behavior

### 13.3 Guardrail Coverage Focus

- contradictory LLM action correction by status
- template fallback completion for missing LLM outputs
- metadata correctness for warnings/stop reasons/trace logs

---

## 14. Local Development & Setup Guide

### 14.1 Runtime

- Python virtual environment required.
- Install dependencies from `requirements.txt`.
- Start local Ollama endpoint and ensure model is installed.

### 14.2 Typical Commands

- CLI run via `main.py`
- Streamlit run via `app.py`
- tests via `pytest`

### 14.3 Key Files

- `main.py`, `app.py`
- `agent/graph.py`, `agent/state.py`, `agent/nodes/*`
- `tools/server.py`, `tools/query_graph.py`
- `knowledge/networkx_graph.py`, `knowledge/cache_layer.py`
- `ui/tabs.py`, `ui/preflight.py`, `ui/runner.py`

---

## 15. Future Roadmap (Post-MVP)

1. Expand graph-derived risk features.
2. Improve planner model compatibility profiles.
3. Add stronger regression matrix for action-policy correctness.
4. Add per-SKU on-demand raw CoT capture with redaction controls.
5. Add benchmark framework for latency/quality tradeoff by mode and model.
