"""Polished Streamlit UI for Inventory Optimization AI Agent."""

from __future__ import annotations

import json
import io
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import pandas as pd
import streamlit as st

from agent.nodes.apply_rules import apply_rules_node
from agent.nodes.calculate_metrics import calculate_metrics_node
from agent.nodes.enrich_context import enrich_context_node
from agent.nodes.explain_llm import stream_explain_llm_batches
from agent.nodes.format_output import format_output_node
from agent.nodes.generate_recs import generate_recs_node
from agent.nodes.load_data import load_data_node
from agent.nodes.template_explanation import template_explanation_node
from agent.nodes.validate_output import validate_output_node
from agent.state import AgentState
from main import DISCLAIMER, run_analysis
from tools.load_data import load_threshold_config


CLOUD_MODEL_NAME = "gemma4:31b-cloud"
CLOUD_MODEL_ALIAS = "gemma4:31b:cloud"


def _append_log(logs: List[str], level: str, message: str) -> None:
    """Append timestamped log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {level.upper():<5} {message}"
    logs.append(line)
    print(line, flush=True)


def _render_logs(log_box, logs: List[str], keep_last: int = 20) -> None:
    """Render rolling log output in Streamlit."""
    log_box.code("\n".join(logs[-keep_last:]) if logs else "No logs yet.", language="text")


def _get_sku_options(data_mode: str, uploaded_file) -> List[str]:
    """Return SKU options from selected source for scope controls."""
    try:
        if data_mode == "Upload file" and uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower()
            raw = uploaded_file.getvalue()
            if suffix == ".json":
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, list):
                    ids = [str(item.get("sku_id", "")) for item in data if isinstance(item, dict)]
                elif isinstance(data, dict):
                    ids = [str(item.get("sku_id", "")) for item in data.get("records", []) if isinstance(item, dict)]
                else:
                    ids = []
            else:
                frame = pd.read_csv(io.BytesIO(raw))
                ids = [str(item) for item in frame.get("sku_id", []).tolist()]
            return sorted({item for item in ids if item})

        frame = pd.read_csv("data/inventory_mock.csv")
        return sorted({str(item) for item in frame.get("sku_id", []).tolist() if str(item).strip()})
    except Exception:
        return []


def _filter_payload_for_skus(payload: Dict[str, Any], selected_skus: List[str]) -> Dict[str, Any]:
    """Filter payload recommendations/summary for selected SKU list."""
    if not selected_skus:
        return payload

    out = json.loads(json.dumps(payload))
    recs = out.get("recommendations", [])
    filtered = [rec for rec in recs if str(rec.get("sku_id", "")) in set(selected_skus)]
    out["recommendations"] = filtered

    summary = out.get("summary", {})
    summary["total_skus_analyzed"] = len(filtered)
    summary["critical_count"] = sum(1 for rec in filtered if rec.get("status") == "critical")
    summary["watch_count"] = sum(1 for rec in filtered if rec.get("status") == "watch")
    summary["healthy_count"] = sum(1 for rec in filtered if rec.get("status") == "healthy")
    summary["overstock_count"] = sum(1 for rec in filtered if rec.get("status") == "overstock")
    summary["skus_skipped"] = max(0, int(summary.get("skus_skipped", 0)))
    summary["top_priority_skus"] = [
        item.get("sku_id", "")
        for item in sorted(filtered, key=lambda item: float(item.get("reorder_urgency_days", 0.0)))
        if item.get("status") in {"critical", "watch"}
    ][:5]
    out["summary"] = summary
    return out


def _normalize_selected_skus(analysis_scope: str, selected_skus: List[str]) -> List[str]:
    """Return cleaned selected SKU list for config-level scoping."""
    if analysis_scope == "All SKUs":
        return []
    return [str(item).strip() for item in selected_skus if str(item).strip()]


def _get_ollama_models(base_url: str, api_key: str) -> Tuple[bool, List[str], str]:
    """Fetch installed Ollama models from local runtime."""
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    try:
        with httpx.Client(timeout=2.5, headers=headers) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        if CLOUD_MODEL_NAME not in models:
            models.append(CLOUD_MODEL_NAME)
        if CLOUD_MODEL_ALIAS not in models:
            models.append(CLOUD_MODEL_ALIAS)
        return True, models, "Ollama reachable"
    except Exception as exc:
        return False, [CLOUD_MODEL_NAME, CLOUD_MODEL_ALIAS], f"Ollama unreachable: {exc}"


def _build_config(
    base_cfg: Dict[str, Any],
    data_path: str,
    ollama_base_url: str,
    ollama_model: str,
    ollama_api_key: str,
    timeout_ms: int,
    temperature: float,
    num_predict: int,
    agent_mode: str,
    scenario_overrides: Dict[str, float],
) -> Dict[str, Any]:
    """Build run config from base + UI settings."""
    cfg = json.loads(json.dumps(base_cfg))
    cfg["data_path"] = data_path
    cfg["config_path"] = "config/thresholds.yaml"
    cfg["kg_seed_path"] = "data/kg_seed.json"

    cfg.setdefault("ollama", {})["base_url"] = ollama_base_url
    cfg.setdefault("ollama", {})["model"] = ollama_model
    cfg.setdefault("ollama", {})["api_key"] = ollama_api_key
    cfg.setdefault("ollama", {})["timeout_ms"] = timeout_ms
    cfg.setdefault("ollama", {})["planner_timeout_ms"] = timeout_ms
    cfg.setdefault("ollama", {})["temperature"] = temperature
    cfg.setdefault("ollama", {})["num_predict"] = num_predict
    cfg["agent_mode"] = agent_mode
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent", {}), dict) else {}
    cfg["agent_max_steps"] = int(agent_cfg.get("max_steps", cfg.get("agent_max_steps", 3)))

    if scenario_overrides:
        cfg["scenario_overrides"] = scenario_overrides
    return cfg


def _initial_state(config: Dict[str, Any]) -> AgentState:
    """Create fresh agent state for streamed execution."""
    return {
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "raw_records": [],
        "sku_records": [],
        "sku_metrics": [],
        "sku_contexts": [],
        "rule_results": {},
        "recommendations": [],
        "llm_prompts": {},
        "llm_responses": {},
        "llm_retries": {},
        "agent_step_count": 0,
        "agent_max_steps": int(config.get("agent_max_steps", 3)),
        "agent_scratchpad": [],
        "agent_tool_history": [],
        "agent_done": False,
        "agent_pending_action": None,
        "agent_fallback_reason": "",
        "current_node": "",
        "errors": [],
        "warnings": [],
        "partial_data": False,
        "graph_source": "default",
        "output_valid": False,
        "final_output": None,
    }


def _run_analysis_streaming(config: Dict[str, Any], status_box, progress_bar, log_box) -> Dict[str, Any]:
    """Run nodes sequentially and stream LLM batch progress to UI."""
    state = _initial_state(config)
    logs: List[str] = []

    _append_log(logs, "info", "Run started in deterministic mode.")
    _append_log(logs, "info", f"Data path: {config.get('data_path', 'unknown')}")
    _render_logs(log_box, logs)

    status_box.info("Loading and validating data...")
    t0 = time.perf_counter()
    state = load_data_node(state)
    _append_log(
        logs,
        "info",
        f"load_data completed in {int((time.perf_counter() - t0) * 1000)} ms | records={len(state['sku_records'])} warnings={len(state['warnings'])}",
    )
    _render_logs(log_box, logs)
    progress_bar.progress(10)

    status_box.info("Calculating SKU metrics...")
    t0 = time.perf_counter()
    state = calculate_metrics_node(state)
    _append_log(
        logs,
        "info",
        f"calculate_metrics completed in {int((time.perf_counter() - t0) * 1000)} ms | metrics={len(state['sku_metrics'])}",
    )
    _render_logs(log_box, logs)
    progress_bar.progress(25)

    status_box.info("Enriching graph context...")
    t0 = time.perf_counter()
    state = enrich_context_node(state)
    _append_log(
        logs,
        "info",
        f"enrich_context completed in {int((time.perf_counter() - t0) * 1000)} ms | source={state.get('graph_source', 'unknown')}",
    )
    _render_logs(log_box, logs)
    progress_bar.progress(40)

    status_box.info("Applying inventory rules...")
    t0 = time.perf_counter()
    state = apply_rules_node(state)
    _append_log(logs, "info", f"apply_rules completed in {int((time.perf_counter() - t0) * 1000)} ms")
    _render_logs(log_box, logs)
    progress_bar.progress(50)

    status_box.info("Preparing LLM prompts...")
    t0 = time.perf_counter()
    state = generate_recs_node(state)
    _append_log(
        logs,
        "info",
        f"generate_recs completed in {int((time.perf_counter() - t0) * 1000)} ms | prompts={len(state['llm_prompts'])}",
    )
    _render_logs(log_box, logs)
    progress_bar.progress(55)

    status_box.info("Generating LLM explanations in batches of 5...")
    total_batches = max(1, (len(state["llm_prompts"]) + 4) // 5)
    for event in stream_explain_llm_batches(state):
        completed = int(event.get("batch_index", 0))
        batch_total = int(event.get("batch_total", total_batches))
        success = bool(event.get("batch_success", False))
        marker = "info" if success else "warn"
        detail = str(event.get("detail", ""))
        _append_log(logs, marker, f"llm_batch {completed}/{batch_total}: {detail}")
        _render_logs(log_box, logs)
        progress_pct = 55 + int((completed / max(1, batch_total)) * 30)
        progress_bar.progress(min(progress_pct, 85))

    status_box.info("Applying template fallback for missing SKUs...")
    t0 = time.perf_counter()
    state = template_explanation_node(state)
    _append_log(logs, "info", f"template_explanation completed in {int((time.perf_counter() - t0) * 1000)} ms")
    _render_logs(log_box, logs)
    progress_bar.progress(90)

    status_box.info("Formatting and validating final output...")
    t0 = time.perf_counter()
    state = format_output_node(state)
    state = validate_output_node(state)
    _append_log(logs, "info", f"format/validate completed in {int((time.perf_counter() - t0) * 1000)} ms")
    if state.get("warnings"):
        _append_log(logs, "warn", f"warnings={len(state['warnings'])}")
    if state.get("errors"):
        _append_log(logs, "error", f"errors={len(state['errors'])}")
    _render_logs(log_box, logs)
    progress_bar.progress(100)
    status_box.success("Run completed.")

    if state.get("final_output"):
        return state["final_output"]  # type: ignore[return-value]

    return {
        "run_id": state["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_skus_analyzed": 0,
            "critical_count": 0,
            "watch_count": 0,
            "healthy_count": 0,
            "overstock_count": 0,
            "skus_skipped": 0,
            "overall_health": "poor",
            "top_priority_skus": [],
        },
        "recommendations": [],
        "metadata": {
            "llm_model": config.get("ollama", {}).get("model", "llama3.2:1b"),
            "graph_source": "default",
            "config_version": "1.0.0",
            "execution_time_ms": 0,
            "partial_data": True,
            "agent_mode": config.get("agent_mode", "deterministic"),
            "agent_steps_executed": state.get("agent_step_count", 0),
            "agent_tool_history": state.get("agent_tool_history", []),
            "agent_fallback_reason": state.get("agent_fallback_reason", ""),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        },
        "disclaimer": DISCLAIMER,
    }


def _payload_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert recommendations to a dataframe for filtering and display."""
    rows = payload.get("recommendations", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    keep_cols = [
        "sku_id",
        "name",
        "category",
        "status",
        "status_emoji",
        "days_of_stock",
        "reorder_qty",
        "reorder_urgency_days",
        "velocity_trend",
        "seasonal_factor",
        "category_avg_dos",
        "context_source",
        "confidence",
        "recommended_action",
        "plain_english_explanation",
    ]
    ordered = [col for col in keep_cols if col in frame.columns]
    return frame[ordered]


def _render_status_panel(payload: Dict[str, Any], elapsed_ms: float | None) -> None:
    """Render summary metrics and runtime diagnostics."""
    summary = payload.get("summary", {})
    warnings = payload.get("metadata", {}).get("warnings", [])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Critical", int(summary.get("critical_count", 0)))
    col2.metric("Watch", int(summary.get("watch_count", 0)))
    col3.metric("Healthy", int(summary.get("healthy_count", 0)))
    col4.metric("Overstock", int(summary.get("overstock_count", 0)))
    col5.metric("Runtime (ms)", int(elapsed_ms) if elapsed_ms is not None else 0)

    fallback_used = any("fallback" in str(item).lower() for item in warnings)
    if fallback_used:
        st.warning("LLM fallback used for all or part of this run. Check Diagnostics tab.")
    else:
        st.success("LLM path active: no fallback warning detected in metadata.")


def _render_recommendations(payload: Dict[str, Any]) -> None:
    """Render filterable recommendation table and cards."""
    frame = _payload_to_df(payload)
    if frame.empty:
        st.info("No recommendations available for display.")
        return

    c1, c2, c3 = st.columns(3)
    status_filter = c1.multiselect("Status", sorted(frame["status"].dropna().unique().tolist()))
    category_filter = c2.multiselect("Category", sorted(frame["category"].dropna().unique().tolist()))
    context_filter = c3.multiselect(
        "Context Source",
        sorted(frame["context_source"].dropna().unique().tolist()),
    )
    search = st.text_input("Search SKU / name", value="").strip().lower()

    filtered = frame.copy()
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if context_filter:
        filtered = filtered[filtered["context_source"].isin(context_filter)]
    if search:
        filtered = filtered[
            filtered["sku_id"].str.lower().str.contains(search)
            | filtered["name"].str.lower().str.contains(search)
        ]

    filtered = filtered.sort_values(by=["status", "reorder_urgency_days"], ascending=[True, True])
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name=f"inventory_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.subheader("Top Priority SKUs")
    card_data = filtered.sort_values(by="reorder_urgency_days").head(10)
    for _, rec in card_data.iterrows():
        with st.container(border=True):
            st.markdown(f"**{rec['sku_id']} - {rec['name']}**")
            st.caption(
                f"{rec['status_emoji']} {rec['status']} | Category: {rec['category']} | "
                f"Context: {rec['context_source']} | Confidence: {rec['confidence']}"
            )
            st.write(
                f"Days of stock: {float(rec['days_of_stock']):.2f} | "
                f"Reorder qty: {float(rec['reorder_qty']):.2f} | "
                f"Urgency days: {float(rec['reorder_urgency_days']):.2f}"
            )
            st.write(f"Action: {rec['recommended_action']}")
            st.write(rec["plain_english_explanation"])


def _render_diagnostics(payload: Dict[str, Any], model_name: str) -> None:
    """Render diagnostic details for troubleshooting fallback and errors."""
    metadata = payload.get("metadata", {})
    warnings = metadata.get("warnings", [])
    errors = metadata.get("errors", [])

    st.write(f"Model configured: `{model_name}`")
    st.write(f"Graph source used: `{metadata.get('graph_source', 'unknown')}`")
    st.write(f"Partial data: `{metadata.get('partial_data', False)}`")
    st.write(f"Agent mode: `{metadata.get('agent_mode', 'deterministic')}`")
    st.write(f"Agent steps executed: `{metadata.get('agent_steps_executed', 0)}`")
    fallback_reason = str(metadata.get("agent_fallback_reason", "")).strip()
    if fallback_reason:
        st.info(f"Agent fallback reason: {fallback_reason}")

    if warnings:
        st.warning("Warnings")
        st.code("\n".join(str(item) for item in warnings), language="text")
    else:
        st.success("No warnings reported.")

    if errors:
        st.error("Errors")
        st.code(json.dumps(errors, indent=2, ensure_ascii=False), language="json")
    else:
        st.success("No errors reported.")

    tool_history = metadata.get("agent_tool_history", [])
    if tool_history:
        st.subheader("Agent Tool History")
        st.json(tool_history)


def main() -> None:
    """Run Streamlit application."""
    st.set_page_config(page_title="Inventory Optimization Agent", layout="wide")
    st.title("Inventory Optimization AI Agent")
    st.caption("Local-first decision-support workflow with hybrid graph context and batched LLM explanations.")
    st.warning(DISCLAIMER)

    base_cfg = load_threshold_config("config/thresholds.yaml")

    with st.sidebar:
        st.header("Run Configuration")

        data_mode = st.radio("Data Source", ["Upload file", "Use mock dataset"], horizontal=False)
        uploaded = None
        if data_mode == "Upload file":
            uploaded = st.file_uploader("Upload inventory CSV or JSON", type=["csv", "json"])

        llm_profile = st.radio("LLM Profile", ["Quality (recommended)", "Strict latency"], index=0)
        default_timeout = 40000 if llm_profile.startswith("Quality") else 4000
        default_predict = 1800 if llm_profile.startswith("Quality") else 900

        base_url = st.text_input("Ollama Base URL", value=base_cfg.get("ollama", {}).get("base_url", "http://localhost:11434"))
        default_key = os.environ.get("OLLAMA_API_KEY", "")
        ollama_api_key = st.text_input("Ollama API Key", value=default_key, type="password")

        ok, models, ollama_msg = _get_ollama_models(base_url, ollama_api_key)
        if ok:
            st.success(ollama_msg)
        else:
            st.error(ollama_msg)

        model_default = base_cfg.get("ollama", {}).get("model", CLOUD_MODEL_NAME)
        if model_default not in models:
            models = [model_default] + models
        if models:
            model = st.selectbox("Ollama Model", options=models, index=models.index(model_default) if model_default in models else 0)
        else:
            model = st.text_input("Ollama Model", value=model_default)

        timeout_ms = int(st.slider("LLM Timeout (ms, 0 = wait indefinitely)", min_value=0, max_value=120000, value=default_timeout, step=1000))
        temperature = float(st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05))
        num_predict = int(st.slider("Max Tokens (num_predict)", min_value=400, max_value=4000, value=default_predict, step=100))

        st.header("Agent Mode")
        mode_options = ["deterministic", "hybrid", "full"]
        mode_default = str(base_cfg.get("agent", {}).get("mode", "deterministic"))
        default_index = mode_options.index(mode_default) if mode_default in mode_options else 0
        agent_mode = st.selectbox("Mode", options=mode_options, index=default_index)
        default_max_steps = int(base_cfg.get("agent", {}).get("max_steps", 3))
        agent_max_steps = int(st.slider("Agent max steps", min_value=1, max_value=8, value=default_max_steps, step=1))

        st.header("Scenario Overrides")
        apply_scenario = st.checkbox("Enable what-if overrides", value=False)
        lead_time_override = float(st.number_input("lead_time_days", min_value=1.0, max_value=90.0, value=7.0, step=1.0))
        safety_stock_override = float(st.number_input("safety_stock", min_value=0.0, max_value=1000.0, value=0.0, step=1.0))

        st.header("SKU Scope")
        analysis_scope = st.radio("Analysis Type", ["All SKUs", "Single SKU", "Multiple SKUs"], index=0)
        sku_options = _get_sku_options(data_mode, uploaded)
        selected_skus: List[str] = []
        if analysis_scope == "Single SKU":
            selected = st.selectbox("Select SKU", options=sku_options) if sku_options else ""
            selected_skus = [selected] if selected else []
        elif analysis_scope == "Multiple SKUs":
            selected_skus = st.multiselect("Select SKUs", options=sku_options)

        if analysis_scope == "All SKUs":
            st.caption(f"Scope: all available SKUs ({len(sku_options)})")
        elif analysis_scope == "Single SKU":
            st.caption(f"Scope: {len(selected_skus)} SKU selected")
        else:
            st.caption(f"Scope: {len(selected_skus)} SKUs selected")

        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    if "last_payload" not in st.session_state:
        st.session_state["last_payload"] = None
    if "last_elapsed_ms" not in st.session_state:
        st.session_state["last_elapsed_ms"] = None

    if run_clicked:
        run_status = st.empty()
        run_logs = st.empty()
        run_progress = st.progress(0)
        with tempfile.TemporaryDirectory() as temp_dir:
            if data_mode == "Upload file":
                if uploaded is None:
                    st.error("Please upload a CSV or JSON file first.")
                    return
                
                #writing it to a temp file.
                data_path = Path(temp_dir) / uploaded.name
                data_path.write_bytes(uploaded.read())
                effective_data_path = str(data_path)
            else:
                effective_data_path = "data/inventory_mock.csv"

            scenario_overrides: Dict[str, float] = {}
            if apply_scenario:
                scenario_overrides = {
                    "lead_time_days": lead_time_override,
                    "safety_stock": safety_stock_override,
                }

            config = _build_config(
                base_cfg=base_cfg,
                data_path=effective_data_path,
                ollama_base_url=base_url,
                ollama_model=model,
                ollama_api_key=ollama_api_key,
                timeout_ms=timeout_ms,
                temperature=temperature,
                num_predict=num_predict,
                agent_mode=agent_mode,
                scenario_overrides=scenario_overrides,
            )
            config.setdefault("ollama", {})["batch_size"] = 5
            config["agent_max_steps"] = agent_max_steps
            config["analysis_sku_ids"] = _normalize_selected_skus(analysis_scope, selected_skus)

            start = time.perf_counter()
            if agent_mode == "deterministic":
                payload = _run_analysis_streaming(config, status_box=run_status, progress_bar=run_progress, log_box=run_logs)
            else:
                hybrid_logs: List[str] = []
                _append_log(hybrid_logs, "info", f"Run started in {agent_mode} mode.")
                _append_log(hybrid_logs, "info", "Executing planner/executor loop via LangGraph...")
                _render_logs(run_logs, hybrid_logs)
                run_status.info(f"Running {agent_mode} mode via planner/executor loop...")
                run_progress.progress(30)
                payload = run_analysis(config)
                metadata = payload.get("metadata", {})
                _append_log(
                    hybrid_logs,
                    "info",
                    f"agent_steps_executed={metadata.get('agent_steps_executed', 0)} tool_calls={len(metadata.get('agent_tool_history', []))}",
                )
                warnings = metadata.get("warnings", [])
                errors = metadata.get("errors", [])
                if warnings:
                    _append_log(hybrid_logs, "warn", f"warnings={len(warnings)}")
                if errors:
                    _append_log(hybrid_logs, "error", f"errors={len(errors)}")
                _render_logs(run_logs, hybrid_logs)
                run_progress.progress(100)
                run_status.success("Run completed.")
            elapsed_ms = (time.perf_counter() - start) * 1000

            if analysis_scope in {"Single SKU", "Multiple SKUs"} and selected_skus:
                payload = _filter_payload_for_skus(payload, selected_skus)
                run_status.info(f"Applied SKU scope filter: {len(selected_skus)} selected SKU(s).")
            elif analysis_scope in {"Single SKU", "Multiple SKUs"} and not selected_skus:
                run_status.warning("No SKU selected for scoped analysis; showing all SKUs.")

            st.session_state["last_payload"] = payload
            st.session_state["last_elapsed_ms"] = elapsed_ms

    payload = st.session_state.get("last_payload")
    elapsed_ms = st.session_state.get("last_elapsed_ms")

    if payload is None:
        st.info("Run an analysis to view results.")
        return

    st.subheader("Run Summary")
    _render_status_panel(payload, elapsed_ms)

    tab1, tab2, tab3 = st.tabs(["Recommendations", "Diagnostics", "Raw JSON"])

    with tab1:
        _render_recommendations(payload)

    with tab2:
        _render_diagnostics(payload, model_name=model)

    with tab3:
        st.download_button(
            "Download JSON Output",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=f"inventory_run_{payload.get('run_id', 'unknown')}.json",
            mime="application/json",
        )
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
