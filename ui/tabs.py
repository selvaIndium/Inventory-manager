"""Tabbed Streamlit views for inventory app."""

from __future__ import annotations

import json
import time
from typing import Any, Dict
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from cli_helpers import generate_report
from main import DISCLAIMER
from ui.config import build_run_config, get_ollama_models, materialize_input_file, new_tempdir, profile_defaults
from ui.formatters import filter_df, now_file_suffix, payload_to_df, summarize_flow
from ui.preflight import run_preflight_checks
from ui.runner import run_once, run_with_scenario
from ui.runner import run_analysis_stream
from ui.session import add_history_entry
from ui.styles import stat_card, status_pill
from tools.load_data import load_inventory_data


def _render_run_gate(is_enabled: bool, reasons: list[str]) -> None:
    """Render run gate status and blocker reasons."""
    if is_enabled:
        status_pill("Run is enabled", "ok")
        return

    status_pill("Run blocked", "warn")
    for reason in reasons:
        st.caption(f"- {reason}")


def _preview_records(data_mode: str, uploaded) -> tuple[list[dict], list[str], str]:
    """Load and normalize dataset for preview and scope selection."""
    try:
        if data_mode == "Use mock dataset":
            result = load_inventory_data("data/inventory_mock.csv")
            return result.get("records", []), result.get("warnings", []), ""

        if uploaded is None:
            return [], [], "Upload a CSV or JSON file to preview records."

        suffix = Path(uploaded.name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded.getvalue())
            temp_path = Path(temp.name)

        try:
            result = load_inventory_data(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        return result.get("records", []), result.get("warnings", []), ""
    except Exception as exc:
        return [], [], f"Dataset preview failed: {exc}"


def _tab_run(base_cfg: Dict[str, Any]) -> None:
    """Render run setup tab and execute analysis."""
    st.markdown("<div class='section-kicker'>Run Setup</div>", unsafe_allow_html=True)
    st.subheader("Configure analysis")

    left, right = st.columns([1.3, 1.0])

    with left:
        data_mode = st.radio("Data source", ["Upload file", "Use mock dataset"], horizontal=True)
        uploaded = None
        if data_mode == "Upload file":
            uploaded = st.file_uploader("Upload inventory CSV/JSON", type=["csv", "json"])

        records, preview_warnings, preview_error = _preview_records(data_mode, uploaded)
        if preview_error:
            st.warning(preview_error)
        if preview_warnings:
            for warning in preview_warnings[:2]:
                st.caption(f"- {warning}")

        st.markdown("#### Dataset Preview")
        if records:
            preview_df = pd.DataFrame(records)
            show_cols = [col for col in ["sku_id", "name", "category", "current_stock", "avg_daily_sales"] if col in preview_df.columns]
            st.dataframe(preview_df[show_cols], width="stretch", hide_index=True, height=220)
        else:
            st.info("No records available yet.")

        analysis_scope = st.radio("Analysis scope", ["All SKUs", "Single SKU", "Custom SKU List"], horizontal=True)
        sku_options = sorted({str(row.get("sku_id", "")).strip() for row in records if str(row.get("sku_id", "")).strip()})
        sku_name_map = {str(row.get("sku_id", "")).strip(): str(row.get("name", "")) for row in records}
        selected_sku_ids: list[str] = []

        if analysis_scope == "Single SKU":
            if sku_options:
                single = st.selectbox(
                    "Select SKU",
                    options=sku_options,
                    format_func=lambda sku: f"{sku} - {sku_name_map.get(sku, '')}",
                )
                selected_sku_ids = [single]
            else:
                st.warning("No SKU options available from previewed dataset.")
        elif analysis_scope == "Custom SKU List":
            selected_sku_ids = st.multiselect(
                "Select SKU list",
                options=sku_options,
                format_func=lambda sku: f"{sku} - {sku_name_map.get(sku, '')}",
            )

        mode = st.radio(
            "Mode",
            ["thinking", "fast"],
            horizontal=True,
            help="Thinking mode uses graph-based context enrichment. Fast mode skips graph enrichment.",
        )
        profile = st.radio("Profile", ["Quality (recommended)", "Strict latency"], horizontal=True)
        defaults = profile_defaults(profile)
        execution_mode = st.radio("Execution", ["Standard", "Live trace"], horizontal=True)

        col_a, col_b = st.columns(2)
        temperature = col_a.slider("Temperature", min_value=0.0, max_value=1.0, value=float(defaults["temperature"]), step=0.05)
        agent_mode = col_b.selectbox("Agent Mode", options=["deterministic", "hybrid", "full"], index=0)

        agent_max_steps = st.slider("Agent max steps", min_value=1, max_value=12, value=4, step=1)

    with right:
        base_url = st.text_input("Ollama Base URL", value=base_cfg.get("ollama", {}).get("base_url", "http://localhost:11434"))
        ollama_ok, available_models, ollama_msg = get_ollama_models(base_url)
        if ollama_ok:
            status_pill(ollama_msg, "ok")
        else:
            status_pill(ollama_msg, "warn")

        if available_models:
            default_model = base_cfg.get("ollama", {}).get("model", available_models[0])
            default_idx = available_models.index(default_model) if default_model in available_models else 0
            model = st.selectbox("Ollama Model", options=available_models, index=default_idx)
        else:
            model = ""
            st.caption("No available models detected from Ollama.")

        fast_template_only = st.checkbox(
            "Full template fallback (optional in fast mode)",
            value=False,
            help="When enabled in fast mode, the run skips LLM explanation calls and uses deterministic templates only.",
        )

        st.caption("Scenario defaults (optional for later tab):")
        lead_time_default = st.number_input("lead_time_days default", min_value=1.0, max_value=90.0, value=7.0, step=1.0)
        safety_default = st.number_input("safety_stock default", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)

    preflight = run_preflight_checks(
        mode=mode,
        agent_mode=("deterministic" if mode == "fast" else agent_mode),
        base_url=base_url,
        model=model,
        records_count=len(records),
        api_key=str(base_cfg.get("ollama", {}).get("api_key", "")),
        planner_timeout_ms=0,
    )
    st.session_state["last_preflight"] = preflight

    st.markdown("#### Preflight")
    preflight_df = pd.DataFrame(preflight.get("checks", []))
    if preflight_df.empty:
        st.caption("No preflight checks available.")
    else:
        st.dataframe(preflight_df, width="stretch", hide_index=True)

    run_enabled = True
    blockers: list[str] = []
    if not records:
        run_enabled = False
        blockers.append("Dataset preview is empty or invalid.")

    if analysis_scope == "Single SKU" and not selected_sku_ids:
        run_enabled = False
        blockers.append("Single SKU scope requires one selected SKU.")
    if analysis_scope == "Custom SKU List" and not selected_sku_ids:
        run_enabled = False
        blockers.append("Custom SKU List scope requires at least one selected SKU.")

    if mode == "thinking":
        if not ollama_ok:
            run_enabled = False
            blockers.append("Thinking mode requires Ollama to be reachable.")
        if not available_models:
            run_enabled = False
            blockers.append("Thinking mode requires at least one installed model.")
    elif mode == "fast" and not fast_template_only:
        if not ollama_ok:
            run_enabled = False
            blockers.append("Fast mode without template-only requires Ollama to be reachable.")
        if not available_models:
            run_enabled = False
            blockers.append("Fast mode without template-only requires at least one installed model.")

    if not preflight.get("ok", False):
        run_enabled = False
        blockers.append(f"Preflight failed: {preflight.get('blocking_reason', 'unknown reason')}")

    if mode == "thinking" and agent_mode == "full" and model == "llama3.2:1b":
        st.warning(
            "`llama3.2:1b` is often too small for strict planner JSON in full-agent mode. "
            "Pick a larger model for reliable thinking+full runs."
        )

    _render_run_gate(run_enabled, blockers)
    run_clicked = st.button("Run Analysis", width="stretch", type="primary", disabled=not run_enabled)

    if not run_clicked:
        return

    try:
        input_snapshot = {
            "type": "mock" if data_mode == "Use mock dataset" else "upload",
            "filename": None if uploaded is None else uploaded.name,
            "bytes": b"" if uploaded is None else uploaded.getvalue(),
        }
    except Exception as exc:
        st.error(f"Could not read selected input: {exc}")
        return

    with st.spinner("Running analysis..."):
        with new_tempdir() as temp_dir:
            data_path = materialize_input_file(input_snapshot, temp_dir)
            settings = {
                "base_url": base_url,
                "model": model,
                "temperature": temperature,
                "mode": mode,
                "agent_mode": ("deterministic" if mode == "fast" else agent_mode),
                "agent_max_steps": agent_max_steps,
                "fast_template_only": fast_template_only,
                "planner_timeout_ms": 0,
                "lead_time_default": lead_time_default,
                "safety_default": safety_default,
                "profile": profile,
                "analysis_scope": analysis_scope,
                "analysis_sku_ids": selected_sku_ids,
                "execution_mode": execution_mode,
            }
            config = build_run_config(base_cfg, data_path, settings=settings)
            if execution_mode == "Live trace":
                st.session_state["live_trace_enabled"] = True
                st.session_state["live_trace_events"] = []
                st.session_state["live_trace_tool_logs"] = []
                st.session_state["live_trace_flow_events"] = []
                st.session_state["live_trace_llm_events"] = []

                progress_slot = st.empty()
                trace_slot = st.empty()
                payload = {}
                elapsed_ms = 0.0
                last_render = 0.0
                throttle_s = max(0.1, float(st.session_state.get("live_trace_refresh_ms", 400)) / 1000.0)

                for event in run_analysis_stream(config):
                    st.session_state["live_trace_events"].append(event)
                    if event.get("event_type") == "node_update":
                        st.session_state["live_trace_flow_events"].extend(event.get("new_flow_events", []))
                        st.session_state["live_trace_tool_logs"].extend(event.get("new_tool_logs", []))
                        st.session_state["live_trace_llm_events"].extend(event.get("new_llm_events", []))

                        now = time.perf_counter()
                        if now - last_render >= throttle_s:
                            progress_slot.info(
                                f"Node: {event.get('node')} | "
                                f"Agent step: {event.get('agent_step_count', 0)} | "
                                f"Tool calls: {len(st.session_state['live_trace_tool_logs'])}"
                            )
                            recent = pd.DataFrame(st.session_state["live_trace_tool_logs"][-12:])
                            with trace_slot.container():
                                st.markdown("#### Live Tool Calls")
                                if recent.empty:
                                    st.caption("No tool calls yet.")
                                else:
                                    st.dataframe(recent, width="stretch", hide_index=True)
                            last_render = now
                    elif event.get("event_type") == "run_complete":
                        payload = event.get("final_payload", {})
                        elapsed_ms = float(event.get("elapsed_ms", 0.0))
                        break
                progress_slot.success("Live trace run complete.")
            else:
                payload, elapsed_ms = run_once(config)

    st.session_state["last_payload"] = payload
    st.session_state["last_elapsed_ms"] = elapsed_ms
    st.session_state["last_run_settings"] = settings
    st.session_state["input_snapshot"] = input_snapshot
    add_history_entry(payload, elapsed_ms, label="standard")

    metadata = payload.get("metadata", {})
    warnings = metadata.get("warnings", [])
    fallback_used = any("fallback" in str(w).lower() for w in warnings)
    if fallback_used:
        st.warning("Run completed with fallback usage. See Diagnostics tab.")
    else:
        st.success("Run completed with active LLM path.")


def _tab_overview(payload: Dict[str, Any], elapsed_ms: float | None, settings: Dict[str, Any] | None) -> None:
    """Render minimal top-level overview."""
    summary = payload.get("summary", {})
    metadata = payload.get("metadata", {})
    warnings = metadata.get("warnings", [])
    fallback_used = any("fallback" in str(item).lower() for item in warnings)

    cols = st.columns(6)
    with cols[0]:
        stat_card("Critical", str(int(summary.get("critical_count", 0))))
    with cols[1]:
        stat_card("Watch", str(int(summary.get("watch_count", 0))))
    with cols[2]:
        stat_card("Healthy", str(int(summary.get("healthy_count", 0))))
    with cols[3]:
        stat_card("Overstock", str(int(summary.get("overstock_count", 0))))
    with cols[4]:
        stat_card("Runtime (ms)", str(int(elapsed_ms or 0)))
    with cols[5]:
        stat_card("Graph", str(metadata.get("graph_source", "unknown")))

    st.caption(
        f"Runtime: {int(elapsed_ms or 0)} ms | "
        f"Mode: {str(metadata.get('mode', (settings or {}).get('mode', 'unknown')))} | "
        f"Scope: {(settings or {}).get('analysis_scope', 'All SKUs')} | "
        f"LLM: {'fallback' if fallback_used else 'active'}"
    )

    frame = payload_to_df(payload)
    if frame.empty:
        st.info("No recommendations available.")
        return

    st.markdown("#### Top Actions")
    preview = frame[["sku_id", "name", "status", "days_of_stock", "reorder_qty", "recommended_action"]].copy()
    preview = preview.head(12)
    st.dataframe(preview, width="stretch", hide_index=True)


def _tab_priority_queue(payload: Dict[str, Any]) -> None:
    """Render focused priority queue for critical/watch SKUs."""
    frame = payload_to_df(payload)
    if frame.empty:
        st.info("No recommendations available.")
        return

    queue = frame[frame["status"].isin(["critical", "watch"])].copy()
    if queue.empty:
        st.success("No critical/watch SKUs in current run.")
        return

    queue = queue.sort_values(by=["priority_score", "reorder_urgency_days"], ascending=[False, True])
    st.dataframe(
        queue,
        width="stretch",
        hide_index=True,
        column_config={
            "priority_score": st.column_config.ProgressColumn(
                "Priority",
                min_value=0.0,
                max_value=max(float(queue["priority_score"].max()), 1.0),
                format="%.1f",
            ),
            "plain_english_explanation": st.column_config.TextColumn("Explanation", width="large"),
        },
    )

    st.markdown("#### Queue Actions")
    reviewed = st.multiselect("Mark reviewed (session only)", options=queue["sku_id"].tolist(), default=[])
    if reviewed:
        st.success(f"Marked as reviewed: {', '.join(reviewed)}")


def _tab_sku_explorer(payload: Dict[str, Any]) -> None:
    """Render full SKU explorer with filters and detail view."""
    frame = payload_to_df(payload)
    if frame.empty:
        st.info("No recommendations available.")
        return

    c1, c2, c3, c4 = st.columns(4)
    status_filter = c1.multiselect("Status", sorted(frame["status"].dropna().unique().tolist()))
    category_filter = c2.multiselect("Category", sorted(frame["category"].dropna().unique().tolist()))
    context_filter = c3.multiselect("Context", sorted(frame["context_source"].dropna().unique().tolist()))
    source_filter = c4.multiselect("Explanation Source", sorted(frame["llm_source"].dropna().unique().tolist()))
    search = st.text_input("Search SKU / name", value="")

    filtered = filter_df(frame, status_filter, category_filter, context_filter, source_filter, search)
    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        column_config={
            "days_of_stock": st.column_config.NumberColumn("Days of Stock", format="%.2f"),
            "reorder_qty": st.column_config.NumberColumn("Reorder Qty", format="%.2f"),
            "reorder_urgency_days": st.column_config.NumberColumn("Urgency Days", format="%.2f"),
            "priority_score": st.column_config.ProgressColumn(
                "Priority",
                min_value=0.0,
                max_value=max(float(filtered["priority_score"].max()) if not filtered.empty else 1.0, 1.0),
                format="%.1f",
            ),
            "plain_english_explanation": st.column_config.TextColumn("Explanation", width="large"),
        },
    )

    if filtered.empty:
        return

    selected = st.selectbox("Select SKU for details", options=filtered["sku_id"].tolist(), index=0)
    row = filtered[filtered["sku_id"] == selected].iloc[0]
    with st.container(border=True):
        st.markdown(f"**{row['sku_id']} - {row['name']}**")
        st.caption(
            f"{row['status_emoji']} {row['status']} | Category: {row['category']} | "
            f"Context: {row['context_source']} | Confidence: {row['confidence']} | Source: {row['llm_source']}"
        )
        st.write(
            f"Days of stock: {float(row['days_of_stock']):.2f} | "
            f"Reorder qty: {float(row['reorder_qty']):.2f} | "
            f"Urgency days: {float(row['reorder_urgency_days']):.2f}"
        )
        st.write(f"Action: {row['recommended_action']}")
        st.write(str(row["plain_english_explanation"]))


def _tab_scenario_lab(base_cfg: Dict[str, Any]) -> None:
    """Render what-if scenario controls and comparison output."""
    settings = st.session_state.get("last_run_settings")
    snapshot = st.session_state.get("input_snapshot")
    if not settings or not snapshot:
        st.info("Run at least one standard analysis in the Run tab before using Scenario Lab.")
        return

    st.markdown("#### Scenario Overrides")
    c1, c2, c3 = st.columns(3)
    lead_time = float(c1.number_input("lead_time_days", min_value=1.0, max_value=90.0, value=float(settings.get("lead_time_default", 7.0)), step=1.0))
    safety_stock = float(c2.number_input("safety_stock", min_value=0.0, max_value=1000.0, value=float(settings.get("safety_default", 0.0)), step=1.0))
    healthy_min = float(c3.number_input("healthy_dos_min", min_value=1.0, max_value=120.0, value=14.0, step=1.0))

    run_scenario = st.button("Run Scenario Comparison", type="primary")
    if not run_scenario:
        return

    overrides = {
        "lead_time_days": lead_time,
        "safety_stock": safety_stock,
        "healthy_dos_min": healthy_min,
    }

    with st.spinner("Running baseline and scenario..."):
        with new_tempdir() as temp_dir:
            data_path = materialize_input_file(snapshot, temp_dir)
            run_cfg = build_run_config(base_cfg, data_path, settings=settings)
            baseline_payload, baseline_ms, scenario_payload, scenario_ms = run_with_scenario(run_cfg, overrides)

    st.session_state["scenario_baseline_payload"] = baseline_payload
    st.session_state["scenario_payload"] = scenario_payload
    st.session_state["scenario_elapsed_ms"] = scenario_ms
    add_history_entry(scenario_payload, scenario_ms, label="scenario")

    base_summary = baseline_payload.get("summary", {})
    scen_summary = scenario_payload.get("summary", {})
    delta_cols = st.columns(4)
    for idx, field in enumerate(["critical_count", "watch_count", "healthy_count", "overstock_count"]):
        left = int(base_summary.get(field, 0))
        right = int(scen_summary.get(field, 0))
        delta_cols[idx].metric(field, right, delta=right - left)

    st.caption(f"Baseline runtime: {int(baseline_ms)} ms | Scenario runtime: {int(scenario_ms)} ms")

    base_df = payload_to_df(baseline_payload)
    scen_df = payload_to_df(scenario_payload)
    merged = base_df[["sku_id", "status", "reorder_qty", "reorder_urgency_days"]].merge(
        scen_df[["sku_id", "status", "reorder_qty", "reorder_urgency_days"]],
        on="sku_id",
        suffixes=("_base", "_scenario"),
    )
    merged["reorder_qty_delta"] = merged["reorder_qty_scenario"] - merged["reorder_qty_base"]
    merged["urgency_delta"] = merged["reorder_urgency_days_scenario"] - merged["reorder_urgency_days_base"]
    st.dataframe(merged, width="stretch", hide_index=True)

    st.download_button(
        "Download Scenario Comparison CSV",
        data=merged.to_csv(index=False),
        file_name=f"scenario_comparison_{now_file_suffix()}.csv",
        mime="text/csv",
    )


def _tab_diagnostics(payload: Dict[str, Any], settings: Dict[str, Any] | None) -> None:
    """Render diagnostics and internal metadata."""
    metadata = payload.get("metadata", {})
    warnings = metadata.get("warnings", [])
    errors = metadata.get("errors", [])

    st.write(f"Model: `{(settings or {}).get('model', metadata.get('llm_model', 'unknown'))}`")
    st.write(f"Mode: `{metadata.get('mode', 'unknown')}`")
    st.write(f"Graph source: `{metadata.get('graph_source', 'unknown')}`")
    st.write(f"Agent mode: `{metadata.get('agent_mode', 'unknown')}`")
    st.write(f"Agent steps executed: `{metadata.get('agent_steps_executed', 0)}`")
    st.write(f"Agent stop reason: `{metadata.get('agent_fallback_reason', '') or 'completed'}`")
    st.write(f"Graph used: `{metadata.get('graph_used', False)}`")
    st.write(f"Planner used: `{metadata.get('planner_used', False)}`")
    if settings:
        st.write(f"Analysis scope: `{settings.get('analysis_scope', 'All SKUs')}`")
        st.write(f"Selected SKUs: `{', '.join(settings.get('analysis_sku_ids', [])) or 'all'}`")

    flow_summary = summarize_flow(metadata)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tool calls", flow_summary["tool_calls_total"])
    c2.metric("Deterministic calls", flow_summary["tool_calls_deterministic"])
    c3.metric("Planner calls", flow_summary["tool_calls_planner"])
    c4.metric("Tool call failures", flow_summary["tool_calls_failed"])
    c5.metric("Duplicate suppressed", flow_summary.get("duplicate_suppressed", 0))

    full_mode_contract_ok = bool(metadata.get("full_mode_contract_ok", True))
    if str(metadata.get("mode", "")).lower() == "thinking" and str(metadata.get("agent_mode", "")).lower() == "full":
        if full_mode_contract_ok:
            st.success("Full-mode contract OK: no deterministic system tool calls.")
        else:
            st.error("Full-mode contract violation: deterministic system tool calls detected.")

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

    history = pd.DataFrame(metadata.get("agent_tool_history", []))
    if not history.empty:
        st.markdown("#### Tool History")
        st.dataframe(history, width="stretch", hide_index=True)

    flow_events = pd.DataFrame(metadata.get("flow_events", []))
    if not flow_events.empty:
        st.markdown("#### Flow Timeline")
        st.dataframe(flow_events, width="stretch", hide_index=True)

    tool_logs = pd.DataFrame(metadata.get("tool_call_logs", []))
    if not tool_logs.empty:
        st.markdown("#### Tool Call Logs")
        left, right, third = st.columns(3)
        caller_filter = left.multiselect("Caller", sorted(tool_logs["caller"].dropna().unique().tolist()), key="diag_caller")
        tool_filter = right.multiselect("Tool", sorted(tool_logs["tool_name"].dropna().unique().tolist()), key="diag_tool")
        status_filter = third.multiselect("Status", sorted(tool_logs["status"].dropna().unique().tolist()), key="diag_status")

        filtered_logs = tool_logs.copy()
        if caller_filter:
            filtered_logs = filtered_logs[filtered_logs["caller"].isin(caller_filter)]
        if tool_filter:
            filtered_logs = filtered_logs[filtered_logs["tool_name"].isin(tool_filter)]
        if status_filter:
            filtered_logs = filtered_logs[filtered_logs["status"].isin(status_filter)]
        st.dataframe(filtered_logs, width="stretch", hide_index=True)

    llm_events = pd.DataFrame(metadata.get("llm_batch_events", []))
    if not llm_events.empty:
        st.markdown("#### LLM Batch Events")
        st.dataframe(llm_events, width="stretch", hide_index=True)

    preflight = st.session_state.get("last_preflight")
    if isinstance(preflight, dict):
        st.markdown("#### Last Preflight Report")
        pre_df = pd.DataFrame(preflight.get("checks", []))
        if not pre_df.empty:
            st.dataframe(pre_df, width="stretch", hide_index=True)


def _tab_live_trace() -> None:
    """Render live trace buffers captured during streaming run."""
    if not st.session_state.get("live_trace_enabled", False):
        st.info("Use Run tab with Execution=Live trace to populate this view.")
        return

    events = st.session_state.get("live_trace_events", [])
    tool_logs = st.session_state.get("live_trace_tool_logs", [])
    flow_logs = st.session_state.get("live_trace_flow_events", [])
    llm_logs = st.session_state.get("live_trace_llm_events", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(events))
    c2.metric("Flow events", len(flow_logs))
    c3.metric("Tool calls", len(tool_logs))
    c4.metric("LLM batches", len(llm_logs))

    if events:
        fallback_reason = ""
        for event in reversed(events):
            reason = event.get("agent_fallback_reason", "")
            if reason:
                fallback_reason = str(reason)
                break
        if fallback_reason:
            st.warning(f"Planner stop reason: {fallback_reason}")
        else:
            st.success("Planner completed without explicit fallback reason.")

    st.markdown("#### Recent Flow Events")
    flow_df = pd.DataFrame(flow_logs)
    if flow_df.empty:
        st.caption("No flow events captured.")
    else:
        st.dataframe(flow_df.tail(30), width="stretch", hide_index=True)

    st.markdown("#### Recent Tool Calls")
    tool_df = pd.DataFrame(tool_logs)
    if tool_df.empty:
        st.caption("No tool calls captured.")
    else:
        left, right = st.columns(2)
        caller_filter = left.multiselect(
            "Caller filter",
            options=sorted(tool_df["caller"].dropna().unique().tolist()) if "caller" in tool_df.columns else [],
            key="live_trace_caller_filter",
        )
        status_filter = right.multiselect(
            "Status filter",
            options=sorted(tool_df["status"].dropna().unique().tolist()) if "status" in tool_df.columns else [],
            key="live_trace_status_filter",
        )
        if caller_filter and "caller" in tool_df.columns:
            tool_df = tool_df[tool_df["caller"].isin(caller_filter)]
        if status_filter and "status" in tool_df.columns:
            tool_df = tool_df[tool_df["status"].isin(status_filter)]
        st.dataframe(tool_df.tail(30), width="stretch", hide_index=True)

    st.markdown("#### Recent LLM Batch Events")
    llm_df = pd.DataFrame(llm_logs)
    if llm_df.empty:
        st.caption("No LLM batch events captured.")
    else:
        st.dataframe(llm_df.tail(30), width="stretch", hide_index=True)


def _tab_exports(payload: Dict[str, Any]) -> None:
    """Render export actions and in-session run history."""
    st.download_button(
        "Download Full JSON",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=f"inventory_run_{payload.get('run_id', 'unknown')}.json",
        mime="application/json",
    )

    frame = payload_to_df(payload)
    if not frame.empty:
        st.download_button(
            "Download Full Recommendations CSV",
            data=frame.to_csv(index=False),
            file_name=f"inventory_recommendations_{now_file_suffix()}.csv",
            mime="text/csv",
        )

    report_md = generate_report(
        payload,
        output_path=st.session_state.get("_exports_report_path")
        or Path("results") / f"{payload.get('run_id', 'unknown')}.json",
        disclaimer=DISCLAIMER,
    ).read_text(encoding="utf-8")
    st.session_state["last_report_markdown"] = report_md
    st.download_button(
        "Download Markdown Report",
        data=report_md,
        file_name=f"report_{now_file_suffix()}.md",
        mime="text/markdown",
    )

    st.markdown("#### Session Run History")
    history = pd.DataFrame(st.session_state.get("run_history", []))
    if history.empty:
        st.info("No runs recorded in this session yet.")
    else:
        st.dataframe(history, width="stretch", hide_index=True)

    st.markdown("#### Trace CSV Exports")
    tool_df = pd.DataFrame(payload.get("metadata", {}).get("tool_call_logs", []))
    flow_df = pd.DataFrame(payload.get("metadata", {}).get("flow_events", []))
    llm_df = pd.DataFrame(payload.get("metadata", {}).get("llm_batch_events", []))
    planner_df = tool_df[tool_df["caller"] == "planner_model"] if not tool_df.empty and "caller" in tool_df.columns else pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        if not flow_df.empty:
            st.download_button(
                "Download flow_timeline.csv",
                data=flow_df.to_csv(index=False),
                file_name="flow_timeline.csv",
                mime="text/csv",
            )
        if not llm_df.empty:
            st.download_button(
                "Download llm_batches.csv",
                data=llm_df.to_csv(index=False),
                file_name="llm_batches.csv",
                mime="text/csv",
            )
    with c2:
        if not tool_df.empty:
            st.download_button(
                "Download tool_calls.csv",
                data=tool_df.to_csv(index=False),
                file_name="tool_calls.csv",
                mime="text/csv",
            )
        if not planner_df.empty:
            st.download_button(
                "Download planner_calls.csv",
                data=planner_df.to_csv(index=False),
                file_name="planner_calls.csv",
                mime="text/csv",
            )

    preflight = st.session_state.get("last_preflight")
    if isinstance(preflight, dict):
        st.download_button(
            "Download preflight_report.json",
            data=json.dumps(preflight, indent=2, ensure_ascii=False),
            file_name="preflight_report.json",
            mime="application/json",
        )


def render_tabs(base_cfg: Dict[str, Any]) -> None:
    """Render tabbed UI and route content to each tab."""
    payload = st.session_state.get("last_payload")
    elapsed_ms = st.session_state.get("last_elapsed_ms")
    settings = st.session_state.get("last_run_settings")

    tabs = st.tabs(
        [
            "Run",
            "Overview",
            "Priority Queue",
            "SKU Explorer",
            "Scenario Lab",
            "Live Trace",
            "Diagnostics",
            "Exports",
        ]
    )

    with tabs[0]:
        _tab_run(base_cfg)

    payload = st.session_state.get("last_payload")
    elapsed_ms = st.session_state.get("last_elapsed_ms")
    settings = st.session_state.get("last_run_settings")

    if payload is None:
        for tab in tabs[1:]:
            with tab:
                st.info("Run an analysis in the Run tab to populate this section.")
        return

    with tabs[1]:
        _tab_overview(payload, elapsed_ms, settings)

    with tabs[2]:
        _tab_priority_queue(payload)

    with tabs[3]:
        _tab_sku_explorer(payload)

    with tabs[4]:
        _tab_scenario_lab(base_cfg)

    with tabs[5]:
        _tab_live_trace()

    with tabs[6]:
        _tab_diagnostics(payload, settings)

    with tabs[7]:
        _tab_exports(payload)
