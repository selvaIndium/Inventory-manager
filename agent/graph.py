"""LangGraph state machine for Phase 1 execution flow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.apply_rules import apply_rules_node
from agent.nodes.calculate_metrics import calculate_metrics_node
from agent.nodes.enrich_context import enrich_context_node
from agent.nodes.execute_action import execute_action_node
from agent.nodes.explain_llm import explain_llm_node
from agent.nodes.format_output import format_output_node
from agent.nodes.generate_recs import generate_recs_node
from agent.nodes.load_data import load_data_node
from agent.nodes.planner_action import planner_action_node
from agent.nodes.template_explanation import template_explanation_node
from agent.nodes.validate_output import validate_output_node
from agent.state import AgentState


def mode_router_node(state: AgentState) -> AgentState:
    """No-op routing node used for mode-based graph branching."""
    state["current_node"] = "mode_router"
    return state


def _route_from_mode(state: AgentState) -> str:
    """Route execution start based on mode and agent_mode."""
    mode = str(state["config"].get("mode", "thinking")).lower()
    agent_mode = str(state["config"].get("agent_mode", "deterministic")).lower()

    if mode == "thinking" and agent_mode == "full":
        return "planner_action"
    return "load_data"


def _route_after_metrics(state: AgentState) -> str:
    """Route metrics stage to graph enrichment or directly to rule application."""
    mode = str(state["config"].get("mode", "thinking")).lower()
    return "apply_rules" if mode == "fast" else "enrich_context"


def _route_after_generate(state: AgentState) -> str:
    """Route to deterministic LLM path or hybrid planner loop."""
    mode = str(state["config"].get("mode", "thinking")).lower()
    if mode == "fast":
        return "explain_llm"

    agent_mode = str(state["config"].get("agent_mode", "deterministic"))
    if agent_mode in {"hybrid", "full"} and not bool(state.get("agent_done", False)):
        return "planner_action"
    return "explain_llm"


def _route_after_planner(state: AgentState) -> str:
    """Route planner output to executor or next deterministic step."""
    action = state.get("agent_pending_action") or {}
    mode = str(state["config"].get("mode", "thinking")).lower()
    agent_mode = str(state["config"].get("agent_mode", "deterministic")).lower()
    if bool(action.get("done", False)) or state.get("agent_done", False):
        if mode == "thinking" and agent_mode == "full":
            return "generate_recs"
        return "explain_llm"
    return "execute_action"


def _route_after_executor(state: AgentState) -> str:
    """Continue loop until done or step cap, then return to LLM explanation."""
    if state.get("agent_done", False):
        return "explain_llm"
    return "planner_action"


def build_graph():
    """Build and compile the Phase 1 LangGraph workflow."""
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("mode_router", mode_router_node)
    graph_builder.add_node("load_data", load_data_node)
    graph_builder.add_node("calculate_metrics", calculate_metrics_node)
    graph_builder.add_node("enrich_context", enrich_context_node)
    graph_builder.add_node("apply_rules", apply_rules_node)
    graph_builder.add_node("generate_recs", generate_recs_node)
    graph_builder.add_node("planner_action", planner_action_node)
    graph_builder.add_node("execute_action", execute_action_node)
    graph_builder.add_node("explain_llm", explain_llm_node)
    graph_builder.add_node("template_explanation", template_explanation_node)
    graph_builder.add_node("format_output", format_output_node)
    graph_builder.add_node("validate_output", validate_output_node)

    graph_builder.add_edge(START, "mode_router")
    graph_builder.add_conditional_edges("mode_router", _route_from_mode)
    graph_builder.add_edge("load_data", "calculate_metrics")
    graph_builder.add_conditional_edges("calculate_metrics", _route_after_metrics)
    graph_builder.add_edge("enrich_context", "apply_rules")
    graph_builder.add_edge("apply_rules", "generate_recs")
    graph_builder.add_conditional_edges("generate_recs", _route_after_generate)
    graph_builder.add_conditional_edges("planner_action", _route_after_planner)
    graph_builder.add_conditional_edges("execute_action", _route_after_executor)
    graph_builder.add_edge("explain_llm", "template_explanation")
    graph_builder.add_edge("template_explanation", "format_output")
    graph_builder.add_edge("format_output", "validate_output")
    graph_builder.add_edge("validate_output", END)

    return graph_builder.compile()
