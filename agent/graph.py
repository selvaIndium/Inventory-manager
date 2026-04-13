"""LangGraph state machine for Phase 1 execution flow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.apply_rules import apply_rules_node
from agent.nodes.calculate_metrics import calculate_metrics_node
from agent.nodes.enrich_context import enrich_context_node
from agent.nodes.explain_llm import explain_llm_node
from agent.nodes.format_output import format_output_node
from agent.nodes.generate_recs import generate_recs_node
from agent.nodes.load_data import load_data_node
from agent.nodes.template_explanation import template_explanation_node
from agent.nodes.validate_output import validate_output_node
from agent.state import AgentState


def build_graph():
    """Build and compile the Phase 1 LangGraph workflow."""
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("load_data", load_data_node)
    graph_builder.add_node("calculate_metrics", calculate_metrics_node)
    graph_builder.add_node("enrich_context", enrich_context_node)
    graph_builder.add_node("apply_rules", apply_rules_node)
    graph_builder.add_node("generate_recs", generate_recs_node)
    graph_builder.add_node("explain_llm", explain_llm_node)
    graph_builder.add_node("template_explanation", template_explanation_node)
    graph_builder.add_node("format_output", format_output_node)
    graph_builder.add_node("validate_output", validate_output_node)

    graph_builder.add_edge(START, "load_data")
    graph_builder.add_edge("load_data", "calculate_metrics")
    graph_builder.add_edge("calculate_metrics", "enrich_context")
    graph_builder.add_edge("enrich_context", "apply_rules")
    graph_builder.add_edge("apply_rules", "generate_recs")
    graph_builder.add_edge("generate_recs", "explain_llm")
    graph_builder.add_edge("explain_llm", "template_explanation")
    graph_builder.add_edge("template_explanation", "format_output")
    graph_builder.add_edge("format_output", "validate_output")
    graph_builder.add_edge("validate_output", END)

    return graph_builder.compile()
