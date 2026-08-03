from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from .state import AgentState


ROUTE_TARGETS = {
    "answerable": "retrieval",
    "clarification": "clarification",
    "escalation": "escalation",
    "out_of_scope": "out_of_scope",
}


def route_by_classification(state: AgentState) -> str:
    classification = state.get("classification", "clarification")
    return classification if classification in ROUTE_TARGETS else "clarification"


def wire_graph_edges(builder: StateGraph[Any]) -> None:
    builder.add_conditional_edges("triage", route_by_classification, ROUTE_TARGETS)
    builder.add_edge("retrieval", "generation")
    builder.add_edge("generation", END)
    builder.add_edge("clarification", END)
    builder.add_edge("escalation", END)
    builder.add_edge("out_of_scope", END)