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

VERIFICATION_TARGETS = {
    "pass": "pass",
    "retry": "retry",
    "safe_failure": "safe_failure",
}


def route_by_classification(state: AgentState) -> str:
    classification = state.get("classification", "clarification")
    return classification if classification in ROUTE_TARGETS else "clarification"


def route_by_verification(state: AgentState) -> str:
    if state.get("verification_passed"):
        return "pass"
    retry_count = int(state.get("retry_count", 0))
    if retry_count < 1:
        return "retry"
    return "safe_failure"


def wire_graph_edges(builder: StateGraph[Any]) -> None:
    builder.add_conditional_edges("triage", route_by_classification, ROUTE_TARGETS)
    builder.add_edge("retrieval", "generation")
    builder.add_edge("generation", "verification")
    builder.add_conditional_edges("verification", route_by_verification, VERIFICATION_TARGETS)
    builder.add_edge("retry", "generation")
    builder.add_edge("pass", END)
    builder.add_edge("clarification", END)
    builder.add_edge("escalation", END)
    builder.add_edge("out_of_scope", END)
    builder.add_edge("safe_failure", END)