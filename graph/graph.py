from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import START, StateGraph

from llm.generator import generate_response
from llm.loader import LocalLLM, load_local_llm
from retrieval.search import RetrievalEngine, build_retrieval_engine
from verifier.verifier import verify_generated_response

from .edges import wire_graph_edges
from .router import clarification_node, escalation_node, generation_node, out_of_scope_node, pass_node, retry_node, retrieval_node, safe_failure_node, triage_node, verification_node
from .state import AgentState


ResponseGeneratorFn = Callable[[str, list[dict[str, Any]], list[dict[str, Any]], LocalLLM | None], dict[str, Any]]
VerificationFn = Callable[[AgentState], dict[str, Any]]


@dataclass(frozen=True)
class GraphDependencies:
    retrieval_engine: RetrievalEngine
    response_generator: ResponseGeneratorFn
    verification_engine: VerificationFn = verify_generated_response
    llm: LocalLLM | None = None


def build_support_graph(dependencies: GraphDependencies | None = None):
    resolved_dependencies = dependencies or _build_default_dependencies()
    builder: StateGraph[AgentState] = StateGraph(AgentState)

    builder.add_node("triage", triage_node)
    builder.add_node("retrieval", lambda state: retrieval_node(state, resolved_dependencies.retrieval_engine))
    builder.add_node("generation", lambda state: generation_node(state, lambda question, docs, cases: resolved_dependencies.response_generator(question, docs, cases, resolved_dependencies.llm)))
    builder.add_node("verification", lambda state: verification_node(state, resolved_dependencies.verification_engine))
    builder.add_node("retry", retry_node)
    builder.add_node("pass", pass_node)
    builder.add_node("safe_failure", safe_failure_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("escalation", escalation_node)
    builder.add_node("out_of_scope", out_of_scope_node)

    builder.add_edge(START, "triage")
    wire_graph_edges(builder)
    return builder.compile()


def _build_default_dependencies() -> GraphDependencies:
    retrieval_engine = build_retrieval_engine()
    llm = load_local_llm()

    def _response_generator(question: str, retrieved_documents: list[dict[str, Any]], retrieved_cases: list[dict[str, Any]], llm_override: LocalLLM | None = None) -> dict[str, Any]:
        active_llm = llm_override or llm
        return generate_response(question, retrieved_documents, retrieved_cases, llm=active_llm)

    return GraphDependencies(
        retrieval_engine=retrieval_engine,
        response_generator=_response_generator,
        verification_engine=verify_generated_response,
        llm=llm,
    )