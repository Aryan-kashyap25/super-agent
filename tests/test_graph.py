from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graph.graph import GraphDependencies, build_support_graph
from graph.state import AgentState


@dataclass
class FakeRetrievalEngine:
    result: dict[str, Any]

    def search(self, question: str) -> dict[str, Any]:
        return self.result


def _fake_generator(question: str, retrieved_documents: list[dict[str, Any]], retrieved_cases: list[dict[str, Any]], llm: Any | None = None) -> dict[str, Any]:
    return {
        "answer": "Generated answer from retrieved evidence.",
        "model_name": "fake-model",
        "generation_time": 0.01,
        "token_count": 12,
        "prompt_used": f"question={question}",
        "model_load_time": 0.0,
        "prompt_creation_time": 0.0,
        "total_response_time": 0.01,
    }


def test_answerable_route_executes_retrieval_and_generation():
    graph = build_support_graph(
        GraphDependencies(
            retrieval_engine=FakeRetrievalEngine(
                result={
                        "documents": [{"document": "KB-1", "document_name": "KB-1", "chunk_id": "KB-1:0", "chunk_text": "Relevant evidence.", "similarity_score": 0.92, "metadata": {}, "source_type": "knowledge_base"}],
                    "resolved_cases": [],
                    "search_latency_seconds": 0.01,
                    "index_size": 1,
                    "embedding_dimension": 8,
                }
            ),
            response_generator=_fake_generator,
        )
    )

    state = graph.invoke({"question": "What do the docs say about API credentials?"})

    assert state["classification"] == "answerable"
    assert "RETRIEVAL" in state["execution_path"]
    assert "GENERATION" in state["execution_path"]
    assert state["sources"]


def test_clarification_route_executes_clarification_node():
    graph = build_support_graph(
        GraphDependencies(
            retrieval_engine=FakeRetrievalEngine(result={"documents": [], "resolved_cases": [], "search_latency_seconds": 0.0, "index_size": 0, "embedding_dimension": 8}),
            response_generator=_fake_generator,
        )
    )

    state = graph.invoke({"question": "Can you help me?"})

    assert state["classification"] == "clarification"
    assert "CLARIFICATION" in state["execution_path"]
    assert "RETRIEVAL" not in state["execution_path"]
    assert "GENERATION" not in state["execution_path"]


def test_escalation_route_executes_escalation_node():
    graph = build_support_graph(
        GraphDependencies(
            retrieval_engine=FakeRetrievalEngine(result={"documents": [], "resolved_cases": [], "search_latency_seconds": 0.0, "index_size": 0, "embedding_dimension": 8}),
            response_generator=_fake_generator,
        )
    )

    state = graph.invoke({"question": "Please create API credentials for my workspace."})

    assert state["classification"] == "escalation"
    assert state["requires_human"] is True
    assert "ESCALATION" in state["execution_path"]
    assert "RETRIEVAL" not in state["execution_path"]


def test_out_of_scope_route_executes_safe_terminal_node():
    graph = build_support_graph(
        GraphDependencies(
            retrieval_engine=FakeRetrievalEngine(result={"documents": [], "resolved_cases": [], "search_latency_seconds": 0.0, "index_size": 0, "embedding_dimension": 8}),
            response_generator=_fake_generator,
        )
    )

    state = graph.invoke({"question": "What is the weather tomorrow?"})

    assert state["classification"] == "out_of_scope"
    assert state["requires_human"] is False
    assert "OUT_OF_SCOPE" in state["execution_path"]
    assert "RETRIEVAL" not in state["execution_path"]