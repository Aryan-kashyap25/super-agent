from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graph.graph import GraphDependencies, build_support_graph
from graph.state import AgentState
from verifier.schema_validator import validate_verification_payload
from verifier.verifier import SAFE_FAILURE_MESSAGE, verify_generated_response


def _supported_state() -> AgentState:
    return {
        "question": "What do the docs say about API credentials?",
        "classification": "answerable",
        "retrieved_docs": [
            {
                "document": "KB-001",
                "document_name": "KB-001",
                "chunk_id": "KB-001:0",
                "chunk_text": "API credential. A secret used by an external application.",
                "similarity_score": 0.96,
                "metadata": {},
                "source_type": "knowledge_base",
            }
        ],
        "retrieved_cases": [],
        "sources": [
            {
                "document": "KB-001",
                "chunk_id": "KB-001:0",
                "passage_id": "KB-001:0",
                "similarity_score": 0.96,
                "metadata": {},
                "source_type": "knowledge_base",
            }
        ],
        "answer": "API credentials are secrets used by external applications.",
        "reason": "Supported by retrieved evidence.",
        "confidence": 0.4,
        "requires_human": False,
        "retry_count": 0,
    }


def test_supported_answer_verifies_successfully():
    result = verify_generated_response(_supported_state())

    assert result["verification_passed"] is True
    assert result["confidence"] >= 0.0
    assert result["sources"]


def test_unsupported_answer_fails_verification():
    state = _supported_state()
    state["answer"] = "The feature automatically issues refunds to customers."

    result = verify_generated_response(state)

    assert result["verification_passed"] is False
    assert result["validation_errors"]


def test_missing_sources_fail_schema_validation():
    valid, errors, schema = validate_verification_payload(
        {
            "classification": "answerable",
            "answer": "Supported answer text.",
            "sources": [],
            "confidence": 0.8,
            "requires_human": False,
            "reason": "ok",
        }
    )

    assert valid is False
    assert errors
    assert schema is None


def test_schema_validation_rejects_malformed_output():
    valid, errors, schema = validate_verification_payload({"answer": "missing fields"})

    assert valid is False
    assert errors
    assert schema is None


@dataclass
class FakeRetrievalEngine:
    result: dict[str, Any]

    def search(self, question: str) -> dict[str, Any]:
        return self.result


def _fake_generator(question: str, retrieved_documents: list[dict[str, Any]], retrieved_cases: list[dict[str, Any]], llm: Any | None = None) -> dict[str, Any]:
    return {
        "answer": "This answer is unsupported by the evidence.",
        "model_name": "fake-model",
        "generation_time": 0.01,
        "token_count": 6,
        "prompt_used": question,
        "model_load_time": 0.0,
        "prompt_creation_time": 0.0,
        "total_response_time": 0.01,
    }


def _retrying_verifier_factory():
    call_count = {"value": 0}

    def _verifier(state: AgentState) -> dict[str, Any]:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "verification_passed": False,
                "verification_reason": "First pass failed.",
                "validation_errors": ["unsupported evidence"],
                "confidence": 0.2,
                "reason": "First pass failed.",
                "requires_human": True,
                "sources": state.get("sources", []),
            }
        return {
            "verification_passed": False,
            "verification_reason": "Second pass failed.",
            "validation_errors": ["unsupported evidence"],
            "confidence": 0.1,
            "reason": "Second pass failed.",
            "requires_human": True,
            "sources": state.get("sources", []),
        }

    return _verifier, call_count


def test_retry_path_and_safe_failure_path():
    verifier, call_count = _retrying_verifier_factory()
    graph = build_support_graph(
        GraphDependencies(
            retrieval_engine=FakeRetrievalEngine(
                result={
                    "documents": [
                        {
                            "document": "KB-1",
                            "document_name": "KB-1",
                            "chunk_id": "KB-1:0",
                            "chunk_text": "Evidence.",
                            "similarity_score": 0.9,
                            "metadata": {},
                            "source_type": "knowledge_base",
                        }
                    ],
                    "resolved_cases": [],
                    "search_latency_seconds": 0.0,
                    "index_size": 1,
                    "embedding_dimension": 8,
                }
            ),
            response_generator=_fake_generator,
            verification_engine=verifier,
        )
    )

    state = graph.invoke({"question": "What do the docs say about API credentials?"})

    assert call_count["value"] == 2
    assert state["retry_count"] == 1
    assert state["verification_passed"] is False
    assert state["answer"] == SAFE_FAILURE_MESSAGE
    assert "RETRY" in state["execution_path"]
    assert "SAFE_FAILURE" in state["execution_path"]