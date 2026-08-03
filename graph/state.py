from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    classification: str
    retrieved_docs: list[dict[str, Any]]
    retrieved_cases: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    requires_human: bool
    reason: str
    logs: list[str]
    execution_path: list[str]
    metadata: dict[str, Any]
    generation_metadata: dict[str, Any]