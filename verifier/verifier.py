from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

from graph.state import AgentState

from .confidence import compute_confidence
from .evidence_checker import EvidenceCheckResult, check_evidence
from .schema_validator import validate_verification_payload


logger = get_logger(__name__)

SAFE_FAILURE_MESSAGE = "I couldn't generate a sufficiently supported answer from the available knowledge base."
CONFIDENCE_THRESHOLD = 0.65


@dataclass(frozen=True)
class SafeFailureResponse:
    classification: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    requires_human: bool
    reason: str
    validation_errors: list[str]

    @classmethod
    def create(cls, state: AgentState, validation_errors: list[str] | None = None) -> "SafeFailureResponse":
        sources = _normalize_sources(state.get("sources", []))
        return cls(
            classification="safe_failure",
            answer=SAFE_FAILURE_MESSAGE,
            sources=sources,
            confidence=0.0,
            requires_human=True,
            reason=SAFE_FAILURE_MESSAGE,
            validation_errors=list(validation_errors or []),
        )


@dataclass(frozen=True)
class VerificationEngine:
    evidence_check: EvidenceCheckResult
    schema_valid: bool
    schema_errors: list[str]
    confidence: float
    passed: bool
    reason: str


def verify_generated_response(state: AgentState) -> dict[str, Any]:
    logger.info("VERIFY")
    logger.info("Evidence Check")

    answer = state.get("answer", "")
    retrieved_docs_raw = [item for item in state.get("retrieved_docs", []) if isinstance(item, dict)]
    retrieved_cases_raw = [item for item in state.get("retrieved_cases", []) if isinstance(item, dict)]
    retrieved_docs = _normalize_sources(retrieved_docs_raw)
    retrieved_cases = _normalize_sources(retrieved_cases_raw)
    sources = _normalize_sources(state.get("sources", []))

    evidence_check = check_evidence(answer, retrieved_docs_raw, retrieved_cases_raw)
    schema_payload = {
        "classification": state.get("classification", "answerable"),
        "answer": answer,
        "sources": sources,
        "confidence": state.get("confidence", 0.0),
        "requires_human": bool(state.get("requires_human", False)),
        "reason": state.get("reason", ""),
    }
    schema_valid, schema_errors, _ = validate_verification_payload(schema_payload)

    evidence_supported = evidence_check.passed
    confidence = compute_confidence(retrieved_docs_raw, retrieved_cases_raw, evidence_supported, schema_valid, evidence_supported and schema_valid)
    passed = evidence_supported and schema_valid and confidence >= CONFIDENCE_THRESHOLD and bool(sources)
    verification_reason = _build_reason(passed, evidence_check, schema_errors, confidence)
    logger.info("Schema Check")
    logger.info("PASS" if schema_valid else "FAIL")
    logger.info("Confidence %.2f", confidence)
    logger.info("Retry %s", int(state.get("retry_count", 0)))

    if passed:
        return {
            "verification_passed": True,
            "verification_reason": verification_reason,
            "validation_errors": [],
            "confidence": confidence,
            "reason": verification_reason,
            "requires_human": bool(state.get("requires_human", False)),
            "sources": sources,
        }

    validation_errors = [*evidence_check.reasons, *schema_errors]
    safe_failure = SafeFailureResponse.create(state, validation_errors=validation_errors)
    return {
        "verification_passed": False,
        "verification_reason": verification_reason,
        "validation_errors": validation_errors,
        "confidence": confidence,
        "reason": verification_reason,
        "requires_human": True,
        "sources": sources,
        "answer": safe_failure.answer if int(state.get("retry_count", 0)) >= 1 else answer,
        "classification": state.get("classification", "answerable"),
    }


def _normalize_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        document = item.get("document") or item.get("document_name")
        chunk_id = item.get("chunk_id") or item.get("passage_id")
        if not document or not chunk_id:
            continue
        normalized.append(
            {
                "document": str(document),
                "chunk_id": str(chunk_id),
                "passage_id": str(item.get("passage_id") or chunk_id),
                "similarity_score": item.get("similarity_score"),
                "metadata": item.get("metadata", {}),
                "source_type": item.get("source_type", "knowledge_base"),
            }
        )
    return normalized


def _build_reason(passed: bool, evidence_check: EvidenceCheckResult, schema_errors: list[str], confidence: float) -> str:
    if passed:
        return f"Verification passed with confidence {confidence:.2f}."

    reasons: list[str] = []
    if evidence_check.unsupported_statements:
        reasons.append("Evidence check failed")
    if schema_errors:
        reasons.append("Schema validation failed")
    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append("Confidence below threshold")
    return "; ".join(reasons) or "Verification failed."