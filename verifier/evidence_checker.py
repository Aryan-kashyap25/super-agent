from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import re


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "about",
    "have",
    "will",
    "should",
    "can",
    "may",
    "must",
    "user",
    "users",
    "answer",
    "question",
    "available",
    "knowledge",
    "base",
}


@dataclass(frozen=True)
class EvidenceCheckResult:
    passed: bool
    supported_statements: list[str]
    unsupported_statements: list[str]
    support_ratio: float
    evidence_text: str
    reasons: list[str] = field(default_factory=list)


def check_evidence(answer: str, retrieved_docs: list[dict[str, Any]], retrieved_cases: list[dict[str, Any]]) -> EvidenceCheckResult:
    evidence_text = _build_evidence_text(retrieved_docs, retrieved_cases)
    if not evidence_text.strip():
        return EvidenceCheckResult(
            passed=False,
            supported_statements=[],
            unsupported_statements=[answer] if answer.strip() else ["empty answer"],
            support_ratio=0.0,
            evidence_text=evidence_text,
            reasons=["No retrieved evidence was available."],
        )

    sentences = _split_sentences(answer)
    if not sentences:
        return EvidenceCheckResult(
            passed=False,
            supported_statements=[],
            unsupported_statements=["empty answer"],
            support_ratio=0.0,
            evidence_text=evidence_text,
            reasons=["The answer did not contain any verifiable statements."],
        )

    supported_statements: list[str] = []
    unsupported_statements: list[str] = []
    evidence_tokens = _token_set(evidence_text)

    for sentence in sentences:
        sentence_tokens = _token_set(sentence)
        if not sentence_tokens:
            continue
        if sentence_tokens & evidence_tokens:
            supported_statements.append(sentence)
        else:
            unsupported_statements.append(sentence)

    support_ratio = len(supported_statements) / max(len(supported_statements) + len(unsupported_statements), 1)
    passed = bool(supported_statements) and not unsupported_statements
    reasons: list[str] = []
    if unsupported_statements:
        reasons.append("One or more answer statements were not supported by retrieved evidence.")
    return EvidenceCheckResult(
        passed=passed,
        supported_statements=supported_statements,
        unsupported_statements=unsupported_statements,
        support_ratio=support_ratio,
        evidence_text=evidence_text,
        reasons=reasons,
    )


def _build_evidence_text(retrieved_docs: list[dict[str, Any]], retrieved_cases: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in [*retrieved_docs, *retrieved_cases]:
        for key in ("chunk_text", "text", "summary", "resolution", "evidence", "notes"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        document = item.get("document") or item.get("document_name") or ""
        chunk_id = item.get("chunk_id") or item.get("passage_id") or ""
        if document:
            parts.append(str(document))
        if chunk_id:
            parts.append(str(chunk_id))
    return "\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _token_set(text: str) -> set[str]:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}