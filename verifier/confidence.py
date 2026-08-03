from __future__ import annotations

from typing import Any


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_confidence(
    retrieved_docs: list[dict[str, Any]],
    retrieved_cases: list[dict[str, Any]],
    evidence_supported: bool,
    schema_valid: bool,
    verification_passed: bool,
) -> float:
    sources = [*retrieved_docs, *retrieved_cases]
    if not sources:
        return 0.0

    similarity_scores = [float(item.get("similarity_score", 0.0) or 0.0) for item in sources]
    max_similarity = max(similarity_scores) if similarity_scores else 0.0
    support_ratio = 1.0 if evidence_supported else 0.0
    source_coverage = min(len(sources) / 5.0, 1.0)
    schema_bonus = 0.1 if schema_valid else -0.2
    verification_bonus = 0.15 if verification_passed else -0.3

    score = (0.45 * max_similarity) + (0.25 * support_ratio) + (0.15 * source_coverage) + schema_bonus + verification_bonus
    return clamp_confidence(score)