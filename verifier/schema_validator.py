from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class SourceReference(BaseModel):
    document: str = Field(min_length=1)
    chunk_id: str | None = None
    passage_id: str | None = None
    similarity_score: float | None = None

    @field_validator("chunk_id", "passage_id")
    @classmethod
    def _normalize_optional_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("document")
    @classmethod
    def _normalize_document(cls, value: str) -> str:
        return value.strip()


class VerificationSchema(BaseModel):
    classification: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    sources: list[SourceReference]
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool
    reason: str = Field(min_length=1)

    @field_validator("answer", "reason", "classification")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("sources")
    @classmethod
    def _reject_empty_sources(cls, value: list[SourceReference]) -> list[SourceReference]:
        if not value:
            raise ValueError("sources must not be empty")
        return value


def validate_verification_payload(payload: dict[str, Any]) -> tuple[bool, list[str], VerificationSchema | None]:
    try:
        schema = VerificationSchema.model_validate(payload)
        return True, [], schema
    except ValidationError as exc:
        return False, [error["msg"] for error in exc.errors()], None