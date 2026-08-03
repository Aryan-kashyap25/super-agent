from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BasePayload(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SupportQuestion(BasePayload):
    question: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolvedCase(BasePayload):
    case_id: str
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)
