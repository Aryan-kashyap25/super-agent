from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = (
    "You are a customer support assistant.\n"
    "Only answer using the supplied evidence.\n"
    "Never invent facts.\n"
    "Never use outside knowledge.\n"
    "If evidence is insufficient, clearly state that the available documentation does not contain enough information."
)

INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I couldn't find enough supporting information in the available knowledge base."
)


@dataclass(frozen=True)
class PromptInputs:
    question: str
    retrieved_documents: list[dict[str, Any]]
    retrieved_cases: list[dict[str, Any]]


def build_prompt(question: str, retrieved_documents: Iterable[Any], retrieved_cases: Iterable[Any]) -> str:
    document_items = [_format_item(item, "Document") for item in retrieved_documents]
    case_items = [_format_item(item, "Resolved Case") for item in retrieved_cases]

    document_section = _format_section("Retrieved Documents", document_items)
    case_section = _format_section("Retrieved Cases", case_items)

    return (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"Context:\n{document_section}\n\n{case_section}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Instructions:\n"
        "Answer the question concisely using only the context provided above.\n"
        "If the context explicitly addresses the topic (e.g. security policies, rules), use it to form your answer.\n"
        "Only if the context contains absolutely no information related to the question, reply exactly with:\n"
        f"{INSUFFICIENT_EVIDENCE_RESPONSE}\n\n"
        "Answer:\n"
    )


def _format_section(title: str, items: list[str]) -> str:
    if not items:
        body = "- None provided"
    else:
        body = "\n".join(f"- {item}" for item in items)
    return f"{title}:\n{body}"


def _format_item(item: Any, label: str) -> str:
    if isinstance(item, dict):
        name = item.get("document_name") or item.get("title") or item.get("chunk_id") or label
        text = item.get("chunk_text") or item.get("text") or item.get("summary") or ""
    else:
        name = getattr(item, "document_name", None) or getattr(item, "title", None) or getattr(item, "chunk_id", None) or label
        text = getattr(item, "chunk_text", None) or getattr(item, "text", None) or getattr(item, "summary", None) or ""

    return f"{name}: {str(text).strip()}"