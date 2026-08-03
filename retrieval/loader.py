from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import KNOWLEDGE_BASE_DIR


@dataclass(frozen=True)
class KnowledgeBaseDocument:
    document_name: str
    file_path: Path
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_knowledge_base_documents(knowledge_base_dir: Path | None = None) -> list[KnowledgeBaseDocument]:
    base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
    if not base_dir.exists():
        return []

    documents: list[KnowledgeBaseDocument] = []
    for markdown_path in sorted(base_dir.glob("*.md")):
        documents.append(_load_markdown_document(markdown_path))
    return documents


def _load_markdown_document(markdown_path: Path) -> KnowledgeBaseDocument:
    raw_text = markdown_path.read_text(encoding="utf-8", errors="replace")
    metadata, content = _split_frontmatter(raw_text)
    title = _extract_title(content, metadata, markdown_path.stem)
    metadata = {**metadata, "file_name": markdown_path.name, "document_title": title}
    return KnowledgeBaseDocument(
        document_name=markdown_path.stem,
        file_path=markdown_path,
        title=title,
        content=content.strip(),
        metadata=metadata,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        return {}, text

    metadata_lines = lines[1:closing_index]
    body = "\n".join(lines[closing_index + 1 :])
    return _parse_metadata(metadata_lines), body


def _parse_metadata(lines: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value.strip())
    return metadata


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        items = [item.strip().strip('"\'') for item in value[1:-1].split(",") if item.strip()]
        return items
    return value.strip('"\'')


def _extract_title(content: str, metadata: dict[str, Any], fallback: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()

    return fallback.replace("_", " ").strip().title()