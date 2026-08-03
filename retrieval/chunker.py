from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .loader import KnowledgeBaseDocument


@dataclass(frozen=True)
class ChunkedDocument:
    document_name: str
    title: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "knowledge_base"


def chunk_document(
    document: KnowledgeBaseDocument,
    chunk_size_words: int = 500,
    overlap_words: int = 100,
) -> list[ChunkedDocument]:
    words = document.content.split()
    if not words:
        return []

    step = max(1, chunk_size_words - overlap_words)
    chunks: list[ChunkedDocument] = []

    for chunk_index, start in enumerate(range(0, len(words), step)):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words).strip()
        chunk_metadata = {
            **document.metadata,
            "source_path": str(document.file_path),
            "chunk_index": chunk_index,
            "word_start": start,
            "word_end": end,
        }
        chunks.append(
            ChunkedDocument(
                document_name=document.document_name,
                title=document.title,
                chunk_id=f"{document.document_name}:{chunk_index}",
                text=chunk_text,
                metadata=chunk_metadata,
            )
        )
        if end >= len(words):
            break

    return chunks


def chunk_documents(documents: Iterable[KnowledgeBaseDocument], chunk_size_words: int = 500, overlap_words: int = 100) -> list[ChunkedDocument]:
    chunks: list[ChunkedDocument] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size_words=chunk_size_words, overlap_words=overlap_words))
    return chunks