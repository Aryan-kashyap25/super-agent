from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import DATA_DIR, OUTPUT_DIR
from utils.helpers import ensure_directory
from utils.logger import get_logger

from .chunker import ChunkedDocument, chunk_documents
from .embedding import EmbeddingService, get_embedding_service
from .loader import KnowledgeBaseDocument, load_knowledge_base_documents
from .vector_store import LocalFaissStore, SearchHit


logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    document_name: str
    chunk_text: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "knowledge_base"


@dataclass
class RetrievalEngine:
    embedding_service: EmbeddingService
    store: LocalFaissStore
    documents: list[KnowledgeBaseDocument]
    chunks: list[ChunkedDocument]
    resolved_cases: list[dict[str, Any]]
    document_load_seconds: float
    chunk_seconds: float
    embedding_seconds: float
    index_seconds: float

    @classmethod
    def build(cls, embedding_service: EmbeddingService | None = None) -> "RetrievalEngine":
        service = embedding_service or get_embedding_service()

        load_start = time.perf_counter()
        documents = load_knowledge_base_documents()
        resolved_cases = _load_resolved_cases()
        document_load_seconds = time.perf_counter() - load_start
        logger.info("Documents Loaded: %s knowledge base docs, %s resolved cases", len(documents), len(resolved_cases))

        chunk_start = time.perf_counter()
        chunks = chunk_documents(documents)
        chunks.extend(_chunk_resolved_cases(resolved_cases))
        chunk_seconds = time.perf_counter() - chunk_start
        logger.info("Chunks Created: %s", len(chunks))

        store = LocalFaissStore.create(service.dimension)
        embed_start = time.perf_counter()
        embeddings = service.embed_texts([chunk.text for chunk in chunks]) if chunks else service.embed_texts([])
        embedding_seconds = time.perf_counter() - embed_start

        index_start = time.perf_counter()
        if chunks:
            store.add_embeddings(chunks, embeddings)
        index_seconds = time.perf_counter() - index_start
        logger.info("Embeddings Generated: %s vectors", store.index.ntotal)
        logger.info("FAISS Index Created: size=%s dim=%s", store.index.ntotal, store.embedding_dimension)

        return cls(
            embedding_service=service,
            store=store,
            documents=documents,
            chunks=chunks,
            resolved_cases=resolved_cases,
            document_load_seconds=document_load_seconds,
            chunk_seconds=chunk_seconds,
            embedding_seconds=embedding_seconds,
            index_seconds=index_seconds,
        )

    @classmethod
    def load_or_build(cls, index_dir: Path | None = None, embedding_service: EmbeddingService | None = None) -> "RetrievalEngine":
        target_dir = index_dir or (OUTPUT_DIR / "retrieval")
        index_path = target_dir / "localmind.faiss"
        metadata_path = target_dir / "localmind_metadata.json"
        service = embedding_service or get_embedding_service()
        if index_path.exists() and metadata_path.exists():
            store = LocalFaissStore.load(target_dir)
            documents = load_knowledge_base_documents()
            resolved_cases = _load_resolved_cases()
            chunks = chunk_documents(documents)
            chunks.extend(_chunk_resolved_cases(resolved_cases))
            return cls(service, store, documents, chunks, resolved_cases, 0.0, 0.0, 0.0, 0.0)
        return cls.build(service)

    def save(self, directory: Path | None = None) -> tuple[Path, Path]:
        return self.store.save(directory)

    def search(self, question: str, top_k_documents: int = 5, top_k_cases: int = 5) -> dict[str, Any]:
        start = time.perf_counter()
        query_embedding = self.embedding_service.embed_text(question)
        hits = self.store.search(query_embedding, top_k=self.store.index.ntotal)

        document_hits = [hit for hit in hits if hit.source_type == "knowledge_base"][:top_k_documents]
        case_hits = [hit for hit in hits if hit.source_type == "resolved_case"][:top_k_cases]
        search_latency = time.perf_counter() - start
        logger.info("Search Results: %s document hits, %s case hits in %.4fs", len(document_hits), len(case_hits), search_latency)

        return {
            "question": question,
            "search_latency_seconds": search_latency,
            "documents": [_to_result(hit) for hit in document_hits],
            "resolved_cases": [_to_result(hit) for hit in case_hits],
            "index_size": self.store.index.ntotal,
            "embedding_dimension": self.store.embedding_dimension,
        }


def build_retrieval_engine(embedding_service: EmbeddingService | None = None) -> RetrievalEngine:
    engine = RetrievalEngine.build(embedding_service=embedding_service)
    ensure_directory(OUTPUT_DIR / "retrieval")
    engine.save(OUTPUT_DIR / "retrieval")
    return engine


def _load_resolved_cases() -> list[dict[str, Any]]:
    resolved_cases_path = DATA_DIR / "resolved_cases.json"
    if not resolved_cases_path.exists():
        return []

    raw_text = resolved_cases_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw_text:
        return []

    payload = json.loads(raw_text)
    if isinstance(payload, list):
        return [case for case in payload if isinstance(case, dict)]
    if isinstance(payload, dict):
        cases = payload.get("cases")
        if isinstance(cases, list):
            return [case for case in cases if isinstance(case, dict)]
    return []


def _chunk_resolved_cases(resolved_cases: list[dict[str, Any]]) -> list[ChunkedDocument]:
    chunks: list[ChunkedDocument] = []
    for index, case in enumerate(resolved_cases):
        case_id = str(case.get("case_id") or case.get("id") or f"resolved-case-{index}")
        document_name = str(case.get("document_name") or case.get("title") or case_id)
        text_parts = [
            str(case.get("title") or ""),
            str(case.get("summary") or case.get("resolution") or ""),
            str(case.get("evidence") or case.get("notes") or ""),
        ]
        text = "\n".join(part for part in text_parts if part).strip()
        if not text:
            continue
        metadata = {**case, "case_id": case_id, "source_type": "resolved_case"}
        chunks.append(
            ChunkedDocument(
                document_name=document_name,
                title=str(case.get("title") or document_name),
                chunk_id=case_id,
                text=text,
                metadata=metadata,
                source_type="resolved_case",
            )
        )
    return chunks


def _to_result(hit: SearchHit) -> SearchResult:
    return SearchResult(
        document_name=hit.document_name,
        chunk_text=hit.text,
        similarity_score=hit.similarity_score,
        metadata=hit.metadata,
        source_type=hit.source_type,
    )