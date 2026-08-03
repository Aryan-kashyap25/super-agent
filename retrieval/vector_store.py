from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from config import OUTPUT_DIR
from .chunker import ChunkedDocument
from .embedding import EmbeddingService


@dataclass(frozen=True)
class SearchHit:
    document_name: str
    chunk_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "knowledge_base"


@dataclass
class LocalFaissStore:
    index: faiss.Index
    records: list[dict[str, Any]]
    embedding_dimension: int

    @classmethod
    def create(cls, embedding_dimension: int) -> "LocalFaissStore":
        index = faiss.IndexFlatIP(embedding_dimension)
        return cls(index=index, records=[], embedding_dimension=embedding_dimension)

    @classmethod
    def build(cls, chunks: Iterable[ChunkedDocument], embedding_service: EmbeddingService) -> "LocalFaissStore":
        chunk_list = list(chunks)
        store = cls.create(embedding_service.dimension)
        if not chunk_list:
            return store

        texts = [chunk.text for chunk in chunk_list]
        embeddings = embedding_service.embed_texts(texts)
        store.add_embeddings(chunk_list, embeddings)
        return store

    def add_embeddings(self, chunks: Sequence[ChunkedDocument], embeddings: np.ndarray) -> None:
        if not len(chunks):
            return
        normalized_embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(normalized_embeddings)
        self.index.add(normalized_embeddings)
        for chunk in chunks:
            self.records.append(asdict(chunk))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indexes = self.index.search(query, top_k)

        hits: list[SearchHit] = []
        for score, record_index in zip(scores[0], indexes[0], strict=False):
            if record_index < 0:
                continue
            record = self.records[int(record_index)]
            hits.append(
                SearchHit(
                    document_name=record["document_name"],
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    similarity_score=float(score),
                    metadata=dict(record.get("metadata", {})),
                    source_type=record.get("source_type", "knowledge_base"),
                )
            )
        return hits

    def save(self, directory: Path | None = None) -> tuple[Path, Path]:
        target_dir = directory or (OUTPUT_DIR / "retrieval")
        target_dir.mkdir(parents=True, exist_ok=True)
        index_path = target_dir / "localmind.faiss"
        metadata_path = target_dir / "localmind_metadata.json"
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps({"embedding_dimension": self.embedding_dimension, "records": self.records}, indent=2),
            encoding="utf-8",
        )
        return index_path, metadata_path

    @classmethod
    def load(cls, directory: Path | None = None) -> "LocalFaissStore":
        target_dir = directory or (OUTPUT_DIR / "retrieval")
        index_path = target_dir / "localmind.faiss"
        metadata_path = target_dir / "localmind_metadata.json"
        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(index=index, records=list(metadata.get("records", [])), embedding_dimension=int(metadata["embedding_dimension"]))