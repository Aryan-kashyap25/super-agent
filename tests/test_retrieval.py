from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from retrieval.chunker import chunk_document, chunk_documents
from retrieval.embedding import EmbeddingService
from retrieval.loader import KnowledgeBaseDocument, load_knowledge_base_documents
from retrieval.search import RetrievalEngine


@dataclass
class FakeEmbeddingModel:
    dimension: int = 8

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            seed = sum(ord(char) for char in text)
            vector = np.array([(seed + offset) % 97 for offset in range(self.dimension)], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vector) or 1.0
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


def test_documents_load_correctly():
    documents = load_knowledge_base_documents()
    assert len(documents) >= 10
    assert all(document.file_path.suffix == ".md" for document in documents)
    assert all(document.title for document in documents)


def test_chunking_works():
    document = KnowledgeBaseDocument(
        document_name="sample",
        file_path=Path("sample.md"),
        title="Sample",
        content=" ".join(f"word{i}" for i in range(650)),
        metadata={"title": "Sample"},
    )
    chunks = chunk_document(document, chunk_size_words=500, overlap_words=100)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "sample:0"
    assert chunks[0].metadata["chunk_index"] == 0


def test_embeddings_generate_and_search_returns_results(monkeypatch):
    fake_model = FakeEmbeddingModel()
    monkeypatch.setattr("retrieval.embedding.get_embedding_model", lambda model_name="sentence-transformers/all-MiniLM-L6-v2": fake_model)
    service = EmbeddingService(model_name="test-model")

    engine = RetrievalEngine.build(embedding_service=service)
    results = engine.search("How do I rotate a secret?", top_k_documents=3, top_k_cases=3)

    assert results["embedding_dimension"] == fake_model.dimension
    assert results["index_size"] > 0
    assert results["documents"]


def test_resolved_cases_are_searchable(monkeypatch, tmp_path):
    fake_model = FakeEmbeddingModel()
    monkeypatch.setattr("retrieval.embedding.get_embedding_model", lambda model_name="sentence-transformers/all-MiniLM-L6-v2": fake_model)
    service = EmbeddingService(model_name="test-model")

    temp_data_dir = tmp_path
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    (temp_data_dir / "resolved_cases.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "CASE-001",
                    "title": "Rotating secrets for integrations",
                    "summary": "Rotate exposed API secrets and notify the owning workspace admin.",
                    "evidence": "Case notes recommend revocation followed by re-issuing a credential.",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("retrieval.search.DATA_DIR", temp_data_dir)

    engine = RetrievalEngine.build(embedding_service=service)
    results = engine.search("Rotate exposed API secrets and notify the owning workspace admin.", top_k_documents=5, top_k_cases=5)

    assert "resolved_cases" in results
    assert isinstance(results["resolved_cases"], list)
    assert results["resolved_cases"]
    assert results["resolved_cases"][0].source_type == "resolved_case"


def test_loader_returns_metadata_for_all_kb_documents():
    documents = load_knowledge_base_documents()
    assert all("file_name" in document.metadata for document in documents)
    assert all("document_title" in document.metadata for document in documents)