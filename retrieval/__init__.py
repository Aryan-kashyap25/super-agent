"""Retrieval engine for LocalMind AI."""

from .chunker import ChunkedDocument, chunk_document, chunk_documents
from .embedding import EmbeddingService, get_embedding_service
from .loader import KnowledgeBaseDocument, load_knowledge_base_documents
from .search import RetrievalEngine, SearchResult, build_retrieval_engine
from .vector_store import LocalFaissStore

__all__ = [
	"ChunkedDocument",
	"EmbeddingService",
	"KnowledgeBaseDocument",
	"LocalFaissStore",
	"RetrievalEngine",
	"SearchResult",
	"build_retrieval_engine",
	"chunk_document",
	"chunk_documents",
	"get_embedding_service",
	"load_knowledge_base_documents",
]
