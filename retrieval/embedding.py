from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbeddingService:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    _text_cache: dict[str, np.ndarray] | None = None

    @property
    def text_cache(self) -> dict[str, np.ndarray]:
        if self._text_cache is None:
            self._text_cache = {}
        return self._text_cache

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model(self.model_name)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        uncached_texts = [text for text in texts if text not in self.text_cache]
        if uncached_texts:
            encoded = self.model.encode(
                uncached_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for text, vector in zip(uncached_texts, encoded, strict=False):
                self.text_cache[text] = np.asarray(vector, dtype=np.float32)

        stacked = [self.text_cache[text] for text in texts]
        return np.vstack(stacked).astype(np.float32, copy=False)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_embedding_service(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingService:
    return EmbeddingService(model_name=model_name)