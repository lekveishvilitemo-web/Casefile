"""Embedding models.

Production uses FastEmbed (ONNX MiniLM) so the vector index does not need a
paid embedding API. Tests inject a deterministic hashing embedder.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Unit-test embedder. Similar strings are not near each other; that is fine.

    Retrieval tests that need semantic neighbors should stub the vector store.
    """

    def __init__(self, dim: int = 32, model_name: str = "hashing-test") -> None:
        self.dim = dim
        self.model_name = model_name

    def _vector(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FastEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self.dim = 384

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
            probe = list(self._model.embed(["dimension probe"]))
            self.dim = int(len(probe[0]))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for vector in model.embed(batch):
                vectors.append(_as_list(vector))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _as_list(vector) -> list[float]:
    return [float(value) for value in vector]
