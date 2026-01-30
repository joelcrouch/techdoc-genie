# from typing import List
# from .base import BaseEmbedder

# class MockEmbedder(BaseEmbedder):
#     """A mock embedding provider for testing purposes."""

#     def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
#         # Return a zero-vector for each text
#         return [[0.0] * 8 for _ in texts]  # 8-dimensional dummy vector

#     def embed_query(self, query: str) -> List[float]:
#         # Return a zero-vector for query
#         return [0.0] * 8


# src/ingestion/providers/mock.py
from typing import List
import hashlib
import random

from .base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for tests.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]

    def embed_documents(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        # batch_size unused here, but required by interface
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_text(query)
