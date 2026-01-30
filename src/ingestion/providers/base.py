from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        pass
    