# src/ingestion/providers/huggingface.py
from typing import List
import os
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class HuggingFaceEmbeddingProvider(BaseEmbedder):
    """
    Hugging Face sentence-transformers embedding provider.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or os.getenv(
            "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_documents(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return [emb.astype(float).tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return embedding.astype(float).tolist()
