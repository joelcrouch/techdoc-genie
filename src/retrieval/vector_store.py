# from typing import List, Tuple, Optional
# from pathlib import Path
# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# from ..ingestion.embedder import DocumentEmbedder
# from ..utils.logger import setup_logger
# from ..utils.config import get_settings

# logger = setup_logger(__name__)

# class VectorStore:
#     def __init__(
#         self,
#         persist_path: str | None = None,
#         embedder: DocumentEmbedder | None = None,
#     ):
#         settings = get_settings()
#         self.persist_path = Path(persist_path or settings.vector_store_path)
#         self.persist_path.mkdir(parents=True, exist_ok=True)

#         self.embedder = embedder or DocumentEmbedder(provider="gemini")
#         self.vectorstore = None

#     def create_from_documents(self, documents: List[Document]) -> None:
#         logger.info(f"Creating vector store from {len(documents)} documents")

#         self.vectorstore = FAISS.from_documents(
#             documents=documents,
#             embedding=self.embedder.provider,  # 👈 interface only
#         )


#src/retrieval/vector_store.py
from typing import List, Tuple, Optional
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from ..utils.logger import setup_logger
from ..utils.config import get_settings
from src.ingestion.providers.base import BaseEmbedder

logger = setup_logger(__name__)


class LangChainEmbeddingAdapter(Embeddings):
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)


class VectorStore:
    """Manage vector storage and retrieval."""

    def __init__(
        self,
        persist_path: str | None = None,
        embedder: BaseEmbedder | None = None,
    ):
        if embedder is None:
            raise ValueError("You must provide an embedder implementing BaseEmbedder")

        settings = get_settings()
        self.persist_path = Path(persist_path or settings.vector_store_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder
        self.embedding = LangChainEmbeddingAdapter(embedder)
        self.vectorstore: Optional[FAISS] = None

    def create_from_documents(self, documents: List[Document]) -> None:
        logger.info(f"Creating vector store from {len(documents)} documents")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding,
        )

        logger.info("✅ Vector store created successfully")

    def save(self, name: str = "index") -> None:
        if not self.vectorstore:
            raise ValueError("No vector store to save")
        save_path = self.persist_path / name
        self.vectorstore.save_local(str(save_path))
        logger.info(f"Vector store saved to {save_path}")

    
    def load(self, name: str = "index") -> None:
        load_path = self.persist_path / name
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        self.vectorstore = FAISS.load_local(
            str(load_path),
            embeddings=self.embedding,  # Changed from 'embedding' to 'embeddings'
            # allow_dangerous_deserialization=True,
        )
        logger.info(f"Vector store loaded from {load_path}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.similarity_search_with_score(query, k=k)
