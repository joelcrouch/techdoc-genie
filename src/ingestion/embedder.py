from typing import List
from .providers.base import BaseEmbedder
from .providers.mock import MockEmbedder
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
class DocumentEmbedder:
    """Wrapper for embedding providers."""
    
    def __init__(self, provider: str = "mock", model: str = None):
        self.provider_name = provider.lower()
        
        if self.provider_name == "mock":
            self.embedder: BaseEmbedder = MockEmbedder()
            logger.info("Using MockEmbedder for testing")
        
        elif self.provider_name == "huggingface":
            from .providers.huggingface import HuggingFaceEmbedder
            self.embedder: BaseEmbedder = HuggingFaceEmbedder(model_name=model)
            logger.info(f"Using HuggingFaceEmbedder with model {model or 'all-MiniLM-L6-v2'}")

        # Placeholder for future providers
        # elif self.provider_name == "gemini":
        #     from .providers.gemini import GeminiEmbeddingProvider
        #     self.embedder: BaseEmbedder = GeminiEmbeddingProvider(model=model)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        return self.embedder.embed_documents(texts, batch_size=batch_size)
    
    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed_query(query)


# from typing import List
# from .providers.base import BaseEmbeddingProvider
# from .providers.gemini import GeminiEmbeddingProvider
# from .providers.mock import MockEmbeddingProvider
# from ..utils.config import get_settings
# from ..utils.logger import setup_logger

# logger = setup_logger(__name__)

# class DocumentEmbedder:
#     """Provider-agnostic embedding interface."""

#     def __init__(self, provider: str = "gemini", model: str | None = None):
#         settings = get_settings()
#         provider = provider.lower()

#         if provider == "gemini":
#             self.provider: BaseEmbeddingProvider = GeminiEmbeddingProvider(
#                 model=model or settings.embedding_model,
#                 api_key=settings.gemini_api_key,
#             )
#         elif provider == "mock":
#             self.provider = MockEmbeddingProvider()
#         else:
#             raise NotImplementedError(
#                 f"Embedding provider '{provider}' not implemented"
#             )

#         logger.info(f"Initialized embedder with provider={provider}")

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.provider.embed_documents(texts)

#     def embed_query(self, query: str) -> List[float]:
#         return self.provider.embed_query(query)
