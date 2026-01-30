from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
from .base import BaseEmbeddingProvider

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model:str, api_key:str):
        self.embeddings=GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=api_key
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query, str) -> List[float]:
        return self.embedding.embed_query(query)