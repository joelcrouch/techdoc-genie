from abc import ABC, abstractmethod
from typing import List

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str])-> List[List[float]]:
        pass


