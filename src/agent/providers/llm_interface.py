from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class LLMInterface:
    """
    Provider-agnostic interface for text generation and embeddings.
    """

    def __init__(self, provider: str, model: str, api_key: str | None = None):
        self.provider_name = provider.lower()
        self.model = model

        if self.provider_name == "mock":
            from .mock_provider import MockLLMProvider
            self.provider = MockLLMProvider()
        elif self.provider_name == "openai":
            raise NotImplementedError("OpenAI provider not implemented yet")
        elif self.provider_name == "claude":
            raise NotImplementedError("Claude provider not implemented yet")
        elif self.provider_name == "gemini":
            raise NotImplementedError("Gemini provider not implemented yet")
        elif self.provider_name == "local":
            raise NotImplementedError("Local provider not implemented yet")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        return self.provider.generate_text(prompt, **kwargs)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.provider.embed_texts(texts)
