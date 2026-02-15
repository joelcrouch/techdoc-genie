from typing import Any, List
import requests
from requests.exceptions import ConnectionError

from .base import BaseLLMProvider
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)

class OpenAIProvider(BaseLLMProvider):
    """
    LLM provider for OpenAI models.
    """
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.settings = get_settings() # For timeout settings, etc.
        
        if not self.api_key:
            logger.error("OpenAI API key is not provided.")
            raise ValueError("OpenAI API key is required.")

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a response from the OpenAI model.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        try:
                      
            response = requests.post(
                f"{self.settings.openai_base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.settings.openai_timeout
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenAI API for model {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI generation: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding generation is not implemented for OpenAIProvider (use dedicated embedder).
        """
        raise NotImplementedError("Embedding generation is not implemented for OpenAIProvider yet. Use dedicated embedder.")

