from typing import Any, List
import requests

from .base import BaseLLMProvider
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)

class ClaudeProvider(BaseLLMProvider):
    """
    LLM provider for Anthropic Claude models.
    """
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.settings = get_settings()
        
        if not self.api_key:
            logger.error("Anthropic API key is not provided.")
            raise ValueError("Anthropic API key is required.")

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a response from the Claude model.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01", # Required for Anthropic API
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(
                f"{self.settings.claude_base_url}/v1/messages", # Assuming claude_base_url in settings
                headers=headers,
                json=payload,
                timeout=self.settings.claude_timeout # Assuming claude_timeout in settings
            )
            response.raise_for_status()
            
            result = response.json()
            # Claude's response structure might differ, this is for "messages" API
            if 'content' in result and isinstance(result['content'], list) and result['content']:
                return result['content'][0]['text']
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Claude API for model {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Claude generation: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding generation is not implemented for ClaudeProvider.
        """
        raise NotImplementedError("Embedding generation is not implemented for ClaudeProvider yet. Use dedicated embedder.")
