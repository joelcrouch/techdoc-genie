from typing import Dict, Any, List
import requests
from requests.exceptions import ConnectionError

from .base import BaseLLMProvider
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)

class OllamaProvider(BaseLLMProvider):
    """
    LLM provider for Ollama models.
    Assumes Ollama server is running locally.
    """
    def __init__(self, model_name: str = "phi3:mini", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.settings = get_settings() # Not strictly needed for Ollama, but keeps consistent with other providers
        
        # Verify Ollama server connectivity
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.settings.ollama_timeout)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama server at {self.base_url}")
            
            available_models = [m['name'] for m in response.json()['models']]
            if self.model_name not in available_models:
                logger.warning(
                    f"Model '{self.model_name}' not found on Ollama server. "
                    f"Available models: {', '.join(available_models)}. "
                    f"Attempting to use anyway, but it might fail."
                )
        except ConnectionError:
            logger.error(
                f"Could not connect to Ollama server at {self.base_url}. "
                "Please ensure Ollama is running and the model is pulled."
            )
            raise
        except Exception as e:
            logger.error(f"Error connecting to Ollama server: {e}")
            raise

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a response from the Ollama model.
        """
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_ctx": kwargs.get("max_tokens", 2048) # Map max_tokens to num_ctx for context window
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.settings.ollama_timeout)
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API for model {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Ollama generation: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Ollama can also provide embeddings if configured, but for now
        this provider only focuses on text generation.
        """
        raise NotImplementedError("Embedding generation is not implemented for OllamaProvider yet.")
