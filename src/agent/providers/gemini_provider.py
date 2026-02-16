from typing import Any, List
import requests
import time # Import time for sleep
from requests.exceptions import HTTPError # Import HTTPError

from .base import BaseLLMProvider
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)

class GeminiProvider(BaseLLMProvider):
    """
    LLM provider for Google Gemini models.
    """
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.settings = get_settings()
        
        if not self.api_key:
            logger.error("Gemini API key is not provided.")
            raise ValueError("Gemini API key is required.")

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a response from the Gemini model with a single retry for 429 rate limits.
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.0),
                "maxOutputTokens": kwargs.get("max_tokens", 2048)
            }
        }
        
        # Function to execute the request and handle common errors
        def _execute_request():
            response = requests.post(
                f"{self.settings.gemini_base_url}/v1beta/models/{self.model_name}:generateContent?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=self.settings.gemini_timeout
            )
            response.raise_for_status()
            return response.json()

        try:
            result = _execute_request()
        except HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Gemini API rate limit hit (429). Retrying in 6 seconds... for model {self.model_name}.")
                time.sleep(6)
                try:
                    result = _execute_request() # Retry once after waiting
                except Exception as retry_e:
                    logger.error(f"Failed second attempt to call Gemini API for model {self.model_name} after 429: {retry_e}")
                    raise # Re-raise if second attempt also fails
            else:
                logger.error(f"Error calling Gemini API for model {self.model_name}: {e}")
                raise # Re-raise other HTTP errors
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Gemini API for model {self.model_name}: {e}")
            raise # Re-raise other request exceptions
        except Exception as e:
            logger.error(f"Unexpected error during Gemini generation: {e}")
            raise
        
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        return ""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        A placeholder implementation to satisfy the BaseLLMProvider interface.
        Embedding generation is handled by a dedicated embedder.
        """
        pass
