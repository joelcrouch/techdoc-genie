from typing import Any, List
import requests

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
        Generates a response from the Gemini model.
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
        
        try:
            # Gemini API URL structure: BASE_URL/v1beta/models/MODEL_NAME:generateContent?key=API_KEY
            response = requests.post(
                f"{self.settings.gemini_base_url}/v1beta/models/{self.model_name}:generateContent?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=self.settings.gemini_timeout # Assuming gemini_timeout in settings
            )
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Gemini API for model {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Gemini generation: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding generation is not implemented for GeminiProvider (use dedicated embedder).
        """
        raise NotImplementedError("Embedding generation is not implemented for GeminiProvider yet. Use dedicated embedder.")
