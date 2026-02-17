from functools import lru_cache
from typing import Optional

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    # OPTIONAL now
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None # Renamed for clarity and consistency

    #  Embedding config
    embedding_provider: str = "huggingface"
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"

    #  Vector store
    vector_store_path: str = "/data"
    collection_name: str = "tech_docs"

    #  Logging
    log_level: str = "INFO"

    #  Chunking (used by your scripts)=> goota change this when i make the experimnet
    # more declarative alos delete htis comment if you have made the experimenttion decalarative
    chunk_size: int = 512
    chunk_overlap: int = 50

    #  Ollama Configuration
    ollama_timeout: int = 180
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_id: str = "phi3:mini" # Default Ollama model for non-evaluation contexts

    # Default LLM configuration for general application use (non-evaluation)
    llm_default_provider: str = "ollama" # Default LLM provider (e.g., "ollama", "openai")
    llm_default_model_id: str = "phi3:mini" # Default LLM model for general use

    # OpenAI Configuration
    openai_model_id: str = "gpt-4-turbo-preview" # Default OpenAI model for non-evaluation contexts
    openai_base_url: str = "https://api.openai.com" # OpenAI API base URL
    openai_timeout: int = 180 # Timeout for OpenAI API calls

    # Anthropic Claude Configuration
    ANTHROPIC_API_KEY: Optional[str] = None
    claude_model_id: str = "claude-3-opus-20240229" # Default Claude model for non-evaluation contexts
    claude_base_url: str = "https://api.anthropic.com" # Claude API base URL
    claude_timeout: int = 180 # Timeout for Claude API calls

    # Google Gemini Configuration
    gemini_model_id: str = "gemini-2.5-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com" # Gemini API base URL
    gemini_timeout: int = 180 # Timeout for Gemini API calls

    model_config = ConfigDict(env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


