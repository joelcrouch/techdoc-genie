from functools import lru_cache
from typing import Optional

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    # OPTIONAL now
    OPENAI_API_KEY: Optional[str] = None
    gemini_embedding_key: Optional[str] = None

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
    ollama_base_url: str = "http://localhost:11434" # Added for consistency

    model_config = ConfigDict(env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


