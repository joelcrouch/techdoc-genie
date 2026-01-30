from functools import lru_cache
from typing import Optional

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    # 🔑 OPTIONAL now
    OPENAI_API_KEY: Optional[str] = None
    gemini_embedding_key: Optional[str] = None

    # 🧠 Embedding config
    embedding_provider: str = "huggingface"
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"

    # 📦 Vector store
    vector_store_path: str = "/data"
    collection_name: str = "tech_docs"

    # 🪵 Logging
    log_level: str = "INFO"

    # ✂️ Chunking (used by your scripts)
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = ConfigDict(env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    return Settings()




# from functools import lru_cache
# from pydantic_settings import BaseSettings
# from pydantic import Field, ConfigDict
# from pydantic_settings import SettingsConfigDict

# class Settings(BaseSettings):
#     OPENAI_API_KEY: str
#     gemini_embedding_key: str
#     embedding_model: str = "text-embedding-3-small"
#     vector_store_path: str = "./data/vector_store"
#     collection_name: str = "tech_docs"
#     log_level: str = "INFO"
#     chunk_size: int = 512
#     chunk_overlap: int = 50

#     model_config = ConfigDict(env_file=".env")

# @lru_cache()
# def get_settings() -> Settings:
#     return Settings()
