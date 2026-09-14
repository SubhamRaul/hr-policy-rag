from functools import lru_cache
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv() 

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)
ENABLE_OLLAMA_FALLBACK = (
    os.getenv("ENABLE_OLLAMA_FALLBACK", "true").lower() == "true"
)


class Settings(BaseSettings):
    gemini_api_key: str | None = None
    llm_model: str = "gemini-3.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = ".chroma"
    collection_name: str = "hr_policies"
    top_k: int = 8
    final_k: int = 5
    min_semantic_sim: float = 0.30
    min_hybrid_score: float = 0.28
    max_upload_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
