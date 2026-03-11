"""
Application configuration — single source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved once at import time so computed paths are consistent
_BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── Application ────────────────────────────────────────────────────────
    APP_NAME: str = "Smart Contract Assistant"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Vector Store Backend ───────────────────────────────────────────────
    VECTOR_STORE_BACKEND: Literal["chroma", "pgvector", "qdrant"] = "chroma"
    CHROMA_COLLECTION_PREFIX: str = "file"
    CHAT_HISTORY_COLLECTION: str = "chat_history"
    EMBEDDING_DIMENSION: int = 384          # matches all-MiniLM-L6-v2

    # ── pgvector (future) ──────────────────────────────────────────────────
    PGVECTOR_HNSW_M: int = 16
    PGVECTOR_HNSW_EF_CONSTRUCTION: int = 64
    PGVECTOR_HNSW_EF_SEARCH: int = 40

    # ── Qdrant (future) ────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None    # secret — set in .env if needed

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"
    LLM_KEY: str = ""                       # SECRET — must be set in .env
    GROQ_MODEL_ID: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 4096              # authoritative default (was split between yaml/env)

    # ── Embeddings ─────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "sentence_transformers"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MIN_CHUNK_LENGTH: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.30

    # ── Multi-doc ─────────────────────────────────────────────────────────
    MAX_DOCUMENTS: int = 20

    # ── Session / Memory ──────────────────────────────────────────────────
    SESSION_HISTORY_TOP_K: int = 3
    MAX_SESSION_TURNS: int = 6
    HISTORY_SCORE_THRESHOLD: float = 0.25
    HISTORY_TTL_DAYS: int = 7

    # ── Server ────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    GRADIO_HOST: str = "0.0.0.0"
    GRADIO_PORT: int = 7860
    GRADIO_SHARE: bool = False

    # ── Evaluation ────────────────────────────────────────────────────────
    EVAL_NUM_QUESTIONS: int = 10
    EVAL_BASELINE_DOC_FRACTION: float = 0.1

    # ── Paths (derived from project root; override via env vars if needed) ─
    UPLOAD_DIR: Path = _BASE_DIR / "data" / "uploads"
    VECTOR_STORE_DIR: Path = _BASE_DIR / "data" / "vector_store"
    LOG_DIR: Path = _BASE_DIR / "data" / "logs"


# ── Singleton instance ─────────────────────────────────────────────────────
settings = Settings()

# Ensure required directories exist at import time
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
