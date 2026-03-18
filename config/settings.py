"""
Application configuration — single source of truth.

Architecture:
  - Secrets (LLM_KEY, DATABASE_URL) come exclusively from .env
  - Deployment overrides (hosts, ports, log level) can also be set in .env
  - All other values have validated defaults here; no .env entry is required
  - config.yaml is kept as human-readable documentation only (not loaded at runtime)

Usage:
    from config.settings import settings
    settings.CHUNK_SIZE       # → 800
    settings.DATABASE_URL     # → value from .env
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    APP_NAME: str = "Smart Contract Assistant"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ───────────────────────────────────────────────────────────
    # SECRET — must be set in .env
    DATABASE_URL: str = ""
    DATABASE_POOL_SIZE: int = 20

    # ── pgvector ───────────────────────────────────────────────────────────
    EMBEDDING_DIMENSION: int = 384          # must match SENTENCE_TRANSFORMER_MODEL output
    PGVECTOR_HNSW_M: int = 16
    PGVECTOR_HNSW_EF_CONSTRUCTION: int = 64
    PGVECTOR_HNSW_EF_SEARCH: int = 40

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"
    LLM_KEY: str = ""                       # SECRET — must be set in .env
    GROQ_MODEL_ID: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 4096

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

    # ── Paths ──────────────────────────────────────────────────────────────
    UPLOAD_DIR: Path = _BASE_DIR / "data" / "uploads"
    VECTOR_STORE_DIR: Path = _BASE_DIR / "data" / "vector_store"
    LOG_DIR: Path = _BASE_DIR / "data" / "logs"


# ── Singleton instance ─────────────────────────────────────────────────────
settings = Settings()

# Ensure required directories exist at import time
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
