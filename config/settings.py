"""
Loads from config.yaml first, then .env overrides, then environment variables.
Uses pathlib.Path for cross-platform compatibility.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)


def _load_yaml_defaults() -> dict:
    """Load defaults from config/config.yaml if it exists."""
    yaml_path = BASE_DIR / "config" / "config.yaml"
    if yaml_path.exists():
        try:
            import yaml  # type: ignore

            with open(yaml_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            logger.debug("PyYAML not installed; skipping config.yaml.")
    return {}


_yaml = _load_yaml_defaults()


def _get(key: str, default):
    """Read: env var > yaml > default."""
    return os.getenv(key) or _yaml.get(key.lower()) or default


# ── Paths ──────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR: Path = BASE_DIR / "data" / "vector_store"
UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
LOG_DIR: Path = BASE_DIR / "data" / "logs"

for _d in (VECTOR_STORE_DIR, UPLOAD_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = _get("LLM_PROVIDER", "groq")
LLM_KEY: str = _get("LLM_KEY", "")
GROQ_MODEL_ID: str = _get("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(_get("LLM_MAX_TOKENS", "1024"))

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER: str = _get("EMBEDDING_PROVIDER", "sentence_transformers")
SENTENCE_TRANSFORMER_MODEL: str = _get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "150"))

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K: int = int(_get("TOP_K", "5"))
SIMILARITY_THRESHOLD: float = float(_get("SIMILARITY_THRESHOLD", "0.30"))

# ── Vector Store ──────────────────────────────────────────────────────────
VECTOR_STORE_BACKEND: str = _get("VECTOR_STORE_BACKEND", "chroma")
CHROMA_COLLECTION_PREFIX: str = _get("CHROMA_COLLECTION_PREFIX", "file")
CHAT_HISTORY_COLLECTION: str = _get("CHAT_HISTORY_COLLECTION", "chat_history")

# ── Multi-doc ─────────────────────────────────────────────────────────────
MAX_DOCUMENTS: int = int(_get("MAX_DOCUMENTS", "20"))

# ── Session / Memory ──────────────────────────────────────────────────────
SESSION_HISTORY_TOP_K: int = int(_get("SESSION_HISTORY_TOP_K", "3"))
MAX_SESSION_TURNS: int = int(_get("MAX_SESSION_TURNS", "6"))

# Separate threshold for semantic history retrieval.
HISTORY_SCORE_THRESHOLD: float = float(_get("HISTORY_SCORE_THRESHOLD", "0.25"))

# History turns older than this many days are purged at startup.
HISTORY_TTL_DAYS: int = int(_get("HISTORY_TTL_DAYS", "7"))

# ── FastAPI / LangServe ───────────────────────────────────────────────────
API_HOST: str = _get("API_HOST", "0.0.0.0")
API_PORT: int = int(_get("API_PORT", "8000"))

# ── Gradio ────────────────────────────────────────────────────────────────
GRADIO_HOST: str = _get("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT: int = int(_get("GRADIO_PORT", "7860"))
GRADIO_SHARE: bool = _get("GRADIO_SHARE", "false").lower() == "true"

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")

# ── Evaluation ────────────────────────────────────────────────────────────
EVAL_NUM_QUESTIONS: int = int(_get("EVAL_NUM_QUESTIONS", "10"))
EVAL_BASELINE_DOC_FRACTION: float = float(_get("EVAL_BASELINE_DOC_FRACTION", "0.1"))
