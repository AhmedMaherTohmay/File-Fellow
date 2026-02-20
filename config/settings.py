"""
Central configuration for Smart Contract Assistant.
All tuneable parameters live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq") 
LLM_KEY: str = os.getenv("LLM_KEY")
# Defaulting to llama-3.3-70b-versatile as per your requirements
GROQ_MODEL_ID: str = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
SENTENCE_TRANSFORMER_MODEL: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

# ── Vector Store ──────────────────────────────────────────────────────────
VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "chroma")  # "chroma" | "faiss"
CHROMA_COLLECTION_PREFIX: str = "File_Fellow"

# ── FastAPI / LangServe ───────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Gradio ────────────────────────────────────────────────────────────────
GRADIO_HOST: str = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"