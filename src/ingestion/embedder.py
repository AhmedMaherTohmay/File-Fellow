"""
Embedding factory.
Returns a cached LangChain-compatible embeddings object.
Supports SentenceTransformers (default, local) with future extension points.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached LangChain embeddings instance.

    Uses SentenceTransformers for fully local, no-API-key embeddings.

    Returns:
        A LangChain-compatible embeddings object.

    Raises:
        RuntimeError: If no embedding provider is available.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "sentence_transformers":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(
                model_name=settings.SENTENCE_TRANSFORMER_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                "Using SentenceTransformer embeddings: %s", settings.SENTENCE_TRANSFORMER_MODEL
            )
            return emb
        except Exception as e:
            logger.error("SentenceTransformer embeddings failed: %s", e)
            raise RuntimeError("SentenceTransformer embedding provider unavailable.") from e

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{settings.EMBEDDING_PROVIDER}'. "
        "Only 'sentence_transformers' is currently supported."
    )
