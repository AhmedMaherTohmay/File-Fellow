"""
Embedding factory.
Returns a LangChain-compatible embeddings object.
Prefers the provider set in settings; falls back automatically.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config.settings import (
    EMBEDDING_PROVIDER,
    SENTENCE_TRANSFORMER_MODEL,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached LangChain embeddings instance.
    """
    provider = EMBEDDING_PROVIDER

    if provider == "sentence_transformers":
    # Local SentenceTransformers
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(model_name=SENTENCE_TRANSFORMER_MODEL)
            logger.info("Using SentenceTransformer embeddings (%s).", SENTENCE_TRANSFORMER_MODEL)
            return emb
        except Exception as e:
            logger.error("SentenceTransformer embeddings failed: %s", e)
            raise RuntimeError("No embedding provider available.") from e
