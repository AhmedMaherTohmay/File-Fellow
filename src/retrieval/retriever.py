"""
Semantic retrieval from the vector store.
Returns the top-k relevant chunks for a given query, filtered by a similarity threshold.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from config.settings import TOP_K, SIMILARITY_THRESHOLD
from src.ingestion.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """Perform semantic similarity search.

    Args:
        query: User's question.
        top_k: Number of chunks to retrieve.
        threshold: Minimum cosine similarity score to keep.

    Returns:
        List of ``(Document, score)`` tuples, sorted by relevance.
    """
    store = get_vector_store()

    results: List[Tuple[Document, float]] = store.similarity_search_with_relevance_scores(
        query, k=top_k
    )

    # Filter by threshold
    filtered = [(doc, score) for doc, score in results if score >= threshold]

    if not filtered:
        logger.warning(
            "No chunks above threshold %.2f; returning top result anyway.", threshold
        )
        filtered = results[:1] if results else []

    logger.debug("Retrieved %d chunks for query: '%s'.", len(filtered), query[:80])
    return filtered
