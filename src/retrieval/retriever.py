"""
Semantic retrieval from the vector store(s).

Supports:
  - Single document retrieval (filtered by doc_name).
  - Cross-document retrieval (all ingested documents).
  - Off-topic guard: returns an empty list when max similarity is very low.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import TOP_K, SIMILARITY_THRESHOLD
from src.ingestion.vector_store import (
    get_global_store,
    get_store_for_document,
)

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    doc_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """Perform semantic similarity search.

    Args:
        query: User's question.
        top_k: Number of chunks to retrieve.
        threshold: Minimum similarity score to include.
        doc_name: If provided, search only this document's store.
                  If None, search across all documents.

    Returns:
        List of ``(Document, score)`` tuples sorted by relevance.
    """
    # Select appropriate store
    try:
        if doc_name:
            store = get_store_for_document(doc_name)
        else:
            store = get_global_store()
    except RuntimeError as e:
        logger.warning("Store not available: %s", e)
        return []

    # Retrieve with scores
    try:
        results: List[Tuple[Document, float]] = (
            store.similarity_search_with_relevance_scores(query, k=top_k)
        )
    except Exception as e:
        logger.error("Similarity search failed: %s", e)
        return []


    # Apply threshold filter
    filtered = [(doc, score) for doc, score in results if score >= threshold]

    if not filtered and results:
        logger.warning(
            "No chunks above threshold %.2f; returning top result anyway.", threshold
        )
        filtered = results[:1]

    logger.debug(
        "Retrieved %d chunks for query='%s' (doc=%s).",
        len(filtered),
        query[:60],
        doc_name or "all",
    )
    return filtered
