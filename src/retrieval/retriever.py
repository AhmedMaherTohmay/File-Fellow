from __future__ import annotations

import gc
import logging
import warnings
from typing import List, Optional, Tuple, Dict, Any

from langchain_core.documents import Document

from config.settings import TOP_K, SIMILARITY_THRESHOLD
from src.core.utils import normalise_score
from src.storage.document_store import get_global_store

logger = logging.getLogger(__name__)


def _build_filter(user_id: Optional[str], doc_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma metadata filter for user and/or document scoping.
    """
    conditions: List[Dict[str, Any]] = []

    if user_id:
        conditions.append({"user_id": {"$eq": user_id}})

    if doc_name:
        conditions.append({"source": {"$eq": doc_name}})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def retrieve_chunks(
    query: str,
    user_id: Optional[str] = None,
    doc_name: Optional[str] = None,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """
    Perform semantic retrieval from the global vector store.

    Args:
        query: User query.
        user_id: Optional user scope filter.
        doc_name: Optional document name filter.
        top_k: Number of chunks to retrieve.
        threshold: Minimum similarity score required.

    Returns:
        List of (Document, score) sorted by relevance.
    """
    store = None

    try:
        store = get_global_store()
    except Exception as exc:
        logger.warning("Global store unavailable: %s", exc)
        return []

    chroma_filter = _build_filter(user_id, doc_name)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            kwargs = {"k": top_k}

            if chroma_filter:
                kwargs["filter"] = chroma_filter

            raw_results = store.similarity_search_with_relevance_scores(
                query,
                **kwargs,
            )

    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        return []

    finally:
        if store is not None:
            del store
            gc.collect()

    if not raw_results:
        logger.debug(
            "No results for query='%.60s' (user=%s, doc=%s).",
            query,
            user_id,
            doc_name,
        )
        return []

    # Normalize similarity scores
    results = [(doc, normalise_score(raw)) for doc, raw in raw_results]
    results.sort(key=lambda item: item[1], reverse=True)

    # Apply similarity threshold
    filtered = [(doc, score) for doc, score in results if score >= threshold]

    if not filtered:
        logger.warning(
            "No chunks above threshold %.2f; returning top result as fallback.",
            threshold,
        )
        filtered = results[:1]

    logger.debug(
        "Retrieved %d chunk(s) for query='%.60s' (user=%s, doc=%s).",
        len(filtered),
        query,
        user_id,
        doc_name,
    )

    return filtered