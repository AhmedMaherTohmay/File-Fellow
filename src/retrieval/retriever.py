from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import TOP_K, SIMILARITY_THRESHOLD, VECTOR_STORE_BACKEND
from src.core.utils import normalise_score
from src.storage.document_store import get_global_store

logger = logging.getLogger(__name__)

_FAISS_FILTER_OVERSAMPLE: int = 20


def retrieve_chunks(
    query: str,
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    doc_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    Perform a semantic similarity search over the global vector store.

    Args:
        query:     The user's question or message.
        top_k:     Maximum number of chunks to return after filtering.
        threshold: Chunks below this score are discarded unless no chunk
                   exceeds it.
        doc_name:  When provided, results are restricted to chunks whose
                   ``source`` metadata matches this value exactly.
                   When None, the entire store is searched.

    Returns:
        List of ``(Document, normalised_score)`` tuples ordered by descending
        score.  Scores are always in [0, 1].
        Empty list if the store is unavailable or no chunks are found.
    """
    try:
        store = get_global_store()
    except Exception as exc:
        logger.warning("Global store unavailable: %s", exc)
        return []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw_results = _search(store, query, top_k, doc_name)
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        return []
    finally:
        del store

    if not raw_results:
        logger.debug("No results for query='%.60s' (doc=%s).", query, doc_name or "all")
        return []

    results = [(doc, normalise_score(raw)) for doc, raw in raw_results]
    results.sort(key=lambda t: t[1], reverse=True)

    filtered = [(doc, score) for doc, score in results if score >= threshold]
    if not filtered:
        logger.warning("No chunks above threshold %.3f; returning top result as fallback.", threshold)
        filtered = results[:1]

    logger.debug("Retrieved %d chunk(s) for query='%.60s' (doc=%s).",
                 len(filtered), query, doc_name or "all")
    return filtered


def _search(
    store,
    query: str,
    top_k: int,
    doc_name: Optional[str],
) -> List[Tuple[Document, float]]:
    """Dispatch the similarity search to the correct backend strategy."""
    if doc_name is None:
        return store.similarity_search_with_relevance_scores(query, k=top_k)

    if VECTOR_STORE_BACKEND == "chroma":
        return store.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"source": doc_name},
        )

    fetch_k = top_k * _FAISS_FILTER_OVERSAMPLE
    raw = store.similarity_search_with_relevance_scores(query, k=fetch_k)
    filtered = [(doc, score) for doc, score in raw if doc.metadata.get("source") == doc_name]
    return filtered[:top_k]
