"""
Document retriever — ranking, scoring, and threshold filtering.

This module owns the business logic of retrieval:
  - What score is "good enough"?
  - What do we return if nothing passes the threshold?
  - How do we normalise raw pgvector scores to [0, 1]?

It does NOT own SQL. All database access goes through vector_repo.

The separation matters: if we add re-ranking (e.g. a cross-encoder
model that re-scores the top-K results), it goes here — not in the repo.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from src.core.utils import normalise_score
from src.db.repositories.vector_repo import similarity_search
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    user_id: Optional[str] = None,
    doc_name: Optional[str] = None,
    top_k: int = settings.TOP_K,
    threshold: float = settings.SIMILARITY_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """
    Retrieve and rank document chunks relevant to a query.

    Steps:
      1. Embed the query with the same model used at ingestion time.
      2. Call vector_repo.similarity_search() for raw (chunk, score) pairs.
      3. Normalise scores to [0, 1].
      4. Filter by threshold; fall back to top-1 if nothing passes.
      5. Return sorted by score descending, capped at top_k.

    Returns LangChain Document objects (not Chunk dataclasses) to keep the
    service layer interface unchanged during the migration.

    Args:
        query:     The user's question.
        user_id:   Scope results to this user. Always pass this in production.
        doc_name:  Optionally narrow to a single document.
        top_k:     Maximum results to return.
        threshold: Minimum normalised score to keep a result.
    """
    # ── 1. Embed the query ─────────────────────────────────────────────────
    try:
        query_vector = get_embeddings().embed_query(query)
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        return []

    # ── 2. Vector search ───────────────────────────────────────────────────
    # Fetch more than top_k so the threshold filter still leaves enough results
    raw_results = similarity_search(
        query_vector=query_vector,
        user_id=user_id,
        doc_name=doc_name,
        limit=top_k * 2,
    )

    if not raw_results:
        logger.debug(
            "No results for query='%.60s' (user=%s, doc=%s).",
            query, user_id, doc_name,
        )
        return []

    # ── 3. Normalise scores ────────────────────────────────────────────────
    scored: List[Tuple[Document, float]] = []
    for chunk, raw_score in raw_results:
        norm_score = normalise_score(raw_score)
        doc = Document(
            page_content=chunk.page_content,
            metadata=chunk.to_metadata(),
        )
        scored.append((doc, norm_score))

    scored.sort(key=lambda item: item[1], reverse=True)

    # ── 4. Threshold filtering ─────────────────────────────────────────────
    filtered = [(doc, s) for doc, s in scored if s >= threshold]

    if not filtered:
        logger.warning(
            "No chunks above threshold %.2f for query='%.40s'; returning top result as fallback.",
            threshold, query,
        )
        filtered = scored[:1]

    result = filtered[:top_k]

    logger.debug(
        "Retrieved %d chunk(s) for query='%.60s' (user=%s, doc=%s).",
        len(result), query, user_id, doc_name,
    )
    return result
