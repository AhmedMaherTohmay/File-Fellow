"""
Vector repository — pgvector similarity search against document_chunks.

This is the only place in the codebase that executes the  <=>  cosine
distance operator. The retrieval layer (document_retriever.py) calls this
function and owns the ranking, scoring, and threshold logic.

Why separate from document_repo.py?
  document_repo.py handles CRUD — insert, delete, fetch by ID.
  vector_repo.py handles SEARCH — find by similarity to a vector.
  These are different access patterns with different call sites.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import psycopg2.extras

from src.db.engine import get_connection, vec_to_literal
from src.db.models.document import Chunk
from src.db.repositories.document_repo import _row_to_chunk

logger = logging.getLogger(__name__)


def similarity_search(
    query_vector: List[float],
    user_id: Optional[str] = None,
    doc_name: Optional[str] = None,
    limit: int = 10,
) -> List[Tuple[Chunk, float]]:
    """
    Find the most similar chunks to a query vector using cosine distance.

    Returns (Chunk, raw_score) pairs where raw_score = 1 - cosine_distance,
    in the range [-1, 1]. The retrieval layer normalises these to [0, 1].

    The caller (document_retriever.py) is responsible for:
      - normalising scores
      - applying a threshold
      - deciding how many results to return to the service

    This function only handles the SQL and the row hydration.

    Args:
        query_vector: Embedded query (same model used at ingestion time).
        user_id:      Scope results to a specific user. Always pass this —
                      without it the search spans all users in the DB.
        doc_name:     Optionally narrow to a single document.
        limit:        Max rows to return from pgvector. Pass a larger value
                      than you need so the retrieval layer can apply a
                      threshold and still have enough results.
    """
    vec_literal = vec_to_literal(query_vector)

    # Build the WHERE clause dynamically based on which filters are provided
    conditions: List[str] = ["embedding IS NOT NULL"]
    params: List[Any] = []

    if user_id:
        conditions.append("user_id = %s")
        params.append(user_id)

    if doc_name:
        conditions.append("source = %s")
        params.append(doc_name)

    where_clause = " AND ".join(conditions)

    # Parameter order:
    #   %s for the score column (1 - distance)
    #   WHERE params
    #   %s for the ORDER BY column (distance)
    #   %s for LIMIT
    sql = f"""
        SELECT
            chunk_id, doc_fk, user_id, doc_name, page_content,
            source, file_type, page, chunk_index,
            global_chunk_index, chunk_size, chunk_overlap, uploaded_at,
            1 - (embedding <=> %s::vector) AS raw_score
        FROM document_chunks
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    query_params = [vec_literal] + params + [vec_literal, limit]

    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, query_params)
                rows = cur.fetchall()
    except Exception as exc:
        logger.error("Similarity search failed: %s", exc)
        return []

    if not rows:
        logger.debug(
            "No results from similarity_search (user=%s, doc=%s).", user_id, doc_name
        )
        return []

    results = []
    for row in rows:
        raw_score = float(row["raw_score"])
        chunk = _row_to_chunk({k: v for k, v in row.items() if k != "raw_score"})
        results.append((chunk, raw_score))

    return results
