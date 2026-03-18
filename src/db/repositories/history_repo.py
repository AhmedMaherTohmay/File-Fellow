"""
History repository — all SQL for conversation_turns and conversations tables.

Responsibilities:
  - Persist user/assistant turns with their embeddings.
  - Fetch recent turns for a conversation (recency window).
  - Search past turns by semantic similarity (pgvector).
  - Purge turns older than the configured TTL.
  - Ensure conversation and user rows exist before inserting turns (FK).

What this file does NOT do:
  - Score normalisation (that belongs to history_retriever.py).
  - Threshold filtering (same — retrieval layer concern).
  - Embedding generation (ingestion layer concern — embedder.py).
  - Prompt formatting (service layer concern — qa_service.py).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import psycopg2.extras

from config.settings import settings
from src.db.engine import get_connection, vec_to_literal
from src.db.models.conversation import ConversationTurn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation and user anchors
# ──────────────────────────────────────────────────────────────────────────────

def ensure_conversation(user_id: str, conversation_id: Optional[str] = None) -> str:
    """
    Ensure a conversation row exists and return its ID.

    If conversation_id is provided → upsert it and return it.
    If None → generate a new UUID, insert it, return it.

    This is where conversation IDs are created — server-side, in the
    data layer. The UI/API never generate IDs; they receive them back.

    Also ensures the user row exists (upsert) so the FK is satisfied.
    """
    # Ensure user exists first (referenced by conversations FK)
    _upsert_user(user_id)

    conv_id = conversation_id or str(uuid.uuid4())

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (id, user_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (conv_id, user_id),
            )

    return conv_id


def _upsert_user(user_id: str) -> None:
    """Ensure a users row exists for this user_id. Safe to call repeatedly."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────

def add_turn(
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    user_vec: Optional[List[float]] = None,
    assistant_vec: Optional[List[float]] = None,
) -> None:
    """
    Persist one user/assistant exchange as two rows.

    Embeddings are optional — if the embedder failed upstream, we store
    the text without a vector. The turn is still useful for recency recall;
    it just won't appear in semantic search results (embedding IS NOT NULL
    filter in similarity queries excludes it).

    This is intentionally non-fatal on failure. A history write error
    should never crash a Q&A response.
    """
    rows = [
        (
            user_id,
            conversation_id,
            "user",
            user_message,
            vec_to_literal(user_vec) if user_vec is not None else None,
        ),
        (
            user_id,
            conversation_id,
            "assistant",
            assistant_message[:1000],   # cap assistant messages to 1k chars
            vec_to_literal(assistant_vec) if assistant_vec is not None else None,
        ),
    ]

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO conversation_turns
                        (user_id, conversation_id, role, content, embedding)
                    VALUES %s
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s::vector)",
                )
    except Exception as exc:
        logger.warning("History persistence failed (non-fatal): %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Read — recency
# ──────────────────────────────────────────────────────────────────────────────

def get_recent_turns(
    user_id: str,
    conversation_id: str,
    n: int = settings.MAX_SESSION_TURNS,
) -> List[ConversationTurn]:
    """
    Return the last *n* turns for a conversation in chronological order.

    Uses a subquery to get the N most recent rows DESC, then re-orders
    them ASC so the LLM sees the conversation in the correct time order.
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, user_id, conversation_id, role, content, created_at
                    FROM (
                        SELECT id, user_id, conversation_id, role, content, created_at
                        FROM conversation_turns
                        WHERE user_id = %s AND conversation_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    ) recent
                    ORDER BY created_at ASC
                    """,
                    (user_id, conversation_id, n * 2),
                )
                rows = cur.fetchall()
    except Exception as exc:
        logger.warning("get_recent_turns failed: %s", exc)
        return []

    return [_row_to_turn(row) for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Read — semantic similarity
# ──────────────────────────────────────────────────────────────────────────────

def search_turns_by_vector(
    user_id: str,
    query_vector: List[float],
    exclude_conversation_id: Optional[str] = None,
    limit: int = settings.SESSION_HISTORY_TOP_K,
) -> List[Tuple[ConversationTurn, float]]:
    """
    Find past turns semantically similar to a query vector.

    Returns (ConversationTurn, raw_score) pairs.
    raw_score = 1 - cosine_distance, range [-1, 1].
    The retrieval layer normalises and filters these.

    Turns from the current conversation are excluded — we don't want
    the LLM to see its own ongoing thread injected as "past context".
    """
    vec_literal = vec_to_literal(query_vector)

    conditions = ["user_id = %s", "embedding IS NOT NULL"]
    params: list = [user_id]

    if exclude_conversation_id:
        conditions.append(
            "(conversation_id IS NULL OR conversation_id != %s)"
        )
        params.append(exclude_conversation_id)

    where_clause = " AND ".join(conditions)

    sql = f"""
        SELECT id, user_id, conversation_id, role, content, created_at,
               1 - (embedding <=> %s::vector) AS raw_score
        FROM conversation_turns
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
        logger.warning("History similarity search failed: %s", exc)
        return []

    results = []
    for row in rows:
        raw_score = float(row["raw_score"])
        turn = _row_to_turn({k: v for k, v in row.items() if k != "raw_score"})
        results.append((turn, raw_score))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────────────────

def purge_old_turns(days: int = settings.HISTORY_TTL_DAYS) -> int:
    """
    Delete turns older than *days*. Non-fatal — never blocks startup.

    Returns the number of rows deleted.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_turns WHERE created_at < %s",
                    (cutoff,),
                )
                deleted = cur.rowcount

        if deleted:
            logger.info("Purged %d turn(s) older than %d day(s).", deleted, days)
        else:
            logger.debug("History purge: no turns older than %d day(s).", days)

        return deleted

    except Exception as exc:
        logger.warning("History purge failed (non-fatal): %s", exc)
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Row hydration
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_turn(row: dict) -> ConversationTurn:
    return ConversationTurn(
        id=              row["id"],
        user_id=         row["user_id"],
        conversation_id= row.get("conversation_id", ""),
        role=            row["role"],
        content=         row["content"],
        created_at=      row["created_at"],
    )
