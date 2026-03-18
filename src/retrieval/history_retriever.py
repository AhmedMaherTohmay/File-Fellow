"""
History retriever — semantic and recency-based conversation recall.

This module owns:
  - Score normalisation for history turns.
  - Threshold filtering for semantic recall.
  - Returning structured turn data for the service layer.

It does NOT own:
  - SQL queries            (history_repo.py)
  - Embedding generation   (embedder.py)
  - Persistence of turns   (history_repo.add_turn)
  - Prompt string building (prompt_builder.py)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from config.settings import settings
from src.core.utils import normalise_score
from src.db.models.conversation import ConversationTurn
from src.db.repositories.history_repo import get_recent_turns, search_turns_by_vector
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)


def retrieve_relevant_history(
    query: str,
    user_id: str,
    exclude_conversation_id: Optional[str] = None,
    k: int = settings.SESSION_HISTORY_TOP_K,
    score_threshold: float = settings.HISTORY_SCORE_THRESHOLD,
) -> List[dict]:
    """
    Find past conversation turns semantically relevant to the current query.

    Turns from the current conversation are excluded — injecting the
    ongoing thread as "past context" would confuse the LLM.

    Returns a list of dicts with keys: role, content, timestamp, score.
    prompt_builder.format_semantic_history() consumes this shape.
    """
    try:
        query_vector = get_embeddings().embed_query(query)
    except Exception as exc:
        logger.warning("History embedding failed, skipping semantic recall: %s", exc)
        return []

    raw_results = search_turns_by_vector(
        user_id=user_id,
        query_vector=query_vector,
        exclude_conversation_id=exclude_conversation_id,
        limit=k * 2,
    )

    relevant = []
    for turn, raw_score in raw_results:
        score = normalise_score(raw_score)
        if score < score_threshold:
            continue
        relevant.append({
            "role":      turn.role,
            "content":   turn.content,
            "timestamp": turn.created_at.isoformat() if turn.created_at else "",
            "score":     round(score, 4),
        })

    relevant.sort(key=lambda x: x["score"], reverse=True)

    logger.debug(
        "Retrieved %d relevant history turn(s) for user '%s'.",
        len(relevant[:k]), user_id,
    )
    return relevant[:k]


def retrieve_recent_turns(
    user_id: str,
    conversation_id: str,
    n: int = settings.MAX_SESSION_TURNS,
) -> List[dict]:
    """
    Return the most recent *n* turns in a conversation, oldest first.

    Returns dicts with keys: role, content.
    prompt_builder.format_recent_history() consumes this shape.
    """
    turns: List[ConversationTurn] = get_recent_turns(
        user_id=user_id,
        conversation_id=conversation_id,
        n=n,
    )
    return [{"role": t.role, "content": t.content} for t in turns]
