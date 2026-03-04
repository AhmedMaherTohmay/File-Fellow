"""
Semantic chat history persistence with strict per-user isolation.

Architecture overview
---------------------
Each conversation turn (user + assistant pair) is embedded and stored in a
dedicated Chroma collection ("chat_history").  At query time, relevant past
turns for the *current user* are retrieved by semantic similarity and injected
into the LLM prompt as cross-session context.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import SESSION_HISTORY_TOP_K, HISTORY_SCORE_THRESHOLD, HISTORY_TTL_DAYS
from src.core.utils import normalise_score
from src.ingestion.vector_store import get_history_store, _WRITE_LOCK

logger = logging.getLogger(__name__)


class HistoryStore:
    """
    Manages per-user semantic chat history in a vector store.

    Args:
        user_id:         Stable user identifier.  Only turns stored under
                         this ID are visible to this instance.
        conversation_id: Current conversation UUID.  When provided,
                         ``retrieve_relevant`` excludes turns from this
                         conversation so they do not duplicate the in-session
                         history already passed to the LLM.
    """

    def __init__(self, user_id: str = "default", conversation_id: Optional[str] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._store = None

    def _get_store(self):
        if self._store is None:
            self._store = get_history_store()
        return self._store

    def retrieve_relevant(
        self,
        query: str,
        k: int = SESSION_HISTORY_TOP_K,
        score_threshold: float = HISTORY_SCORE_THRESHOLD,
        exclude_conversation_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve semantically relevant past turns for the current query.

        Only turns belonging to this user_id are considered.  Turns from
        ``exclude_conversation_id`` (typically the current conversation) are
        skipped so the LLM does not see the same dialogue twice.

        Args:
            query:                  Current user question.
            k:                      Max turns to return.
            score_threshold:        Minimum normalised score to include.
            exclude_conversation_id: Conversation ID whose turns to skip.

        Returns:
            List of dicts: {role, content, timestamp, user_id, score}.
        """
        effective_exclude = exclude_conversation_id or self.conversation_id

        try:
            store = self._get_store()
            # Over-fetch before filtering so we reliably return up to k items.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                raw_results = store.similarity_search_with_relevance_scores(
                    query, k=k * 6
                )

            relevant = []
            for doc, raw in raw_results:
                meta = doc.metadata

                # Skip the bootstrap sentinel document.
                if meta.get("type") == "init":
                    continue

                # Strict user isolation — always applied, no bypass.
                # Backward-compat: old turns used "session_id" as the field name.
                stored_user = meta.get("user_id") or meta.get("session_id", "")
                if stored_user != self.user_id:
                    continue

                # Exclude the current conversation to prevent the LLM from
                # seeing in-session turns twice (once in history, once here).
                if effective_exclude and meta.get("conversation_id") == effective_exclude:
                    continue

                score = normalise_score(raw)
                if score < score_threshold:
                    continue

                relevant.append({
                    "role": meta.get("role", "user"),
                    "content": doc.page_content,
                    "timestamp": meta.get("timestamp", ""),
                    "user_id": stored_user,
                    "score": round(score, 4),
                })
                if len(relevant) >= k:
                    break

            logger.debug("Retrieved %d relevant history items for user '%s'.",
                         len(relevant), self.user_id)
            return relevant

        except Exception as exc:
            logger.warning("History retrieval failed: %s", exc)
            return []

    def add_turn(self, user: str, assistant: str) -> None:
        """
        Persist a conversation turn (user + assistant pair).

        Turns are tagged with both user_id (for retrieval filtering) and
        conversation_id (for current-conversation exclusion).  The assistant
        text stored here must be the plain answer — no source citation footers
        — so retrieved context remains clean for the LLM on the next turn.

        Args:
            user:      User's message.
            assistant: Plain answer text (no citation markdown appended).
        """
        try:
            store = self._get_store()
            ts = datetime.now(timezone.utc).isoformat()

            docs = [
                Document(
                    page_content=user,
                    metadata={
                        "role": "user",
                        # Store as both user_id (new) and session_id (legacy compat)
                        "user_id": self.user_id,
                        "session_id": self.user_id,
                        "conversation_id": self.conversation_id or "",
                        "timestamp": ts,
                        "type": "turn",
                    },
                ),
                Document(
                    page_content=assistant[:1000],  # Truncate very long answers
                    metadata={
                        "role": "assistant",
                        "user_id": self.user_id,
                        "session_id": self.user_id,
                        "conversation_id": self.conversation_id or "",
                        "timestamp": ts,
                        "type": "turn",
                    },
                ),
            ]

            with _WRITE_LOCK:
                store.add_documents(docs)

            logger.debug("Persisted turn to history store (user='%s', conv='%s').",
                         self.user_id, self.conversation_id)
        except Exception as exc:
            logger.warning("History persistence failed: %s", exc)

    def add_bulk(self, history: List[dict]) -> None:
        """Bulk-persist a completed conversation history."""
        for i in range(0, len(history) - 1, 2):
            user_msg = (history[i].get("content", "")
                        if history[i].get("role") == "user" else "")
            asst_msg = (history[i + 1].get("content", "")
                        if i + 1 < len(history) and history[i + 1].get("role") == "assistant"
                        else "")
            if user_msg and asst_msg:
                self.add_turn(user_msg, asst_msg)

    def format_for_prompt(self, relevant_history: List[dict]) -> str:
        """Format retrieved history as a string for inclusion in the LLM prompt."""
        if not relevant_history:
            return "No relevant conversation history found."

        lines = ["[Relevant past conversation context:]"]
        for item in relevant_history:
            role_label = "User" if item["role"] == "user" else "Assistant"
            ts = item.get("timestamp", "")[:10]
            lines.append(f"[{ts}] {role_label}: {item['content'][:300]}")
        return "\n".join(lines)


# ── Module-level cleanup function ──────────────────────────────────────────

def purge_old_turns(days: int = HISTORY_TTL_DAYS) -> int:
    """
    Delete history turns older than *days* days from the vector store.

    Call this once at application startup (see main.py).  ISO 8601 timestamps
    are lexicographically sortable, so string comparison is used to identify
    stale records without deserialising every timestamp.

    Args:
        days: Turns with a timestamp older than this many days are removed.

    Returns:
        Number of turns deleted (0 on any failure).
    """
    from src.ingestion.vector_store import VECTOR_STORE_BACKEND

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        store = get_history_store()

        if VECTOR_STORE_BACKEND != "chroma":
            # FAISS history store: no server-side filtering — skip for now.
            logger.debug("History purge skipped: FAISS backend does not support it.")
            return 0

        # Fetch only the metadata for all turn-type documents (avoids loading
        # large embedding vectors into memory).
        raw = store._collection.get(
            where={"type": "turn"},
            include=["metadatas"],
        )

        old_ids = [
            doc_id
            for doc_id, meta in zip(raw["ids"], raw["metadatas"])
            if meta.get("timestamp", "9999-99-99") < cutoff_iso
        ]

        if old_ids:
            with _WRITE_LOCK:
                store._collection.delete(ids=old_ids)
            logger.info("Purged %d history turn(s) older than %d day(s).",
                        len(old_ids), days)
        else:
            logger.debug("History purge: no turns older than %d day(s).", days)

        del store
        return len(old_ids)

    except Exception as exc:
        logger.warning("History purge failed (non-fatal): %s", exc)
        return 0
