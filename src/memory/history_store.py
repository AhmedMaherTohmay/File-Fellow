"""
Semantic chat history persistence.

Architecture:
  - Each conversation turn (user + assistant pair) is embedded and stored
    in a dedicated Chroma collection ("chat_history").
  - At the start of each conversation, relevant past turns are retrieved
    by semantic similarity to the current query.
  - After a conversation ends (or on each turn), new turns are persisted.
  - Sessions are identified by a session_id for per-user isolation.

Usage:
    from src.memory.history_store import HistoryStore

    store = HistoryStore(session_id="user-abc-123")
    past_context = store.retrieve_relevant(query="termination clauses", k=3)
    store.add_turn(user="What is the payment schedule?", assistant="...")
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import SESSION_HISTORY_TOP_K, SIMILARITY_THRESHOLD
from src.ingestion.vector_store import get_history_store

logger = logging.getLogger(__name__)


class HistoryStore:
    """Manages per-session semantic chat history in a vector store."""

    def __init__(self, session_id: str = "default"):
        """
        Args:
            session_id: Unique identifier for this conversation session.
                        Used as metadata filter for isolation.
        """
        self.session_id = session_id
        self._store = None

    def _get_store(self):
        if self._store is None:
            self._store = get_history_store()
        return self._store

    def retrieve_relevant(
        self,
        query: str,
        k: int = SESSION_HISTORY_TOP_K,
        score_threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[dict]:
        """Retrieve semantically relevant past turns for a query.

        Args:
            query: The current user query.
            k: Max number of past turns to retrieve.
            score_threshold: Minimum similarity to include.

        Returns:
            List of dicts: {role, content, timestamp, session_id}.
        """
        try:
            store = self._get_store()
            results = store.similarity_search_with_relevance_scores(query, k=k * 2)

            relevant = []
            for doc, score in results:
                meta = doc.metadata
                # Filter by session if session_id is set (skip init marker)
                if meta.get("type") == "init":
                    continue
                if self.session_id != "default" and meta.get("session_id") not in (
                    self.session_id,
                    "default",
                    None,
                ):
                    continue
                if score >= score_threshold:
                    relevant.append(
                        {
                            "role": meta.get("role", "user"),
                            "content": doc.page_content,
                            "timestamp": meta.get("timestamp", ""),
                            "session_id": meta.get("session_id", ""),
                            "score": round(score, 4),
                        }
                    )
                if len(relevant) >= k:
                    break

            logger.debug(
                "Retrieved %d relevant history items for session '%s'.",
                len(relevant),
                self.session_id,
            )
            return relevant

        except Exception as e:
            logger.warning("History retrieval failed: %s", e)
            return []

    def add_turn(self, user: str, assistant: str) -> None:
        """Persist a conversation turn (user + assistant pair).

        Args:
            user: User's message.
            assistant: Assistant's response.
        """
        try:
            store = self._get_store()
            ts = datetime.now(timezone.utc).isoformat()

            docs = [
                Document(
                    page_content=user,
                    metadata={
                        "role": "user",
                        "session_id": self.session_id,
                        "timestamp": ts,
                        "type": "turn",
                    },
                ),
                Document(
                    page_content=assistant[:1000],  # Truncate long answers
                    metadata={
                        "role": "assistant",
                        "session_id": self.session_id,
                        "timestamp": ts,
                        "type": "turn",
                    },
                ),
            ]
            store.add_documents(docs)
            logger.debug(
                "Persisted turn to history store (session='%s').", self.session_id
            )
        except Exception as e:
            logger.warning("History persistence failed: %s", e)

    def add_bulk(self, history: List[dict]) -> None:
        """Bulk-persist a completed conversation history.

        Args:
            history: List of {role, content} dicts.
        """
        for i in range(0, len(history) - 1, 2):
            user_msg = history[i].get("content", "") if history[i].get("role") == "user" else ""
            asst_msg = history[i + 1].get("content", "") if i + 1 < len(history) and history[i + 1].get("role") == "assistant" else ""
            if user_msg and asst_msg:
                self.add_turn(user_msg, asst_msg)

    def format_for_prompt(self, relevant_history: List[dict]) -> str:
        """Format retrieved history as a string for inclusion in the prompt.

        Args:
            relevant_history: Output from retrieve_relevant().

        Returns:
            Multi-line string of past conversation context.
        """
        if not relevant_history:
            return "No relevant conversation history found."

        lines = ["[Relevant past conversation context:]"]
        for item in relevant_history:
            role_label = "User" if item["role"] == "user" else "Assistant"
            ts = item.get("timestamp", "")[:10]  # Just the date
            lines.append(f"[{ts}] {role_label}: {item['content'][:300]}")
        return "\n".join(lines)
