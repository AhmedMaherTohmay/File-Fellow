from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import SESSION_HISTORY_TOP_K, HISTORY_SCORE_THRESHOLD, HISTORY_TTL_DAYS
from src.core.utils import normalise_score
from src.storage.document_store import get_history_store, _WRITE_LOCK

logger = logging.getLogger(__name__)


class HistoryStore:
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
        effective_exclude = exclude_conversation_id or self.conversation_id
        try:
            store = self._get_store()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                raw_results = store.similarity_search_with_relevance_scores(query, k=k * 6)

            relevant = []
            for doc, raw in raw_results:
                meta = doc.metadata
                if meta.get("type") == "init":
                    continue
                stored_user = meta.get("user_id") or meta.get("session_id", "")
                if stored_user != self.user_id:
                    continue
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

            logger.debug("Retrieved %d history items for user '%s'.", len(relevant), self.user_id)
            return relevant
        except Exception as exc:
            logger.warning("History retrieval failed: %s", exc)
            return []

    def add_turn(self, user: str, assistant: str) -> None:
        try:
            store = self._get_store()
            ts = datetime.now(timezone.utc).isoformat()
            docs = [
                Document(
                    page_content=user,
                    metadata={
                        "role": "user",
                        "user_id": self.user_id,
                        "session_id": self.user_id,
                        "conversation_id": self.conversation_id or "",
                        "timestamp": ts,
                        "type": "turn",
                    },
                ),
                Document(
                    page_content=assistant[:1000],
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
            logger.debug("Persisted turn (user='%s', conv='%s').", self.user_id, self.conversation_id)
        except Exception as exc:
            logger.warning("History persistence failed: %s", exc)

    def format_for_prompt(self, relevant_history: List[dict]) -> str:
        if not relevant_history:
            return "No relevant conversation history found."
        lines = ["[Relevant past conversation context:]"]
        for item in relevant_history:
            role_label = "User" if item["role"] == "user" else "Assistant"
            ts = item.get("timestamp", "")[:10]
            lines.append(f"[{ts}] {role_label}: {item['content'][:300]}")
        return "\n".join(lines)


def purge_old_turns(days: int = HISTORY_TTL_DAYS) -> int:
    from src.storage.document_store import VECTOR_STORE_BACKEND

    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        store = get_history_store()
        if VECTOR_STORE_BACKEND != "chroma":
            logger.debug("History purge skipped: FAISS backend does not support it.")
            return 0

        raw = store._collection.get(where={"type": "turn"}, include=["metadatas"])
        old_ids = [
            doc_id
            for doc_id, meta in zip(raw["ids"], raw["metadatas"])
            if meta.get("timestamp", "9999-99-99") < cutoff_iso
        ]
        if old_ids:
            with _WRITE_LOCK:
                store._collection.delete(ids=old_ids)
            logger.info("Purged %d history turn(s) older than %d day(s).", len(old_ids), days)
        else:
            logger.debug("History purge: no turns older than %d day(s).", days)

        del store
        return len(old_ids)
    except Exception as exc:
        logger.warning("History purge failed (non-fatal): %s", exc)
        return 0
