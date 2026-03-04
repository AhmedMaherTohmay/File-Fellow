"""
RAG-based Q&A chain with:
  - Multi-document support (query one or all docs).
  - Graceful normal conversation even when no documents are ingested.
  - Semantic cross-session history retrieval from vector store.
  - Source citations in every response (kept out of the LLM context).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import MAX_SESSION_TURNS
from src.retrieval.retriever import retrieve_chunks
from src.llm.llm_factory import get_llm
from src.llm.prompts import QA_PROMPT
from src.memory.history_store import HistoryStore

logger = logging.getLogger(__name__)


# ── Formatting helpers ─────────────────────────────────────────────────────

def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved chunks for the LLM prompt.

    Scores are intentionally omitted from the text sent to the LLM.
    Floating-point score values have been observed to bias the model's
    confidence about which chunk to trust.  Source and page metadata are
    preserved so the LLM can produce accurate citations.
    """
    if not chunks:
        return "No document context available."

    parts = []
    for doc, _score in chunks:
        m = doc.metadata
        citation = f"[Source: {m.get('source', '?')}, Page: {m.get('page', '?')}]"
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[dict]) -> str:
    """
    Format the last MAX_SESSION_TURNS turns for the LLM prompt.

    Uses MAX_SESSION_TURNS from config — a single source of truth — so the
    in-prompt history window and the UI display window stay in sync.
    """
    if not history:
        return "No recent conversation."
    recent = history[-(MAX_SESSION_TURNS * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')[:300]}")
    return "\n".join(lines)


# ── Main Q&A function ──────────────────────────────────────────────────────

def answer_question(
    question: str,
    history: Optional[List[dict]] = None,
    doc_name: Optional[str] = None,
    session_id: str = "default",
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> dict:
    """
    Generate a response to the user's message.

    Args:
        question:        The user's message or question text.
        history:         Recent in-session conversation as [{role, content}]
                         dicts.  Must contain only plain answer text — no
                         citation footers — so the LLM gets clean context.
        doc_name:        Target document. None = search all documents.
        session_id:      Legacy parameter kept for API backward-compatibility.
                         New callers should use ``user_id`` instead.
        user_id:         Stable user identifier for history store filtering.
                         Falls back to ``session_id`` if not provided.
        conversation_id: Current conversation UUID.  Used to exclude the
                         current conversation's turns from semantic history
                         retrieval (they are already in ``history``).

    Returns:
        Dict with keys:
          - ``answer``           (str)  Plain LLM response, no footer.
          - ``sources``          (list) Citation dicts for UI display.
          - ``retrieved_chunks`` (list) Raw (Document, score) pairs.
    """
    history = history or []

    # Resolve the effective user identifier for history store access.
    # ``user_id`` is preferred; fall back to ``session_id`` for callers
    # that pre-date the user_id field (e.g. the LangServe playground).
    effective_user_id = user_id or session_id

    # ── Semantic history retrieval ─────────────────────────────────────────
    # Fetch semantically relevant past turns from *other* conversations.
    # The current conversation is excluded (``conversation_id`` hint) because
    # it is already present in the ``history`` argument — passing it twice
    # would waste tokens and potentially confuse the model.
    history_mgr = HistoryStore(
        user_id=effective_user_id,
        conversation_id=conversation_id,
    )
    relevant_history = history_mgr.retrieve_relevant(
        question,
        # Only exclude the current conversation when there is already active
        # in-session history.  If history is empty this is the first turn and
        # there is nothing to exclude.
        exclude_conversation_id=conversation_id if history else None,
    )
    semantic_history_str = history_mgr.format_for_prompt(relevant_history)

    # ── Document retrieval ────────────────────────────────────────────────
    retrieved = retrieve_chunks(question, doc_name=doc_name)
    context_str = _format_context(retrieved)  # scores omitted from LLM context
    history_str = _format_history(history)

    # ── LLM call ──────────────────────────────────────────────────────────
    llm = get_llm()
    chain = QA_PROMPT | llm
    response = chain.invoke({
        "context": context_str,
        "history": history_str,
        "semantic_history": semantic_history_str,
        "question": question,
    })
    answer = response.content if hasattr(response, "content") else str(response)

    # ── Source citations (for UI only — NOT part of the answer string) ─────
    sources = []
    for doc, score in retrieved:
        m = doc.metadata
        sources.append({
            "source": m.get("source", "?"),
            "page": m.get("page", "?"),
            "score": round(score, 4),
            "snippet": doc.page_content[:250],
            "doc_id": m.get("doc_id", "?"),
        })

    # ── Persist this turn ─────────────────────────────────────────────────
    # Store the plain answer — not the answer + sources markdown — so that
    # future semantic retrievals see clean, uncluttered assistant text.
    history_mgr.add_turn(user=question, assistant=answer)

    logger.debug("Q: '%s' | chunks=%d | user=%s | conv=%s",
                 question[:60], len(retrieved), effective_user_id, conversation_id)
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
    }
