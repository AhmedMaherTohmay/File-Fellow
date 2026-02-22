"""
RAG-based Q&A chain with:
  - Multi-document support (query one or all docs).
  - Graceful normal conversation even when no documents are ingested.
  - Semantic session history retrieval from vector store.
  - Source citations in every response.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.retrieval.retriever import retrieve_chunks
from src.llm.llm_factory import get_llm
from src.llm.prompts import QA_PROMPT
from src.memory.history_store import HistoryStore

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6  # Recent turns to include in the prompt


# ── Formatting helpers ─────────────────────────────────────────────────────

def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    """Format retrieved chunks with source citations for the prompt."""
    if not chunks:
        return "No document context available."

    parts = []
    for doc, score in chunks:
        m = doc.metadata
        citation = (
            f"[Source: {m.get('source', '?')}, "
            f"Page: {m.get('page', '?')}, "
            f"Score: {score:.3f}]"
        )
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[dict]) -> str:
    """Format the last N conversation turns for the prompt."""
    if not history:
        return "No recent conversation."
    # Take only the most recent turns to keep the prompt concise
    recent = history[-(MAX_HISTORY_TURNS * 2):]
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
) -> dict:
    """Generate a response to the user's message.

    Handles two modes automatically:
      - **Conversational**: No documents loaded, or a casual greeting/question.
        The LLM responds naturally without document grounding.
      - **Document Q&A**: Documents are loaded. Retrieved chunks are injected
        into the prompt and the LLM cites its sources.

    Args:
        question: The user's message or question text.
        history: Recent conversation as a list of {role, content} dicts.
        doc_name: Target document name. None = query across all documents.
        session_id: Session identifier used for history store isolation.

    Returns:
        Dict with keys:
          - ``answer`` (str): The LLM's response.
          - ``sources`` (list): Source citation dicts (empty for casual chat).
          - ``retrieved_chunks`` (list): Raw (Document, score) pairs retrieved.
    """
    history = history or []

    # ── Semantic history retrieval ─────────────────────────────────────────
    # Fetch past turns from the persistent vector history store that are
    # semantically relevant to the current question.
    history_mgr = HistoryStore(session_id=session_id)
    relevant_history = history_mgr.retrieve_relevant(question)
    semantic_history_str = history_mgr.format_for_prompt(relevant_history)

    # ── Document retrieval ────────────────────────────────────────────────
    # retrieve_chunks returns [] if no store is ready or nothing passes threshold.
    retrieved = retrieve_chunks(question, doc_name=doc_name)

    # Build context string — "No document context available." when empty,
    # which the prompt handles gracefully for normal conversation.
    context_str = _format_context(retrieved)
    history_str = _format_history(history)

    # ── LLM call ──────────────────────────────────────────────────────────
    llm = get_llm()
    chain = QA_PROMPT | llm

    response = chain.invoke(
        {
            "context": context_str,
            "history": history_str,
            "semantic_history": semantic_history_str,
            "question": question,
        }
    )

    answer = response.content if hasattr(response, "content") else str(response)

    # ── Source citations ───────────────────────────────────────────────────
    sources = []
    for doc, score in retrieved:
        m = doc.metadata
        sources.append(
            {
                "source": m.get("source", "?"),
                "page": m.get("page", "?"),
                "score": round(score, 4),
                "snippet": doc.page_content[:250],
                "doc_id": m.get("doc_id", "?"),
            }
        )

    # ── Persist this turn to the history store ────────────────────────────
    history_mgr.add_turn(user=question, assistant=answer)

    logger.debug(
        "Q: '%s' | chunks=%d | session=%s",
        question[:60],
        len(retrieved),
        session_id,
    )
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
    }
