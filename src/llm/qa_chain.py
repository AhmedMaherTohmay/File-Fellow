"""
RAG-based Q&A chain.
Retrieves relevant chunks, builds a grounded prompt, and calls the LLM.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from src.retrieval.retriever import retrieve_chunks
from src.llm.llm_factory import get_llm
from src.llm.prompts import QA_PROMPT

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 6  # Keep last N user/assistant pairs


def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    """Convert retrieved chunks into a formatted context string with citations."""
    parts = []
    for doc, score in chunks:
        m = doc.metadata
        citation = f"[Source: {m.get('source','?')}, Page: {m.get('page','?')}, Chunk: {m.get('chunk_id','?')}]"
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[dict]) -> str:
    """Render conversation history for the prompt."""
    if not history:
        return "No previous conversation."
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def answer_question(
    question: str,
    history: List[dict] | None = None,
) -> dict:
    """Generate a grounded answer to the user's question.

    Args:
        question: The user's question text.
        history: List of ``{role, content}`` dicts representing conversation.

    Returns:
        Dict with ``answer``, ``sources``, ``retrieved_chunks``.
    """
    history = history or []

    # Retrieval
    retrieved = retrieve_chunks(question)

    if not retrieved:
        return {
            "answer": "Information not found in the document.",
            "sources": [],
            "retrieved_chunks": [],
        }

    context_str = _format_context(retrieved)
    history_str = _format_history(history)

    # Build and invoke chain
    llm = get_llm()
    chain = QA_PROMPT | llm

    response = chain.invoke(
        {"context": context_str, "history": history_str, "question": question}
    )

    # Extract content from AIMessage or plain string
    answer = response.content if hasattr(response, "content") else str(response)

    # Collect source citations
    sources = []
    for doc, score in retrieved:
        m = doc.metadata
        sources.append(
            {
                "source": m.get("source", "?"),
                "page": m.get("page", "?"),
                "chunk_id": m.get("chunk_id", "?"),
                "score": round(score, 4),
                "snippet": doc.page_content[:200],
            }
        )

    logger.debug("Q: %s | chunks=%d", question[:60], len(retrieved))
    return {"answer": answer, "sources": sources, "retrieved_chunks": retrieved}
