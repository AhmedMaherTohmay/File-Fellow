from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import MAX_SESSION_TURNS
from src.retrieval.retriever import retrieve_chunks
from src.llm.llm_factory import get_llm
from src.llm.prompts import QA_PROMPT
from src.storage.history_store import HistoryStore

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    if not chunks:
        return "No document context available."
    parts = []
    for doc, _score in chunks:
        m = doc.metadata
        citation = f"[Source: {m.get('source', '?')}, Page: {m.get('page', '?')}]"
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_history(history: List[dict]) -> str:
    if not history:
        return "No recent conversation."
    recent = history[-(MAX_SESSION_TURNS * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')[:300]}")
    return "\n".join(lines)


def answer_question(
    question: str,
    history: Optional[List[dict]] = None,
    doc_name: Optional[str] = None,
    session_id: str = "default",
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> dict:
    history = history or []
    effective_user_id = user_id or session_id

    history_mgr = HistoryStore(user_id=effective_user_id, conversation_id=conversation_id)
    relevant_history = history_mgr.retrieve_relevant(
        question,
        exclude_conversation_id=conversation_id if history else None,
    )
    semantic_history_str = history_mgr.format_for_prompt(relevant_history)

    retrieved = retrieve_chunks(question, doc_name=doc_name)
    context_str = _format_context(retrieved)
    history_str = _format_history(history)

    llm = get_llm()
    chain = QA_PROMPT | llm
    response = chain.invoke({
        "context": context_str,
        "history": history_str,
        "semantic_history": semantic_history_str,
        "question": question,
    })
    answer = response.content if hasattr(response, "content") else str(response)

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

    history_mgr.add_turn(user=question, assistant=answer)

    logger.debug("Q: '%s' | chunks=%d | user=%s | conv=%s",
                 question[:60], len(retrieved), effective_user_id, conversation_id)
    return {"answer": answer, "sources": sources, "retrieved_chunks": retrieved}
