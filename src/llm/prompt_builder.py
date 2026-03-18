"""
Prompt input assembly — the last step before the LLM call.

This module owns the translation from retrieved data structures into
the plain strings the LLM prompt template expects. It sits at the
boundary between the retrieval/service world and the LLM world.

Why this exists as a separate module
--------------------------------------
qa_service.py orchestrates the pipeline (retrieve → assemble → call → persist).
prompts.py holds the prompt templates.
prompt_builder.py bridges them: it knows the shape of retrieved data and
the shape the prompt templates expect, and it converts between the two.

This separation means:
  - The service stays focused on pipeline flow, not string formatting.
  - Prompt formatting is independently testable: feed it chunks, get a string.
  - When the prompt template changes, only this file changes — not the service.
  - When the retrieval output shape changes, only this file changes — not the
    prompt template.

What lives here
---------------
  format_document_context(chunks)     → context block for the prompt
  format_recent_history(turns)        → recent conversation block
  format_semantic_history(turns)      → past relevant context block
  build_prompt_inputs(...)            → final dict passed to the LLM chain
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings


def format_document_context(
    chunks: List[Tuple[Document, float]],
) -> str:
    """
    Format retrieved document chunks into the context block injected into
    the LLM prompt.

    Each chunk is prefixed with a citation so the model can reference the
    source document and page number in its answer.

    Args:
        chunks: List of (Document, score) pairs from the document retriever.

    Returns:
        A formatted string ready for the {context} placeholder in QA_PROMPT.
        Returns a "no context" message if the list is empty — the prompt
        template always receives a non-empty string.
    """
    if not chunks:
        return "No document context available."

    parts = []
    for doc, _score in chunks:
        m = doc.metadata
        citation = f"[Source: {m.get('source', '?')}, Page: {m.get('page', '?')}]"
        parts.append(f"{citation}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def format_recent_history(
    turns: List[dict],
    max_turns: Optional[int] = None,
) -> str:
    """
    Format the recent conversation window for the {history} placeholder.

    Turns are role/content dicts — the same shape returned by
    history_retriever.retrieve_recent_turns() and appended by the UI.

    Args:
        turns:     List of {"role": "user"|"assistant", "content": str} dicts.
        max_turns: Cap on the number of turns to include. Defaults to
                   settings.MAX_SESSION_TURNS * 2 (pairs of user + assistant).

    Returns:
        Multi-line string "User: ...\nAssistant: ..." ready for the prompt.
    """
    if not turns:
        return "No recent conversation."

    cap = (max_turns or settings.MAX_SESSION_TURNS) * 2
    lines = []
    for msg in turns[-cap:]:
        role  = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')[:300]}")

    return "\n".join(lines)


def format_semantic_history(
    relevant_history: List[dict],
) -> str:
    """
    Format semantically retrieved past turns for the {semantic_history}
    placeholder.

    Past turns are retrieved from previous conversations — not the current
    one — to give the model continuity across sessions.

    Args:
        relevant_history: List of dicts with keys: role, content, timestamp, score.
                          Returned by history_retriever.retrieve_relevant_history().

    Returns:
        Formatted string with timestamped past exchanges.
    """
    if not relevant_history:
        return "No relevant conversation history found."

    lines = ["[Relevant past conversation context:]"]
    for item in relevant_history:
        role_label = "User" if item["role"] == "user" else "Assistant"
        ts = item.get("timestamp", "")[:10]
        lines.append(f"[{ts}] {role_label}: {item['content'][:300]}")

    return "\n".join(lines)


def build_prompt_inputs(
    question: str,
    chunks: List[Tuple[Document, float]],
    recent_turns: List[dict],
    semantic_history: List[dict],
) -> dict:
    """
    Assemble the complete input dict for the QA_PROMPT chain.

    This is the single call the service makes — it receives the raw
    retrieved data structures and returns the string-keyed dict that
    QA_PROMPT | llm expects.

    Args:
        question:         The user's current question.
        chunks:           Retrieved document chunks with scores.
        recent_turns:     Recent turns from the current conversation.
        semantic_history: Semantically relevant past turns.

    Returns:
        Dict with keys: context, history, semantic_history, question.
        All values are plain strings — no LangChain objects.
    """
    return {
        "context":          format_document_context(chunks),
        "history":          format_recent_history(recent_turns),
        "semantic_history": format_semantic_history(semantic_history),
        "question":         question,
    }
