from __future__ import annotations

import logging
from typing import List, Optional

from src.db.repositories.document_repo import get_chunks_for_doc
from src.llm.llm_factory import get_llm
from src.llm.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

SUMMARY_SEGMENT_CHARS = 12_000
MAX_SUMMARY_SEGMENTS  = 5


def _summarize_text(text: str) -> str:
    llm = get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"contract_text": text})
    return response.content if hasattr(response, "content") else str(response)


def _group_chunks_into_segments(chunks) -> tuple[List[str], bool]:
    """
    Group chunks into segments of at most SUMMARY_SEGMENT_CHARS each.

    Returns (segments, was_truncated).
    was_truncated is True when the document exceeded MAX_SUMMARY_SEGMENTS
    segments and the tail was dropped.
    """
    all_segments: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for chunk in chunks:
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        if current_len + len(text) > SUMMARY_SEGMENT_CHARS and current_parts:
            all_segments.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(text)
        current_len += len(text)

    if current_parts:
        all_segments.append("\n\n".join(current_parts))

    truncated = len(all_segments) > MAX_SUMMARY_SEGMENTS
    return all_segments[:MAX_SUMMARY_SEGMENTS], truncated


def summarize_document(filename: str, user_id: Optional[str] = None) -> str:
    chunks = get_chunks_for_doc(filename, user_id=user_id)
    if not chunks:
        raise ValueError(
            f"No chunks found for '{filename}'. "
            "The document may not be ingested yet."
        )

    segments, was_truncated = _group_chunks_into_segments(chunks)

    if was_truncated:
        logger.warning(
            "Document '%s' exceeds the %d-segment summary limit (~%d chars). "
            "The summary covers the first ~%d characters only.",
            filename,
            MAX_SUMMARY_SEGMENTS,
            MAX_SUMMARY_SEGMENTS * SUMMARY_SEGMENT_CHARS,
            MAX_SUMMARY_SEGMENTS * SUMMARY_SEGMENT_CHARS,
        )

    truncation_note = (
        "\n\n---\n*Note: This document is large. "
        "The summary covers the first portion of the content only.*"
        if was_truncated else ""
    )

    if len(segments) == 1:
        logger.info("Summarizing '%s' in a single pass (%d chunks).", filename, len(chunks))
        return _summarize_text(segments[0]) + truncation_note

    logger.info(
        "Summarizing '%s' in %d segment(s) (%d chunks, map-reduce).",
        filename, len(segments), len(chunks),
    )
    partial_summaries = [_summarize_text(seg) for seg in segments]
    combined = "\n\n===\n\n".join(partial_summaries)

    llm = get_llm()
    response = llm.invoke(
        "Merge these partial summaries into one coherent, structured summary:\n\n" + combined
    )
    result = response.content if hasattr(response, "content") else str(response)
    return result + truncation_note
