from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.storage.document_store import get_chunks_for_doc
from src.llm.llm_factory import get_llm
from src.llm.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

SUMMARY_SEGMENT_CHARS = 12_000
MAX_SUMMARY_SEGMENTS = 5


def _summarize_text(text: str) -> str:
    llm = get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"contract_text": text})
    return response.content if hasattr(response, "content") else str(response)


def _group_chunks_into_segments(chunks: List[Document]) -> List[str]:
    segments: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for chunk in chunks:
        text = chunk.page_content
        if current_len + len(text) > SUMMARY_SEGMENT_CHARS and current_parts:
            segments.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(text)
        current_len += len(text)

    if current_parts:
        segments.append("\n\n".join(current_parts))

    return segments[:MAX_SUMMARY_SEGMENTS]


def summarize_document(filename: str, user_id: Optional[str] = None) -> str:
    chunks = get_chunks_for_doc(filename, user_id=user_id)
    if not chunks:
        raise ValueError(
            f"No chunks found for '{filename}'. "
            "The document may not be ingested or the vector store may be unavailable."
        )

    segments = _group_chunks_into_segments(chunks)

    if len(segments) == 1:
        logger.info("Summarizing '%s' in a single pass (%d chunks).", filename, len(chunks))
        return _summarize_text(segments[0])

    logger.info(
        "Summarizing '%s' in %d segments (%d chunks, map-reduce).",
        filename, len(segments), len(chunks),
    )
    partial_summaries = [_summarize_text(seg) for seg in segments]
    combined = "\n\n===\n\n".join(partial_summaries)

    llm = get_llm()
    response = llm.invoke(
        "Merge these partial summaries into one coherent, structured summary:\n\n" + combined
    )
    return response.content if hasattr(response, "content") else str(response)
