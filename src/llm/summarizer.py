"""
Document summarization using map-reduce over stored chunks.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from src.ingestion.vector_store import get_chunks_for_doc
from src.llm.llm_factory import get_llm
from src.llm.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Target character budget per summarization segment.
SUMMARY_SEGMENT_CHARS = 12_000

# Maximum number of map segments before the reduce step.
# Prevents runaway LLM calls on extremely large documents.
MAX_SUMMARY_SEGMENTS = 5


def _summarize_text(text: str) -> str:
    """Summarize a single text block using the summary prompt."""
    llm = get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"contract_text": text})
    return response.content if hasattr(response, "content") else str(response)


def _group_chunks_into_segments(chunks: List[Document]) -> List[str]:
    """
    Group ordered chunks into segments of at most SUMMARY_SEGMENT_CHARS.
    """
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


def summarize_document(filename: str) -> str:
    """
    Produce a structured summary of an ingested document.

    Retrieves the document's chunks from the vector store (ordered by
    global_chunk_index), groups them into segments, and applies map-reduce
    summarization.
    """
    chunks = get_chunks_for_doc(filename)

    if not chunks:
        raise ValueError(
            f"No chunks found for '{filename}'. "
            "The document may not be ingested or the vector store may be unavailable."
        )

    segments = _group_chunks_into_segments(chunks)

    if len(segments) == 1:
        logger.info("Summarizing '%s' in a single pass (%d chunks).", filename, len(chunks))
        return _summarize_text(segments[0])

    # Map phase: summarize each segment independently.
    logger.info("Summarizing '%s' in %d segments (%d chunks total, map-reduce).",
                filename, len(segments), len(chunks))
    partial_summaries = [_summarize_text(seg) for seg in segments]

    # Reduce phase: merge partial summaries into one coherent summary.
    combined = "\n\n===\n\n".join(partial_summaries)
    merge_prompt = (
        "You have several partial summaries of the same document. "
        "Merge them into one coherent, structured summary covering all key aspects:\n\n"
        + combined
    )
    llm = get_llm()
    response = llm.invoke(merge_prompt)
    return response.content if hasattr(response, "content") else str(response)
