"""
Contract summarization using map-reduce for large documents.
Supports summarizing a specific document or all ingested documents.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import UPLOAD_DIR
from src.ingestion.parser import parse_document
from src.llm.llm_factory import get_llm
from src.llm.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

SUMMARY_CHUNK_CHARS = 12_000
MAX_SUMMARY_CHUNKS = 5


def _summarize_text(text: str) -> str:
    """Summarize a single text block."""
    llm = get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"contract_text": text})
    return response.content if hasattr(response, "content") else str(response)


def summarize_document(filename: str) -> str:
    """Produce a structured summary of an ingested document.

    Uses map-reduce: summarizes large docs in chunks then merges results.

    Args:
        filename: Name of the previously uploaded file (in UPLOAD_DIR).

    Returns:
        Structured summary text.

    Raises:
        FileNotFoundError: If the file is not in the uploads directory.
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"'{filename}' not found in uploads directory.")

    pages = parse_document(file_path)
    full_text = "\n\n".join(p["text"] for p in pages)

    if not full_text.strip():
        return "Could not extract text from this document."

    # Split into at most MAX_SUMMARY_CHUNKS segments
    step = max(len(full_text) // MAX_SUMMARY_CHUNKS, SUMMARY_CHUNK_CHARS)
    segments = [full_text[i : i + step] for i in range(0, len(full_text), step)]
    segments = segments[:MAX_SUMMARY_CHUNKS]

    if len(segments) == 1:
        logger.info("Summarizing '%s' in a single pass.", filename)
        return _summarize_text(segments[0])

    # Map: summarize each segment
    logger.info("Summarizing '%s' in %d segments (map-reduce).", filename, len(segments))
    partial_summaries = [_summarize_text(seg) for seg in segments]

    # Reduce: merge partial summaries
    combined = "\n\n===\n\n".join(partial_summaries)
    merge_prompt = (
        "You have several partial summaries of the same document. "
        "Merge them into one coherent, structured summary covering all key aspects:\n\n"
        + combined
    )
    llm = get_llm()
    response = llm.invoke(merge_prompt)
    return response.content if hasattr(response, "content") else str(response)
