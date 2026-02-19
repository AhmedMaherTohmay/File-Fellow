"""
Contract summarization.
Uses a map-reduce approach for large documents to stay within context limits.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import UPLOAD_DIR
from src.ingestion.parser import parse_document
from src.llm.llm_factory import get_llm
from src.llm.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Max characters sent to the LLM in a single summarization call
SUMMARY_CHUNK_CHARS = 12_000
MAX_SUMMARY_CHUNKS = 5  # For map-reduce on very large docs


def _summarize_text(text: str) -> str:
    """Summarize a single text block."""
    llm = get_llm()
    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"contract_text": text})
    return response.content if hasattr(response, "content") else str(response)


def summarize_document(filename: str) -> str:
    """Produce a structured summary of an ingested document.

    Uses map-reduce: summarizes chunks individually then merges.

    Args:
        filename: Name of the file previously uploaded (in UPLOAD_DIR).

    Returns:
        Summary text.
    """
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"'{filename}' not found in uploads directory.")

    pages = parse_document(file_path)
    full_text = "\n\n".join(p["text"] for p in pages)

    # Split into at most MAX_SUMMARY_CHUNKS pieces
    step = max(len(full_text) // MAX_SUMMARY_CHUNKS, SUMMARY_CHUNK_CHARS)
    segments = [full_text[i : i + step] for i in range(0, len(full_text), step)]
    segments = segments[:MAX_SUMMARY_CHUNKS]

    if len(segments) == 1:
        logger.info("Summarizing document in a single pass.")
        return _summarize_text(segments[0])

    # Map
    logger.info("Summarizing document in %d segments (map-reduce).", len(segments))
    partial_summaries = [_summarize_text(seg) for seg in segments]

    # Reduce
    combined = "\n\n===\n\n".join(partial_summaries)
    final_prompt = (
        "You have several partial summaries of the same contract. "
        "Merge them into one coherent, structured summary:\n\n" + combined
    )
    llm = get_llm()
    response = llm.invoke(final_prompt)
    return response.content if hasattr(response, "content") else str(response)
