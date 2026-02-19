"""
Top-level ingestion pipeline.
Orchestrates: parse → chunk → embed → store.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from config.settings import UPLOAD_DIR
from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_pages
from src.ingestion.vector_store import build_vector_store

logger = logging.getLogger(__name__)


def ingest_document(file_path: str | Path) -> dict:
    """Full ingestion pipeline for a single PDF or DOCX.

    Args:
        file_path: Path to the uploaded document.

    Returns:
        Dict with ``num_pages``, ``num_chunks``, ``filename``.
    """
    file_path = Path(file_path)

    # Copy to uploads dir for provenance
    dest = UPLOAD_DIR / file_path.name
    if file_path.resolve() != dest.resolve():
        shutil.copy2(file_path, dest)

    logger.info("Starting ingestion for '%s'.", file_path.name)

    pages = parse_document(file_path)
    if not pages:
        raise ValueError(f"No text could be extracted from '{file_path.name}'.")

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("Chunking produced no output.")

    build_vector_store(chunks)

    result = {
        "filename": file_path.name,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
    }
    logger.info("Ingestion complete: %s", result)
    return result
