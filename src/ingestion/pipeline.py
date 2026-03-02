"""
Document ingestion pipeline.

Orchestrates the full ingest flow:
  sanitize → copy to UPLOAD_DIR → parse → chunk → store

Public API
----------
ingest_document(file_path) → dict
    Run the full pipeline for a single file.  Returns metadata about
    the ingested document (filename, num_pages, num_chunks).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from config.settings import UPLOAD_DIR
from src.core.utils import sanitize_filename
from src.core.exceptions import ExtractionError
from src.ingestion.chunker import chunk_pages
from src.ingestion.parser import parse_document
from src.ingestion.vector_store import add_document

logger = logging.getLogger(__name__)


def ingest_document(file_path: "str | Path") -> Dict[str, Any]:
    """
    Full ingestion pipeline: sanitize → copy → parse → chunk → store.

    Args:
        file_path: Path to the file to ingest.  May be anywhere on disk;
                   the pipeline copies it into UPLOAD_DIR under its
                   sanitised name before processing.

    Returns:
        Dict with keys: ``filename``, ``num_pages``, ``num_chunks``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ExtractionError:   If the parser returns no usable text.
        ValueError:        If the file extension is not supported.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    # ── 1. Sanitize filename ──────────────────────────────────────────────
    safe_name = sanitize_filename(file_path.name)

    if safe_name != file_path.name:
        logger.info(
            "Filename sanitized: '%s' → '%s'", file_path.name, safe_name
        )

    # ── 2. Ensure file lives in UPLOAD_DIR under its safe name ───────────
    dest = UPLOAD_DIR / safe_name

    if file_path != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        logger.debug("Copied '%s' → '%s'.", file_path, dest)

        # Remove the unsafe-named original only if it was already inside
        # UPLOAD_DIR (avoids deleting files the caller still owns).
        if file_path.parent.resolve() == UPLOAD_DIR.resolve():
            try:
                file_path.unlink()
                logger.debug("Removed unsafe-named original '%s'.", file_path)
            except OSError as exc:
                logger.warning(
                    "Could not remove unsafe-named original '%s': %s",
                    file_path, exc,
                )

    # ── 3. Parse ─────────────────────────────────────────────────────────
    pages = parse_document(dest)
    if not pages:
        # Use the typed exception so the API/UI can catch it specifically
        raise ExtractionError(safe_name)

    # ── 4. Chunk ─────────────────────────────────────────────────────────
    # doc_id uses the sanitised name so chunk IDs are stable across re-ingests
    chunks = chunk_pages(pages, doc_id=safe_name)

    # ── 5. Store ─────────────────────────────────────────────────────────
    add_document(
        doc_name=safe_name,
        chunks=chunks,
        metadata={
            "num_pages": len(pages),
            "num_chunks": len(chunks),
        },
    )

    logger.info(
        "Ingested '%s': %d page(s), %d chunk(s).",
        safe_name, len(pages), len(chunks),
    )
    return {
        "filename": safe_name,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
    }