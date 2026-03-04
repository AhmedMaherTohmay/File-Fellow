"""
Document ingestion pipeline.

Orchestrates the full ingest flow:
  sanitize → content-hash check → copy to UPLOAD_DIR → parse → chunk → store
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict

from config.settings import UPLOAD_DIR
from src.core.utils import sanitize_filename, file_content_hash
from src.core.exceptions import ExtractionError
from src.ingestion.chunker import chunk_pages
from src.ingestion.parser import parse_document
from src.ingestion.vector_store import add_document, get_document_registry

logger = logging.getLogger(__name__)


def ingest_document(file_path: "str | Path") -> Dict[str, Any]:
    """
    Full ingestion pipeline: sanitize → dedup check → copy → parse → chunk → store.

    Args:
        file_path: Path to the file to ingest.  May be anywhere on disk;
                   the pipeline copies it into UPLOAD_DIR under its
                   sanitised name before processing.

    Returns:
        Dict with keys: ``filename``, ``num_pages``, ``num_chunks``,
        ``duplicate`` (bool), ``duplicate_of`` (str or None).

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
        logger.info("Filename sanitized: '%s' → '%s'", file_path.name, safe_name)

    # ── 2. Content-hash deduplication ────────────────────────────────────
    content_hash = file_content_hash(file_path)
    registry = get_document_registry()

    for existing_name, existing_meta in registry.items():
        if existing_meta.get("content_hash") == content_hash:
            # Exact same bytes already in the store — skip re-ingestion.
            logger.info(
                "Duplicate detected: '%s' has the same content as '%s'; skipping.",
                file_path.name, existing_name,
            )
            return {
                "filename": existing_name,
                "num_pages": existing_meta.get("num_pages", 0),
                "num_chunks": existing_meta.get("num_chunks", 0),
                "duplicate": True,
                "duplicate_of": existing_name,
            }

    # ── 3. Resolve name collision (different content, same sanitised name) ─
    if safe_name in registry and registry[safe_name].get("content_hash") != content_hash:
        # A different document already occupies this sanitised name.
        # Append the first 8 chars of the content hash to make it unique.
        stem = Path(safe_name).stem
        ext = Path(safe_name).suffix
        safe_name = f"{stem}_{content_hash[:8]}{ext}"
        logger.info(
            "Name collision resolved: '%s' already exists with different content; "
            "using '%s' for the new file.",
            Path(safe_name).stem.rsplit("_", 1)[0] + ext, safe_name,
        )

    # ── 4. Ensure file lives in UPLOAD_DIR under its safe name ───────────
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
                logger.warning("Could not remove unsafe-named original '%s': %s",
                               file_path, exc)

    # ── 5. Parse ─────────────────────────────────────────────────────────
    pages = parse_document(dest)
    if not pages:
        raise ExtractionError(safe_name)

    # ── 6. Chunk ─────────────────────────────────────────────────────────
    chunks = chunk_pages(pages, doc_id=safe_name)

    # ── 7. Store ─────────────────────────────────────────────────────────
    add_document(
        doc_name=safe_name,
        chunks=chunks,
        metadata={
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "content_hash": content_hash,   # Stored for future dedup checks
        },
    )

    logger.info("Ingested '%s': %d page(s), %d chunk(s).", safe_name, len(pages), len(chunks))
    return {
        "filename": safe_name,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "duplicate": False,
        "duplicate_of": None,
    }
