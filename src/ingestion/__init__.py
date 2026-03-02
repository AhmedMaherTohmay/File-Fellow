"""
Top-level ingestion pipeline.
Orchestrates: sanitize → parse → chunk → embed → store.

Public surface:
  ingest_document(file_path)  – full pipeline, returns result metadata dict
  sanitize_filename(filename)  – exported so callers (API, UI) can derive the
                                 safe name *before* writing the file to disk,
                                 avoiding an unsafe name ever touching the FS.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from config.settings import UPLOAD_DIR
from src.ingestion.chunker import chunk_pages
from src.ingestion.parser import parse_document
from src.ingestion.vector_store import add_document

logger = logging.getLogger(__name__)


# ── Filename sanitization ──────────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    # Step 1: strip directory components — Path.name gives the bare filename
    bare = Path(filename).name

    # Step 2: split stem and extension; normalise extension case
    stem = Path(bare).stem
    suffix = Path(bare).suffix.lower()   # e.g. ".PDF" → ".pdf"

    # Step 3–4: replace unsafe chars, collapse underscores, strip edges
    safe_stem = re.sub(r"[^\w\-]", "_", stem)       # \w = [a-zA-Z0-9_]
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("_")

    # Step 5–6: length cap and empty-stem fallback
    safe_stem = safe_stem[:100] or "unnamed"

    return f"{safe_stem}{suffix}"


# ── Ingestion pipeline ─────────────────────────────────────────────────────

def ingest_document(file_path: "str | Path") -> Dict[str, Any]:
    """
    Full ingestion pipeline: sanitize → copy → parse → chunk → store.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    # ── 1. Sanitize filename ───────────────────────────────────────────────
    safe_name = sanitize_filename(file_path.name)

    if safe_name != file_path.name:
        logger.info(
            "Filename sanitized: '%s' → '%s'", file_path.name, safe_name
        )

    # ── 2. Ensure file lives in UPLOAD_DIR under its safe name ────────────
    dest = UPLOAD_DIR / safe_name

    if file_path != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        logger.debug("Copied '%s' → '%s'.", file_path, dest)

        # Remove the unsafe-named original only if it was already inside
        if file_path.parent.resolve() == UPLOAD_DIR.resolve():
            try:
                file_path.unlink()
                logger.debug("Removed unsafe-named original '%s'.", file_path)
            except OSError as exc:
                logger.warning(
                    "Could not remove unsafe-named original '%s': %s",
                    file_path, exc,
                )

    # ── 3. Parse ──────────────────────────────────────────────────────────
    pages = parse_document(dest)
    if not pages:
        raise ValueError(
            f"No text could be extracted from '{safe_name}'. "
            "The file may be empty, image-only, or password-protected."
        )

    # ── 4. Chunk ──────────────────────────────────────────────────────────
    # doc_id uses the sanitized name so chunk IDs are stable across re-ingests
    chunks = chunk_pages(pages, doc_id=safe_name)

    # ── 5. Store ──────────────────────────────────────────────────────────
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