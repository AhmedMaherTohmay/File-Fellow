"""
Document ingestion pipeline.

Orchestrates the full ingest flow:

    validate → parse → chunk → embed → store
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.core.exceptions import ExtractionError
from src.core.utils import file_content_hash
from src.db.repositories.document_repo import (
    add_document,
    document_exists_by_hash,
    get_document_registry,
)
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import get_embeddings
from src.ingestion.parser import parse_document
from src.ingestion.validators import prepare_upload

logger = logging.getLogger(__name__)


def ingest_document(file_path: "str | Path", user_id: str = "default") -> Dict[str, Any]:
    """
    Full ingestion pipeline: validate → parse → chunk → embed → store.

    Args:
        file_path: Path to the file to ingest.
        user_id:   Owner of the document.

    Returns:
        Dict with keys: filename, num_pages, num_chunks, duplicate, duplicate_of.

    Raises:
        FileNotFoundError: If file_path does not exist.
        UnsupportedFileType: If the file extension is not allowed.
        FileTooLarge: If the file exceeds MAX_FILE_MB.
        ExtractionError: If the parser returns no usable text.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    # ── 1. Fast duplicate check — one targeted query, not a full registry load ──
    content_hash = file_content_hash(file_path)
    existing_doc = document_exists_by_hash(content_hash, user_id=user_id)

    if existing_doc:
        logger.info(
            "Duplicate upload for user '%s': '%s' matches existing doc '%s'.",
            user_id, file_path.name, existing_doc,
        )
        return {
            "filename":     existing_doc,
            "num_pages":    0,
            "num_chunks":   0,
            "duplicate":    True,
            "duplicate_of": existing_doc,
        }

    # ── 2. Validate extension/size, sanitize name, resolve collision, copy file ──
    # Registry is fetched scoped to this user only — name collision detection
    # needs to know what filenames this user already has.
    prepared = prepare_upload(
        file_path,
        get_document_registry(user_id=user_id),
        user_id=user_id,
    )
    # prepared.is_duplicate will always be False here — we checked above.

    # ── 3. Parse ───────────────────────────────────────────────────────────────
    pages = parse_document(prepared.dest)

    if not pages:
        raise ExtractionError(prepared.safe_name)

    # ── 4. Chunk (user_id injected into every chunk's metadata) ───────────────
    chunks = chunk_pages(
        pages,
        doc_id=prepared.safe_name,
        user_id=user_id,
    )

    # ── 5. Embed ───────────────────────────────────────────────────────────────
    vectors = get_embeddings().embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    # ── 6. Store (atomic transaction: upsert doc → delete old chunks → insert) ─
    add_document(
        doc_name=prepared.safe_name,
        chunks=chunks,
        embeddings=vectors,
        user_id=user_id,
        doc_metadata={
            "num_pages":    len(pages),
            "num_chunks":   len(chunks),
            "content_hash": prepared.content_hash,
            "file_path":    str(prepared.dest),
        },
    )

    logger.info(
        "Ingested '%s' for user '%s': %d page(s), %d chunk(s).",
        prepared.safe_name, user_id, len(pages), len(chunks),
    )

    return {
        "filename":     prepared.safe_name,
        "num_pages":    len(pages),
        "num_chunks":   len(chunks),
        "duplicate":    False,
        "duplicate_of": None,
    }
