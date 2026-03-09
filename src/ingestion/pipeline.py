"""
Document ingestion pipeline.

Orchestrates the full ingest flow:

    validate → parse → chunk → embed → store

Key design changes from the original implementation:
  - Upload validation, deduplication, and file copying are handled by
    ``prepare_upload()`` from the validators module.
  - Embeddings are computed inside this pipeline rather than inside
    the vector store layer.
  - Documents are now scoped by ``user_id`` to support multi-user
    environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.core.exceptions import ExtractionError
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import get_embeddings
from src.ingestion.parser import parse_document
from src.ingestion.validators import prepare_upload
from src.storage.document_store import add_document, get_document_registry

logger = logging.getLogger(__name__)


def ingest_document(file_path: "str | Path", user_id: str = "default") -> Dict[str, Any]:
    """
    Full ingestion pipeline.

    Steps:
        validate → parse → chunk → embed → store

    Args:
        file_path:
            Path to the file to ingest.

        user_id:
            Identifier for the user uploading the document. This value is
            injected into chunk metadata and used for registry isolation.

    Returns:
        Dict with keys:
            filename
            num_pages
            num_chunks
            duplicate (bool)
            duplicate_of (str or None)

    Raises:
        FileNotFoundError:
            If *file_path* does not exist.

        ValueError:
            If the file extension is unsupported or validation fails.

        ExtractionError:
            If the parser returns no usable text.
    """

    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: '{file_path}'")

    # ── 1. Validate upload, sanitize name, detect duplicates, copy to UPLOAD_DIR ──
    prepared = prepare_upload(
        file_path,
        get_document_registry(),
        user_id=user_id,
    )

    # If this file's content already exists in the store we skip ingestion
    if prepared.is_duplicate:
        logger.info(
            "Duplicate upload detected for user '%s': '%s' → '%s'.",
            user_id,
            file_path.name,
            prepared.duplicate_of,
        )

        return {
            "filename": prepared.safe_name,
            "num_pages": 0,
            "num_chunks": 0,
            "duplicate": True,
            "duplicate_of": prepared.duplicate_of,
        }

    # ── 2. Parse document ──────────────────────────────────────────────────
    pages = parse_document(prepared.dest)

    if not pages:
        raise ExtractionError(prepared.safe_name)

    # ── 3. Chunk document (user_id injected into chunk metadata) ───────────
    chunks = chunk_pages(
        pages,
        doc_id=prepared.safe_name,
        user_id=user_id,
    )

    # ── 4. Generate embeddings for each chunk ─────────────────────────────
    embeddings_model = get_embeddings()

    vectors = embeddings_model.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    # ── 5. Store chunks and embeddings ────────────────────────────────────
    add_document(
        doc_name=prepared.safe_name,
        chunks=chunks,
        embeddings=vectors,
        user_id=user_id,
        doc_metadata={
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "content_hash": prepared.content_hash,
        },
    )

    logger.info(
        "Ingested '%s' for user '%s': %d page(s), %d chunk(s).",
        prepared.safe_name,
        user_id,
        len(pages),
        len(chunks),
    )

    return {
        "filename": prepared.safe_name,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "duplicate": False,
        "duplicate_of": None,
    }