"""
Top-level ingestion pipeline.
Orchestrates: parse → chunk → embed → store.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Any

from config.settings import UPLOAD_DIR
from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_pages
from src.ingestion.vector_store import add_document

logger = logging.getLogger(__name__)

def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst)


def ingest_document(file_path: "str | Path") -> Dict[str, Any]:
    """Parse, chunk, embed, and store a document.

    Args:
        file_path: Path to a PDF or DOCX file.

    Returns:
        Dict with ``filename``, ``num_pages``, ``num_chunks``.
    """
    file_path = Path(file_path)

    # Ensure file is in uploads directory
    dest = UPLOAD_DIR / file_path.name
    if file_path.resolve() != dest.resolve():
        # dest.write_bytes(file_path.read_bytes())
        shutil.copy2(str(file_path), str(dest))
        logger.info("Copied '%s' to uploads directory.", file_path.name)

    # 1. Parse
    pages = parse_document(dest)
    if not pages:
        raise ValueError(f"No text could be extracted from '{file_path.name}'.")

    # 2. Flat chunking
    chunks = chunk_pages(pages, doc_id=file_path.name)

    # 3. Build vector store(s)
    add_document(
        doc_name=file_path.name,
        chunks=chunks,
        metadata={
            "num_pages": len(pages),
            "num_chunks": len(chunks),
        },
    )

    logger.info(
        "Ingested '%s': %d pages, %d chunks.",
        file_path.name,
        len(pages),
        len(chunks),
    )
    return {
        "filename": file_path.name,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
    }