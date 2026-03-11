"""
Text chunking utilities.

Splits parsed document pages into overlapping chunks with stable IDs
and rich metadata for RAG retrieval, evaluation, and multi-user isolation.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Stable chunk ID generator
# ──────────────────────────────────────────────────────────────


def _stable_chunk_id(
    *,
    doc_id: str,
    page: int,
    chunk_index: int,
    text: str,
) -> str:
    """
    Generate a deterministic chunk ID using SHA-256.

    The ID is derived from position + content so it remains stable across
    re-ingestion of identical documents and unique across documents.
    SHA-256 is used instead of SHA-1 to avoid linter/security warnings.
    Output is truncated to 40 hex chars to maintain backward-compatible ID length.
    """
    raw = f"{doc_id}:{page}:{chunk_index}:{text[:100]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


# ──────────────────────────────────────────────────────────────
# Chunking pipeline
# ──────────────────────────────────────────────────────────────


def chunk_pages(
    pages: List[Dict[str, Any]],
    *,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    doc_id: Optional[str] = None,
    user_id: str = "default",
    uploaded_at: Optional[str] = None,
    min_chunk_length: int = settings.MIN_CHUNK_LENGTH,
) -> List[Document]:
    """
    Split parsed document pages into overlapping text chunks.

    Each chunk receives rich metadata required for:
      - semantic retrieval
      - document reconstruction
      - multi-user filtering
      - evaluation pipelines

    Args:
        pages:
            Output from the parser module.
        chunk_size:
            Maximum characters per chunk.
        chunk_overlap:
            Overlap characters between adjacent chunks.
        doc_id:
            Stable document identifier.
        user_id:
            Owner of the document (used for multi-user isolation).
        uploaded_at:
            Optional ISO timestamp for ingestion. Defaults to current UTC time.
        min_chunk_length:
            Minimum character count; shorter chunks are discarded as they
            degrade retrieval quality.

    Returns:
        List of LangChain ``Document`` objects.
    """
    timestamp = uploaded_at or datetime.now(timezone.utc).isoformat()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    global_chunk_index = 0

    for page_info in pages:
        text = page_info.get("text", "").strip()
        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for local_index, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()

            # ── Quality filter ─────────────────────────────
            if len(chunk_text) < min_chunk_length:
                continue

            resolved_doc_id = doc_id or page_info["source"]

            chunk_id = _stable_chunk_id(
                doc_id=resolved_doc_id,
                page=page_info["page"],
                chunk_index=local_index,
                text=chunk_text,
            )

            doc = Document(
                page_content=chunk_text,
                metadata={
                    # ── Identity ──────────────────────────
                    "chunk_id":            chunk_id,
                    "doc_id":              resolved_doc_id,
                    "user_id":             user_id,

                    # ── Source information ─────────────────
                    "source":              page_info["source"],
                    "file_type":           page_info.get("file_type"),

                    # ── Position within the document ───────
                    "page":                page_info["page"],
                    "chunk_index":         local_index,
                    "global_chunk_index":  global_chunk_index,

                    # ── Chunking parameters ────────────────
                    "chunk_size":          chunk_size,
                    "chunk_overlap":       chunk_overlap,

                    # ── Ingestion metadata ─────────────────
                    "uploaded_at":         timestamp,
                },
            )

            all_chunks.append(doc)
            global_chunk_index += 1

    logger.info(
        "Created %d chunk(s) (size=%d, overlap=%d) for doc='%s' user='%s'.",
        len(all_chunks),
        chunk_size,
        chunk_overlap,
        doc_id or "auto",
        user_id,
    )

    return all_chunks
