"""
Text chunking utilities.
Splits parsed document pages into overlapping chunks with stable IDs
and rich metadata for RAG & evaluation.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def _stable_chunk_id(
    *,
    doc_id: str,
    page: int,
    chunk_index: int,
    text: str,
) -> str:
    """Generate deterministic chunk ID."""
    raw = f"{doc_id}:{page}:{chunk_index}:{text[:100]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chunk_pages(
    pages: List[Dict[str, Any]],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    doc_id: str | None = None,
    min_chunk_length: int = 50,
) -> List[Document]:
    """
    Split page texts into overlapping chunks with rich metadata.

    Args:
        pages: Output from the parser.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters.
        doc_id: Stable document identifier.
        min_chunk_length: Filter very small chunks.

    Returns:
        List of LangChain Document objects.
    """

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

            # ---- Quality filter ----
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
                    # Identity
                    "chunk_id": chunk_id,
                    "doc_id": resolved_doc_id,

                    # Provenance
                    "source": page_info["source"],
                    "file_type": page_info.get("file_type"),
                    "file_path": page_info.get("file_path"),

                    # Positioning
                    "page": page_info["page"],
                    "chunk_index": local_index,
                    "global_chunk_index": global_chunk_index,

                    # Chunking params
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
            )

            all_chunks.append(doc)
            global_chunk_index += 1

    logger.info(
        "Created %d chunks (size=%d, overlap=%d) for doc='%s'",
        len(all_chunks),
        chunk_size,
        chunk_overlap,
        doc_id or "auto",
    )

    return all_chunks