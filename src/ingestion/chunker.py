"""
Text chunking utilities.
Splits parsed document pages into overlapping flat chunks for retrieval.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    doc_id: str = "",
) -> List[Document]:
    """Split page texts into overlapping chunks.

    Args:
        pages: Output from the parser (list of {text, page, source}).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.
        doc_id: Optional document identifier for metadata (collection scoping).

    Returns:
        List of LangChain ``Document`` objects with rich metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    chunk_id = 0

    for page_info in pages:
        raw_chunks = splitter.split_text(page_info["text"])
        for chunk_text in raw_chunks:
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": page_info["source"],
                    "page": page_info["page"],
                    "chunk_id": chunk_id,
                    "doc_id": doc_id or page_info["source"],
                },
            )
            all_chunks.append(doc)
            chunk_id += 1

    logger.info(
        "Created %d chunks (size=%d, overlap=%d) for doc='%s'.",
        len(all_chunks),
        chunk_size,
        chunk_overlap,
        doc_id or "unknown",
    )
    return all_chunks
