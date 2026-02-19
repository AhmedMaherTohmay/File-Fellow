"""
Vector store management (Chroma or FAISS).
Provides functions to build, persist, and load the store.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from config.settings import (
    VECTOR_STORE_BACKEND,
    VECTOR_STORE_DIR,
    CHROMA_COLLECTION,
)
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)

# Module-level store cache
_store = None


def _chroma_store(docs: List[Document] | None = None):
    """Create or load a Chroma vector store."""
    from langchain_community.vectorstores import Chroma

    persist_path = str(VECTOR_STORE_DIR / "chroma")
    embeddings = get_embeddings()

    if docs is not None:
        # Wipe and rebuild
        if Path(persist_path).exists():
            shutil.rmtree(persist_path)
        store = Chroma.from_documents(
            docs,
            embeddings,
            collection_name=CHROMA_COLLECTION,
            persist_directory=persist_path,
        )
        logger.info("Chroma store built with %d chunks.", len(docs))
        return store

    # Load existing
    store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=persist_path,
    )
    logger.info("Chroma store loaded from disk.")
    return store


def _faiss_store(docs: List[Document] | None = None):
    """Create or load a FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    persist_path = VECTOR_STORE_DIR / "faiss"
    embeddings = get_embeddings()

    if docs is not None:
        store = FAISS.from_documents(docs, embeddings)
        persist_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_path))
        logger.info("FAISS store built with %d chunks.", len(docs))
        return store

    store = FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS store loaded from disk.")
    return store


def build_vector_store(docs: List[Document]):
    """Build (or rebuild) the vector store from a list of Documents.

    Args:
        docs: Chunked LangChain Documents with metadata.

    Returns:
        The vector store instance.
    """
    global _store
    if VECTOR_STORE_BACKEND == "faiss":
        _store = _faiss_store(docs)
    else:
        _store = _chroma_store(docs)
    return _store


def get_vector_store():
    """Return the in-memory store, loading from disk if necessary.

    Returns:
        The vector store instance.

    Raises:
        RuntimeError: If no store has been built yet.
    """
    global _store
    if _store is not None:
        return _store

    # Try loading from disk
    try:
        if VECTOR_STORE_BACKEND == "faiss":
            _store = _faiss_store()
        else:
            _store = _chroma_store()
        return _store
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "No vector store found. Please upload and ingest a document first."
        ) from e


def store_is_ready() -> bool:
    """Return True if a vector store exists (in-memory or on disk)."""
    try:
        get_vector_store()
        return True
    except RuntimeError:
        return False
