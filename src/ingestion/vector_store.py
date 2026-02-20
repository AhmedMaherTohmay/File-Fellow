"""
Vector store management — Stage 1 & 2.

Stage 1:
- One vector store per document.

Stage 2:
- One global vector store containing all documents
  (for cross-document queries).

Supports:
- Chroma (multi-collection)
- FAISS (single global fallback)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from config.settings import (
    VECTOR_STORE_BACKEND,
    VECTOR_STORE_DIR,
    CHROMA_COLLECTION,
)
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# In-memory caches
# ─────────────────────────────────────────────────────────────

_doc_stores: Dict[str, object] = {}   # doc_name -> vector store
_global_store = None                  # all-documents store


# ─────────────────────────────────────────────────────────────
# Chroma helpers
# ─────────────────────────────────────────────────────────────

def _chroma_store(collection_name: str, docs: List[Document] | None = None):
    """Create or load a Chroma collection."""
    from langchain_community.vectorstores import Chroma

    persist_path = VECTOR_STORE_DIR / "chroma" / collection_name
    embeddings = get_embeddings()

    if docs is not None:
        # Rebuild collection
        if persist_path.exists():
            shutil.rmtree(persist_path)

        store = Chroma.from_documents(
            docs,
            embeddings,
            collection_name=collection_name,
            persist_directory=str(persist_path),
        )
        logger.info("Chroma collection '%s' built with %d chunks.", collection_name, len(docs))
        return store

    # Load existing
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )


def _chroma_exists(collection_name: str) -> bool:
    return (VECTOR_STORE_DIR / "chroma" / collection_name).exists()


# ─────────────────────────────────────────────────────────────
# FAISS helpers (global-only)
# ─────────────────────────────────────────────────────────────

def _faiss_store(docs: List[Document] | None = None):
    """Create or load a FAISS store (global only)."""
    from langchain_community.vectorstores import FAISS

    persist_path = VECTOR_STORE_DIR / "faiss"
    embeddings = get_embeddings()

    if docs is not None:
        store = FAISS.from_documents(docs, embeddings)
        persist_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_path))
        logger.info("FAISS store built with %d chunks.", len(docs))
        return store

    return FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def add_document(doc_name: str, docs: List[Document]) -> None:
    """
    Add a document to:
    - its own vector store (Stage 1)
    - the global vector store (Stage 2)
    """
    global _global_store

    # ── Stage 1: per-document store ──────────────────────────
    if VECTOR_STORE_BACKEND == "chroma":
        collection_name = f"{CHROMA_COLLECTION}_{doc_name}"
        store = _chroma_store(collection_name, docs)
        _doc_stores[doc_name] = store

    # ── Stage 2: global store ─────────────────────────────────
    if VECTOR_STORE_BACKEND == "chroma":
        global_collection = f"{CHROMA_COLLECTION}__all__"

        if _chroma_exists(global_collection):
            global_store = _chroma_store(global_collection)
            global_store.add_documents(docs)
            _global_store = global_store
        else:
            _global_store = _chroma_store(global_collection, docs)

    else:
        # FAISS: rebuild global store
        all_docs = list(docs)
        try:
            existing = _faiss_store()
            # Simple strategy: rebuild
        except Exception:
            pass

        _global_store = _faiss_store(all_docs)


def get_store_for_document(doc_name: str):
    """Return the vector store for a specific document."""
    if doc_name in _doc_stores:
        return _doc_stores[doc_name]

    if VECTOR_STORE_BACKEND == "chroma":
        collection_name = f"{CHROMA_COLLECTION}_{doc_name}"
        if _chroma_exists(collection_name):
            store = _chroma_store(collection_name)
            _doc_stores[doc_name] = store
            return store

    raise RuntimeError(f"No vector store found for document '{doc_name}'.")


def get_global_store():
    """Return the global (all-documents) vector store."""
    global _global_store

    if _global_store is not None:
        return _global_store

    if VECTOR_STORE_BACKEND == "chroma":
        global_collection = f"{CHROMA_COLLECTION}__all__"
        if _chroma_exists(global_collection):
            _global_store = _chroma_store(global_collection)
            return _global_store

    try:
        _global_store = _faiss_store()
        return _global_store
    except Exception as e:
        raise RuntimeError(
            "No global vector store found. Please ingest a document first."
        ) from e


def store_is_ready() -> bool:
    """Return True if at least one store exists."""
    try:
        get_global_store()
        return True
    except RuntimeError:
        return False