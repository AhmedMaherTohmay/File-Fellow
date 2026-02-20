"""
Vector store management — multi-document edition.

Key design decisions:
  - Each document gets its own Chroma collection (contract_<safe_name>).
  - A global "all_documents" collection holds all chunks for cross-doc queries.
  - A "chat_history" collection persists conversation turns semantically.
  - Supports Chroma (default) and FAISS (single-store fallback).
  - Uses pathlib.Path throughout for cross-platform compatibility.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from config.settings import (
    CHROMA_COLLECTION_PREFIX,
    VECTOR_STORE_BACKEND,
    VECTOR_STORE_DIR,
    CHAT_HISTORY_COLLECTION,
)
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)

# ── In-memory caches ───────────────────────────────────────────────────────
_doc_stores: Dict[str, object] = {}   # doc_name → vector store
_global_store = None                  # all-documents store
_history_store = None                 # chat history store
_registry_path = VECTOR_STORE_DIR / "doc_registry.json"


# ── Registry helpers ───────────────────────────────────────────────────────

def _load_registry() -> Dict[str, dict]:
    if _registry_path.exists():
        try:
            return json.loads(_registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    _registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def get_document_registry() -> Dict[str, dict]:
    """Return metadata for all ingested documents."""
    return _load_registry()


def _safe_name(filename: str) -> str:
    """Convert a filename to a safe Chroma collection name."""
    base = Path(filename).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", base)[:40]
    return safe or "doc"


# ── Chroma helpers ──────────────────────────────────────────────────────────

def _chroma_collection_name(doc_name: str) -> str:
    return f"{CHROMA_COLLECTION_PREFIX}{_safe_name(doc_name)}"


def _get_chroma(collection_name: str, docs: Optional[List[Document]] = None):
    """Create or load a Chroma vector store for a given collection."""
    from langchain_community.vectorstores import Chroma

    persist_path = str(VECTOR_STORE_DIR / "chroma" / collection_name)
    embeddings = get_embeddings()

    if docs is not None:
        # Wipe and rebuild
        chroma_dir = Path(persist_path)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        store = Chroma.from_documents(
            docs,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_path,
        )
        logger.info("Chroma collection '%s' built with %d docs.", collection_name, len(docs))
        return store

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_path,
    )
    return store


def _get_faiss(docs: Optional[List[Document]] = None):
    """Create or load a global FAISS store."""
    from langchain_community.vectorstores import FAISS

    persist_path = VECTOR_STORE_DIR / "faiss"
    embeddings = get_embeddings()

    if docs is not None:
        store = FAISS.from_documents(docs, embeddings)
        persist_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_path))
        return store

    return FAISS.load_local(
        str(persist_path), embeddings, allow_dangerous_deserialization=True
    )


# ── Public API ─────────────────────────────────────────────────────────────

def add_document(
    doc_name: str,
    chunks: List[Document],
    metadata: Optional[dict] = None,
) -> None:
    """Ingest a document into the vector store(s).

    Args:
        doc_name: Original filename (used for collection naming and registry).
        chunks: Flat chunks for retrieval and embedding.
        metadata: Extra metadata saved to the registry (num_pages, num_chunks...).
    """
    global _global_store

    # ── Per-document store ────────────────────────────────────────────────
    if VECTOR_STORE_BACKEND == "chroma":
        coll_name = _chroma_collection_name(doc_name)
        _doc_stores[doc_name] = _get_chroma(coll_name, chunks)
    # (FAISS uses global store only)

    # ── Global store (all docs together, for cross-doc queries) ───────────
    if VECTOR_STORE_BACKEND == "chroma":
        global_coll = f"all_{CHROMA_COLLECTION_PREFIX}s"
        existing_global = _get_chroma_safe(global_coll)
        if existing_global is not None:
            existing_global.add_documents(chunks)
            _global_store = existing_global
        else:
            _global_store = _get_chroma(global_coll, chunks)
    else:
        # FAISS: rebuild global store
        _global_store = _get_faiss(chunks)

    # ── Registry ──────────────────────────────────────────────────────────
    registry = _load_registry()
    registry[doc_name] = {
        "doc_name": doc_name,
        "collection": _chroma_collection_name(doc_name) if VECTOR_STORE_BACKEND == "chroma" else "faiss_global",
        "num_chunks": len(chunks),
        **(metadata or {}),
    }
    _save_registry(registry)
    logger.info("Document '%s' registered. Total docs: %d.", doc_name, len(registry))


def _get_chroma_safe(collection_name: str):
    """Return Chroma store if the persist directory exists, else None."""
    persist_path = VECTOR_STORE_DIR / "chroma" / collection_name
    if persist_path.exists():
        try:
            return _get_chroma(collection_name)
        except Exception:
            pass
    return None


def get_store_for_doc(doc_name: str):
    """Return vector store for a specific document."""
    if doc_name in _doc_stores:
        return _doc_stores[doc_name]

    if VECTOR_STORE_BACKEND == "chroma":
        coll_name = _chroma_collection_name(doc_name)
        store = _get_chroma_safe(coll_name)
        if store:
            _doc_stores[doc_name] = store
            return store

    raise RuntimeError(f"No vector store found for document '{doc_name}'.")


def get_global_store():
    """Return the global (all-documents) vector store."""
    global _global_store

    if _global_store is not None:
        return _global_store

    if VECTOR_STORE_BACKEND == "chroma":
        global_coll = f"{CHROMA_COLLECTION_PREFIX}__all__"
        store = _get_chroma_safe(global_coll)
        if store:
            _global_store = store
            return store

    try:
        _global_store = _get_faiss()
        return _global_store
    except Exception:
        pass

    raise RuntimeError("No vector store found. Please upload a document first.")


def get_history_store():
    """Return (or create) the semantic chat history vector store."""
    global _history_store

    if _history_store is not None:
        return _history_store

    if VECTOR_STORE_BACKEND == "chroma":
        persist_path = VECTOR_STORE_DIR / "chroma" / CHAT_HISTORY_COLLECTION
        persist_path.mkdir(parents=True, exist_ok=True)
        _history_store = _get_chroma(CHAT_HISTORY_COLLECTION)
        return _history_store

    # FAISS fallback
    hist_path = VECTOR_STORE_DIR / "faiss_history"
    if hist_path.exists():
        from langchain_community.vectorstores import FAISS
        _history_store = FAISS.load_local(
            str(hist_path), get_embeddings(), allow_dangerous_deserialization=True
        )
    else:
        dummy = Document(page_content="[history store initialized]", metadata={"type": "init"})
        from langchain_community.vectorstores import FAISS
        _history_store = FAISS.from_documents([dummy], get_embeddings())
        hist_path.mkdir(parents=True, exist_ok=True)
        _history_store.save_local(str(hist_path))

    return _history_store


def store_is_ready() -> bool:
    """Return True if at least one document has been ingested."""
    registry = _load_registry()
    return bool(registry)


def get_ingested_documents() -> List[str]:
    """Return list of ingested document names."""
    return list(_load_registry().keys())


def remove_document(doc_name: str) -> bool:
    """Remove a document from all stores and the registry.

    Args:
        doc_name: The original filename.

    Returns:
        True if successfully removed.
    """
    global _global_store

    # Remove per-doc collection (Chroma)
    if VECTOR_STORE_BACKEND == "chroma":
        coll_name = _chroma_collection_name(doc_name)
        coll_path = VECTOR_STORE_DIR / "chroma" / coll_name
        if coll_path.exists():
            shutil.rmtree(coll_path)

    # Clear from in-memory cache
    _doc_stores.pop(doc_name, None)
    _global_store = None  # Force rebuild on next access

    # Update registry
    registry = _load_registry()
    if doc_name in registry:
        del registry[doc_name]
        _save_registry(registry)
        logger.info("Document '%s' removed from registry.", doc_name)
        return True

    return False
