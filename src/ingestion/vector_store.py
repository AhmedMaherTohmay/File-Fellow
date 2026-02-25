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
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Iterator

from langchain_core.documents import Document

from config.settings import (
    CHROMA_COLLECTION_PREFIX,
    VECTOR_STORE_BACKEND,
    VECTOR_STORE_DIR,
    CHAT_HISTORY_COLLECTION,
)
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# Registry
# ───────────────────────────────────────────────────────────────

_registry_path = VECTOR_STORE_DIR / "doc_registry.json"


def _load_registry() -> Dict[str, dict]:
    if _registry_path.exists():
        try:
            return json.loads(_registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    """Atomic save with backup."""
    _registry_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _registry_path.with_suffix(".tmp")
    
    # Write to temp first
    temp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    
    # Backup existing
    if _registry_path.exists():
        shutil.copy2(_registry_path, _registry_path.with_suffix(".json.bak"))
    
    # Atomic rename
    temp_path.replace(_registry_path)


def get_document_registry() -> Dict[str, dict]:
    return _load_registry()


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

def _safe_name(filename: str, max_len: int = 30) -> str:
    """
    Generate collision-resistant safe name.
    Includes hash suffix to guarantee uniqueness.
    """
    path = Path(filename)
    base = path.stem or "unnamed"
    
    # Create hash from full filename for uniqueness
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    # Clean base name
    safe_base = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    safe_base = re.sub(r"_+", "_", safe_base).strip("_")
    
    # Reserve space for hash suffix
    available = max_len - len(name_hash) - 1
    truncated = safe_base[:available] if len(safe_base) > available else safe_base
    
    return f"{truncated}_{name_hash}" if truncated else f"doc_{name_hash}"


def _chroma_collection_name(doc_name: str) -> str:
    return f"{CHROMA_COLLECTION_PREFIX}_{_safe_name(doc_name)}"


# ───────────────────────────────────────────────────────────────
# Chroma / FAISS loaders (STATELESS)
# ───────────────────────────────────────────────────────────────

def _get_chroma(collection_name: str):
    """Always open a fresh Chroma instance (Windows-safe)."""
    from langchain_community.vectorstores import Chroma

    persist_path = VECTOR_STORE_DIR / "chroma" / collection_name
    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_path),
    )


def _get_faiss(docs: Optional[List[Document]] = None):
    from langchain_community.vectorstores import FAISS

    persist_path = VECTOR_STORE_DIR / "faiss"
    embeddings = get_embeddings()

    if docs is not None:
        store = FAISS.from_documents(docs, embeddings)
        persist_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_path))
        return store

    return FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ───────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────

def add_document(
    doc_name: str,
    chunks: List[Document],
    metadata: Optional[dict] = None,
) -> None:
    """
    Ingest a document into vector stores (Windows-safe).
    """
    registry = _load_registry()

    # ── Chroma backend ─────────────────────────────────────────
    if VECTOR_STORE_BACKEND == "chroma":
        # Per-document collection
        coll_name = _chroma_collection_name(doc_name)
        doc_store = _get_chroma(coll_name)
        doc_store.add_documents(chunks)
        # IMPORTANT: release file handles
        del doc_store

        # Global collection
        global_coll = f"all_{CHROMA_COLLECTION_PREFIX}s"
        global_store = _get_chroma(global_coll)
        global_store.add_documents(chunks)
        del global_store

        registry[doc_name] = {
            "doc_name": doc_name,
            "collection": coll_name,
            "num_chunks": len(chunks),
            **(metadata or {}),
        }

    # ── FAISS backend ──────────────────────────────────────────
    else:
        # FAISS uses a single global store
        faiss_store = _get_faiss(chunks)
        del faiss_store

        registry[doc_name] = {
            "doc_name": doc_name,
            "collection": "faiss_global",
            "num_chunks": len(chunks),
            **(metadata or {}),
        }

    _save_registry(registry)

    logger.info(
        "Document '%s' ingested (%d chunks). Total docs: %d",
        doc_name,
        len(chunks),
        len(registry),
    )


def get_store_for_doc(doc_name: str):
    if VECTOR_STORE_BACKEND == "chroma":
        coll_name = _chroma_collection_name(doc_name)
        return _get_chroma(coll_name)

    raise RuntimeError("FAISS does not support per-document stores.")


def get_global_store():
    if VECTOR_STORE_BACKEND == "chroma":
        global_coll = f"all_{CHROMA_COLLECTION_PREFIX}s"
        return _get_chroma(global_coll)

    return _get_faiss()


def get_history_store():
    if VECTOR_STORE_BACKEND == "chroma":
        return _get_chroma(CHAT_HISTORY_COLLECTION)

    from langchain_community.vectorstores import FAISS

    hist_path = VECTOR_STORE_DIR / "faiss_history"
    embeddings = get_embeddings()

    if hist_path.exists():
        return FAISS.load_local(
            str(hist_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    dummy = Document(page_content="[history initialized]", metadata={"type": "init"})
    store = FAISS.from_documents([dummy], embeddings)
    hist_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(hist_path))
    return store


def store_is_ready() -> bool:
    return bool(_load_registry())


def get_ingested_documents() -> List[str]:
    return list(_load_registry().keys())


def remove_document(doc_name: str) -> bool:
    registry = _load_registry()

    if VECTOR_STORE_BACKEND == "chroma":
        coll_name = _chroma_collection_name(doc_name)
        coll_path = VECTOR_STORE_DIR / "chroma" / coll_name
        if coll_path.exists():
            shutil.rmtree(coll_path)

    if doc_name in registry:
        del registry[doc_name]
        _save_registry(registry)
        logger.info("Document '%s' removed.", doc_name)
        return True

    return False