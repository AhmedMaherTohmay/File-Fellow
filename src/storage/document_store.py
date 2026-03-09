"""
Vector store management — multi-user single-store architecture.

Design decisions:
  - All document chunks live in ONE global Chroma collection.
  - Each chunk carries metadata:
        source  -> document name
        user_id -> owner of the document
  - Per-user isolation is enforced via metadata filters at query time.
  - Registry keys follow the format:  user_id:doc_name

Embedding behaviour:
  - Embeddings are computed upstream in the ingestion pipeline.
  - This module ONLY writes vectors to Chroma and never runs the model.

Concurrency:
  - _WRITE_LOCK serialises all writes to avoid SQLite locking errors.
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from config.settings import (
    CHROMA_COLLECTION_PREFIX,
    VECTOR_STORE_DIR,
    CHAT_HISTORY_COLLECTION,
)

from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Write lock
# ──────────────────────────────────────────────────────────────

_WRITE_LOCK = threading.RLock()

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_GLOBAL_COLLECTION: str = f"all_{CHROMA_COLLECTION_PREFIX}s"
_registry_path: Path = VECTOR_STORE_DIR / "doc_registry.json"

# ──────────────────────────────────────────────────────────────
# Registry helpers
# ──────────────────────────────────────────────────────────────


def _registry_key(user_id: str, doc_name: str) -> str:
    """Return the unique registry key for a user document."""
    return f"{user_id}:{doc_name}"


def _load_registry() -> Dict[str, dict]:
    """Load the registry file from disk."""
    if _registry_path.exists():
        try:
            return json.loads(_registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load registry from '%s': %s", _registry_path, exc)
    return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    """
    Atomically save registry with a backup file.

    Write order:
        temp file → backup → atomic rename
    """
    _registry_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = _registry_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    if _registry_path.exists():
        shutil.copy2(_registry_path, _registry_path.with_suffix(".json.bak"))

    temp_path.replace(_registry_path)


def get_document_registry(user_id: Optional[str] = None) -> Dict[str, dict]:
    """
    Return the document registry.

    If user_id is provided only documents owned by that user are returned.
    """
    registry = _load_registry()

    if user_id is None:
        return registry

    return {
        key: value
        for key, value in registry.items()
        if key.startswith(f"{user_id}:")
    }


# ──────────────────────────────────────────────────────────────
# Windows-safe filesystem helpers
# ──────────────────────────────────────────────────────────────


def _safe_rmtree(path: Path) -> None:
    """Retry directory removal to avoid Windows PermissionError."""
    gc.collect()

    for attempt in range(3):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            if attempt < 2:
                wait = 0.3 * (attempt + 1)
                logger.debug(
                    "rmtree('%s') attempt %d failed; retrying in %.1fs: %s",
                    path,
                    attempt + 1,
                    wait,
                    exc,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Could not fully delete '%s' after 3 attempts: %s", path, exc
                )


# ──────────────────────────────────────────────────────────────
# Chroma constructor
# ──────────────────────────────────────────────────────────────


def _get_chroma(collection_name: str):
    """
    Open a fresh Chroma instance.

    A new object is created each call to ensure file handles
    are released properly on Windows.
    """
    from langchain_community.vectorstores import Chroma

    persist_path = VECTOR_STORE_DIR / "chroma" / collection_name
    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_path),
    )


# ──────────────────────────────────────────────────────────────
# Chunk deletion
# ──────────────────────────────────────────────────────────────


def _delete_doc_chunks(doc_name: str, user_id: str) -> None:
    """
    Remove all chunks belonging to a specific document/user pair.
    """
    coll_path = VECTOR_STORE_DIR / "chroma" / _GLOBAL_COLLECTION

    if not coll_path.exists():
        return

    try:
        store = _get_chroma(_GLOBAL_COLLECTION)

        store._collection.delete(
            where={
                "$and": [
                    {"source": {"$eq": doc_name}},
                    {"user_id": {"$eq": user_id}},
                ]
            }
        )

        del store
        gc.collect()

        logger.debug(
            "Deleted Chroma chunks for source='%s' user='%s'.",
            doc_name,
            user_id,
        )

    except Exception as exc:
        logger.warning(
            "Failed to delete Chroma chunks for '%s'/'%s': %s",
            doc_name,
            user_id,
            exc,
        )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def add_document(
    doc_name: str,
    chunks: List[Document],
    embeddings: List[List[float]],
    user_id: str = "default",
    doc_metadata: Optional[dict] = None,
) -> None:
    """
    Ingest a document's chunks into the global vector store.

    Embeddings MUST already be computed upstream.

    The entire operation (purge + write + registry update)
    is protected by _WRITE_LOCK.
    """

    if not chunks:
        raise ValueError(f"Cannot ingest '{doc_name}': chunk list is empty.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})."
        )

    registry_key = _registry_key(user_id, doc_name)

    with _WRITE_LOCK:
        registry = _load_registry()

        if registry_key in registry:
            logger.info(
                "Re-ingesting '%s' for user '%s': purging %d previous chunks.",
                doc_name,
                user_id,
                registry[registry_key].get("num_chunks", "?"),
            )

            _delete_doc_chunks(doc_name, user_id)

        store = _get_chroma(_GLOBAL_COLLECTION)

        store._collection.add(
            documents=[c.page_content for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
            ids=[c.metadata["chunk_id"] for c in chunks],
        )

        del store
        gc.collect()

        registry[registry_key] = {
            "doc_name": doc_name,
            "user_id": user_id,
            "collection": _GLOBAL_COLLECTION,
            "num_chunks": len(chunks),
            **(doc_metadata or {}),
        }

        _save_registry(registry)

    logger.info(
        "Ingested '%s' for user '%s': %d chunks into '%s'.",
        doc_name,
        user_id,
        len(chunks),
        _GLOBAL_COLLECTION,
    )


def remove_document(doc_name: str, user_id: str = "default") -> bool:
    """Remove a document and all its chunks."""
    registry_key = _registry_key(user_id, doc_name)

    with _WRITE_LOCK:
        registry = _load_registry()

        if registry_key not in registry:
            logger.warning(
                "remove_document: '%s' not found for user '%s'.",
                doc_name,
                user_id,
            )
            return False

        _delete_doc_chunks(doc_name, user_id)

        del registry[registry_key]

        _save_registry(registry)

    logger.info("Document '%s' removed for user '%s'.", doc_name, user_id)

    return True


def get_global_store():
    """Return the shared Chroma store."""
    return _get_chroma(_GLOBAL_COLLECTION)


def get_chunks_for_doc(
    doc_name: str,
    user_id: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve all stored chunks for a document.

    Results are ordered by global_chunk_index so
    text reconstruction matches the original document.
    """

    where: dict = {"source": {"$eq": doc_name}}

    if user_id:
        where = {
            "$and": [
                {"source": {"$eq": doc_name}},
                {"user_id": {"$eq": user_id}},
            ]
        }

    try:
        store = _get_chroma(_GLOBAL_COLLECTION)

        raw = store._collection.get(
            where=where,
            include=["documents", "metadatas"],
        )

        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]

        docs.sort(key=lambda d: d.metadata.get("global_chunk_index", 0))

        del store
        gc.collect()

        return docs

    except Exception as exc:
        logger.warning("get_chunks_for_doc('%s') failed: %s", doc_name, exc)
        return []


def get_history_store():
    """Return the vector store used for semantic chat history."""
    return _get_chroma(CHAT_HISTORY_COLLECTION)


def store_is_ready(user_id: Optional[str] = None) -> bool:
    """Return True if at least one document exists."""
    return bool(get_document_registry(user_id=user_id))


def get_ingested_documents(user_id: Optional[str] = None) -> List[str]:
    """Return list of ingested document names."""
    registry = get_document_registry(user_id=user_id)

    return [
        v.get("doc_name", k.split(":", 1)[-1])
        for k, v in registry.items()
    ]


# ──────────────────────────────────────────────────────────────
# Migration
# ──────────────────────────────────────────────────────────────


def migrate_per_doc_collections() -> int:
    """
    Remove legacy per-document collections.

    Older versions created one Chroma collection per document.
    """
    chroma_root = VECTOR_STORE_DIR / "chroma"

    if not chroma_root.exists():
        return 0

    protected = {_GLOBAL_COLLECTION, CHAT_HISTORY_COLLECTION}

    removed = 0

    for entry in sorted(chroma_root.iterdir()):
        if entry.is_dir() and entry.name not in protected:
            logger.info("Migration: removing stale collection '%s'.", entry.name)

            _safe_rmtree(entry)

            removed += 1

    registry = _load_registry()

    patched = False

    for key, meta in registry.items():
        if meta.get("collection") != _GLOBAL_COLLECTION:
            meta["collection"] = _GLOBAL_COLLECTION
            patched = True

    if patched:
        _save_registry(registry)

    if removed or patched:
        logger.info(
            "Migration complete: %d collection(s) removed, registry %s.",
            removed,
            "patched" if patched else "already up-to-date",
        )

    return removed