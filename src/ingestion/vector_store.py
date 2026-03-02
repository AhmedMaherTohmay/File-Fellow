"""
Vector store management — single-store architecture.

Design decisions:
  - All document chunks live in ONE global Chroma/FAISS collection.
  - Per-document scoping is handled via metadata filter {"source": doc_name}
    at query time — no per-document collections are created or maintained.
  - A separate "chat_history" collection persists conversation turns.
  - Re-ingesting an existing document automatically replaces its old chunks.
  - Windows-safe: Chroma instances are opened fresh per request and explicitly
    deleted to release SQLite file handles before any filesystem operations.
  - migrate_per_doc_collections() handles upgrading installs that were running
    the old dual-store architecture.
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import time
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

# ── Constants ──────────────────────────────────────────────────────────────

# The single collection that holds all document chunks.
# Named for backward-compatibility with the previous global collection so
# existing data on disk is picked up automatically without re-ingestion.
_GLOBAL_COLLECTION: str = f"all_{CHROMA_COLLECTION_PREFIX}s"

_registry_path: Path = VECTOR_STORE_DIR / "doc_registry.json"


# ── Registry ───────────────────────────────────────────────────────────────

def _load_registry() -> Dict[str, dict]:
    """Load the document registry from disk. Returns {} on any failure."""
    if _registry_path.exists():
        try:
            return json.loads(_registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(
                "Failed to load registry from '%s': %s", _registry_path, exc
            )
    return {}


def _save_registry(registry: Dict[str, dict]) -> None:
    """
    Atomically persist the registry with a .bak safety copy.

    Write order: temp file → backup existing → atomic rename.
    This guarantees the registry is never left in a corrupt state even if
    the process is killed mid-write.
    """
    _registry_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _registry_path.with_suffix(".tmp")

    temp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    if _registry_path.exists():
        shutil.copy2(_registry_path, _registry_path.with_suffix(".json.bak"))

    temp_path.replace(_registry_path)


def get_document_registry() -> Dict[str, dict]:
    """Return the full document registry keyed by document name."""
    return _load_registry()


# ── Windows-safe filesystem helpers ───────────────────────────────────────

def _safe_rmtree(path: Path) -> None:
    """
    Remove a directory tree, retrying on Windows PermissionError.

    Chroma/SQLite holds OS-level file locks that may outlive the Python
    object. gc.collect() encourages CPython to finalise any lingering
    Chroma handles before the first removal attempt.
    """
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
                    path, attempt + 1, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Could not fully delete '%s' after 3 attempts: %s",
                    path, exc,
                )


def _delete_chroma_dir(collection_name: str) -> None:
    """
    Properly tear down a persisted Chroma collection directory.

    Steps:
      1. Ask chromadb's own client to delete the collection, which
         flushes WAL and closes SQLite file handles cleanly.
      2. Force CPython GC to finalise the client object immediately.
      3. Remove the directory from disk with retry logic for Windows locks.
    """
    import chromadb

    coll_path = VECTOR_STORE_DIR / "chroma" / collection_name
    if not coll_path.exists():
        return

    try:
        client = chromadb.PersistentClient(path=str(coll_path))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection may not exist in this path — still clean up dir
        del client
        gc.collect()
    except Exception as exc:
        logger.warning(
            "chromadb teardown for '%s' failed (%s) — proceeding to rmtree",
            collection_name, exc,
        )

    _safe_rmtree(coll_path)


# ── Backend constructors ───────────────────────────────────────────────────

def _get_chroma(collection_name: str):
    """
    Open a fresh, stateless Chroma instance for the given collection.

    A new Python object is created on every call, backed by the same
    persisted directory. Callers must ``del`` the returned instance (and
    optionally call ``gc.collect()``) once done to release SQLite handles.
    """
    from langchain_community.vectorstores import Chroma

    persist_path = VECTOR_STORE_DIR / "chroma" / collection_name
    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_path),
    )


def _get_faiss(new_docs: Optional[List[Document]] = None):
    """
    Open or update the single FAISS global store.

    - ``new_docs=None``:  load and return the existing persisted store.
    - ``new_docs=[...]``: merge the new documents into the existing store
      (creating a fresh one if none exists) then persist to disk.

    The merge strategy prevents the old bug where every ingest replaced the
    entire store with only the newly uploaded document's chunks.
    """
    from langchain_community.vectorstores import FAISS

    persist_path = VECTOR_STORE_DIR / "faiss"
    embeddings = get_embeddings()

    if new_docs is not None:
        new_store = FAISS.from_documents(new_docs, embeddings)
        persist_path.mkdir(parents=True, exist_ok=True)
        index_file = persist_path / "index.faiss"

        if index_file.exists():
            try:
                existing = FAISS.load_local(
                    str(persist_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                existing.merge_from(new_store)
                existing.save_local(str(persist_path))
                return existing
            except Exception as exc:
                logger.warning(
                    "Could not merge into existing FAISS store (%s); "
                    "saving new store as replacement.",
                    exc,
                )

        new_store.save_local(str(persist_path))
        return new_store

    return FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── Internal chunk management ─────────────────────────────────────────────

def _chroma_delete_doc_chunks(doc_name: str) -> None:
    """
    Delete all Chroma chunks whose ``source`` metadata equals ``doc_name``.

    Uses Chroma's native where-clause delete so only the matching entries are
    touched. All other documents in the global collection remain intact.
    """
    coll_path = VECTOR_STORE_DIR / "chroma" / _GLOBAL_COLLECTION
    if not coll_path.exists():
        return

    try:
        store = _get_chroma(_GLOBAL_COLLECTION)
        store._collection.delete(where={"source": doc_name})
        del store
        gc.collect()
        logger.debug("Deleted Chroma chunks for source='%s'.", doc_name)
    except Exception as exc:
        logger.warning(
            "Failed to delete Chroma chunks for '%s': %s", doc_name, exc
        )


def _faiss_rebuild_without(doc_name: str) -> None:
    """
    Rebuild the FAISS global store excluding all chunks from ``doc_name``.

    FAISS provides no server-side delete, so a full rebuild from the
    remaining documents is necessary. If the rebuilt store would be empty
    the directory is removed entirely so future loads don't fail.
    """
    from langchain_community.vectorstores import FAISS

    persist_path = VECTOR_STORE_DIR / "faiss"
    if not (persist_path / "index.faiss").exists():
        return

    embeddings = get_embeddings()
    try:
        store = FAISS.load_local(
            str(persist_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        logger.warning("Cannot load FAISS store for rebuild: %s", exc)
        return

    remaining: List[Document] = [
        doc
        for doc in store.docstore._dict.values()
        if doc.metadata.get("source") != doc_name
    ]

    if not remaining:
        shutil.rmtree(persist_path)
        logger.info("FAISS store removed — no documents remain after deleting '%s'.", doc_name)
        return

    rebuilt = FAISS.from_documents(remaining, embeddings)
    rebuilt.save_local(str(persist_path))
    logger.info(
        "Rebuilt FAISS store: removed '%s', %d chunks remain.",
        doc_name, len(remaining),
    )


# ── Public API ─────────────────────────────────────────────────────────────

def add_document(
    doc_name: str,
    chunks: List[Document],
    metadata: Optional[dict] = None,
) -> None:
    """
    Ingest a document's chunks into the global vector store.

    If the document was previously ingested its old chunks are purged first,
    making this operation idempotent and safe for re-upload workflows.

    Args:
        doc_name:  Original filename used as the ``source`` metadata value
                   on every chunk and as the registry key.
        chunks:    LangChain Document objects produced by the chunker.
        metadata:  Optional extra fields stored in the registry entry
                   (e.g. num_pages, num_chunks).
    """
    if not chunks:
        raise ValueError(f"Cannot ingest '{doc_name}': chunk list is empty.")

    registry = _load_registry()

    # Purge stale chunks before re-ingesting to avoid duplicates
    if doc_name in registry:
        logger.info("Re-ingesting '%s': purging %d previous chunks.", doc_name, registry[doc_name].get("num_chunks", "?"))
        if VECTOR_STORE_BACKEND == "chroma":
            _chroma_delete_doc_chunks(doc_name)
        else:
            _faiss_rebuild_without(doc_name)

    # Write chunks to the single global store
    if VECTOR_STORE_BACKEND == "chroma":
        store = _get_chroma(_GLOBAL_COLLECTION)
        store.add_documents(chunks)
        del store
        gc.collect()
    else:
        _get_faiss(new_docs=chunks)

    registry[doc_name] = {
        "doc_name": doc_name,
        "collection": _GLOBAL_COLLECTION,
        "num_chunks": len(chunks),
        **(metadata or {}),
    }
    _save_registry(registry)

    logger.info(
        "Ingested '%s': %d chunks into '%s'. Registry: %d document(s).",
        doc_name, len(chunks), _GLOBAL_COLLECTION, len(registry),
    )


def remove_document(doc_name: str) -> bool:
    """
    Remove a document and all its chunks from the vector store and registry.

    Args:
        doc_name: Exact filename as stored in the registry.

    Returns:
        True if the document existed and was removed, False if not found.
    """
    registry = _load_registry()

    if doc_name not in registry:
        logger.warning("remove_document: '%s' not found in registry.", doc_name)
        return False

    if VECTOR_STORE_BACKEND == "chroma":
        _chroma_delete_doc_chunks(doc_name)
    else:
        _faiss_rebuild_without(doc_name)

    del registry[doc_name]
    _save_registry(registry)
    logger.info("Document '%s' removed from store and registry.", doc_name)
    return True


def get_global_store():
    """
    Return the single vector store holding all document chunks.

    For Chroma: callers must ``del`` the returned object when done to
    release SQLite file handles on Windows.
    """
    if VECTOR_STORE_BACKEND == "chroma":
        return _get_chroma(_GLOBAL_COLLECTION)
    return _get_faiss()


def get_history_store():
    """
    Return the vector store used for semantic chat-history persistence.

    For FAISS: bootstraps with a sentinel document on first use so the
    index file exists before any similarity searches are attempted.
    """
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

    # Bootstrap an empty-ish store on first use
    sentinel = Document(
        page_content="[history initialised]",
        metadata={"type": "init"},
    )
    store = FAISS.from_documents([sentinel], embeddings)
    hist_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(hist_path))
    return store


def store_is_ready() -> bool:
    """Return True if at least one document has been ingested."""
    return bool(_load_registry())


def get_ingested_documents() -> List[str]:
    """Return the list of all ingested document names."""
    return list(_load_registry().keys())


# ── Migration ──────────────────────────────────────────────────────────────

def migrate_per_doc_collections() -> int:
    """
    One-time migration for installs running the old dual-store architecture.

    The previous design wrote every document's chunks into both a dedicated
    per-document Chroma collection AND the global collection. The global
    collection is the single source of truth in the new architecture, so
    the per-document collections are redundant and safe to delete
    (all data is already present in the global store).

    Steps:
      1. Scan ``vector_store/chroma/`` for directories that are not the
         global collection or the chat-history collection.
      2. Delete each stale directory using the Windows-safe helper.
      3. Patch any registry entry whose ``collection`` field still points
         to an old per-document collection name.

    Returns:
        Number of stale per-document collection directories that were removed.
    """
    chroma_root = VECTOR_STORE_DIR / "chroma"
    if not chroma_root.exists():
        return 0

    protected = {_GLOBAL_COLLECTION, CHAT_HISTORY_COLLECTION}
    removed = 0

    for entry in sorted(chroma_root.iterdir()):
        if entry.is_dir() and entry.name not in protected:
            logger.info(
                "Migration: removing stale per-doc collection '%s'.", entry.name
            )
            _delete_chroma_dir(entry.name)
            removed += 1

    # Patch registry entries that still reference old per-doc collection names
    registry = _load_registry()
    patched = False
    for doc_name, meta in registry.items():
        if meta.get("collection") != _GLOBAL_COLLECTION:
            logger.info(
                "Migration: updating registry entry '%s': '%s' → '%s'.",
                doc_name, meta.get("collection"), _GLOBAL_COLLECTION,
            )
            meta["collection"] = _GLOBAL_COLLECTION
            patched = True

    if patched:
        _save_registry(registry)

    if removed or patched:
        logger.info(
            "Migration complete: %d collection(s) removed, registry %s.",
            removed, "patched" if patched else "already up-to-date",
        )
    else:
        logger.debug("Migration: nothing to do.")

    return removed