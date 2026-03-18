"""
Document repository — all SQL for the documents and document_chunks tables.

Rules this file lives by:
  - Every function takes plain Python values, returns domain model instances.
  - No business logic. No ranking. No LLM calls. Pure data access.
  - Every write operation is wrapped in get_connection() which commits on
    success and rolls back on any exception.
  - User isolation is enforced at the SQL level (WHERE user_id = %s),
    never assumed from the call site.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2.extras

from config.settings import settings
from src.db.engine import get_connection, vec_to_literal
from src.db.models.document import Chunk, Document

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# User anchor — required before inserting documents (migration 002 adds FK)
# ──────────────────────────────────────────────────────────────────────────────

def upsert_user(user_id: str) -> None:
    """
    Ensure a user row exists for this user_id.

    Called by the ingestion pipeline before inserting a document so the
    FK constraint documents.user_id → users.id is satisfied.
    Safe to call multiple times — ON CONFLICT DO NOTHING is a no-op.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Write operations
# ──────────────────────────────────────────────────────────────────────────────

def add_document(
    doc_name: str,
    chunks: list,            # List[langchain_core.documents.Document]
    embeddings: list,        # List[List[float]]
    user_id: str = "default",
    doc_metadata: Optional[dict] = None,
) -> None:
    """
    Ingest a document atomically.

    The entire operation — document upsert, old-chunk deletion, new-chunk
    bulk insert — runs inside a single transaction. On any failure the
    transaction rolls back; no partial state is written.

    Why one transaction?
    If we committed the document row first and then the chunk insert failed,
    we'd have an empty document in the registry with no searchable content.
    A user refreshing would see a document they can't query. The atomic
    transaction prevents that state from ever existing.

    Args:
        doc_name:     Sanitised filename — stable document key.
        chunks:       LangChain Document objects from the chunker.
        embeddings:   Pre-computed vectors (one per chunk, same order).
        user_id:      Owner of the document.
        doc_metadata: Extra fields: num_pages, num_chunks, content_hash, file_path.
    """
    if not chunks:
        raise ValueError(f"Cannot ingest '{doc_name}': chunk list is empty.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})."
        )

    meta         = doc_metadata or {}
    num_pages    = meta.get("num_pages", 0)
    num_chunks   = len(chunks)
    content_hash = meta.get("content_hash", "")

    # Store path relative to UPLOAD_DIR — not absolute — so the project
    # can be moved or deployed in a container without breaking file links.
    raw_path = meta.get("file_path", "")
    if raw_path:
        try:
            file_path = str(Path(raw_path).relative_to(settings.UPLOAD_DIR))
        except ValueError:
            file_path = raw_path   # already relative or outside UPLOAD_DIR
    else:
        file_path = ""

    # Ensure the user row exists before inserting the document (FK constraint).
    upsert_user(user_id)

    with get_connection() as conn:
        with conn.cursor() as cur:

            # ── 1. Upsert the document row ──────────────────────────────
            cur.execute(
                """
                INSERT INTO documents
                    (user_id, doc_name, content_hash, num_pages, num_chunks, file_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, doc_name) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    num_pages    = EXCLUDED.num_pages,
                    num_chunks   = EXCLUDED.num_chunks,
                    file_path    = EXCLUDED.file_path,
                    ingested_at  = NOW()
                RETURNING id
                """,
                (user_id, doc_name, content_hash, num_pages, num_chunks, file_path),
            )
            doc_id: int = cur.fetchone()[0]

            # ── 2. Delete previous chunks (re-ingest = fresh start) ────
            cur.execute("DELETE FROM document_chunks WHERE doc_fk = %s", (doc_id,))

            # ── 3. Bulk-insert new chunks in one round-trip ─────────────
            rows = [
                (
                    chunk.metadata["chunk_id"],
                    doc_id,
                    user_id,
                    doc_name,
                    chunk.page_content,
                    vec_to_literal(emb),
                    chunk.metadata.get("source"),
                    chunk.metadata.get("file_type"),
                    chunk.metadata.get("page"),
                    chunk.metadata.get("chunk_index"),
                    chunk.metadata.get("global_chunk_index"),
                    chunk.metadata.get("chunk_size"),
                    chunk.metadata.get("chunk_overlap"),
                    chunk.metadata.get("uploaded_at"),
                )
                for chunk, emb in zip(chunks, embeddings)
            ]

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO document_chunks
                    (chunk_id, doc_fk, user_id, doc_name, page_content, embedding,
                     source, file_type, page, chunk_index, global_chunk_index,
                     chunk_size, chunk_overlap, uploaded_at)
                VALUES %s
                """,
                rows,
                template=(
                    "(%s, %s, %s, %s, %s, %s::vector, "
                    " %s, %s, %s, %s, %s, %s, %s, %s)"
                ),
            )

    logger.info("Ingested '%s' for user '%s': %d chunk(s).", doc_name, user_id, num_chunks)


def remove_document(doc_name: str, user_id: str = "default") -> bool:
    """
    Delete a document and all its chunks (CASCADE handles the chunks).

    Returns True if the document existed and was deleted, False if not found.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE user_id = %s AND doc_name = %s",
                (user_id, doc_name),
            )
            deleted = cur.rowcount > 0

    if deleted:
        logger.info("Document '%s' removed for user '%s'.", doc_name, user_id)
    else:
        logger.warning("remove_document: '%s' not found for user '%s'.", doc_name, user_id)

    return deleted


# ──────────────────────────────────────────────────────────────────────────────
# Read operations
# ──────────────────────────────────────────────────────────────────────────────

def document_exists_by_hash(content_hash: str, user_id: Optional[str] = None) -> Optional[str]:
    """
    Check whether a file with this content hash is already ingested.

    Returns the doc_name if found, None otherwise.

    This replaces the previous pattern of loading the entire registry dict
    (SELECT * FROM documents) just to scan for a hash match. One targeted
    query is always better than loading all rows into Python.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if user_id is not None:
                cur.execute(
                    """
                    SELECT doc_name FROM documents
                    WHERE content_hash = %s AND user_id = %s
                    LIMIT 1
                    """,
                    (content_hash, user_id),
                )
            else:
                cur.execute(
                    "SELECT doc_name FROM documents WHERE content_hash = %s LIMIT 1",
                    (content_hash,),
                )
            row = cur.fetchone()
            return row[0] if row else None


def get_document_registry(user_id: Optional[str] = None) -> Dict[str, dict]:
    """
    Return the document registry as a dict keyed by 'user_id:doc_name'.

    Format is kept compatible with the old JSON registry so validators.py
    and pipeline.py don't need changes during the transition.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if user_id is not None:
                cur.execute(
                    """
                    SELECT user_id, doc_name, content_hash, num_pages,
                           num_chunks, ingested_at
                    FROM documents
                    WHERE user_id = %s
                    ORDER BY ingested_at DESC
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT user_id, doc_name, content_hash, num_pages,
                           num_chunks, ingested_at
                    FROM documents
                    ORDER BY ingested_at DESC
                    """
                )
            rows = cur.fetchall()

    return {
        f"{row['user_id']}:{row['doc_name']}": {
            "doc_name":     row["doc_name"],
            "user_id":      row["user_id"],
            "content_hash": row["content_hash"],
            "num_pages":    row["num_pages"],
            "num_chunks":   row["num_chunks"],
            "ingested_at":  row["ingested_at"].isoformat() if row["ingested_at"] else None,
        }
        for row in rows
    }


def get_chunks_for_doc(
    doc_name: str,
    user_id: Optional[str] = None,
) -> List[Chunk]:
    """
    Return all stored chunks for a document ordered by position.

    Used by the summariser to reconstruct document text for map-reduce
    summarisation, bypassing the vector search path entirely.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if user_id is not None:
                cur.execute(
                    """
                    SELECT chunk_id, doc_fk, user_id, doc_name, page_content,
                           source, file_type, page, chunk_index,
                           global_chunk_index, chunk_size, chunk_overlap, uploaded_at
                    FROM document_chunks
                    WHERE doc_name = %s AND user_id = %s
                    ORDER BY global_chunk_index ASC
                    """,
                    (doc_name, user_id),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_id, doc_fk, user_id, doc_name, page_content,
                           source, file_type, page, chunk_index,
                           global_chunk_index, chunk_size, chunk_overlap, uploaded_at
                    FROM document_chunks
                    WHERE doc_name = %s
                    ORDER BY global_chunk_index ASC
                    """,
                    (doc_name,),
                )
            rows = cur.fetchall()

    return [_row_to_chunk(row) for row in rows]


def store_is_ready(user_id: Optional[str] = None) -> bool:
    """Return True if at least one document exists (optionally scoped to user)."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if user_id is not None:
                    cur.execute(
                        "SELECT 1 FROM documents WHERE user_id = %s LIMIT 1",
                        (user_id,),
                    )
                else:
                    cur.execute("SELECT 1 FROM documents LIMIT 1")
                return cur.fetchone() is not None
    except Exception as exc:
        logger.warning("store_is_ready check failed: %s", exc)
        return False


def get_ingested_documents(user_id: Optional[str] = None) -> List[str]:
    """Return document names, optionally scoped to a user, newest first."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if user_id is not None:
                cur.execute(
                    "SELECT doc_name FROM documents WHERE user_id = %s ORDER BY ingested_at DESC",
                    (user_id,),
                )
            else:
                cur.execute("SELECT doc_name FROM documents ORDER BY ingested_at DESC")
            return [row[0] for row in cur.fetchall()]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_chunks(
    doc_name: Optional[str] = None,
    user_id: Optional[str] = None,
) -> int:
    """Return total chunk count, optionally filtered."""
    conditions, params = [], []
    if user_id is not None:
        conditions.append("user_id = %s")
        params.append(user_id)
    if doc_name is not None:
        conditions.append("doc_name = %s")
        params.append(doc_name)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM document_chunks {where}", params)
            return cur.fetchone()[0]


def get_chunks_by_ids(chunk_ids: List[str]) -> List[Chunk]:
    """Fetch specific chunks by their chunk_id values."""
    if not chunk_ids:
        return []

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_fk, user_id, doc_name, page_content,
                       source, file_type, page, chunk_index,
                       global_chunk_index, chunk_size, chunk_overlap, uploaded_at
                FROM document_chunks
                WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            rows = cur.fetchall()

    return [_row_to_chunk(row) for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Row hydration — database rows → domain model instances
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_chunk(row: dict) -> Chunk:
    """Convert a RealDictCursor row into a Chunk domain model instance."""
    return Chunk(
        chunk_id=           row["chunk_id"],
        doc_fk=             row["doc_fk"],
        user_id=            row["user_id"],
        doc_name=           row["doc_name"],
        page_content=       row["page_content"],
        source=             row.get("source"),
        file_type=          row.get("file_type"),
        page=               row.get("page"),
        chunk_index=        row.get("chunk_index"),
        global_chunk_index= row.get("global_chunk_index"),
        chunk_size=         row.get("chunk_size"),
        chunk_overlap=      row.get("chunk_overlap"),
        uploaded_at=        row.get("uploaded_at"),
    )
