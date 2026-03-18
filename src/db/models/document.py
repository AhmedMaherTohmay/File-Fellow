"""
Domain models for documents and chunks.

These are plain Python dataclasses — not ORM models, not SQL schemas.
They represent the shape of data as the application understands it,
independent of how it's stored in the database.

Repositories return these types. Everything above the repository layer
(retrievers, services, API) works with these types, not with raw dicts
or psycopg2 rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    """
    A user-uploaded document as recorded in the documents table.

    This is the registry entry — it tracks metadata about the file,
    not the file content itself (chunks hold the content).
    """
    id:           int
    user_id:      str
    doc_name:     str
    content_hash: str
    num_pages:    int
    num_chunks:   int
    ingested_at:  datetime
    file_path:    Optional[str] = None   # relative path from UPLOAD_DIR


@dataclass
class Chunk:
    """
    A single text chunk from a document, including its embedding vector.

    This is what gets stored in document_chunks and searched via pgvector.
    The embedding field is None when a chunk is retrieved without its
    vector (e.g. for display purposes), and populated when returned from
    a similarity search.
    """
    chunk_id:            str
    doc_fk:              int
    user_id:             str
    doc_name:            str
    page_content:        str
    source:              Optional[str]   = None
    file_type:           Optional[str]   = None
    page:                Optional[int]   = None
    chunk_index:         Optional[int]   = None
    global_chunk_index:  Optional[int]   = None
    chunk_size:          Optional[int]   = None
    chunk_overlap:       Optional[int]   = None
    uploaded_at:         Optional[datetime] = None
    embedding:           Optional[list[float]] = field(default=None, repr=False)
    # repr=False keeps __repr__ readable — embeddings are 384-float lists

    def to_metadata(self) -> dict:
        """
        Return the chunk fields that callers treat as LangChain-style metadata.

        Historically the application used LangChain Document objects with a
        metadata dict. This method produces the same shape so the service layer
        doesn't need to change.
        """
        uploaded_str = (
            self.uploaded_at.isoformat() if self.uploaded_at else None
        )
        return {
            "chunk_id":           self.chunk_id,
            "doc_id":             self.doc_name,   # historic key used by prompts
            "user_id":            self.user_id,
            "source":             self.source,
            "file_type":          self.file_type,
            "page":               self.page,
            "chunk_index":        self.chunk_index,
            "global_chunk_index": self.global_chunk_index,
            "chunk_size":         self.chunk_size,
            "chunk_overlap":      self.chunk_overlap,
            "uploaded_at":        uploaded_str,
        }
