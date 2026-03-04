"""
Pydantic schemas for the File Fellow API.

Keeping models in a dedicated module (instead of inside server.py) lets
other layers (tests, the Gradio UI, future CLI tools) import just the
type definitions without loading the full FastAPI application.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────

class QARequest(BaseModel):
    """Payload for POST /qa."""

    question: str = Field(..., min_length=1, description="User's question or message")
    history: Optional[List[dict]] = Field(
        default=[],
        description="Recent conversation history as [{role, content}, …] dicts. "
                    "Must contain only plain message text — no citation footers.",
    )
    doc_name: Optional[str] = Field(
        default=None,
        description="Target document name.  None = search across all documents.",
    )
    session_id: str = Field(
        default="default",
        description="[Legacy] Session identifier.  Prefer user_id for new integrations.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Stable user identifier for history store filtering. "
                    "Falls back to session_id if not provided.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Current conversation UUID.  When provided, the current "
                    "conversation's turns are excluded from semantic history "
                    "retrieval to prevent the LLM seeing the same dialogue twice.",
    )


class SummarizeRequest(BaseModel):
    """Payload for POST /summarize."""

    filename: str = Field(
        ..., description="Exact filename of the ingested document to summarise"
    )


# ── Response models ────────────────────────────────────────────────────────

class QAResponse(BaseModel):
    """Response body for POST /qa."""

    answer: str
    sources: List[dict]
    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class IngestResponse(BaseModel):
    """Response body for POST /ingest (single file)."""

    filename: str
    num_pages: int
    num_chunks: int
    message: str
    duplicate: bool = False
    duplicate_of: Optional[str] = None


class BatchIngestResponse(BaseModel):
    """Response body for POST /ingest/batch."""

    results: List[IngestResponse]
    errors: List[dict]
