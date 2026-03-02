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
        description="Recent conversation history as [{role, content}, …] dicts",
    )
    doc_name: Optional[str] = Field(
        default=None,
        description="Target document name.  None = search across all documents.",
    )
    session_id: str = Field(
        default="default",
        description="Session identifier used for semantic memory isolation",
    )


class SummarizeRequest(BaseModel):
    """Payload for POST /summarize."""

    filename: str = Field(..., description="Exact filename of the ingested document to summarise")


# ── Response models ────────────────────────────────────────────────────────

class QAResponse(BaseModel):
    """Response body for POST /qa."""

    answer: str
    sources: List[dict]
    session_id: str


class IngestResponse(BaseModel):
    """Response body for POST /ingest (single file)."""

    filename: str
    num_pages: int
    num_chunks: int
    message: str


class BatchIngestResponse(BaseModel):
    """Response body for POST /ingest/batch."""

    results: List[IngestResponse]
    errors: List[dict]