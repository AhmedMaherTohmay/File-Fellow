from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    filename: str
    num_pages: int
    num_chunks: int
    message: str
    duplicate: bool = False
    duplicate_of: Optional[str] = None


class BatchIngestResponse(BaseModel):
    results: list[IngestResponse]
    errors: list[dict]
