from __future__ import annotations

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    filename: str = Field(..., description="Exact filename of the ingested document to summarise")


class SummarizeResponse(BaseModel):
    filename: str
    summary: str
