from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    filename: str = Field(..., description="Exact filename of the ingested document to summarise.")
    user_id: Optional[str] = Field(default="default", description="Owner of the document.")


class SummarizeResponse(BaseModel):
    filename: str
    summary: str
