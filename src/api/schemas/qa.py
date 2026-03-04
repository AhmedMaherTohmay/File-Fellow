from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[dict]] = Field(default=[])
    doc_name: Optional[str] = Field(default=None)
    session_id: str = Field(default="default")
    user_id: Optional[str] = Field(default=None)
    conversation_id: Optional[str] = Field(default=None)


class QAResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
