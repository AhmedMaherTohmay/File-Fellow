from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import require_store
from src.api.schemas.qa import QARequest, QAResponse
from src.services.qa import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(tags=["qa"])


@router.post("/qa", response_model=QAResponse, dependencies=[Depends(require_store)])
async def qa(req: QARequest):
    try:
        result = await asyncio.to_thread(
            answer_question,
            question=req.question,
            history=req.history,
            doc_name=req.doc_name,
            session_id=req.session_id,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )

        return QAResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=req.session_id,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )

    except Exception as exc:
        logger.error("Q&A error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Q&A error: {exc}") from exc