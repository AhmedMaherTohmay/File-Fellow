from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas.summary import SummarizeRequest, SummarizeResponse
from src.services.summary import summarize_document

logger = logging.getLogger(__name__)

router = APIRouter(tags=["summary"])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    try:
        summary = await asyncio.to_thread(
            summarize_document,
            filename=req.filename,
            user_id=req.user_id,
        )

        return SummarizeResponse(
            filename=req.filename,
            summary=summary,
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except Exception as exc:
        logger.error("Summarization error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Summarization error: {exc}",
        ) from exc