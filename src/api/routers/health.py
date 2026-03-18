from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from src.db.repositories.document_repo import store_is_ready, get_ingested_documents

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(user_id: str = Query("default")):
    # Both calls are blocking psycopg2 — wrap in to_thread so the
    # FastAPI event loop is not blocked during DB I/O.
    docs, ready = await asyncio.gather(
        asyncio.to_thread(get_ingested_documents, user_id=user_id),
        asyncio.to_thread(store_is_ready),
    )

    return {
        "status":             "ok",
        "vector_store_ready": ready,
        "num_documents":      len(docs),
        "documents":          docs[:10],
    }
