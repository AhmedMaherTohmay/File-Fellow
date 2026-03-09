from __future__ import annotations

from fastapi import APIRouter, Query

from src.storage.document_store import store_is_ready, get_ingested_documents

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(user_id: str = Query("default")):
    docs = get_ingested_documents(user_id=user_id)

    return {
        "status": "ok",
        "vector_store_ready": store_is_ready(),
        "num_documents": len(docs),
        "documents": docs[:10],
    }