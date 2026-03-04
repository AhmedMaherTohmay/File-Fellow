from __future__ import annotations

from fastapi import APIRouter

from src.storage.document_store import store_is_ready, get_ingested_documents

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    docs = get_ingested_documents()
    return {
        "status": "ok",
        "vector_store_ready": store_is_ready(),
        "num_documents": len(docs),
        "documents": docs[:10],
    }
