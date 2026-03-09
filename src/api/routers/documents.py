from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query

from src.api.deps import save_upload
from src.api.schemas.documents import IngestResponse, BatchIngestResponse
from src.core.exceptions import ExtractionError
from src.ingestion.pipeline import ingest_document
from src.storage.document_store import get_document_registry, remove_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


def _ingest_response(result: dict) -> IngestResponse:
    msg = (
        f"Duplicate of '{result['duplicate_of']}' — skipped."
        if result.get("duplicate")
        else f"Ingested '{result['filename']}' "
             f"({result['num_pages']} pages, {result['num_chunks']} chunks)."
    )
    return IngestResponse(**result, message=msg)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    dest: Path = Depends(save_upload),
    user_id: str = Query("default"),
):
    try:
        result = await asyncio.to_thread(
            ingest_document,
            file_path=dest,
            user_id=user_id,
        )
    except (ValueError, ExtractionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", dest.name, exc)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc

    return _ingest_response(result)


@router.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    user_id: str = Query("default"),
):
    results: List[IngestResponse] = []
    errors: List[dict] = []

    for file in files:
        try:
            dest = await save_upload(file)
        except HTTPException as exc:
            errors.append({"filename": file.filename, "error": exc.detail})
            continue

        try:
            result = await asyncio.to_thread(
                ingest_document,
                file_path=dest,
                user_id=user_id,
            )
            results.append(_ingest_response(result))
        except (ValueError, ExtractionError) as exc:
            errors.append({"filename": file.filename, "error": str(exc)})
        except Exception as exc:
            logger.error("Ingestion error for '%s': %s", file.filename, exc)
            errors.append({"filename": file.filename, "error": str(exc)})

    return BatchIngestResponse(results=results, errors=errors)


@router.get("")
async def list_documents(user_id: str = Query("default")):
    logger.debug("Listing documents for user '%s'", user_id)
    registry = get_document_registry(user_id=user_id)
    return {"documents": registry, "count": len(registry)}


@router.delete("/{doc_name}")
async def delete_document(
    doc_name: str,
    user_id: str = Query("default"),
):
    removed = await asyncio.to_thread(
        remove_document,
        doc_name=doc_name,
        user_id=user_id,
    )

    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")

    return {"message": f"Document '{doc_name}' removed successfully."}