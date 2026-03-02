"""
FastAPI backend with LangServe endpoints.

Endpoints
---------
POST   /ingest                 — Upload & ingest a single document
POST   /ingest/batch           — Upload multiple documents at once
GET    /documents              — List all ingested documents with metadata
DELETE /documents/{name}       — Remove a document
POST   /qa                     — Ask a question (single doc or cross-doc)
POST   /summarize              — Summarise a document
GET    /health                 — Health check + system status
/qa-langserve/playground       — LangServe interactive playground

Schemas have been moved to src/api/schema.py for reusability.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from config.settings import UPLOAD_DIR, API_HOST, API_PORT
from src.core.utils import sanitize_filename          # moved from ingestion/__init__
from src.core.exceptions import ExtractionError
from src.api.schema import (                          # schemas live in schema.py
    QARequest,
    QAResponse,
    IngestResponse,
    BatchIngestResponse,
    SummarizeRequest,
)
from src.ingestion import ingest_document
from src.ingestion.vector_store import (
    store_is_ready,
    get_ingested_documents,
    get_document_registry,
    remove_document,
)
from src.llm.qa_chain import answer_question
from src.llm.summarizer import summarize_document

logger = logging.getLogger(__name__)

app = FastAPI(
    title="File Fellow",
    description="Multi-document RAG-powered document assistant with session memory",
    version="2.0.0",
)

# NOTE: CORS is intentionally permissive for local development.
# Before deploying publicly, restrict allow_origins to specific trusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check: returns system status and list of ingested documents."""
    docs = get_ingested_documents()
    return {
        "status": "ok",
        "vector_store_ready": store_is_ready(),
        "num_documents": len(docs),
        "documents": docs[:10],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF or DOCX document.

    The filename is sanitised before being written to disk so that
    path-traversal sequences and OS-unsafe characters never touch the FS.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: '{suffix}'")

    safe_name = sanitize_filename(file.filename)
    dest = UPLOAD_DIR / safe_name

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except OSError as exc:
        logger.error("Failed to save uploaded file '%s': %s", safe_name, exc)
        raise HTTPException(500, f"Could not save file: {exc}") from exc

    try:
        result = ingest_document(dest)
    except ExtractionError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", safe_name, exc)
        raise HTTPException(500, f"Ingestion error: {exc}") from exc

    return IngestResponse(
        **result,
        message=(
            f"Successfully ingested '{result['filename']}' "
            f"({result['num_pages']} pages, {result['num_chunks']} chunks)."
        ),
    )


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(files: List[UploadFile] = File(...)):
    """
    Upload and ingest multiple documents in a single request.

    Files that fail type-validation or ingestion are collected in ``errors``
    so the rest of the batch still completes.
    """
    results: List[IngestResponse] = []
    errors: List[dict] = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            errors.append({"filename": file.filename, "error": f"Unsupported file type: '{suffix}'"})
            continue

        safe_name = sanitize_filename(file.filename)
        dest = UPLOAD_DIR / safe_name

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
        except OSError as exc:
            errors.append({"filename": file.filename, "error": f"Could not save file: {exc}"})
            logger.error("Failed to save '%s': %s", safe_name, exc)
            continue

        try:
            result = ingest_document(dest)
            results.append(IngestResponse(
                **result,
                message=f"Ingested '{result['filename']}' ({result['num_chunks']} chunks).",
            ))
        except ExtractionError as exc:
            errors.append({"filename": file.filename, "error": str(exc)})
        except Exception as exc:
            errors.append({"filename": file.filename, "error": str(exc)})
            logger.error("Ingestion error for '%s': %s", safe_name, exc)

    return BatchIngestResponse(results=results, errors=errors)


@app.get("/documents")
async def list_documents():
    """Return all ingested documents with their registry metadata."""
    registry = get_document_registry()
    return {"documents": registry, "count": len(registry)}


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Remove an ingested document and all its chunks from the system."""
    success = remove_document(doc_name)
    if not success:
        raise HTTPException(404, f"Document '{doc_name}' not found.")
    return {"message": f"Document '{doc_name}' removed successfully."}


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    """
    Answer a question grounded in one or all ingested documents.

    Returns the answer, source citations, and the session_id so callers
    can maintain session continuity across requests.
    """
    if not store_is_ready():
        raise HTTPException(400, "No documents ingested yet. Please upload a document first.")

    try:
        result = answer_question(
            question=req.question,
            history=req.history,
            doc_name=req.doc_name,
            session_id=req.session_id,
        )
        return QAResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=req.session_id,
        )
    except Exception as exc:
        logger.error("Q&A error: %s", exc)
        raise HTTPException(500, f"Q&A error: {exc}") from exc


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Generate a structured summary of the specified document."""
    try:
        summary = summarize_document(req.filename)
        return {"filename": req.filename, "summary": summary}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.error("Summarization error: %s", exc)
        raise HTTPException(500, f"Summarization error: {exc}") from exc


# ── LangServe playground ───────────────────────────────────────────────────

def _qa_runnable(inputs: dict) -> dict:
    """LangServe-compatible wrapper around the Q&A chain."""
    return answer_question(
        question=inputs.get("question", ""),
        history=inputs.get("history", []),
        doc_name=inputs.get("doc_name"),
        session_id=inputs.get("session_id", "default"),
    )


add_routes(
    app,
    RunnableLambda(_qa_runnable),
    path="/qa-langserve",
    enabled_endpoints=["invoke", "playground"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host=API_HOST, port=API_PORT, reload=False)