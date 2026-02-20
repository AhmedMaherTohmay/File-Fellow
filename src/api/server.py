"""
FastAPI backend with LangServe endpoints.

Endpoints:
  POST /ingest                    – Upload & ingest a document (async)
  POST /ingest/batch              – Upload multiple documents
  GET  /documents                 – List all ingested documents
  DELETE /documents/{name}        – Remove a document
  POST /qa                        – Ask a question (single/cross-doc)
  POST /summarize                 – Summarize a document
  POST /sessions/{id}/end         – End a session and persist history
  GET  /health                    – Health check
  /qa-langserve/playground        – LangServe interactive playground
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from config.settings import UPLOAD_DIR, API_HOST, API_PORT
from src.ingestion import ingest_document
from src.ingestion.vector_store import (
    store_is_ready,
    get_ingested_documents,
    get_document_registry,
    remove_document,
)
from src.llm.qa_chain import answer_question
from src.llm.summarizer import summarize_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="File Fellow",
    description="Multi-document RAG-powered contract assistant with session memory",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    history: Optional[List[dict]] = Field(default=[], description="Recent conversation history")
    doc_name: Optional[str] = Field(default=None, description="Target document (None = all docs)")
    session_id: str = Field(default="default", description="Session ID for memory")


class QAResponse(BaseModel):
    answer: str
    sources: List[dict]
    session_id: str


class IngestResponse(BaseModel):
    filename: str
    num_pages: int
    num_chunks: int
    message: str


class BatchIngestResponse(BaseModel):
    results: List[IngestResponse]
    errors: List[dict]


class SummarizeRequest(BaseModel):
    filename: str


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check with system status."""
    docs = get_ingested_documents()
    return {
        "status": "ok",
        "vector_store_ready": store_is_ready(),
        "num_documents": len(docs),
        "documents": docs[:10],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a PDF or DOCX document synchronously."""
    allowed = {".pdf", ".docx", ".doc"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: '{suffix}'")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_document(dest)
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(500, f"Ingestion error: {e}") from e

    return IngestResponse(
        **result,
        message=f"Successfully ingested '{result['filename']}' "
                f"({result['num_pages']} pages, {result['num_chunks']} chunks).",
    )


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(files: List[UploadFile] = File(...)):
    """Upload and ingest multiple documents at once."""
    allowed = {".pdf", ".docx", ".doc"}
    results = []
    errors = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            errors.append({"filename": file.filename, "error": f"Unsupported type: '{suffix}'"})
            continue

        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = ingest_document(dest)
            results.append(IngestResponse(
                **result,
                message=f"Ingested '{file.filename}' ({result['num_chunks']} chunks).",
            ))
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
            logger.error("Batch ingestion error for '%s': %s", file.filename, e)

    return BatchIngestResponse(results=results, errors=errors)


@app.get("/documents")
async def list_documents():
    """List all ingested documents with metadata."""
    registry = get_document_registry()
    return {"documents": registry, "count": len(registry)}


@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Remove an ingested document from the system."""
    success = remove_document(doc_name)
    if not success:
        raise HTTPException(404, f"Document '{doc_name}' not found.")
    return {"message": f"Document '{doc_name}' removed successfully."}


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    """Answer a question grounded in one or all ingested documents."""
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
    except Exception as e:
        logger.error("Q&A error: %s", e)
        raise HTTPException(500, f"Q&A error: {e}") from e


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize the specified document."""
    try:
        summary = summarize_document(req.filename)
        return {"filename": req.filename, "summary": summary}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        logger.error("Summarization error: %s", e)
        raise HTTPException(500, f"Summarization error: {e}") from e


# ── LangServe playground ───────────────────────────────────────────────────

def _qa_runnable(inputs: dict) -> dict:
    """LangServe-compatible wrapper for the Q&A chain."""
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
