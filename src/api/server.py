"""
FastAPI backend with LangServe endpoints.
Provides:
  POST /ingest          – Upload & ingest a document
  POST /qa              – Ask a question
  POST /summarize       – Summarize the current document
  GET  /health          – Health check
  /qa/playground        – LangServe interactive playground
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from config.settings import UPLOAD_DIR, API_HOST, API_PORT
from src.ingestion import ingest_document
from src.llm.qa_chain import answer_question
from src.llm.summarizer import summarize_document
from src.ingestion.vector_store import store_is_ready

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Contract Q&A API",
    description="RAG-powered contract assistant",
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
    question: str
    history: Optional[List[dict]] = []

class QAResponse(BaseModel):
    answer: str
    sources: List[dict]

class SummarizeRequest(BaseModel):
    filename: str

class IngestResponse(BaseModel):
    filename: str
    num_pages: int
    num_chunks: int
    message: str

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "vector_store_ready": store_is_ready()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a PDF or DOCX document."""
    allowed = {".pdf", ".docx", ".doc"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: '{suffix}'")

    # Save to temp file then ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Rename to preserve original filename
        dest = UPLOAD_DIR / file.filename
        tmp_path.rename(dest)
        result = ingest_document(dest)
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(500, f"Ingestion error: {e}") from e

    return IngestResponse(
        **result,
        message=f"Successfully ingested '{result['filename']}' "
                f"({result['num_pages']} pages, {result['num_chunks']} chunks).",
    )


@app.post("/qa", response_model=QAResponse)
async def qa(req: QARequest):
    """Answer a question grounded in the ingested document."""
    if not store_is_ready():
        raise HTTPException(400, "No document ingested yet. Please upload a document first.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        result = answer_question(req.question, req.history)
        return QAResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error("Q&A error: %s", e)
        raise HTTPException(500, f"Q&A error: {e}") from e


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize the specified document."""
    try:
        summary = summarize_document(req.filename)
        return {"summary": summary}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        logger.error("Summarization error: %s", e)
        raise HTTPException(500, f"Summarization error: {e}") from e


# ── LangServe playground ───────────────────────────────────────────────────

def _qa_runnable(inputs: dict) -> dict:
    """Wrapper to make Q&A compatible with LangServe."""
    return answer_question(
        question=inputs.get("question", ""),
        history=inputs.get("history", []),
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
