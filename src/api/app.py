from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from src.api.routers import health, documents, qa, summary
from src.services.qa import answer_question

app = FastAPI(
    title="File Fellow",
    description="Multi-document RAG-powered document assistant with session memory",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(summary.router)


def _qa_runnable(inputs: dict) -> dict:
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
