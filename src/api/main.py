"""FastAPI query service for the local RAG MVP.

Endpoints:
  GET  /health  -> {status, rag_loaded, llm_backend}
  GET  /stats   -> {points, backend, collection}
  POST /query   -> {answer, citations, fallback, ...} (retrieve → LLM)

Ingest stays in scripts/ingest.py (offline batch). This service only reads
an already-ingested --rag-dir.

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:
  RAG_DIR=.cache/rag  RAG_BACKEND=local  RAG_EMBEDDER=hash
  LLM_BACKEND=fake     LLM_MODEL=...      LLM_BASE_URL=http://localhost:8080/v1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="TP RAG system (local)")


class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = Field(default=5, ge=1, le=10)
    threshold: float | None = None
    doc_id: str | None = None
    chunk_type: str | None = Field(default=None, pattern="^(text|table|figure)?$")


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = []
    fallback: bool = False
    reason: str | None = None
    standalone_question: str = ""
    chunks: list[dict[str, Any]] = []


_state: dict[str, Any] = {"retriever": None, "llm": None, "memory": None, "error": None}


def _settings() -> dict[str, str]:
    return {
        "rag_dir": os.environ.get("RAG_DIR", str(ROOT / ".cache" / "rag")),
        "backend": os.environ.get("RAG_BACKEND", "local"),
        "embedder": os.environ.get("RAG_EMBEDDER", "hash"),
        "llm_backend": os.environ.get("LLM_BACKEND", "fake"),
        "llm_model": os.environ.get("LLM_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"),
        "llm_base_url": os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        "qdrant_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
    }


def get_components() -> dict[str, Any]:
    if _state["retriever"] is None and _state["error"] is None:
        try:
            import sys

            sys.path.insert(0, str(ROOT / "src"))
            from rag.answer import answer_question  # noqa: F401
            from rag.llm import make_llm
            from rag.memory import SessionMemory
            from rag.retrieve import HybridRetriever

            settings = _settings()
            _state["retriever"] = HybridRetriever.load(
                Path(settings["rag_dir"]), embedder_name=settings["embedder"],
                backend=settings["backend"], qdrant_url=settings["qdrant_url"],
            )
            _state["llm"] = make_llm(
                settings["llm_backend"], model=settings["llm_model"], base_url=settings["llm_base_url"])
            _state["memory"] = SessionMemory()
        except Exception as exc:
            _state["error"] = str(exc)
    return _state


@app.get("/health")
def health() -> dict[str, Any]:
    components = get_components()
    settings = _settings()
    return {"status": "ok" if components["retriever"] is not None else "no-rag",
            "rag_loaded": components["retriever"] is not None,
            "llm_backend": settings["llm_backend"],
            "error": components["error"]}


@app.get("/stats")
def stats() -> dict[str, Any]:
    components = get_components()
    if components["retriever"] is None:
        raise HTTPException(status_code=503, detail=components["error"] or "RAG index not loaded")
    store = components["retriever"].store
    points = len(store) if hasattr(store, "__len__") else "qdrant"
    return {"points": points, "bm25_docs": len(components["retriever"].bm25)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    components = get_components()
    if components["retriever"] is None:
        raise HTTPException(status_code=503, detail=components["error"] or "RAG index not loaded")
    from rag.answer import answer_question

    filters = {}
    if request.doc_id:
        filters["doc_id"] = request.doc_id
    if request.chunk_type:
        filters["type"] = request.chunk_type
    try:
        result = answer_question(
            components["retriever"], request.question, components["llm"], components["memory"],
            session_id=request.session_id, top_k=request.top_k,
            threshold=request.threshold, filters=filters or None,
        )
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return QueryResponse(
        answer=result["answer"], citations=result["citations"], fallback=result["fallback"],
        reason=result.get("reason"), standalone_question=result.get("standalone_question", ""),
        chunks=[{"chunk_id": c.chunk_id, "citation": c.citation, "fused_score": c.fused_score,
                 "type": c.payload.get("type"), "page": c.payload.get("page"),
                 "content": str(c.payload.get("content", ""))[:800]} for c in result.get("chunks", [])],
    )
