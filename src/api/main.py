"""FastAPI query service for the local RAG MVP.

Endpoints:
  GET  /health  -> {status, rag_loaded, llm_backend}
  GET  /stats   -> {points, backend, collection}
  POST /query   -> {answer, citations, fallback, ...} (retrieve → LLM)
  GET/POST /sessions -> durable transcripts with per-session working context

Ingest stays in scripts/ingest.py (offline batch). Production reads the
published Qdrant alias without local index artifacts.

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:
  RAG_BACKEND=qdrant  RAG_EMBEDDER=bge-m3  QDRANT_COLLECTION=docs_current
  LLM_BACKEND=server  LLM_BASE_URL=http://localhost:8080/v1
"""

from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from project_settings import configured_path, setting
from rag.conversations import ConversationConflict, ConversationStore
from rag.memory import SessionMemory
from rag.requests import RequestHistory
from rag.source_assets import (IMAGE_TYPES, attach_image_links, resolve_source_image,
                               serialize_sources, source_images)

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="TP RAG system (local)")


class QueryRequest(BaseModel):
    question: str
    session_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,128}$")
    workspace_id: str = Field(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")
    top_k: int = Field(default_factory=lambda: setting("retrieval.top_k"), ge=1, le=setting("api.max_top_k"))
    threshold: float | None = Field(default=None, ge=0, allow_inf_nan=False,
                                   description="Minimum dense cosine relevance")
    rerank: bool = Field(default_factory=lambda: setting("retrieval.rerank"))
    doc_id: str | None = None
    chunk_type: str | None = Field(default=None, pattern="^(text|table|figure)?$")


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = []
    fallback: bool = False
    reason: str | None = None
    standalone_question: str = ""
    chunks: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    request_id: str | None = None
    metrics: dict[str, Any] = {}


class SessionRequest(BaseModel):
    workspace_id: str = Field(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")


@lru_cache(maxsize=8)
def _conversation_store(path: str) -> ConversationStore:
    return ConversationStore(Path(path))


def get_conversations() -> ConversationStore:
    return _conversation_store(str(configured_path("memory.storage_path")))


@lru_cache(maxsize=8)
def _request_history(path: str) -> RequestHistory:
    return RequestHistory(Path(path))


def get_requests() -> RequestHistory:
    return _request_history(str(get_conversations().path))


@app.middleware("http")
async def record_query_request(request: Request, call_next):
    """Include successful, rejected and failed queries in durable telemetry."""
    if request.method != "POST" or request.url.path != "/query":
        return await call_next(request)
    started = perf_counter()
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = {}
    if not isinstance(body, dict):
        body = {}

    def identifier(key, default):
        value = body.get(key, default)
        return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) else default

    history = get_requests()
    request_id = history.begin(
        workspace_id=identifier("workspace_id", "local"),
        session_id=identifier("session_id", "default"),
        question=str(body.get("question", "")), model=setting("llm.model", "LLM_MODEL"),
        rerank=bool(body.get("rerank", setting("retrieval.rerank"))),
    )
    request.state.request_id = request_id
    request.state.started = started
    request.state.metrics = {}
    request.state.details = {}
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request.state.metrics["runtime_ms"] = (perf_counter() - started) * 1000
        details = request.state.details
        status = ("error" if status_code >= 400 else
                  "fallback" if details.get("fallback") else "success")
        if status == "error" and not details.get("error"):
            details["error"] = f"HTTP {status_code}"
        history.finish(request_id, status=status, http_status=status_code,
                       metrics=request.state.metrics, details=details)


@app.get("/requests")
def list_requests(
    workspace_id: str = Query(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$"),
    limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None, pattern=r"^(running|success|fallback|error)$"),
    session_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$"),
):
    return get_requests().list(workspace_id, limit=limit, offset=offset,
                               status=status, session_id=session_id)


@app.get("/requests/{request_id}")
def read_request(request_id: str, workspace_id: str = Query(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    try:
        record = get_requests().get(workspace_id, request_id)
        details = record["details"]
        if details.get("message_id"):
            details["chunks"] = attach_image_links(details.get("chunks", []), workspace_id,
                                                    record["session_id"], details["message_id"])
        return record
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def context_info(context: dict) -> dict:
    turns = context.get("turns", [])
    return {"recent_messages": len(turns),
            "recent_chars": sum(len(t["content"]) for t in turns),
            "max_messages": setting("memory.max_turns"),
            "max_chars": setting("memory.max_chars"),
            "summary": context.get("summary", "")}


def session_detail(session: dict) -> dict:
    return {**session, "context": context_info(session["context"]), "messages": [
        {**message, "sources": attach_image_links(message.get("sources", []),
            session["workspace_id"], session["id"], message["id"])}
        for message in session.get("messages", [])]}


@app.get("/sessions")
def list_sessions(workspace_id: str = Query(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    return {"sessions": get_conversations().list(workspace_id)}


@app.post("/sessions")
def create_session(request: SessionRequest):
    return session_detail(get_conversations().create(request.workspace_id))


@app.get("/sessions/{session_id}")
def read_session(session_id: str, workspace_id: str = Query(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    try:
        return session_detail(get_conversations().get(workspace_id, session_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/sessions/{session_id}/context/reset")
def reset_context(session_id: str, request: SessionRequest):
    try:
        return session_detail(get_conversations().reset_context(request.workspace_id, session_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, workspace_id: str = Query(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    try:
        get_conversations().delete(workspace_id, session_id)
        return {"deleted": session_id}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/sessions/{session_id}/messages/{message_id}/sources/{source_index}/images/{image_index}")
def read_source_image(session_id: str, message_id: int, source_index: int, image_index: int,
                      workspace_id: str = Query(default="local", pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    try:
        session = get_conversations().get(workspace_id, session_id)
        message = next(m for m in session["messages"] if m["id"] == message_id)
        if source_index < 0 or image_index < 0:
            raise IndexError
        source = message["sources"][source_index]
        image = source_images(source)[image_index]
        path = resolve_source_image(image)
        if path is None:
            raise LookupError("Ảnh nguồn chưa có trên máy chủ.")
    except (LookupError, StopIteration) as exc:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh nguồn.") from exc
    return FileResponse(path, media_type=IMAGE_TYPES[path.suffix.lower()],
                        headers={"Cache-Control": "private, no-store"})


_state: dict[str, Any] = {"retriever": None, "llm": None, "error": None}
_initialization_lock = threading.Lock()


def _settings() -> dict[str, Any]:
    return {
        "rag_dir": str(configured_path("ingestion.rag_dir", "RAG_DIR")),
        "backend": setting("qdrant.backend", "RAG_BACKEND"),
        "embedder": setting("embedding.backend", "RAG_EMBEDDER"),
        "collection": setting("qdrant.alias", "QDRANT_COLLECTION"),
        "llm_backend": setting("llm.backend", "LLM_BACKEND"),
        "llm_model": setting("llm.model", "LLM_MODEL"),
        "llm_base_url": setting("llm.base_url", "LLM_BASE_URL"),
        "qdrant_url": os.environ.get("QDRANT_URL"),
        "qdrant_api_key": os.environ.get("QDRANT_API_KEY"),
    }


def get_components() -> dict[str, Any]:
    if _state["retriever"] is not None:
        return _state
    with _initialization_lock:
        if _state["retriever"] is not None:
            return _state
        try:
            import sys

            sys.path.insert(0, str(ROOT / "src"))
            from rag.llm import make_llm
            from rag.retrieve import BgeReranker, HybridRetriever

            settings = _settings()
            retriever = HybridRetriever.load(
                Path(settings["rag_dir"]), embedder_name=settings["embedder"],
                backend=settings["backend"], qdrant_url=settings["qdrant_url"],
                qdrant_api_key=settings["qdrant_api_key"],
                collection=settings["collection"],
            )
            if settings["backend"] != "local":
                retriever.store.count()  # confirm the published collection is reachable
            llm = make_llm(
                settings["llm_backend"], model=settings["llm_model"], base_url=settings["llm_base_url"])
            _state.update(retriever=retriever, llm=llm,
                          reranker=BgeReranker(), error=None)
        except Exception as exc:
            _state["error"] = str(exc)
    return _state


@app.get("/health")
def health() -> dict[str, Any]:
    components = get_components()
    settings = _settings()
    retriever = components["retriever"]
    calibrated = retriever is not None and retriever.threshold is not None
    status = "ok" if calibrated else "needs-calibration" if retriever is not None else "no-rag"
    return {"status": status,
            "rag_loaded": retriever is not None,
            "relevance_configured": calibrated,
            "relevance_threshold": retriever.threshold if retriever is not None else None,
            "llm_backend": settings["llm_backend"],
            "error": components["error"]}


@app.get("/stats")
def stats() -> dict[str, Any]:
    components = get_components()
    if components["retriever"] is None:
        raise HTTPException(status_code=503, detail=components["error"] or "RAG index not loaded")
    store = components["retriever"].store
    points = len(store) if hasattr(store, "__len__") else store.count()
    bm25 = components["retriever"].bm25
    return {"points": points, "backend": _settings()["backend"],
            "collection": getattr(store, "collection", None),
            "bm25_docs": len(bm25) if bm25 is not None else 0}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    metrics = http_request.state.metrics
    components = get_components()
    if components["retriever"] is None:
        http_request.state.details = {"error": components["error"] or "RAG index not loaded"}
        raise HTTPException(status_code=503, detail=components["error"] or "RAG index not loaded")
    from rag.answer import answer_question

    filters = {}
    if request.doc_id:
        filters["doc_id"] = request.doc_id
    if request.chunk_type:
        filters["type"] = request.chunk_type
    store = get_conversations()
    try:
        with store.lock(request.session_id):
            session = store.create(request.workspace_id, request.session_id)
            # Load only this session's context; no process-global chat state.
            memory = SessionMemory()
            memory.restore(request.session_id, session["context"])
            result = answer_question(
                components["retriever"], request.question, components["llm"], memory,
                session_id=request.session_id, top_k=request.top_k,
                threshold=request.threshold, filters=filters or None,
                reranker=components["reranker"].rerank if request.rerank else None,
                metrics=metrics, request_started=http_request.state.started,
            )
            context = memory.snapshot(request.session_id)
            result["sources"] = serialize_sources(result.get("chunks", []))
            # Save the transcript (including fallbacks) and working context together.
            message_id = store.record_exchange(request.workspace_id, request.session_id, request.question,
                                               result, context, session["revision"])
    except ConnectionError as exc:
        http_request.state.details = {"error": str(exc)}
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        http_request.state.details = {"error": str(exc)}
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    metrics["runtime_ms"] = (perf_counter() - http_request.state.started) * 1000
    response = QueryResponse(
        request_id=http_request.state.request_id, metrics=metrics,
        answer=result["answer"], citations=result["citations"], fallback=result["fallback"],
        reason=result.get("reason"), standalone_question=result.get("standalone_question", ""),
        context=context_info(context),
        chunks=attach_image_links(result["sources"], request.workspace_id, request.session_id, message_id),
    )
    http_request.state.details = {"answer": result["answer"], "citations": result["citations"],
                                  "fallback": result["fallback"], "reason": result.get("reason"),
                                  "chunks": response.chunks, "message_id": message_id}
    return response
