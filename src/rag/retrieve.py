"""BGE-M3 dense/sparse → Qdrant prefetch + RRF → optional rerank → top-K.

Production needs only the published Qdrant collection, with no local lexical
index or chunk files. Local dense + BM25 remains an explicit development mode.
Dense cosine relevance thresholds must be calibrated for the selected corpus
and embedder; RRF scores are used only for ranking.
"""

from __future__ import annotations

import math
import threading
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .bm25 import BM25Index
from .embeddings import BaseEmbedder, make_embedder
from project_settings import configured_path, setting

from .store import CURRENT_ALIAS, LocalVectorStore, _cosine, get_store

FALLBACK_MESSAGE = "không đủ thông tin"


def validate_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Relevance threshold must be finite and non-negative")
    return value


@dataclass
class RetrievedChunk:
    chunk_id: str
    fused_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        name = self.payload.get("file_name", "")
        page = self.payload.get("page")
        return f"[{name}, trang {page}]" if page else f"[{name}]"


def rrf_fuse(
    dense_ranked: Sequence[str],
    sparse_ranked: Sequence[str],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over two ranked id lists."""
    fused: dict[str, float] = {}
    for rank, cid in enumerate(dense_ranked, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ranked, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


RerankerFn = Callable[[str, list[RetrievedChunk]], list[RetrievedChunk]]


class BgeReranker:
    """Shared lazy cross-encoder; score the candidate pool before selecting top 5."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or setting("retrieval.reranker_model")
        self.device = device or setting("retrieval.reranker_device")
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("pip install 'sentence-transformers>=2.7' for reranking") from exc
            self._model = CrossEncoder(
                self.model_name, device=self.device,
                max_length=setting("retrieval.reranker_max_length"),
                cache_folder=str(configured_path("retrieval.reranker_cache_dir")),
            )
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.payload.get("content", "")) for c in candidates]
        # Serialize lazy loading and CPU inference across concurrent API requests.
        with self._lock:
            model = self._load()
            scores = model.predict(pairs, batch_size=setting("retrieval.reranker_batch_size"),
                                   show_progress_bar=False)
        if len(scores) != len(candidates) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError("Reranker returned invalid scores")
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        ordered = sorted(zip(candidates, scores), key=lambda t: float(t[1]), reverse=True)
        return [c for c, _ in ordered]


class HybridRetriever:
    """Qdrant-native hybrid search with an explicit local development backend."""

    def __init__(
        self,
        store: LocalVectorStore | Any,
        bm25: BM25Index | None = None,
        embedder: BaseEmbedder | None = None,
        payloads: dict[str, dict] | None = None,
        threshold: float | None = None,
    ) -> None:
        if embedder is None:
            raise ValueError("An embedder is required")
        if isinstance(store, LocalVectorStore) and bm25 is None:
            raise ValueError("The local development backend requires a BM25 index")
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self._payloads = payloads or {}
        configured = setting("retrieval.relevance_threshold", "RAG_RELEVANCE_THRESHOLD") if threshold is None else threshold
        self.threshold = validate_threshold(configured) if configured is not None else None

    @classmethod
    def load(
        cls,
        rag_dir: Path | None = None,
        embedder_name: str | None = None,
        backend: str | None = None,
        qdrant_url: str | None = None,
        collection: str | None = None,
        qdrant_api_key: str | None = None,
        threshold: float | None = None,
        **embedder_kwargs,
    ) -> "HybridRetriever":
        backend = (backend or setting("qdrant.backend", "RAG_BACKEND")).strip().lower()
        embedder_name = embedder_name or setting("embedding.backend", "RAG_EMBEDDER")
        collection = collection or setting("qdrant.alias", "QDRANT_COLLECTION")
        embedder = make_embedder(embedder_name, **embed_kwargs(embedder_kwargs))
        bm25 = None
        payloads = None
        if backend == "local":
            if rag_dir is None:
                raise ValueError("rag_dir is required for the local development backend")
            rag_dir = Path(rag_dir)
            bm25_path = rag_dir / "bm25.json"
            if not bm25_path.exists():
                raise FileNotFoundError(
                    f"BM25 index not found: {bm25_path} (run local ingestion first)"
                )
            bm25 = BM25Index.load(bm25_path)
            store = LocalVectorStore.load(rag_dir)
            payloads = {str(p.get("chunk_id")): dict(p) for p in store.payloads}
        else:
            from .store import resolve_qdrant_settings

            url, api_key = resolve_qdrant_settings(qdrant_url, qdrant_api_key)
            store = get_store("qdrant" if backend == "auto" else backend,
                              url=url,
                              collection=collection, api_key=api_key)
        return cls(store=store, bm25=bm25, embedder=embedder, payloads=payloads,
                   threshold=threshold)

    def retrieve(
        self,
        query: str,
        dense_top: int | None = None,
        sparse_top: int | None = None,
        top_k: int | None = None,
        rrf_k: int | None = None,
        filters: dict | None = None,
        threshold: float | None = None,
        reranker: RerankerFn | None = None,
        metrics: dict | None = None,
    ) -> dict[str, Any]:
        """Gate cosine relevance before reranking/generation; RRF ranks results.

        A missing calibrated threshold fails closed. An explicit request may
        tighten the configured threshold but cannot weaken the deployment gate.
        """
        started = perf_counter()
        metrics = metrics if metrics is not None else {}
        metrics.update(retrieved_count=0, reranked_count=0, selected_count=0, rerank_ms=0.0)
        dense_top = dense_top if dense_top is not None else setting("retrieval.dense_top")
        sparse_top = sparse_top if sparse_top is not None else setting("retrieval.sparse_top")
        top_k = top_k if top_k is not None else setting("retrieval.top_k")
        rrf_k = rrf_k if rrf_k is not None else setting("retrieval.local_rrf_k")
        if min(top_k, dense_top, sparse_top) <= 0:
            raise ValueError("top_k, dense_top and sparse_top must be positive")
        effective_threshold = self.threshold
        if threshold is not None:
            override = validate_threshold(threshold)
            effective_threshold = max(override, self.threshold or 0.0)
        if not query.strip():
            return {"query": query, "chunks": [], "fallback": True, "reason": "empty query",
                    "fallback_message": FALLBACK_MESSAGE}
        if isinstance(self.store, LocalVectorStore):
            candidates, counts = self._retrieve_local(
                query, dense_top, sparse_top, rrf_k, filters
            )
        else:
            # One query encoding and one Qdrant Query API call. Fusion and
            # payload retrieval happen in Qdrant, including filtered prefetch.
            embedding = self.embedder.embed_hybrid([query])[0]
            candidate_limit = max(top_k, setting("retrieval.candidate_top"))
            hits = self.store.search_hybrid_scored(
                embedding.dense, embedding.sparse_indices, embedding.sparse_values,
                top_k=candidate_limit, dense_top=dense_top, sparse_top=sparse_top,
                filters=filters,
            )
            candidates = [
                RetrievedChunk(
                    chunk_id=hit.chunk_id, fused_score=hit.fused_score,
                    dense_score=hit.dense_score, payload=hit.payload,
                ) for hit in hits
            ]
            counts = {"hybrid_count": len(candidates)}
        candidates = candidates[:max(top_k, setting("retrieval.candidate_top"))]
        metrics.update(retrieved_count=len(candidates), retrieval_ms=(perf_counter() - started) * 1000)
        if not candidates:
            return {"query": query, "chunks": [], "fallback": True,
                    "reason": "no matching chunks", "fallback_message": FALLBACK_MESSAGE,
                    **counts}
        # Ignore malformed/non-finite/empty evidence even if a backend ranks it.
        candidates = [c for c in candidates if (
            c.dense_score is not None and math.isfinite(c.dense_score)
            and math.isfinite(c.fused_score) and str(c.payload.get("content") or "").strip()
        )]
        best_score = max((c.dense_score for c in candidates), default=0.0)
        diagnostics = {"best_score": best_score, "score_metric": "dense_cosine",
                       "threshold": effective_threshold, **counts}
        if effective_threshold is None:
            return {"query": query, "chunks": [], "fallback": True,
                    "reason": "relevance threshold not configured",
                    "fallback_message": FALLBACK_MESSAGE, **diagnostics}
        candidates = [c for c in candidates
                      if c.dense_score > 0 and c.dense_score >= effective_threshold]
        if not candidates:
            return {"query": query, "chunks": [], "fallback": True,
                    "reason": "below relevance threshold",
                    "fallback_message": FALLBACK_MESSAGE, **diagnostics}
        if reranker is not None:
            metrics["reranked_count"] = len(candidates)
            rerank_started = perf_counter()
            try:
                candidates = reranker(query, candidates)
            finally:
                metrics["rerank_ms"] = (perf_counter() - rerank_started) * 1000
        top = candidates[:top_k]
        metrics["selected_count"] = len(top)
        if not top:
            return {"query": query, "chunks": [], "fallback": True,
                    "reason": "no matching chunks", "fallback_message": FALLBACK_MESSAGE,
                    **diagnostics}
        return {"query": query, "chunks": top, "fallback": False,
                **diagnostics}

    def _retrieve_local(self, query, dense_top, sparse_top, rrf_k, filters):
        """Legacy offline backend; production always delegates fusion to Qdrant."""
        query_vector = self.embedder.embed_texts([query])[0]
        dense_hits = self.store.search_dense(query_vector, top_k=dense_top, filters=filters)
        dense_ids = [cid for cid, _, _ in dense_hits]
        dense_scores = {cid: float(s) for cid, s, _ in dense_hits}
        # Filter before taking top-N so other documents cannot exhaust the quota.
        sparse_hits = self.bm25.search(query, top_k=len(self.bm25))
        if filters:
            sparse_hits = [(cid, s) for cid, s in sparse_hits
                           if _match_filters(self._lookup(cid), filters)]
        sparse_hits = sparse_hits[:sparse_top]
        sparse_ids = [cid for cid, _ in sparse_hits]
        sparse_scores = dict(sparse_hits)
        # Sparse-only candidates still need a semantic score for the gate.
        dense_scores = {
            cid: _cosine(query_vector, vec)
            for cid, vec in zip(self.store.ids, self.store.vectors)
        }
        fused = rrf_fuse(dense_ids, sparse_ids, k=rrf_k)
        candidates = [
            RetrievedChunk(
                chunk_id=cid, fused_score=score, dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid), payload=self._lookup(cid, dense_hits),
            )
            for cid, score in fused.items()
        ]
        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        return candidates, {"dense_count": len(dense_ids), "sparse_count": len(sparse_ids)}

    def _lookup(self, chunk_id: str, dense_hits: list | None = None) -> dict:
        if dense_hits:
            for cid, _, payload in dense_hits:
                if cid == chunk_id:
                    return dict(payload)
        if chunk_id in self._payloads:
            return dict(self._payloads[chunk_id])
        # Local store fallback: scan payloads by chunk_id.
        payloads = getattr(self.store, "payloads", [])
        for payload in payloads:
            if str(payload.get("chunk_id")) == chunk_id:
                return dict(payload)
        return {"chunk_id": chunk_id}


def _match_filters(payload: dict, filters: dict) -> bool:
    if filters.get("doc_id") and payload.get("doc_id") != filters["doc_id"]:
        return False
    if filters.get("type") and payload.get("type") != filters["type"]:
        return False
    return True


def embed_kwargs(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k in {
        "dim", "model_name", "revision", "device", "batch_size", "max_length",
        "cache", "cache_dir", "model",
    }}
