"""Step-3 hybrid retrieval (§4.4): dense + BM25 → RRF → top-K + fallback.

Spec: encode question (dense + sparse), Qdrant prefetch dense top20 +
sparse top20, RRF fuse, cross-encoder rerank keep top5. If top score
below a tuned threshold → fallback "không đủ thông tin", no LLM call.
Threshold must be tuned on an eval set, never hardcoded as final.

This step implements BM25 as the sparse side (per user decision) and
keeps the fusion/threshold/rerank-hook API stable so a later Qdrant
native-sparse upgrade is drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .bm25 import BM25Index
from .embeddings import BaseEmbedder, make_embedder
from .store import LocalVectorStore, get_store

FALLBACK_MESSAGE = "không đủ thông tin"


@dataclass
class RetrievedChunk:
    chunk_id: str
    fused_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

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
    """Optional cross-encoder rerank (bge-reranker-v2-m3). Lazy, CPU.

    Disabled by default (heavy on 13GB CPU). Enable in production with:
      retriever.retrieve(query, reranker=BgeReranker().rerank)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("pip install 'sentence-transformers>=2.7' for reranking") from exc
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.payload.get("content", "")) for c in candidates]
        scores = model.predict(pairs)
        ordered = sorted(zip(candidates, scores), key=lambda t: float(t[1]), reverse=True)
        return [c for c, _ in ordered]


class HybridRetriever:
    """Dense (Local/Qdrant) + BM25 hybrid with RRF fusion."""

    def __init__(
        self,
        store: LocalVectorStore | Any,
        bm25: BM25Index,
        embedder: BaseEmbedder,
        payloads: dict[str, dict] | None = None,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self._payloads = payloads or {}

    @classmethod
    def load(
        cls,
        rag_dir: Path,
        embedder_name: str = "hash",
        backend: str = "local",
        qdrant_url: str = "http://localhost:6333",
        collection: str = "docs",
        **embedder_kwargs,
    ) -> "HybridRetriever":
        rag_dir = Path(rag_dir)
        bm25_path = rag_dir / "bm25.json"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path} (run scripts/ingest.py first)")
        bm25 = BM25Index.load(bm25_path)
        embedder = make_embedder(embedder_name, **embed_kwargs(embedder_kwargs))
        if backend == "local":
            store = LocalVectorStore.load(rag_dir)
            payloads = {str(p.get("chunk_id")): dict(p) for p in store.payloads}
        else:
            store = get_store(backend, url=qdrant_url, collection=collection)
            payloads = _payloads_from_chunks(rag_dir / "chunks.jsonl")
        return cls(store=store, bm25=bm25, embedder=embedder, payloads=payloads)

    def retrieve(
        self,
        query: str,
        dense_top: int = 20,
        sparse_top: int = 20,
        top_k: int = 5,
        rrf_k: int = 60,
        filters: dict | None = None,
        threshold: float | None = None,
        reranker: RerankerFn | None = None,
    ) -> dict[str, Any]:
        """Return {chunks: top-K, fallback: bool, ...}. Empty query → fallback."""
        if not query.strip():
            return {"query": query, "chunks": [], "fallback": True, "reason": "empty query",
                    "fallback_message": FALLBACK_MESSAGE}
        query_vector = self.embedder.embed_texts([query])[0]
        dense_hits = self.store.search_dense(query_vector, top_k=dense_top, filters=filters)
        dense_ids = [cid for cid, _, _ in dense_hits]
        dense_scores = {cid: float(s) for cid, s, _ in dense_hits}
        sparse_hits = self.bm25.search(query, top_k=sparse_top)
        if filters:
            sparse_hits = [(cid, s) for cid, s in sparse_hits if _match_filters(self._lookup(cid), filters)]
        sparse_ids = [cid for cid, _ in sparse_hits]
        sparse_scores = dict(sparse_hits)

        fused = rrf_fuse(dense_ids, sparse_ids, k=rrf_k)
        candidates = [
            RetrievedChunk(
                chunk_id=cid,
                fused_score=score,
                dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid),
                payload=self._lookup(cid, dense_hits),
            )
            for cid, score in fused.items()
        ]
        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        if reranker is not None:
            candidates = reranker(query, candidates)
        top = candidates[:top_k]
        if threshold is not None and (not top or top[0].fused_score < threshold):
            return {"query": query, "chunks": [], "fallback": True, "reason": "below threshold",
                    "threshold": threshold, "fallback_message": FALLBACK_MESSAGE,
                    "best_score": top[0].fused_score if top else 0.0}
        return {"query": query, "chunks": top, "fallback": False,
                "dense_count": len(dense_ids), "sparse_count": len(sparse_ids)}

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


def _payloads_from_chunks(path: Path) -> dict[str, dict]:
    import json

    payloads: dict[str, dict] = {}
    if not path.exists():
        return payloads
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            payloads[str(item.get("chunk_id"))] = item
    return payloads


def embed_kwargs(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k in {"dim", "model_name", "device", "batch_size"}}
