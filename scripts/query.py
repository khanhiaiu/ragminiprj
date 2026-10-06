#!/usr/bin/env python3
"""Step-3 query CLI: hybrid retrieve over an ingested .cache/rag directory.

Examples:
  python scripts/query.py --rag-dir .cache/rag --query "Phí thường niên thẻ chuẩn là bao nhiêu?"
  python scripts/query.py --rag-dir .cache/rag --query "..." --top-k 5 --threshold 0.02 --as-json
  python scripts/query.py --rag-dir .cache/rag --query "..." --filter-type table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.retrieve import FALLBACK_MESSAGE, HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-dir", type=Path, default=ROOT / ".cache" / "rag")
    parser.add_argument("--query", required=True)
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--backend", default="local")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--dense-top", type=int, default=20)
    parser.add_argument("--sparse-top", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fallback if best fused score < threshold. Tune on eval/questions.json.")
    parser.add_argument("--filter-doc", default=None)
    parser.add_argument("--filter-type", default=None, choices=["text", "table", "figure"])
    parser.add_argument("--rerank", action="store_true", help="Use bge-reranker-v2-m3 (needs model download)")
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    retriever = HybridRetriever.load(
        args.rag_dir, embedder_name=args.embedder, backend=args.backend,
        qdrant_url=args.qdrant_url, collection=args.collection,
    )
    filters = {}
    if args.filter_doc:
        filters["doc_id"] = args.filter_doc
    if args.filter_type:
        filters["type"] = args.filter_type
    reranker = None
    if args.rerank:
        from rag.retrieve import BgeReranker

        _r = BgeReranker()
        reranker = _r.rerank
    result = retriever.retrieve(
        args.query, dense_top=args.dense_top, sparse_top=args.sparse_top,
        top_k=args.top_k, filters=filters or None, threshold=args.threshold, reranker=reranker,
    )
    if args.as_json:
        print(json.dumps({
            "query": result["query"], "fallback": result["fallback"],
            "chunks": [{"chunk_id": c.chunk_id, "fused_score": c.fused_score,
                        "dense_score": c.dense_score, "sparse_score": c.sparse_score,
                        "payload": c.payload, "citation": c.citation} for c in result["chunks"]],
            "reason": result.get("reason"), "fallback_message": FALLBACK_MESSAGE if result["fallback"] else None,
        }, ensure_ascii=False, indent=2))
        return 0
    if result["fallback"]:
        print(FALLBACK_MESSAGE + (f" ({result.get('reason')})" if result.get("reason") else ""))
        return 0
    for i, chunk in enumerate(result["chunks"], start=1):
        payload = chunk.payload
        print(f"[{i}] {chunk.citation} score={chunk.fused_score:.4f} "
              f"(dense={chunk.dense_score}, sparse={chunk.sparse_score}) {payload.get('type')}")
        print(payload.get("content", "")[:800])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
