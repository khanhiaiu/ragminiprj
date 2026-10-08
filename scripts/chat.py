#!/usr/bin/env python3
"""Step-4 chat CLI: rewrite → retrieve → LLM answer (or fallback, no LLM call).

Single question:
  python scripts/chat.py --query "Phí thường niên thẻ chuẩn?"
Interactive:
  python scripts/chat.py
Production LLM (llama-server with Qwen3.5-2B Q4_K_M on :8080):
  python scripts/chat.py --query "..." --rerank
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_settings import configure_cli, configured_path, setting  # noqa: E402

from rag.answer import answer_question  # noqa: E402
from rag.llm import make_llm  # noqa: E402
from rag.memory import SessionMemory  # noqa: E402
from rag.retrieve import BgeReranker, HybridRetriever  # noqa: E402


def ask(retriever, llm, memory, session_id: str, question: str, top_k: int, threshold,
        reranker=None) -> None:
    result = answer_question(retriever, question, llm, memory,
                             session_id=session_id, top_k=top_k, threshold=threshold,
                             reranker=reranker)
    print(result["answer"])
    if not result["fallback"] and result.get("standalone_question") != question:
        print(f"(standalone: {result['standalone_question']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_cli(parser)
    parser.add_argument("--rag-dir", type=Path, default=configured_path("ingestion.rag_dir", "RAG_DIR"))
    parser.add_argument("--query", default=None)
    parser.add_argument("--llm", default=setting("llm.backend", "LLM_BACKEND"), choices=["fake", "server"])
    parser.add_argument("--llm-model", default=setting("llm.model", "LLM_MODEL"))
    parser.add_argument("--llm-url", default=setting("llm.base_url", "LLM_BASE_URL"))
    parser.add_argument("--embedder", default=setting("embedding.backend", "RAG_EMBEDDER"), choices=["bge-m3", "hash"])
    parser.add_argument("--backend", default=setting("qdrant.backend", "RAG_BACKEND"), choices=["qdrant", "local"])
    parser.add_argument("--qdrant-url", default=None,
                        help="Qdrant URL (default: QDRANT_URL env / .env, else http://localhost:6333)")
    parser.add_argument("--qdrant-api-key", default=None,
                        help="Qdrant API key (default: QDRANT_API_KEY env / .env)")
    parser.add_argument("--collection", default=setting("qdrant.alias", "QDRANT_COLLECTION"))
    parser.add_argument("--top-k", type=int, default=setting("retrieval.top_k"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum dense cosine relevance; defaults to RAG_RELEVANCE_THRESHOLD")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=setting("retrieval.rerank"))
    parser.add_argument("--session", default="cli")
    args = parser.parse_args()

    retriever = HybridRetriever.load(
        args.rag_dir, embedder_name=args.embedder, backend=args.backend,
        qdrant_url=args.qdrant_url, collection=args.collection,
        qdrant_api_key=args.qdrant_api_key,
    )
    llm = make_llm(args.llm, model=args.llm_model, base_url=args.llm_url)
    memory = SessionMemory()
    reranker = BgeReranker().rerank if args.rerank else None
    if args.query:
        ask(retriever, llm, memory, args.session, args.query, args.top_k, args.threshold, reranker)
        return 0
    print("Nhập câu hỏi (trống để thoát):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(retriever, llm, memory, args.session, question, args.top_k, args.threshold, reranker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
