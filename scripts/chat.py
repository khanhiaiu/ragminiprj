#!/usr/bin/env python3
"""Step-4 chat CLI: rewrite → retrieve → LLM answer (or fallback, no LLM call).

Single question:
  python scripts/chat.py --rag-dir .cache/rag --query "Phí thường niên thẻ chuẩn?"
Interactive:
  python scripts/chat.py --rag-dir .cache/rag
Production LLM (llama-server with Qwen3-4B Q4_K_M on :8080):
  python scripts/chat.py --rag-dir .cache/rag --llm server --query "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.answer import answer_question  # noqa: E402
from rag.llm import make_llm  # noqa: E402
from rag.memory import SessionMemory  # noqa: E402
from rag.retrieve import HybridRetriever  # noqa: E402


def ask(retriever, llm, memory, session_id: str, question: str, top_k: int, threshold) -> None:
    result = answer_question(retriever, question, llm, memory,
                             session_id=session_id, top_k=top_k, threshold=threshold)
    print(result["answer"])
    if not result["fallback"] and result.get("standalone_question") != question:
        print(f"(standalone: {result['standalone_question']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-dir", type=Path, default=ROOT / ".cache" / "rag")
    parser.add_argument("--query", default=None)
    parser.add_argument("--llm", default="fake", help="'fake' or 'server'")
    parser.add_argument("--llm-model", default="unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")
    parser.add_argument("--llm-url", default="http://localhost:8080/v1")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--backend", default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--session", default="cli")
    args = parser.parse_args()

    retriever = HybridRetriever.load(args.rag_dir, embedder_name=args.embedder, backend=args.backend)
    llm = make_llm(args.llm, model=args.llm_model, base_url=args.llm_url)
    memory = SessionMemory()
    if args.query:
        ask(retriever, llm, memory, args.session, args.query, args.top_k, args.threshold)
        return 0
    print("Nhập câu hỏi (trống để thoát):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(retriever, llm, memory, args.session, question, args.top_k, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
