#!/usr/bin/env python3
"""Eval runner: score hybrid retrieval against a questions JSON file.

Each item: {"q": ..., "expect_contains": str|null, "expect_type": text|table|figure|null,
            "expect_fallback": bool}

Pass criteria:
  - expect_fallback: result must be fallback.
  - otherwise: not fallback, top-K contains expect_contains (if set),
    and some chunk matches expect_type (if set).

Exit 0 when all pass, 1 otherwise. Use --threshold sweeps to tune the
fallback threshold per SystemDocuments §4.4.

Example:
  python scripts/eval_retrieval.py --rag-dir .cache/rag --backend qdrant --embedder bge-m3
  python scripts/eval_retrieval.py --questions eval/questions_demo.json --threshold 0.02 --as-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.retrieve import HybridRetriever  # noqa: E402


def grade(item: dict, result: dict, top_k: int) -> dict:
    chunks = result["chunks"]
    row = {
        "q": item["q"],
        "fallback": result["fallback"],
        "reason": result.get("reason"),
        "best_score": round(chunks[0].fused_score, 4) if chunks else 0.0,
        "top_citation": chunks[0].citation if chunks else None,
    }
    if item.get("expect_fallback"):
        row["pass"] = result["fallback"] is True
        return row
    if result["fallback"]:
        row["pass"] = False
        return row
    ok = True
    want = item.get("expect_contains")
    if want:
        ok = any(want in str(c.payload.get("content", "")) for c in chunks[:top_k])
        row["contains"] = ok
    want_type = item.get("expect_type")
    if want_type:
        type_ok = any(c.payload.get("type") == want_type for c in chunks[:top_k])
        row["type_match"] = type_ok
        ok = ok and type_ok
    row["pass"] = ok
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-dir", type=Path, default=ROOT / ".cache" / "rag")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--backend", default="local")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.questions.read_text(encoding="utf-8"))
    items = spec["questions"]
    retriever = HybridRetriever.load(
        args.rag_dir, embedder_name=args.embedder, backend=args.backend,
        qdrant_url=args.qdrant_url, collection=args.collection,
        qdrant_api_key=args.qdrant_api_key,
    )
    rows = []
    for item in items:
        result = retriever.retrieve(item["q"], top_k=args.top_k, threshold=args.threshold)
        rows.append(grade(item, result, args.top_k))
    passed = sum(1 for r in rows if r["pass"])
    summary = {"total": len(rows), "passed": passed, "failed": len(rows) - passed,
               "threshold": args.threshold, "questions": str(args.questions)}
    if args.as_json:
        print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"[{mark}] score={r['best_score']} fallback={r['fallback']} {r['top_citation']}")
            print(f"       Q: {r['q']}")
        print(f"{passed}/{len(rows)} passed (threshold={args.threshold})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
