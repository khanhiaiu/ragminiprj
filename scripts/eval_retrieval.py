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
  python scripts/eval_retrieval.py --collection docs_current
  python scripts/eval_retrieval.py --questions eval/questions.json --as-json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_settings import configure_cli, configured_path, setting  # noqa: E402

from rag.retrieve import HybridRetriever  # noqa: E402


def supporting_chunks(item: dict, chunks: list) -> list:
    """Match answer text, type and expected document in the same source chunk."""
    source_doc = (item.get("source") or {}).get("doc_id")
    return [chunk for chunk in chunks if (
        (not item.get("expect_contains") or item["expect_contains"] in str(chunk.payload.get("content", "")))
        and (not item.get("expect_type") or chunk.payload.get("type") == item["expect_type"])
        and (not source_doc or chunk.payload.get("doc_id") == source_doc)
    )]


def calibrate_threshold(items: list[dict], results: list[dict]) -> dict:
    """Choose a separating cosine threshold; never invent a production default.

    Positive questions need actual matching source evidence. Both supported
    and unsupported questions are required, and overlapping score ranges fail
    calibration rather than publishing a misleading threshold.
    """
    positive_scores, negative_scores = [], []
    for item, result in zip(items, results):
        if item.get("expect_fallback"):
            negative_scores.append(float(result.get("best_score", 0.0)))
            continue
        if not item.get("expect_contains"):
            raise ValueError("Every supported calibration question needs expect_contains")
        supporting = [
            c.dense_score for c in supporting_chunks(item, result["chunks"])
            if c.dense_score is not None
        ]
        if not supporting:
            raise ValueError(f"No expected source evidence retrieved for: {item['q']}")
        positive_scores.append(max(supporting))
    if not positive_scores or not negative_scores:
        raise ValueError("Calibration requires supported and unsupported questions")
    if not all(math.isfinite(s) for s in [*positive_scores, *negative_scores]):
        raise ValueError("Calibration scores must be finite")
    highest_negative = max(negative_scores)
    lowest_positive = min(positive_scores)
    if highest_negative >= lowest_positive:
        raise ValueError(
            "No cosine threshold separates the labeled questions; improve retrieval "
            "or the evaluation set before enabling generation"
        )
    return {
        "score_metric": "dense_cosine",
        "threshold": (max(0.0, highest_negative) + lowest_positive) / 2,
        "highest_unsupported_score": highest_negative,
        "lowest_supported_score": lowest_positive,
        "supported_questions": len(positive_scores),
        "unsupported_questions": len(negative_scores),
    }


def grade(item: dict, result: dict, top_k: int) -> dict:
    chunks = result["chunks"]
    row = {
        "q": item["q"],
        "fallback": result["fallback"],
        "reason": result.get("reason"),
        "best_score": result.get("best_score", 0.0),
        "score_metric": result.get("score_metric", "dense_cosine"),
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
    # A matching type and matching text must belong to the same source chunk.
    ok = ok and bool(supporting_chunks(item, chunks[:top_k]))
    row["pass"] = ok
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_cli(parser)
    parser.add_argument("--rag-dir", type=Path, default=configured_path("ingestion.rag_dir", "RAG_DIR"))
    parser.add_argument("--questions", type=Path, default=configured_path("ingestion.questions"))
    parser.add_argument("--embedder", default=setting("embedding.backend", "RAG_EMBEDDER"), choices=["bge-m3", "hash"])
    parser.add_argument("--backend", default=setting("qdrant.backend", "RAG_BACKEND"), choices=["qdrant", "local"])
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection", default=setting("qdrant.alias", "QDRANT_COLLECTION"))
    parser.add_argument("--top-k", type=int, default=setting("retrieval.top_k"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibrate-output", type=Path,
                        help="Write a measured cosine threshold from labeled questions")
    parser.add_argument("--as-json", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.questions.read_text(encoding="utf-8"))
    items = spec["questions"]
    retriever = HybridRetriever.load(
        args.rag_dir, embedder_name=args.embedder, backend=args.backend,
        qdrant_url=args.qdrant_url, collection=args.collection,
        qdrant_api_key=args.qdrant_api_key,
        # Calibration observes positive cosine evidence independently of any
        # deployment threshold inherited from the environment.
        threshold=0.0 if args.calibrate_output else args.threshold,
    )
    results = []
    for item in items:
        results.append(retriever.retrieve(item["q"], top_k=args.top_k))
    calibration = None
    if args.calibrate_output:
        try:
            calibration = calibrate_threshold(items, results)
        except ValueError as exc:
            print(f"Calibration failed: {exc}", file=sys.stderr)
            return 1
        # Verify final top-K behavior with the chosen threshold, including
        # removal of irrelevant chunks before ranking reaches the prompt.
        results = [retriever.retrieve(item["q"], top_k=args.top_k,
                                      threshold=calibration["threshold"]) for item in items]
    rows = [grade(item, result, args.top_k) for item, result in zip(items, results)]
    passed = sum(1 for r in rows if r["pass"])
    summary = {"total": len(rows), "passed": passed, "failed": len(rows) - passed,
               "threshold": calibration["threshold"] if calibration else retriever.threshold,
               "score_metric": "dense_cosine", "questions": str(args.questions)}
    if calibration and passed == len(rows):
        calibration.update(
            backend=args.backend, collection=args.collection, embedder=args.embedder,
            questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        )
        args.calibrate_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibrate_output.write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.as_json:
        print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
    else:
        for r in rows:
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"[{mark}] score={r['best_score']} fallback={r['fallback']} {r['top_citation']}")
            print(f"       Q: {r['q']}")
        print(f"{passed}/{len(rows)} passed (cosine threshold={summary['threshold']})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
