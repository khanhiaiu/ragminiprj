#!/usr/bin/env python3
"""Run dense, sparse and hybrid smoke queries against a Qdrant generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_settings import configure_cli, configured_path, setting  # noqa: E402

from rag.embeddings import make_embedder  # noqa: E402
from rag.indexing import GenerationIndexer  # noqa: E402
from rag.store import QdrantStore, resolve_qdrant_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_cli(parser)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--expected-points", required=True, type=int)
    parser.add_argument("--query", default="Điều khoản và số liệu trong tài liệu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    url, api_key = resolve_qdrant_settings(args.qdrant_url, args.qdrant_api_key)
    store = QdrantStore(url=url, api_key=api_key, collection=args.collection)
    query = make_embedder(setting("embedding.backend", "RAG_EMBEDDER")).embed_hybrid([args.query])[0]
    indexer = GenerationIndexer(store)
    result = indexer.verify(args.expected_points, query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.publish:
        indexer.publish(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
