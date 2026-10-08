#!/usr/bin/env python3
"""Ensure a Qdrant dense/sparse collection exists (requires qdrant-client + server).

Usage:
  python scripts/init_qdrant.py --collection my_generation
  python scripts/init_qdrant.py --memory   # no server needed, validates schema only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_settings import configure_cli, configured_path, setting  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_cli(parser)
    parser.add_argument("--qdrant-url", default=None,
                        help="Qdrant URL (default: QDRANT_URL env / .env, else http://localhost:6333)")
    parser.add_argument("--qdrant-api-key", default=None,
                        help="Qdrant API key (default: QDRANT_API_KEY env / .env)")
    parser.add_argument("--collection", default=setting("qdrant.collection_prefix"))
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    from rag.store import get_store, resolve_qdrant_settings

    backend = "qdrant-memory" if args.memory else "qdrant"
    url, api_key = resolve_qdrant_settings(args.qdrant_url, args.qdrant_api_key)
    store = get_store(backend, url=url,
                      collection=args.collection, api_key=api_key)
    store.ensure_collection()
    print(f"Collection '{args.collection}' ready (backend={backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
