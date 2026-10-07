#!/usr/bin/env python3
"""Ensure the Qdrant collection for Step 2 exists (requires qdrant-client + server).

Usage:
  python scripts/init_qdrant.py [--qdrant-url http://localhost:6333] [--collection docs]
  python scripts/init_qdrant.py --memory   # no server needed, validates schema only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=None,
                        help="Qdrant URL (default: QDRANT_URL env / .env, else http://localhost:6333)")
    parser.add_argument("--qdrant-api-key", default=None,
                        help="Qdrant API key (default: QDRANT_API_KEY env / .env)")
    parser.add_argument("--collection", default="docs")
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    from rag.store import get_store, resolve_qdrant_settings

    backend = "qdrant-memory" if args.memory else "qdrant"
    url, api_key = resolve_qdrant_settings(args.qdrant_url, args.qdrant_api_key)
    store = get_store(backend, url=url or "http://localhost:6333",
                      collection=args.collection, api_key=api_key)
    store.ensure_collection()
    print(f"Collection '{args.collection}' ready (backend={backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
