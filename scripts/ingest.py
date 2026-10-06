#!/usr/bin/env python3
"""Step-2 ingest: parsed documents -> chunks -> bge-m3 dense + BM25 -> store.

Inputs (one of):
  --input parsed_test_document        directory of */document.json (CanonicalDocument)
  --chunks chunks.jsonl               pre-chunked output from Step-1 owner (Chunk JSONL)

The Step-1 owner can pass --chunks to skip the temporary fallback chunker.
Otherwise each document.json is chunked with rag.chunk_contract.chunk_documents.

Outputs in --output (default .cache/rag):
  chunks.jsonl    all ingested chunks (Chunk schema)
  vectors.json + payloads.jsonl   LocalVectorStore persistence (--backend local)
  bm25.json       lexical index for hybrid search
  manifest.json   counts, embedder, backend, collection

Dedup: content_hash already in store is skipped (re-ingest safe per §4.2).

Examples:
  python scripts/ingest.py --input parsed_test_document --embedder hash --backend local
  python scripts/ingest.py --input parsed_test_document --embedder bge-m3 --backend qdrant --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.bm25 import BM25Index  # noqa: E402
from rag.chunk_contract import chunk_documents  # noqa: E402
from rag.embeddings import make_embedder  # noqa: E402
from rag.schemas import Chunk  # noqa: E402
from rag.store import COLLECTION, LocalVectorStore, get_store  # noqa: E402


def discover_documents(input_dir: Path) -> list[Path]:
    docs = sorted(input_dir.glob("*/document.json"))
    if input_dir.joinpath("document.json").exists():
        docs.append(input_dir / "document.json")
    return docs


def load_chunks_file(path: Path) -> list[Chunk]:
    chunks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "parsed_test_document")
    parser.add_argument("--chunks", type=Path, default=None, help="Pre-chunked Chunk JSONL (Step-1 output)")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "rag")
    parser.add_argument("--embedder", default="hash", help="'hash' (offline/tests) or 'bge-m3' (production)")
    parser.add_argument("--backend", default="local", help="'local' | 'qdrant' | 'qdrant-memory' | 'auto'")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.chunks is not None:
        chunks = load_chunks_file(args.chunks)
    else:
        docs = discover_documents(args.input)
        if not docs:
            print(f"No document.json found under {args.input}", file=sys.stderr)
            print("Hint: run scripts/parse_test_documents.py first, or pass --chunks chunks.jsonl", file=sys.stderr)
            return 1
        chunks = []
        for doc_path in docs:
            data = json.loads(doc_path.read_text(encoding="utf-8"))
            chunks.extend(chunk_documents(data))
    if not chunks:
        print("No chunks to ingest", file=sys.stderr)
        return 1

    backend = args.backend
    if backend == "local":
        store = LocalVectorStore()
        prev = LocalVectorStore.load(output)
        store.ids, store.vectors, store.payloads = prev.ids, prev.vectors, prev.payloads
        store._by_hash = dict(prev._by_hash)
        qdrant = None
    else:
        url = None if backend == "qdrant-memory" else args.qdrant_url
        store = get_store(backend, url=url, collection=args.collection)
        store.ensure_collection()
        qdrant = store if backend != "local" else None

    known_hashes = store.existing_hashes()
    fresh = [c for c in chunks if c.content_hash not in known_hashes]
    # De-duplicate within this batch as well.
    seen: set[str] = set(known_hashes)
    batch: list[Chunk] = []
    for chunk in fresh:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        batch.append(chunk)

    embedder = make_embedder(args.embedder)
    vectors: list[list[float]] = []
    for i in range(0, len(batch), args.batch_size):
        vectors.extend(embedder.embed_texts([c.content for c in batch[i : i + args.batch_size]]))

    payloads = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "file_name": c.file_name,
            "page": c.page,
            "section": c.section,
            "type": c.type,
            "image_path": c.image_path,
            "content": c.content,
            "content_hash": c.content_hash,
            "ingested_at": c.ingested_at,
            **({"metadata": c.metadata} if c.metadata else {}),
        }
        for c in batch
    ]
    inserted = store.upsert([c.chunk_id for c in batch], vectors, payloads) if batch else 0

    # BM25 covers the full store (previous + new) for local backend;
    # for Qdrant backend it covers this ingest view (full rebuild needs scroll).
    if isinstance(store, LocalVectorStore):
        all_ids = list(store.ids)
        all_texts = [str(p.get("content", "")) for p in store.payloads]
        all_chunks = [Chunk.model_validate(_payload_to_chunk(p)) for p in store.payloads]
    else:
        all_ids = [c.chunk_id for c in chunks]
        all_texts = [c.content for c in chunks]
        all_chunks = chunks
    bm25 = BM25Index()
    if all_ids:
        bm25.add(all_ids, all_texts)
        bm25.save(output / "bm25.json")

    if isinstance(store, LocalVectorStore):
        store.save(output)
    (output / "chunks.jsonl").write_text(
        "\n".join(c.model_dump_json() for c in all_chunks) + ("\n" if all_chunks else ""), encoding="utf-8"
    )
    manifest = {
        "input": str(args.input if args.chunks is None else args.chunks),
        "backend": backend,
        "collection": args.collection,
        "embedder": args.embedder,
        "embed_dim": embedder.dim,
        "chunks_seen": len(chunks),
        "chunks_new": len(batch),
        "chunks_inserted": inserted,
        "chunks_skipped_dup": len(chunks) - len(batch),
        "total_points": len(store) if isinstance(store, LocalVectorStore) else "qdrant",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    _ = qdrant
    return 0


def _payload_to_chunk(payload: dict) -> dict:
    return {
        "chunk_id": payload.get("chunk_id", ""),
        "doc_id": payload.get("doc_id", ""),
        "file_name": payload.get("file_name", ""),
        "page": payload.get("page"),
        "section": payload.get("section"),
        "type": payload.get("type", "text"),
        "content": payload.get("content", ""),
        "image_path": payload.get("image_path"),
        "content_hash": payload.get("content_hash", ""),
        "ingested_at": payload.get("ingested_at", "1970-01-01T00:00:00+00:00"),
        "metadata": payload.get("metadata", {}),
    }


if __name__ == "__main__":
    raise SystemExit(main())
