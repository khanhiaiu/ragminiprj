#!/usr/bin/env python3
"""Prepare type-aware chunks or build and publish a verified hybrid Qdrant generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document_parser.normalization.schema import CanonicalDocument  # noqa: E402
from document_parser.retrieval.preprocessor import RetrievalPreprocessor  # noqa: E402
from rag.bm25 import BM25Index  # noqa: E402
from rag.chunk_contract import chunk_documents  # noqa: E402
from rag.chunking import TypeAwareChunker  # noqa: E402
from rag.chunking.chunker import ChunkingConfig  # noqa: E402
from rag.embeddings import EmbeddingCache, HashEmbedder, make_embedder  # noqa: E402
from rag.enrichment.tokenizer import (  # noqa: E402
    BGE_M3_MODEL,
    BGE_M3_REVISION,
    load_bge_m3_tokenizer,
)
from rag.indexing import GenerationIndexer  # noqa: E402
from rag.schemas import Chunk  # noqa: E402
from rag.store import COLLECTION, LocalVectorStore, get_store, resolve_qdrant_settings  # noqa: E402


class WhitespaceTokenizer:
    """Reversible tokenizer reserved for deterministic fake/offline tests."""

    name_or_path = "test/whitespace"

    def __init__(self):
        self.forward = {}
        self.reverse = {}

    def encode(self, text, *, add_special_tokens=False):
        values = []
        for token in re.findall(r"\S+", text):
            if token not in self.forward:
                index = len(self.forward) + 1
                self.forward[token] = index
                self.reverse[index] = token
            values.append(self.forward[token])
        return values

    def decode(self, values, **kwargs):
        return " ".join(self.reverse[value] for value in values)


def discover_documents(input_dir: Path) -> list[Path]:
    docs = sorted(input_dir.glob("*/document.json"))
    if input_dir.joinpath("document.json").exists():
        docs.append(input_dir / "document.json")
    return docs


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunks_file(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(value) for value in load_jsonl(path)]


def load_captions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    records = load_jsonl(path)
    unfinished = [value for value in records if value.get("status") not in {
        "completed", "needs_review", "excluded"
    }]
    if not records or unfinished:
        raise ValueError(
            f"Caption file is incomplete: {len(unfinished)}/{len(records)} records pending/error; "
            "generate captions before preparing chunks"
        )
    captions = {value["image_element_id"]: value for value in records}
    if len(captions) != len(records):
        raise ValueError("Duplicate image_element_id in caption file")
    return captions


def payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "document_version": chunk.document_version,
        "file_name": chunk.file_name,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "section": chunk.section,
        "heading_path": chunk.heading_path,
        "sheet": chunk.sheet,
        "cell_range": chunk.cell_range,
        "type": chunk.type,
        "image_path": chunk.image_path,
        "image_hash": chunk.image_hash,
        "image": [item.model_dump(mode="json") for item in chunk.image],
        "content": chunk.content,
        "text_for_embedding": chunk.text_for_embedding or chunk.content,
        "source_element_ids": chunk.source_element_ids,
        "source_spans": chunk.source_spans,
        "uncertainty_flags": chunk.uncertainty_flags,
        "content_hash": chunk.content_hash,
        "ingested_at": chunk.ingested_at,
        "chunker_version": chunk.chunker_version,
        "metadata": chunk.metadata,
    }


def prepare(args, output: Path) -> list[Chunk]:
    if args.chunks:
        return load_chunks_file(args.chunks)
    documents = discover_documents(args.input)
    if not documents:
        raise FileNotFoundError(f"No document.json found under {args.input}")
    captions = load_captions(args.captions)
    tokenizer = (
        WhitespaceTokenizer()
        if args.embedder == "hash"
        else load_bge_m3_tokenizer(local_files_only=args.local_files_only)
    )
    chunker = TypeAwareChunker(tokenizer, ChunkingConfig(merge_image_captions=bool(args.captions)))
    chunks = []
    retrieval_dir = output / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    for path in documents:
        canonical = CanonicalDocument.model_validate_json(path.read_text(encoding="utf-8"))
        if args.captions:
            missing = [
                element.element_id for element in canonical.elements
                if element.element_type == "image"
                and not element.metadata.get("exclude_from_content")
                and element.element_id not in captions
            ]
            if missing:
                raise ValueError(f"Missing captions for images in {canonical.document_id}: {missing}")
        derived = RetrievalPreprocessor().process(canonical, captions=captions, asset_base=path.parent)
        (retrieval_dir / f"{canonical.document_id}.json").write_text(
            derived.model_dump_json(indent=2), encoding="utf-8"
        )
        chunks.extend(chunker.chunk(derived))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "parsed_test_document/all_documents_gpu")
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--chunks", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ingestion")
    parser.add_argument("--run-id", help="Create a versioned output subdirectory")
    parser.add_argument("--mode", choices=["prepare-only", "full-ingest"], default="full-ingest")
    parser.add_argument("--prepare-only", action="store_true", help="Compatibility alias for --mode prepare-only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedder", choices=["hash", "bge-m3"], default="bge-m3")
    parser.add_argument("--backend", choices=["local", "qdrant", "qdrant-memory", "auto"], default="qdrant")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-api-key")
    parser.add_argument("--collection")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        args.mode = "prepare-only"
    output = args.output / args.run_id if args.run_id else args.output
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    try:
        chunks = prepare(args, output)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not chunks:
        print("No valid chunks were produced", file=sys.stderr)
        return 1
    (output / "chunks.jsonl").write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8"
    )
    planned_fingerprint = {
        "model": BGE_M3_MODEL if args.embedder == "bge-m3" else "hash",
        "revision": BGE_M3_REVISION if args.embedder == "bge-m3" else "hash-v2",
        "dense_dimensions": 1024,
        "dense": True,
        "sparse": True,
        "colbert": False,
        "status": "not_generated" if args.mode == "prepare-only" or args.dry_run else "pending",
    }
    (output / "embedding_fingerprint.json").write_text(
        json.dumps(planned_fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "run_id": args.run_id,
        "started_at": started.isoformat(),
        "input": str(args.chunks or args.input),
        "captions": str(args.captions) if args.captions else None,
        "mode": args.mode,
        "backend": args.backend,
        "embedder": args.embedder,
        "chunks_seen": len(chunks),
        "chunks_new": len(chunks),
        "chunks_inserted": 0,
        "chunks_skipped_dup": 0,
        "published": False,
    }
    if args.mode == "prepare-only" or args.dry_run:
        manifest["status"] = "prepared"
        (output / "verification_report.json").write_text(
            json.dumps(
                {
                    "status": "not_run",
                    "reason": "prepare-only/dry-run does not write a Qdrant generation",
                    "passed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    embedder = make_embedder(
        args.embedder,
        batch_size=args.batch_size,
        cache=EmbeddingCache(output / "embedding_cache") if args.embedder == "bge-m3" else None,
    )
    texts = [chunk.text_for_embedding or chunk.content for chunk in chunks]
    hybrid = []
    for index in range(0, len(texts), args.batch_size):
        hybrid.extend(embedder.embed_hybrid(texts[index : index + args.batch_size]))
    fingerprint = {
        "model": getattr(embedder, "model_name", "hash"),
        "revision": getattr(embedder, "revision", "hash-v2"),
        "encoding_config": getattr(embedder, "encoding_config", "hash"),
    }
    (output / "embedding_fingerprint.json").write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payloads = [payload(chunk) for chunk in chunks]

    if args.backend == "local":
        # Explicit test/development backend. Production defaults to Qdrant and never falls back.
        store = LocalVectorStore()
        previous = LocalVectorStore.load(output) if args.resume else LocalVectorStore()
        store.ids, store.vectors, store.payloads = previous.ids, previous.vectors, previous.payloads
        store._by_hash = dict(previous._by_hash)
        known = store.existing_hashes()
        fresh_indexes = [i for i, chunk in enumerate(chunks) if chunk.content_hash not in known]
        inserted = store.upsert(
            [chunks[i].chunk_id for i in fresh_indexes],
            [hybrid[i].dense for i in fresh_indexes],
            [payloads[i] for i in fresh_indexes],
        )
        store.save(output)
        bm25 = BM25Index()
        if store.ids:
            bm25.add(store.ids, [str(item.get("content", "")) for item in store.payloads])
            bm25.save(output / "bm25.json")
        manifest.update(
            chunks_new=len(fresh_indexes),
            chunks_inserted=inserted,
            chunks_skipped_dup=len(chunks) - len(fresh_indexes),
            total_points=len(store),
            status="completed",
        )
    else:
        generation = args.collection or f"{COLLECTION}_{started.strftime('%Y%m%dT%H%M%SZ')}"
        url, api_key = resolve_qdrant_settings(args.qdrant_url, args.qdrant_api_key)
        store = get_store(
            args.backend,
            url=url or "http://localhost:6333",
            api_key=api_key,
            collection=generation,
        )
        store.ensure_collection()
        indexer = GenerationIndexer(store)
        inserted = indexer.upsert([chunk.chunk_id for chunk in chunks], hybrid, payloads)
        verification = indexer.verify(len(chunks), hybrid[0])
        (output / "verification_report.json").write_text(
            json.dumps(verification.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not args.no_publish:
            indexer.publish(verification)
        manifest.update(
            collection=generation,
            chunks_inserted=inserted,
            total_points=verification.actual_points,
            verification_passed=verification.passed,
            published=verification.passed and not args.no_publish,
            status="completed" if verification.passed else "verification_failed",
        )
        if not verification.passed:
            (output / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return 1
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
