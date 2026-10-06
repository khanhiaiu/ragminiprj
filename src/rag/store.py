"""Vector storage for Step 2 (§4.3 Qdrant collection ``docs``).

Two backends share one interface so tests/CI run without Docker:

- ``LocalVectorStore`` (default): numpy cosine search + JSONL persistence.
  No server needed. Used by ``scripts/ingest.py --backend local`` and all
  unit tests.
- ``QdrantStore``: real Qdrant (Docker at ``localhost:6333`` or
  ``:memory:``). Requires ``qdrant-client``. Used with
  ``--backend qdrant`` / ``--backend auto`` (falls back to local when
  the server/client is unavailable).

Qdrant layout (created by ``ensure_collection``):
  collection ``docs``, vector ``dense`` size 1024 distance Cosine,
  payload indexes ``doc_id`` keyword + ``type`` keyword.
Payload per point follows §4.3: doc_id, file_name, page, section, type,
image_path, content, content_hash, ingested_at.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

COLLECTION = "docs"
VECTOR_NAME = "dense"
VECTOR_SIZE = 1024


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class LocalVectorStore:
    """In-memory + on-disk dense store. Cosine search, hash dedup."""

    def __init__(self, dim: int = VECTOR_SIZE) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.payloads: list[dict] = []
        self._by_hash: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def ensure_collection(self) -> None:  # parity with QdrantStore
        return None

    def existing_hashes(self) -> set[str]:
        return set(self._by_hash)

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
    ) -> int:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids/vectors/payloads must align")
        inserted = 0
        for pid, vec, pay in zip(ids, vectors, payloads):
            if len(vec) != self.dim:
                raise ValueError(f"vector dim {len(vec)} != {self.dim}")
            content_hash = str(pay.get("content_hash", ""))
            if content_hash and content_hash in self._by_hash:
                continue  # re-ingest dedup per §4.2
            if pid in self.ids:
                idx = self.ids.index(str(pid))
                self.vectors[idx] = list(map(float, vec))
                self.payloads[idx] = dict(pay)
                continue
            self.ids.append(str(pid))
            self.vectors.append(list(map(float, vec)))
            self.payloads.append(dict(pay))
            if content_hash:
                self._by_hash[content_hash] = str(pid)
            inserted += 1
        return inserted

    def search_dense(
        self,
        query_vector: Sequence[float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        scored = []
        for pid, vec, pay in zip(self.ids, self.vectors, self.payloads):
            if filters:
                if filters.get("doc_id") and pay.get("doc_id") != filters["doc_id"]:
                    continue
                if filters.get("type") and pay.get("type") != filters["type"]:
                    continue
            scored.append((pid, _cosine(query_vector, vec), pay))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "payloads.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in self.payloads) + ("\n" if self.payloads else ""),
            encoding="utf-8",
        )
        (directory / "vectors.json").write_text(json.dumps({"ids": self.ids, "vectors": self.vectors}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, dim: int = VECTOR_SIZE) -> "LocalVectorStore":
        store = cls(dim=dim)
        directory = Path(directory)
        payloads_path = directory / "payloads.jsonl"
        vectors_path = directory / "vectors.json"
        if vectors_path.exists():
            data = json.loads(vectors_path.read_text(encoding="utf-8"))
            store.ids = list(data.get("ids", []))
            store.vectors = [list(map(float, v)) for v in data.get("vectors", [])]
        if payloads_path.exists():
            store.payloads = [
                json.loads(line) for line in payloads_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        # Rebuild hash index; ids/vectors/payloads align by position.
        for pid, pay in zip(store.ids, store.payloads):
            h = str(pay.get("content_hash", ""))
            if h:
                store._by_hash[h] = pid
        return store


class QdrantStore:
    """Thin wrapper over qdrant-client. Lazy import so tests skip it."""

    def __init__(
        self,
        url: str | None = None,
        collection: str = COLLECTION,
        vector_size: int = VECTOR_SIZE,
        in_memory: bool = False,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("Install qdrant-client>=1.9 for --backend qdrant") from exc
        if in_memory or not url:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={VECTOR_NAME: VectorParams(size=self.vector_size, distance=Distance.COSINE)},
            )
        for field in ("doc_id", "type"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass  # index already exists or backend ignores duplicates

    def existing_hashes(self) -> set[str]:
        hashes: set[str] = set()
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            for point in batch:
                payload = point.payload or {}
                if payload.get("content_hash"):
                    hashes.add(str(payload["content_hash"]))
            if offset is None:
                break
        return hashes

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
    ) -> int:
        import hashlib

        from qdrant_client.models import PointStruct

        existing = self.existing_hashes()
        points: list[Any] = []
        skipped = 0
        for pid, vec, pay in zip(ids, vectors, payloads):
            h = str(pay.get("content_hash", ""))
            if h and h in existing:
                skipped += 1
                continue
            # Qdrant point ids must be int or UUID; chunk_ids are strings
            # like "{doc}-c0001", so derive a deterministic UUID from them.
            digest = hashlib.md5(str(pid).encode("utf-8")).hexdigest()
            point_id = f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
            points.append(PointStruct(id=point_id, vector={VECTOR_NAME: list(map(float, vec))}, payload=dict(pay)))
        for i in range(0, len(points), 128):
            self.client.upsert(collection_name=self.collection, points=points[i : i + 128])
        return len(points)

    def search_dense(
        self,
        query_vector: Sequence[float],
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter = None
        if filters:
            must = []
            if filters.get("doc_id"):
                must.append(FieldCondition(key="doc_id", match=MatchValue(value=filters["doc_id"])))
            if filters.get("type"):
                must.append(FieldCondition(key="type", match=MatchValue(value=filters["type"])))
            if must:
                qdrant_filter = Filter(must=must)
        # qdrant-client>=1.9 uses query_points; keep back-compat with search().
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection,
                query=list(map(float, query_vector)),
                using=VECTOR_NAME,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            points = result.points
        else:  # pragma: no cover - older client
            points = self.client.search(
                collection_name=self.collection,
                query_vector=(VECTOR_NAME, list(map(float, query_vector))),
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        out = []
        for p in points:
            payload = dict(p.payload or {})
            out.append((str(payload.get("chunk_id", p.id)), float(p.score), payload))
        return out

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count


def get_store(backend: str = "auto", **kwargs) -> LocalVectorStore | QdrantStore:
    """backend: local | qdrant | qdrant-memory | auto (qdrant if reachable else local)."""
    key = backend.strip().lower()
    if key == "local":
        return LocalVectorStore(dim=int(kwargs.get("dim", VECTOR_SIZE)))
    if key in {"qdrant-memory", "memory"}:
        return QdrantStore(in_memory=True, **{k: v for k, v in kwargs.items() if k in {"collection", "vector_size"}})
    if key == "qdrant":
        return QdrantStore(url=kwargs.get("url"), collection=str(kwargs.get("collection", COLLECTION)))
    if key == "auto":
        url = kwargs.get("url")
        try:
            store = QdrantStore(url=url, collection=str(kwargs.get("collection", COLLECTION)))
            store.ensure_collection()
            return store
        except Exception:
            return LocalVectorStore(dim=int(kwargs.get("dim", VECTOR_SIZE)))
    raise ValueError(f"Unknown backend {backend!r}")
