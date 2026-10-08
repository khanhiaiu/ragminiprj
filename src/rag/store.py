"""Vector storage for Step 2 (§4.3 Qdrant collection ``docs``).

Two backends share one interface so tests/CI run without Docker:

- ``LocalVectorStore`` (default): numpy cosine search + JSONL persistence.
  No server needed. Used by ``scripts/ingest.py --backend local`` and all
  unit tests.
- ``QdrantStore``: real Qdrant (local Docker, self-hosted with TLS +
  API key, or Qdrant Cloud). Requires ``qdrant-client``. Used with
  ``--backend qdrant`` / ``--backend auto`` (falls back to local when
  the server/client is unavailable). Connection settings resolve in
  order: explicit argument → ``QDRANT_URL`` / ``QDRANT_API_KEY`` env →
  repo-root ``.env`` file (also accepts ``CLUSTER_URL`` and
  ``QDRANT__SERVICE__API_KEY`` aliases). REST transport is forced
  (``prefer_grpc=False``) so Qdrant Cloud works without the gRPC port.

Qdrant layout (created by ``ensure_collection``):
  collection ``docs``, vector ``dense`` size 1024 distance Cosine,
  payload indexes ``doc_id`` keyword + ``type`` keyword.
Payload per point follows §4.3: doc_id, file_name, page, section, type,
image_path, content, content_hash, ingested_at.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

COLLECTION = "docs"
CURRENT_ALIAS = "docs_current"
VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE = 1024

# Env / .env keys accepted for Qdrant connections (explicit args win).
_URL_KEYS = ("QDRANT_URL", "CLUSTER_URL")
_API_KEY_KEYS = ("QDRANT_API_KEY", "QDRANT__SERVICE__API_KEY")


def _load_dotenv_file() -> None:
    """Fill missing Qdrant settings from the repo-root ``.env`` (no dependency).

    Accepts both ``KEY=value`` and ``KEY: value`` lines, strips quotes, and
    never overrides real environment variables. Only the keys in
    ``_URL_KEYS`` / ``_API_KEY_KEYS`` are imported.
    """
    wanted = set(_URL_KEYS) | set(_API_KEY_KEYS)
    if all(os.environ.get(k) for k in ("QDRANT_URL", "QDRANT_API_KEY")):
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if not sep:
            key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key not in wanted or os.environ.get(key):
            continue
        value = value.strip().strip("\"").strip("'")
        if value:
            os.environ[key] = value


def resolve_qdrant_settings(
    url: str | None = None, api_key: str | None = None
) -> tuple[str | None, str | None]:
    """Resolve Qdrant URL / API key: arg → env → .env aliases."""
    _load_dotenv_file()
    if not url:
        url = os.environ.get("QDRANT_URL") or os.environ.get("CLUSTER_URL")
    if not api_key:
        api_key = os.environ.get("QDRANT_API_KEY") or os.environ.get("QDRANT__SERVICE__API_KEY")
    return url, api_key


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
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            self.collection = collection
            self.vector_size = vector_size
            return
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("Install qdrant-client>=1.9 for --backend qdrant") from exc
        url, api_key = resolve_qdrant_settings(url, api_key)
        if in_memory or not url:
            self.client = QdrantClient(":memory:")
        else:
            # REST only: Qdrant Cloud free tier blocks the gRPC port.
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=30)
        self.collection = collection
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        from qdrant_client.models import (
            Distance,
            PayloadSchemaType,
            SparseVectorParams,
            VectorParams,
        )

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={VECTOR_NAME: VectorParams(size=self.vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
            )
        for field in ("doc_id", "document_version", "type", "page", "sheet", "image_hash"):
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

    @staticmethod
    def deterministic_point_id(chunk_id: str) -> str:
        import hashlib

        digest = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
        return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

    def upsert_hybrid(
        self,
        ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_indices: Sequence[Sequence[int]],
        sparse_values: Sequence[Sequence[float]],
        payloads: Sequence[dict],
        *,
        batch_size: int = 128,
    ) -> int:
        """Upsert one deterministic Qdrant point per chunk with both named vectors."""
        from qdrant_client.models import PointStruct, SparseVector

        lengths = {len(ids), len(dense_vectors), len(sparse_indices), len(sparse_values), len(payloads)}
        if len(lengths) != 1:
            raise ValueError("hybrid vectors and payloads must align")
        points = []
        for chunk_id, dense, indices, values, payload in zip(
            ids, dense_vectors, sparse_indices, sparse_values, payloads
        ):
            if len(dense) != self.vector_size:
                raise ValueError(f"dense vector dim {len(dense)} != {self.vector_size}")
            if len(indices) != len(values):
                raise ValueError("sparse indices and values must align")
            enriched = dict(payload)
            enriched["chunk_id"] = chunk_id
            points.append(
                PointStruct(
                    id=self.deterministic_point_id(str(chunk_id)),
                    vector={
                        VECTOR_NAME: list(map(float, dense)),
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=list(map(int, indices)), values=list(map(float, values))
                        ),
                    },
                    payload=enriched,
                )
            )
        for index in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[index : index + batch_size],
                wait=True,
            )
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

    def search_sparse(
        self,
        indices: Sequence[int],
        values: Sequence[float],
        *,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        from qdrant_client.models import SparseVector

        result = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(indices=list(indices), values=list(values)),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=self._filter(filters),
            with_payload=True,
        )
        return self._points(result.points)

    def search_hybrid(
        self,
        dense: Sequence[float],
        sparse_indices: Sequence[int],
        sparse_values: Sequence[float],
        *,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        query_filter = self._filter(filters)
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=list(map(float, dense)), using=VECTOR_NAME, limit=max(top_k, 20)),
                Prefetch(
                    query=SparseVector(
                        indices=list(map(int, sparse_indices)),
                        values=list(map(float, sparse_values)),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=max(top_k, 20),
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return self._points(result.points)

    @staticmethod
    def _points(points) -> list[tuple[str, float, dict]]:
        output = []
        for point in points:
            payload = dict(point.payload or {})
            output.append((str(payload.get("chunk_id", point.id)), float(point.score), payload))
        return output

    @staticmethod
    def _filter(filters: dict | None):
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
            if value is not None
        ]
        return Filter(must=conditions) if conditions else None

    def publish_alias(self, alias: str = CURRENT_ALIAS) -> None:
        """Atomically point the stable alias at this verified generation."""
        from qdrant_client.models import (
            CreateAlias,
            CreateAliasOperation,
            DeleteAlias,
            DeleteAliasOperation,
        )

        aliases = {item.alias_name for item in self.client.get_aliases().aliases}
        operations = []
        if alias in aliases:
            operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
        operations.append(
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=self.collection, alias_name=alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)

    def verify_count(self, expected: int) -> dict[str, Any]:
        actual = int(self.count())
        return {"expected_points": expected, "actual_points": actual, "passed": actual == expected}


def get_store(backend: str = "qdrant", **kwargs) -> LocalVectorStore | QdrantStore:
    """Select a store explicitly. Production failures never fall back silently."""
    key = backend.strip().lower()
    if key == "local":
        return LocalVectorStore(dim=int(kwargs.get("dim", VECTOR_SIZE)))
    if key in {"qdrant-memory", "memory"}:
        return QdrantStore(in_memory=True, **{k: v for k, v in kwargs.items() if k in {"collection", "vector_size"}})
    if key == "qdrant":
        return QdrantStore(
            url=kwargs.get("url"), collection=str(kwargs.get("collection", COLLECTION)),
            api_key=kwargs.get("api_key"),
        )
    if key == "auto":
        url = kwargs.get("url")
        try:
            store = QdrantStore(
                url=url, collection=str(kwargs.get("collection", COLLECTION)),
                api_key=kwargs.get("api_key"),
            )
            store.ensure_collection()
            return store
        except Exception as exc:
            raise RuntimeError("Qdrant auto-discovery failed; refusing local fallback") from exc
    raise ValueError(f"Unknown backend {backend!r}")
