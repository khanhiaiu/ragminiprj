"""Generation index orchestration with a publish-after-verification gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .embeddings import HybridEmbedding
from .store import QdrantStore


@dataclass
class VerificationResult:
    expected_points: int
    actual_points: int
    dense_results: int
    sparse_results: int
    hybrid_results: int
    passed: bool
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class GenerationIndexer:
    def __init__(self, store: QdrantStore):
        self.store = store

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[HybridEmbedding],
        payloads: Sequence[dict],
    ) -> int:
        return self.store.upsert_hybrid(
            ids,
            [item.dense for item in embeddings],
            [item.sparse_indices for item in embeddings],
            [item.sparse_values for item in embeddings],
            payloads,
        )

    def verify(
        self,
        expected: int,
        query: HybridEmbedding,
        *,
        filters: dict | None = None,
    ) -> VerificationResult:
        errors = []
        actual = int(self.store.count())
        if actual != expected:
            errors.append(f"point_count:{actual}!={expected}")
        dense = self.store.search_dense(query.dense, top_k=3, filters=filters)
        sparse = self.store.search_sparse(
            query.sparse_indices, query.sparse_values, top_k=3, filters=filters
        )
        hybrid = self.store.search_hybrid(
            query.dense,
            query.sparse_indices,
            query.sparse_values,
            top_k=3,
            filters=filters,
        )
        if expected and not dense:
            errors.append("dense_smoke_empty")
        if expected and not sparse:
            errors.append("sparse_smoke_empty")
        if expected and not hybrid:
            errors.append("hybrid_smoke_empty")
        return VerificationResult(
            expected_points=expected,
            actual_points=actual,
            dense_results=len(dense),
            sparse_results=len(sparse),
            hybrid_results=len(hybrid),
            passed=not errors,
            errors=errors,
        )

    def publish(self, verification: VerificationResult) -> None:
        if not verification.passed:
            raise RuntimeError("refusing to publish an incomplete or unverified collection")
        self.store.publish_alias()
