"""Dense embeddings for Step 2: BM25 + bge-m3 semantic search.

Spec §4.3 uses bge-m3 dense 1024 + sparse. Per user decision this step
implements **BM25 (local, lexical) + bge-m3 dense (semantic)**; the dense
side is exactly bge-m3 so a later sparse-vector upgrade stays compatible.

Design for 13GB CPU, no GPU, tests without network:
- ``BaseEmbedder`` protocol: ``dim`` + ``embed_texts`` (L2-normalized).
- ``HashEmbedder``: deterministic, no downloads — default for tests and
  ``scripts/ingest.py --embedder hash``.
- ``BgeM3Embedder``: lazy ``sentence-transformers`` load of
  ``BAAI/bge-m3`` on CPU, batch 8, normalized. Import errors are raised
  only when this embedder is actually constructed.
- ``make_embedder("hash" | "bge-m3")`` factory.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence


class BaseEmbedder(Protocol):
    dim: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic stand-in embedding (no model download).

    Uses SHA256 word-hashing into ``dim`` buckets + L2 normalization so
    cosine search behaves sanely in tests. NOT for production retrieval.
    """

    def __init__(self, dim: int = 1024) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class BgeM3Embedder:
    """bge-m3 dense embeddings on CPU via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 8,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = 1024
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install the rag embed extra: pip install 'sentence-transformers>=2.7' "
                    "(pulls torch CPU, ~2GB; not needed for --embedder hash)"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, row)) for row in vectors]


def make_embedder(name: str, **kwargs) -> BaseEmbedder:
    key = name.strip().lower()
    if key in {"hash", "test", "offline"}:
        return HashEmbedder(dim=int(kwargs.get("dim", 1024)))
    if key in {"bge-m3", "bge_m3", "bge", "dense"}:
        return BgeM3Embedder(
            model_name=str(kwargs.get("model_name", "BAAI/bge-m3")),
            device=str(kwargs.get("device", "cpu")),
            batch_size=int(kwargs.get("batch_size", 8)),
        )
    raise ValueError(f"Unknown embedder {name!r}; expected 'hash' or 'bge-m3'")
