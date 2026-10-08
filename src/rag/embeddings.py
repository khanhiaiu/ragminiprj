"""Validated BGE-M3 dense and lexical-sparse embeddings with caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

from project_settings import setting

from .enrichment.tokenizer import BGE_M3_MODEL, BGE_M3_REVISION, embedding_token_count


@dataclass(frozen=True)
class HybridEmbedding:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    text_fingerprint: str


class BaseEmbedder(Protocol):
    dim: int

    def embed_hybrid(self, texts: Sequence[str]) -> list[HybridEmbedding]: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


def embedding_text_fingerprint(text: str, model_revision: str, config: str) -> str:
    return hashlib.sha256(f"{model_revision}\n{config}\n{text}".encode("utf-8")).hexdigest()


def _validate(item: HybridEmbedding, *, dim: int = setting("embedding.dense_dimensions")) -> None:
    if len(item.dense) != dim:
        raise ValueError(f"dense vector dimension {len(item.dense)} != {dim}")
    if not all(math.isfinite(value) for value in item.dense):
        raise ValueError("dense vector contains non-finite values")
    if len(item.sparse_indices) != len(item.sparse_values):
        raise ValueError("sparse indices and values are misaligned")
    if any(index < 0 for index in item.sparse_indices):
        raise ValueError("sparse indices must be non-negative")
    if len(set(item.sparse_indices)) != len(item.sparse_indices):
        raise ValueError("sparse indices must be unique")
    if not all(math.isfinite(value) and value > 0 for value in item.sparse_values):
        raise ValueError("sparse weights must be finite and positive")


class EmbeddingCache:
    """Atomic, content-addressed per-text cache."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def _path(self, fingerprint: str) -> Path:
        return self.directory / fingerprint[:2] / f"{fingerprint}.json"

    def get(self, fingerprint: str) -> HybridEmbedding | None:
        path = self._path(fingerprint)
        if not path.exists():
            return None
        item = HybridEmbedding(**json.loads(path.read_text(encoding="utf-8")))
        _validate(item, dim=setting("embedding.dense_dimensions"))
        return item

    def put(self, item: HybridEmbedding) -> None:
        _validate(item, dim=setting("embedding.dense_dimensions"))
        path = self._path(item.text_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(item.__dict__, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)


class HashEmbedder:
    """Deterministic dense+sparse stand-in for unit tests only."""

    revision = "hash-v2"

    def __init__(self, dim: int | None = None) -> None:
        dim = dim if dim is not None else setting("embedding.dense_dimensions")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.encoding_config = "hash-dense-sparse-v2"

    def embed_hybrid(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        output = []
        for text in texts:
            dense = [0.0] * self.dim
            sparse: dict[int, float] = {}
            for token in text.lower().split():
                index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dim
                dense[index] += 1.0
                sparse[index] = sparse.get(index, 0.0) + 1.0
            norm = math.sqrt(sum(value * value for value in dense)) or 1.0
            item = HybridEmbedding(
                dense=[value / norm for value in dense],
                sparse_indices=sorted(sparse),
                sparse_values=[sparse[index] for index in sorted(sparse)],
                text_fingerprint=embedding_text_fingerprint(
                    text, self.revision, self.encoding_config
                ),
            )
            _validate(item, dim=self.dim)
            output.append(item)
        return output

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [item.dense for item in self.embed_hybrid(texts)]


class BgeM3Embedder:
    """Single-load FlagEmbedding adapter with dense 1024 + lexical weights."""

    dim = setting("embedding.dense_dimensions")

    def __init__(
        self,
        model_name: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        cache: EmbeddingCache | None = None,
        model: Any | None = None,
    ) -> None:
        model_name = model_name or setting("embedding.model")
        revision = revision or setting("embedding.revision")
        device = device if device is not None else setting("embedding.device")
        batch_size = batch_size if batch_size is not None else setting("embedding.batch_size")
        max_length = max_length if max_length is not None else setting("embedding.max_length")
        self.dim = setting("embedding.dense_dimensions")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = setting("embedding.use_fp16")
        self.cache = cache
        self._model = model
        self.encoding_config = json.dumps(
            {"dense": True, "sparse": True, "colbert": False, "max_length": max_length,
             "model": model_name, "use_fp16": self.use_fp16,
             "input_token_policy": "special_tokens_no_truncation_v1"},
            sort_keys=True,
        )

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "Production BGE-M3 requires FlagEmbedding and huggingface-hub"
                ) from exc
            local_path = snapshot_download(repo_id=self.model_name, revision=self.revision)
            devices = [self.device] if self.device else None
            self._model = BGEM3FlagModel(local_path, devices=devices, use_fp16=self.use_fp16)
        return self._model

    @staticmethod
    def _sparse(value: Any) -> tuple[list[int], list[float]]:
        pairs = sorted((int(index), float(weight)) for index, weight in dict(value).items())
        return [index for index, _ in pairs], [weight for _, weight in pairs]

    def _encode_missing(self, texts: list[str]) -> list[HybridEmbedding]:
        if not texts:
            return []
        model = self._load()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            for text in texts:
                count = embedding_token_count(tokenizer, text)
                if count > self.max_length:
                    raise ValueError(
                        f"embedding input has {count} tokens including special tokens; "
                        f"limit is {self.max_length}; refusing silent truncation"
                    )
        result = model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_values = result.get("dense_vecs")
        lexical_values = result.get("lexical_weights")
        if dense_values is None or lexical_values is None:
            raise ValueError("FlagEmbedding did not return dense_vecs and lexical_weights")
        if len(dense_values) != len(texts) or len(lexical_values) != len(texts):
            raise ValueError("FlagEmbedding output batch is misaligned")
        output = []
        for text, dense, sparse in zip(texts, dense_values, lexical_values):
            indices, values = self._sparse(sparse)
            item = HybridEmbedding(
                dense=[float(value) for value in dense],
                sparse_indices=indices,
                sparse_values=values,
                text_fingerprint=embedding_text_fingerprint(
                    text, self.revision, self.encoding_config
                ),
            )
            _validate(item, dim=self.dim)
            if self.cache:
                self.cache.put(item)
            output.append(item)
        return output

    def embed_hybrid(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        text_list = list(texts)
        output: list[HybridEmbedding | None] = [None] * len(text_list)
        missing_texts: list[str] = []
        missing_positions: list[int] = []
        for index, value in enumerate(text_list):
            fingerprint = embedding_text_fingerprint(value, self.revision, self.encoding_config)
            cached = self.cache.get(fingerprint) if self.cache else None
            if cached is None:
                missing_texts.append(value)
                missing_positions.append(index)
            else:
                output[index] = cached
        for position, item in zip(missing_positions, self._encode_missing(missing_texts)):
            output[position] = item
        return [item for item in output if item is not None]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Backward-compatible dense-only view for existing retriever code."""
        return [item.dense for item in self.embed_hybrid(texts)]

    def stream(
        self, texts: Iterable[str], *, batch_size: int | None = None
    ) -> Iterator[list[HybridEmbedding]]:
        size = batch_size or self.batch_size
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) == size:
                yield self.embed_hybrid(batch)
                batch = []
        if batch:
            yield self.embed_hybrid(batch)


def make_embedder(name: str, **kwargs) -> BaseEmbedder:
    key = name.strip().lower()
    if key in {"hash", "test", "offline"}:
        return HashEmbedder(dim=int(kwargs.get("dim", setting("embedding.dense_dimensions"))))
    if key in {"bge-m3", "bge_m3", "bge", "dense"}:
        cache = kwargs.get("cache")
        if cache is None and kwargs.get("cache_dir"):
            cache = EmbeddingCache(kwargs["cache_dir"])
        return BgeM3Embedder(
            model_name=str(kwargs.get("model_name", setting("embedding.model"))),
            revision=str(kwargs.get("revision", setting("embedding.revision"))),
            device=str(kwargs.get("device", setting("embedding.device"))),
            batch_size=int(kwargs.get("batch_size", setting("embedding.batch_size"))),
            max_length=int(kwargs.get("max_length", setting("embedding.max_length"))),
            cache=cache,
            model=kwargs.get("model"),
        )
    raise ValueError(f"Unknown embedder {name!r}; expected 'hash' or 'bge-m3'")
