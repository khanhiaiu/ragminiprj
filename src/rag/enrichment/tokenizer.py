"""Pinned tokenizer access shared by context extraction and chunking."""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class TokenizerLike(Protocol):
    name_or_path: str

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


def embedding_token_count(tokenizer: TokenizerLike, text: str) -> int:
    """Count the complete model input, including BOS/EOS special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def load_bge_m3_tokenizer(
    model_name: str = BGE_M3_MODEL,
    revision: str = BGE_M3_REVISION,
    *,
    local_files_only: bool = False,
):
    """Load the exact tokenizer revision. Never fall back to another tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("transformers is required for the pinned BGE-M3 tokenizer") from exc
    return AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        use_fast=True,
        local_files_only=local_files_only,
    )


def tokenizer_fingerprint(
    tokenizer: TokenizerLike,
    *,
    model_name: str = BGE_M3_MODEL,
    revision: str = BGE_M3_REVISION,
) -> str:
    payload = f"{model_name}\n{revision}\n{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
