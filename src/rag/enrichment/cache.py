"""Content-addressed caption cache with atomic records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .caption import CaptionRecord


def caption_cache_key(
    *,
    image_hash: str,
    provider: str,
    model: str,
    prompt_version: str,
    context_hash: str,
    preprocessing_version: str,
) -> str:
    values = [image_hash, provider, model, prompt_version, context_hash, preprocessing_version]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


class CaptionCache:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> CaptionRecord | None:
        path = self._path(key)
        if not path.exists():
            return None
        return CaptionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, record: CaptionRecord) -> None:
        path = self._path(record.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
