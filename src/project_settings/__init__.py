"""Shared runtime settings. CLI arguments > environment > config.yaml.

Paths in YAML are relative to that file, independent of the working directory.
The wheel installs the same source YAML under share/rag-document-parser.
"""

from __future__ import annotations

import argparse
import math
import os
import sysconfig
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_SOURCE_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
DEFAULT_CONFIG = (
    _SOURCE_CONFIG if _SOURCE_CONFIG.is_file()
    else Path(sysconfig.get_path("data")) / "share/rag-document-parser/config.yaml"
)
_active_path: Path | None = None
_SECTIONS = {
    "device", "models", "paths", "ocr", "pdf", "max_excel_region_cells",
    "environment", "recognition_download", "postprocessing", "text_correction",
    "ingestion", "tokenizer", "chunking", "embedding", "image_context", "captioning",
    "qdrant", "retrieval", "llm", "memory", "prompt", "guardrails", "api", "ui",
    "pdf_conversion",
}


def config_path() -> Path:
    return (_active_path or Path(os.environ.get("PROJECT_CONFIG", DEFAULT_CONFIG))).expanduser().resolve()


def _read(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read project config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    if data.keys() - _SECTIONS:
        raise ValueError(f"Unknown config fields: {sorted(data.keys() - _SECTIONS)}")
    return data


def _merge(base: dict, overrides: dict, prefix: str = "") -> dict:
    result = deepcopy(base)
    # These maps intentionally accept backend options, vocabulary and env names.
    open_maps = {"ocr.options", "environment", "text_correction.models"}
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key not in base and prefix not in open_maps:
            raise ValueError(f"Unknown config field: {name}")
        original = base.get(key)
        if isinstance(original, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            result[key] = _merge(original, value, name)
        else:
            if original is not None and value is not None:
                valid = isinstance(value, type(original))
                if isinstance(original, float):
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
                elif isinstance(original, int) and not isinstance(original, bool):
                    valid = isinstance(value, int) and not isinstance(value, bool)
                if not valid:
                    raise ValueError(f"Invalid type for {name}: expected {type(original).__name__}")
            if value is None and original is not None and name != "retrieval.relevance_threshold":
                raise ValueError(f"{name} cannot be null")
            result[key] = deepcopy(value)
    return result


@lru_cache(maxsize=8)
def _load(path: Path, modified: int, default_modified: int) -> dict[str, Any]:
    defaults = _read(DEFAULT_CONFIG)
    # Validate the primary file against itself; component constructors validate
    # model-specific constraints. Overrides additionally reject misspelled keys.
    data = _merge(defaults, _read(path))
    _validate(data)
    return data


def _validate(data: dict) -> None:
    for section in _SECTIONS - {"device", "max_excel_region_cells"}:
        if not isinstance(data.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    positive = {
        "embedding": ["batch_size", "max_length", "dense_dimensions"],
        "retrieval": ["dense_top", "sparse_top", "candidate_top", "top_k", "local_rrf_k",
                      "reranker_batch_size", "reranker_max_length"],
        "llm": ["timeout_seconds", "max_tokens"],
        "memory": ["max_turns", "max_chars", "summary_chars", "summary_snippet_chars",
                   "summary_max_tokens", "rewrite_turns", "rewrite_turn_chars", "rewrite_max_tokens"],
        "prompt": ["history_turns", "history_turn_chars", "summary_chars"],
        "guardrails": ["max_question_chars", "max_answer_chars"],
        "api": ["port", "max_top_k", "chunk_preview_chars"],
        "ui": ["query_timeout_seconds", "status_timeout_seconds"],
        "qdrant": ["timeout_seconds"],
        "captioning": ["timeout_seconds", "maximum_image_dimension"],
        "pdf_conversion": ["timeout_seconds", "poll_interval_seconds", "max_output_tokens"],
    }
    for section, names in positive.items():
        for name in names:
            value = data[section].get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} must be finite and positive")
            if not name.endswith("_seconds") and not isinstance(value, int):
                raise ValueError(f"{section}.{name} must be an integer")
    threshold = data["retrieval"]["relevance_threshold"]
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not math.isfinite(threshold) or threshold < 0):
        raise ValueError("retrieval.relevance_threshold must be finite and non-negative, or null")
    if data["retrieval"]["top_k"] > data["api"]["max_top_k"]:
        raise ValueError("retrieval.top_k cannot exceed api.max_top_k")
    if data["retrieval"]["top_k"] > data["retrieval"]["candidate_top"]:
        raise ValueError("retrieval.top_k cannot exceed retrieval.candidate_top")
    if data["api"]["port"] > 65535:
        raise ValueError("api.port must be between 1 and 65535")
    if data["embedding"]["max_length"] < data["chunking"]["hard_max_tokens"]:
        raise ValueError("embedding.max_length cannot be less than chunking.hard_max_tokens")
    if data["embedding"]["backend"] == "bge-m3" and data["embedding"]["dense_dimensions"] != 1024:
        raise ValueError("BGE-M3 dense_dimensions must be 1024")
    if (data["tokenizer"]["model"], data["tokenizer"]["revision"]) != (data["embedding"]["model"], data["embedding"]["revision"]):
        raise ValueError("tokenizer and embedding must use the same model and revision")
    for section, field in [("captioning", "minimum_request_interval_seconds"), ("pdf_conversion", "request_interval_seconds")]:
        if data[section][field] < 4:
            raise ValueError(f"{section}.{field} must be at least 4 seconds")
    for section, field in [("captioning", "maximum_retries"), ("pdf_conversion", "max_retries")]:
        value = data[section][field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{section}.{field} must be a non-negative integer")


def secret(name: str) -> str | None:
    """Read a credential from the environment or an adjacent .env, without mutation."""
    if os.environ.get(name):
        return os.environ[name]
    for path in dict.fromkeys([config_path().parent / ".env", DEFAULT_CONFIG.parent / ".env"]):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip().removeprefix("export ")
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                value = value.strip().strip("\"'")
                if value:
                    return value
    return None


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve() if path is not None else config_path()
    try:
        data = _load(selected, selected.stat().st_mtime_ns, DEFAULT_CONFIG.stat().st_mtime_ns)
    except OSError as exc:
        raise ValueError(f"Cannot read project config {selected}: {exc}") from exc
    return deepcopy(data)


def setting(name: str, env: str | None = None) -> Any:
    value: Any = load_config()
    for part in name.split("."):
        value = value[part]
    if env and env in os.environ:
        raw = os.environ[env]
        if isinstance(value, bool):
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise ValueError(f"{env} must be true or false")
            return raw.lower() in {"true", "1"}
        if isinstance(value, int):
            return int(raw)
        if isinstance(value, float):
            return float(raw)
        return raw
    return value


def configured_path(name: str, env: str | None = None) -> Path:
    return (config_path().parent / Path(setting(name, env)).expanduser()).resolve()


def configure_cli(parser: argparse.ArgumentParser) -> None:
    """Select YAML before the caller adds arguments with configured defaults."""
    global _active_path
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    args, _ = preliminary.parse_known_args()
    _active_path = args.config.expanduser().resolve() if args.config else None
    try:
        load_config()
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument("--config", type=Path, default=config_path(), help="Shared config.yaml")
