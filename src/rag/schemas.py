"""Chunk schema for SystemDocuments §4.2 / §4.3.

Contract agreed with Chunk owner (Step 1):
- Input: ``CanonicalDocument`` (``document_parser``) or Elements JSON
  ``{doc_id, type: text|table|figure, page, section, content, image_path}``.
- Output: list of ``Chunk`` below. Step 2 (this task) only consumes it;
  ``chunk_contract.fallback_chunks_from_canonical`` is a temporary
  stand-in until the full heading-aware chunker lands. It implements the
  same return type so the Step-1 owner can swap it out without touching
  embed/store code.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkType = Literal["text", "table", "figure"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Chunk(BaseModel):
    """Single retrievable unit. One table / one figure caption = one chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Stable id: {doc_id}-c{index:04d}")
    doc_id: str
    document_version: str = "1"
    file_name: str = ""
    page: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section: str | None = None
    type: ChunkType = "text"
    content: str = Field(min_length=1)
    text_for_embedding: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    sheet: str | None = None
    cell_range: str | None = None
    source_element_ids: list[str] = Field(default_factory=list)
    source_spans: list[dict[str, Any]] = Field(default_factory=list)
    image_path: str | None = None
    image_hash: str | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)
    contextual_content: bool = False
    content_hash: str = Field(description="sha256(content) hex, used for re-ingest dedup")
    embedding_fingerprint: str | None = None
    tokenizer_fingerprint: str | None = None
    chunker_version: str = "legacy-adapter-v1"
    ingested_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)
