"""Retrieval-facing schemas that remain independent of embedding providers."""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class RetrievalElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str
    element_type: str
    text: str
    text_for_embedding: str
    document_id: str
    page_number: int | None = None
    section_id: str | None = None
    parent_id: str | None = None
    order: int = Field(ge=0)
    bbox: tuple[float, float, float, float] | None = None
    heading_path: list[str] = Field(default_factory=list)
    source_element_ids: list[str] = Field(default_factory=list)
    source_spans: list[dict[str, Any]] = Field(default_factory=list)
    asset_path: str | None = None
    image_hash: str | None = None
    generated_enrichment: dict[str, Any] | None = None
    uncertainty_flags: list[str] = Field(default_factory=list)
    table_structure: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    document_id: str
    filename: str
    file_type: str
    source_path: str
    document_version: str = "1"
    elements: list[RetrievalElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    text_for_embedding: str
    source_element_ids: list[str]
    page_start: int | None = None
    page_end: int | None = None
    section_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseChunker(Protocol):
    def chunk(self, document: RetrievalDocument) -> list[Chunk]: ...
