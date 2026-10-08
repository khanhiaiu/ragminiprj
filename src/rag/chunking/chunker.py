"""Text, table and figure chunking over the derived retrieval document."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from project_settings import setting

from document_parser.retrieval.models import RetrievalDocument, RetrievalElement

from rag.enrichment.tokenizer import TokenizerLike, embedding_token_count, tokenizer_fingerprint
from rag.schemas import Chunk, ChunkImage, utc_now_iso


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = field(default_factory=lambda: setting("chunking.target_tokens"))
    preferred_min_tokens: int = field(default_factory=lambda: setting("chunking.preferred_min_tokens"))
    preferred_max_tokens: int = field(default_factory=lambda: setting("chunking.preferred_max_tokens"))
    hard_max_tokens: int = field(default_factory=lambda: setting("chunking.hard_max_tokens"))
    overlap_tokens: int = field(default_factory=lambda: setting("chunking.overlap_tokens"))
    merge_image_captions: bool = field(default_factory=lambda: setting("chunking.merge_image_captions"))

    def __post_init__(self):
        if not (0 <= self.overlap_tokens < self.target_tokens <= self.preferred_max_tokens):
            raise ValueError("invalid text token budgets")
        if self.preferred_max_tokens > self.hard_max_tokens:
            raise ValueError("preferred maximum cannot exceed hard maximum")
        if not 0 <= self.preferred_min_tokens <= self.target_tokens:
            raise ValueError("preferred minimum must be between zero and target")


@dataclass(frozen=True)
class _TextPiece:
    text: str
    element: RetrievalElement


class TypeAwareChunker:
    version = "type-aware-v2"

    def __init__(self, tokenizer: TokenizerLike, config: ChunkingConfig | None = None):
        self.tokenizer = tokenizer
        self.config = config or ChunkingConfig()
        self.tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)

    def _tokens(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

    @staticmethod
    def _embedding_text(body: str, headings: list[str]) -> str:
        prefix = "\n".join(dict.fromkeys(h for h in headings if h))
        if not prefix or body == prefix or body.startswith(prefix + "\n"):
            return body
        return prefix + "\n\n" + body

    def _input_count(self, body: str, headings: list[str]) -> int:
        return embedding_token_count(self.tokenizer, self._embedding_text(body, headings))

    def _text_units(self, text: str, headings: list[str]) -> Iterable[str]:
        """Keep paragraphs/sentences intact; split tokens only for oversized sentences."""
        limit = self.config.target_tokens
        capacity = limit - self._input_count("", headings)
        if capacity <= 0:
            raise ValueError("heading leaves no room for chunk content")
        for paragraph in re.split(r"\n\s*\n", text.strip()):
            if not paragraph.strip():
                continue
            if self._input_count(paragraph, headings) <= limit:
                yield paragraph
                continue
            for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", paragraph):
                if not sentence.strip():
                    continue
                if self._input_count(sentence, headings) <= limit:
                    yield sentence
                    continue
                tokens = self._tokens(sentence)
                while tokens:
                    take = min(capacity, len(tokens))
                    while take and self._input_count(self._decode(tokens[:take]), headings) > limit:
                        take -= 1
                    if not take:
                        raise ValueError("unable to fit content in chunk token budget")
                    yield self._decode(tokens[:take])
                    tokens = tokens[take:]

    @staticmethod
    def _id(document_id: str, kind: str, ordinal: int, source_ids: list[str], text: str) -> str:
        value = json.dumps(
            [document_id, kind, ordinal, source_ids, hashlib.sha256(text.encode()).hexdigest()],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"{document_id}-{kind}-{hashlib.sha256(value.encode()).hexdigest()[:20]}"

    def _build(
        self,
        document: RetrievalDocument,
        kind: str,
        ordinal: int,
        content: str,
        embedding: str,
        elements: list[RetrievalElement],
        *,
        sheet: str | None = None,
        cell_range: str | None = None,
    ) -> Chunk:
        count = embedding_token_count(self.tokenizer, embedding)
        if count > self.config.hard_max_tokens:
            raise ValueError(
                f"chunk embedding input has {count} tokens; hard limit is {self.config.hard_max_tokens}"
            )
        source_ids = list(dict.fromkeys(i for e in elements for i in e.source_element_ids))
        pages = [e.page_number for e in elements if e.page_number is not None]
        headings = elements[-1].heading_path if elements else []
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return Chunk(
            chunk_id=self._id(document.document_id, kind, ordinal, source_ids, embedding),
            doc_id=document.document_id,
            document_version=document.document_version,
            file_name=document.filename,
            page=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            section=elements[-1].section_id if elements else None,
            type=kind,
            content=content,
            text_for_embedding=embedding,
            heading_path=headings,
            sheet=sheet,
            cell_range=cell_range,
            source_element_ids=source_ids,
            source_spans=[span for e in elements for span in e.source_spans],
            image_path=elements[0].asset_path if kind == "figure" and elements else None,
            image_hash=elements[0].image_hash if kind == "figure" and elements else None,
            image=[
                ChunkImage(
                    element_id=e.element_id,
                    path=e.asset_path,
                    image_hash=e.image_hash,
                    page=e.page_number,
                    caption=e.text,
                    visible_text=(e.generated_enrichment or {}).get("visible_text") or [],
                    uncertainties=e.uncertainty_flags,
                )
                for e in elements if e.element_type == "image"
            ],
            uncertainty_flags=[flag for e in elements for flag in e.uncertainty_flags],
            contextual_content=any(e.generated_enrichment is not None for e in elements),
            content_hash=content_hash,
            ingested_at=utc_now_iso(),
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            chunker_version=self.version,
            metadata={
                "embedding_token_count": count,
                "token_count_includes_special_tokens": True,
                "split_image_captions": [
                    e.element_id for e in elements if e.metadata.get("caption_split_for_budget")
                ],
            },
        )

    def _text_chunks(
        self, document: RetrievalDocument, elements: list[RetrievalElement], ordinal: int
    ) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        pieces: list[_TextPiece] = []
        has_new_content = False
        current_path: list[str] | None = None

        def body(values: list[_TextPiece]) -> str:
            # Keep heading provenance, but put structural headings only in the
            # embedding prefix when this chunk also has a body. An orphan
            # heading still needs a chunk of its own, otherwise content is lost.
            content = [p for p in values if not (
                p.element.element_type == "heading"
                and p.text in p.element.heading_path
            )]
            return "\n\n".join(piece.text for piece in (content or values))

        def flush(*, retain_overlap: bool = True):
            nonlocal ordinal, pieces, has_new_content
            if not has_new_content:
                if not retain_overlap:
                    pieces = []
                return
            content = body(pieces)
            grouped = list({p.element.element_id: p.element for p in pieces}.values())
            kind = "figure" if all(e.element_type == "image" for e in grouped) else "text"
            chunks.append(self._build(
                document, kind, ordinal, content,
                self._embedding_text(content, current_path or []), grouped,
            ))
            ordinal += 1
            overlap: list[_TextPiece] = []
            remaining = self.config.overlap_tokens if retain_overlap else 0
            for piece in reversed(pieces):
                # A caption is atomic: never carry a caption fragment as overlap.
                if not remaining or piece.element.element_type in {"image", "heading"}:
                    break
                tokens = self._tokens(piece.text)
                take = min(remaining, len(tokens))
                overlap.insert(0, _TextPiece(self._decode(tokens[-take:]), piece.element))
                remaining -= take
            pieces = overlap
            has_new_content = False

        for element in elements:
            if current_path is not None and element.heading_path != current_path:
                flush(retain_overlap=False)
            current_path = element.heading_path
            is_image = element.element_type == "image"
            if is_image and self._input_count(element.text, current_path) <= self.config.hard_max_tokens:
                units = [element.text]
            else:
                if is_image:
                    element = element.model_copy(update={
                        "metadata": {**element.metadata, "caption_split_for_budget": True}
                    })
                title = (element.generated_enrichment or {}).get("title", "").strip() if is_image else ""
                split_headings = [*current_path, title] if title else current_path
                units = self._text_units(element.text, split_headings)
                if title:
                    units = (text if text.startswith(title) else title + "\n" + text for text in units)
            for text in units:
                piece = _TextPiece(text, element)
                candidate = body([*pieces, piece])
                if pieces and self._input_count(candidate, current_path) > self.config.preferred_max_tokens:
                    flush()
                if pieces and self._input_count(body([*pieces, piece]), current_path) > self.config.preferred_max_tokens:
                    pieces = []  # discard overlap rather than cut a new paragraph/caption
                pieces.append(piece)
                has_new_content = True
                if self._input_count(body(pieces), current_path) >= self.config.target_tokens:
                    flush()
        flush(retain_overlap=False)
        return chunks, ordinal

    def _table_chunks(
        self, document: RetrievalDocument, element: RetrievalElement, ordinal: int
    ) -> tuple[list[Chunk], int]:
        lines = self._table_lines(element)
        separator_indexes = [
            index
            for index, line in enumerate(lines[:4])
            if set(line.replace("|", "").strip()) <= {"-", ":", " "}
        ]
        header_count = separator_indexes[0] + 1 if separator_indexes else 1
        headers, rows = lines[:header_count], lines[header_count:]
        if not rows:
            rows = []
        chunks: list[Chunk] = []
        current = list(headers)
        groups = rows or [""]
        for row in groups:
            candidate = "\n".join([*current, row] if row else current)
            embedding = "\n".join([*element.heading_path, candidate])
            if embedding_token_count(self.tokenizer, embedding) > self.config.hard_max_tokens:
                if len(current) == header_count:
                    raise ValueError(
                        f"table row in {element.element_id} exceeds 768 tokens and cannot be cut safely"
                    )
                body = "\n".join(current)
                chunks.append(
                    self._build(
                        document,
                        "table",
                        ordinal,
                        body,
                        "\n".join([*element.heading_path, body]),
                        [element],
                        sheet=element.metadata.get("sheet"),
                        cell_range=element.metadata.get("range"),
                    )
                )
                ordinal += 1
                current = [*headers, row]
            elif row:
                current.append(row)
        if current:
            body = "\n".join(current)
            chunks.append(
                self._build(
                    document,
                    "table",
                    ordinal,
                    body,
                    "\n".join([*element.heading_path, body]),
                    [element],
                    sheet=element.metadata.get("sheet"),
                    cell_range=element.metadata.get("range"),
                )
            )
            ordinal += 1
        return chunks, ordinal

    @staticmethod
    def _table_lines(element: RetrievalElement) -> list[str]:
        """Prefer structured parser cells when display text is monolithic HTML."""
        structure = element.table_structure or {}
        cells = structure.get("cells")
        if not cells or not element.text.lstrip().lower().startswith("<html"):
            return [line for line in element.text.splitlines() if line.strip()]
        row_count = int(structure.get("rows") or (max(c.get("row_index", 0) for c in cells) + 1))
        column_count = int(
            structure.get("columns") or (max(c.get("column_index", 0) for c in cells) + 1)
        )
        grid = [[""] * column_count for _ in range(row_count)]
        rowspans = set()
        for cell in cells:
            row = int(cell.get("row_index", 0))
            column = int(cell.get("column_index", 0))
            text = " ".join(str(cell.get("text") or "").split()).replace("|", "\\|")
            rowspan = max(1, int(cell.get("rowspan", 1)))
            colspan = max(1, int(cell.get("colspan", 1)))
            if rowspan > 1 or colspan > 1:
                rowspans.add(row)
            for y in range(row, min(row_count, row + rowspan)):
                for x in range(column, min(column_count, column + colspan)):
                    if not grid[y][x]:
                        grid[y][x] = text
        lines = ["| " + " | ".join(row) + " |" for row in grid]
        header_rows = 2 if 0 in rowspans and len(lines) > 1 else 1
        separator = "| " + " | ".join(["---"] * column_count) + " |"
        return [*lines[:header_rows], separator, *lines[header_rows:]]

    def chunk(self, document: RetrievalDocument) -> list[Chunk]:
        result: list[Chunk] = []
        ordinal = 0
        text_buffer: list[RetrievalElement] = []

        def flush_text():
            nonlocal ordinal, text_buffer
            if text_buffer:
                made, ordinal = self._text_chunks(document, text_buffer, ordinal)
                result.extend(made)
                text_buffer = []

        for element in sorted(document.elements, key=lambda item: item.order):
            if element.element_type == "image" and self.config.merge_image_captions:
                text_buffer.append(element)
            elif element.element_type in {"table", "image"}:
                flush_text()
                if element.element_type == "table":
                    made, ordinal = self._table_chunks(document, element, ordinal)
                    result.extend(made)
                else:
                    result.append(
                        self._build(
                            document,
                            "figure",
                            ordinal,
                            element.text,
                            element.text_for_embedding,
                            [element],
                        )
                    )
                    ordinal += 1
            else:
                text_buffer.append(element)
        flush_text()
        return result
