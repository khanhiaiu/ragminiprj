"""Text, table and figure chunking over the derived retrieval document."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from document_parser.retrieval.models import RetrievalDocument, RetrievalElement

from rag.enrichment.tokenizer import TokenizerLike, tokenizer_fingerprint
from rag.schemas import Chunk, utc_now_iso


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 500
    preferred_min_tokens: int = 400
    preferred_max_tokens: int = 600
    hard_max_tokens: int = 768
    overlap_tokens: int = 50

    def __post_init__(self):
        if not (0 <= self.overlap_tokens < self.target_tokens <= self.preferred_max_tokens):
            raise ValueError("invalid text token budgets")
        if self.preferred_max_tokens > self.hard_max_tokens:
            raise ValueError("preferred maximum cannot exceed hard maximum")


class TypeAwareChunker:
    version = "type-aware-v1"

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
        count = len(self._tokens(embedding))
        if count > self.config.hard_max_tokens:
            raise ValueError(f"chunk embedding input has {count} tokens; hard limit is 768")
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
            uncertainty_flags=[flag for e in elements for flag in e.uncertainty_flags],
            contextual_content=any(e.generated_enrichment is not None for e in elements),
            content_hash=content_hash,
            ingested_at=utc_now_iso(),
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            chunker_version=self.version,
            metadata={"embedding_token_count": count},
        )

    def _text_chunks(
        self, document: RetrievalDocument, elements: list[RetrievalElement], ordinal: int
    ) -> tuple[list[Chunk], int]:
        chunks: list[Chunk] = []
        grouped: list[RetrievalElement] = []
        group_tokens: list[int] = []

        def flush():
            nonlocal ordinal, grouped, group_tokens
            if not group_tokens:
                return
            content = self._decode(group_tokens)
            chunks.append(self._build(document, "text", ordinal, content, content, grouped))
            ordinal += 1
            tail = group_tokens[-self.config.overlap_tokens :] if self.config.overlap_tokens else []
            # The overlap belongs to the same source elements and section only.
            grouped = grouped[-1:] if tail else []
            group_tokens = list(tail)

        current_path = None
        for element in elements:
            tokens = self._tokens(element.text_for_embedding)
            if current_path is not None and element.heading_path != current_path and group_tokens:
                flush()
                grouped, group_tokens = [], []  # no overlap across section boundaries
            current_path = element.heading_path
            start = 0
            while start < len(tokens):
                room = self.config.target_tokens - len(group_tokens)
                take = min(room, len(tokens) - start)
                group_tokens.extend(tokens[start : start + take])
                if element not in grouped:
                    grouped.append(element)
                start += take
                if len(group_tokens) >= self.config.target_tokens:
                    flush()
            if len(group_tokens) >= self.config.preferred_min_tokens:
                flush()
        flush()
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
            if len(self._tokens(embedding)) > self.config.hard_max_tokens:
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
            if element.element_type in {"table", "image"}:
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
