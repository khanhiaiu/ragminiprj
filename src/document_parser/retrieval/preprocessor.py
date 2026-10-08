"""Conservative, non-mutating preparation of canonical elements for retrieval."""

import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..normalization.ocr_postprocessing import clean_text
from ..normalization.schema import CanonicalDocument, Element
from .models import RetrievalDocument, RetrievalElement

_STRUCTURAL_LINE = re.compile(
    r"^\s*(?:CHƯƠNG\s+[IVXLCDM]+|PHẦN\s+\S+|MỤC\s+\S+|"
    r"Điều\s+\d+[A-Za-z]?|Khoản\s+\d+|Điểm\s+[a-zđ]|\d+[.)]|[a-zđ][.)])",
    re.IGNORECASE,
)
_CHAPTER = re.compile(r"^\s*CHƯƠNG\s+[IVXLCDM]+\b", re.IGNORECASE)
_ARTICLE = re.compile(r"^\s*Điều\s+\d+[A-Za-z]?\b", re.IGNORECASE)
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_SAFE_METADATA = {
    "parser",
    "original_label",
    "coordinate_unit",
    "heading_level",
    "marker",
    "enumerated",
    "sheet",
    "range",
    "workbook",
}


def _dehyphenate(match: re.Match) -> str:
    left, right = match.group(1), match.group(2)
    # Uppercase starts more often indicate a true boundary than a wrapped word.
    return left + right if right.islower() else match.group(0)


def clean_retrieval_text(text: str, *, preserve_table: bool = False) -> str:
    """Clean layout noise without changing Vietnamese semantics or legal markers."""
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_table:
        return text.strip()
    text = re.sub(r"([^\W\d_])-\n([^\W\d_])", _dehyphenate, text)
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    output = []
    for line in lines:
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        if not output or output[-1] == "" or _STRUCTURAL_LINE.match(line):
            output.append(line)
        elif _STRUCTURAL_LINE.match(output[-1]):
            output.append(line)
        else:
            output[-1] += " " + line
    cleaned = "\n".join(output).strip()
    return clean_text(cleaned, punctuation=True)


class RetrievalPreprocessor:
    """Build retrieval data without mutating or copying OCR debug payloads."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.excluded_types = set(self.config.get("excluded_types", ["header", "footer"]))

    @staticmethod
    def _retrieval_metadata(element: Element) -> dict[str, Any]:
        return {
            key: deepcopy(value) for key, value in element.metadata.items() if key in _SAFE_METADATA
        }

    @staticmethod
    def _is_upper_heading(text: str) -> bool:
        letters = [character for character in text if character.isalpha()]
        return bool(letters) and all(not character.islower() for character in letters)

    def _update_heading_path(
        self, path: list[tuple[str, str, int | None]], element: Element, text: str
    ) -> list[tuple[str, str, int | None]]:
        level = element.metadata.get("heading_level")
        level = level if isinstance(level, int) and level > 0 else None
        if level is not None:
            path = [entry for entry in path if entry[2] is None or entry[2] < level]
            return [*path, ("heading", text, level)]
        if _CHAPTER.match(text):
            return [("chapter", text, None)]
        if _ARTICLE.match(text):
            path = [entry for entry in path if entry[0] not in {"article", "heading"}]
            return [*path, ("article", text, None)]
        if path and path[0][0] == "chapter" and self._is_upper_heading(text):
            path = [entry for entry in path if entry[0] != "chapter_title"]
            return [*path, ("chapter_title", text, None)]
        return [("heading", text, None)]

    @staticmethod
    def _embedding_text(text: str, headings: list[str]) -> str:
        values = [*headings]
        if not values or values[-1] != text:
            values.append(text)
        return "\n".join(value for value in values if value)

    @staticmethod
    def _caption_for(source: Element, captions: dict[str, Any] | None) -> dict[str, Any] | None:
        if not captions or source.element_id not in captions:
            return None
        value = captions[source.element_id]
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        status = value.get("status")
        if status not in {"completed", "needs_review"}:
            return None
        output = value.get("caption") or {}
        if not output or not output.get("retrieval_useful"):
            return None
        return value

    @staticmethod
    def _figure_text(caption: dict[str, Any], headings: list[str]) -> tuple[str, str]:
        output = caption["caption"]
        visible = output.get("visible_text") or []
        values = [output.get("title", "").strip(), output.get("caption", "").strip()]
        if visible:
            values.append("Văn bản nhìn thấy: " + "; ".join(map(str, visible)))
        for key, label in (
            ("steps", "Các bước"),
            ("relationships", "Các quan hệ"),
            ("chart_details", "Chi tiết biểu đồ"),
        ):
            value = output.get(key)
            if value:
                items = value if isinstance(value, list) else [value]
                values.extend(
                    label + ": " + (
                        item if isinstance(item, str)
                        else json.dumps(item, ensure_ascii=False, separators=(", ", ": "))
                    )
                    for item in items
                )
        # Neighboring context is intentionally absent from embedding text.
        display = "\n".join(value for value in values if value)
        return display, "\n".join([*headings, display])

    def process(
        self, document: CanonicalDocument, captions: dict[str, Any] | None = None,
        *, asset_base: Path | None = None,
    ) -> RetrievalDocument:
        heading_state: list[tuple[str, str, int | None]] = []
        elements = []
        for source in sorted(document.elements, key=lambda element: element.order):
            if source.metadata.get("exclude_from_content") or source.element_type in self.excluded_types:
                continue
            caption = self._caption_for(source, captions) if source.element_type == "image" else None
            if caption and caption.get("document_id", document.document_id) != document.document_id:
                raise ValueError(f"Caption document mismatch for {source.element_id}")
            if source.element_type == "image" and caption is None:
                continue
            text = clean_retrieval_text(source.text, preserve_table=source.element_type == "table")
            if caption is not None:
                text, embedding_text = self._figure_text(
                    caption, [entry[1] for entry in heading_state]
                )
            if not text:
                continue
            if source.element_type == "heading":
                heading_state = self._update_heading_path(heading_state, source, text)
            headings = [entry[1] for entry in heading_state]
            asset_path = source.metadata.get("asset_path")
            if asset_path and asset_base is not None:
                asset_path = str((asset_base / asset_path).resolve())
            elements.append(
                RetrievalElement(
                    element_id=source.element_id,
                    element_type=source.element_type,
                    text=text,
                    text_for_embedding=(
                        embedding_text if caption is not None else self._embedding_text(text, headings)
                    ),
                    document_id=document.document_id,
                    page_number=source.page_number,
                    section_id=source.section_id,
                    parent_id=source.parent_id,
                    order=source.order,
                    bbox=source.bbox,
                    heading_path=headings,
                    source_element_ids=[source.element_id],
                    source_spans=[
                        {
                            "element_id": source.element_id,
                            "page_number": source.page_number,
                            "bbox": source.bbox,
                            "order": source.order,
                            "sheet": source.metadata.get("sheet"),
                            "cell_range": source.metadata.get("range"),
                        }
                    ],
                    asset_path=asset_path,
                    image_hash=caption.get("image_hash") if caption else None,
                    generated_enrichment=caption.get("caption") if caption else None,
                    uncertainty_flags=(
                        list((caption.get("caption") or {}).get("uncertainties") or [])
                        if caption
                        else []
                    ),
                    table_structure=(
                        {
                            key: deepcopy(source.metadata[key])
                            for key in ("cells", "values", "rows", "columns", "range", "sheet")
                            if key in source.metadata
                        }
                        if source.element_type == "table"
                        else None
                    ),
                    metadata=self._retrieval_metadata(source),
                )
            )
        return RetrievalDocument(
            document_id=document.document_id,
            filename=document.filename,
            file_type=document.file_type,
            source_path=document.source_path,
            document_version=str(document.metadata.get("document_version", document.schema_version)),
            elements=elements,
            metadata={"source_parser": document.metadata.get("parser")},
        )
