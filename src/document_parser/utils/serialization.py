import json
import os
from pathlib import Path

from ..normalization.schema import CanonicalDocument


def document_markdown(document: CanonicalDocument) -> str:
    parts = [f"# {document.filename}"]
    page = None
    sheet = None
    for element in document.elements:
        if element.page_number is not None and element.page_number != page:
            page = element.page_number
            parts.append(f"<!-- Page {page} -->")
        if element.metadata.get("sheet") and element.metadata["sheet"] != sheet:
            sheet = element.metadata["sheet"]
            parts.append(f"## Sheet: {sheet}")
        text = element.text
        if element.element_type == "heading":
            level = max(1, min(6, int(element.metadata.get("heading_level", 2))))
            text = f"{'#' * level} {text}"
        elif element.element_type == "image":
            asset = element.metadata.get("asset_path")
            text = f"[IMAGE: {asset or 'embedded image; see JSON'}]"
        elif element.element_type == "list":
            text = f"- {text}"
        if text:
            parts.append(text)
    return "\n\n".join(parts) + "\n"


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_document(document: CanonicalDocument, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "assets").mkdir(exist_ok=True)
    atomic_text(directory / "document.json", document.model_dump_json(indent=2))
    atomic_text(directory / "document.md", document_markdown(document))


def write_summary(summary: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_text(directory / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
