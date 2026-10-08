import json
import os
from pathlib import Path

from ..normalization.schema import CanonicalDocument


def document_markdown(document: CanonicalDocument) -> str:
    parts = [f"# {document.filename}"]
    page = None
    sheet = None
    for element in document.elements:
        if element.metadata.get("exclude_from_content"):
            continue
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
        elif element.element_type == "table":
            table = element.metadata.get("table") or {}
            # Markdown cannot preserve merged or nested cells. In that case the
            # structured HTML emitted by TableRecognizer is the lossless form.
            text = table.get("markdown") or table.get("html") or text
        if text:
            parts.append(text)
    return "\n\n".join(parts) + "\n"


def structured_document_dict(document: CanonicalDocument) -> dict:
    """Serialize both the canonical list and page-local structured blocks."""
    payload = document.model_dump(mode="json")
    page_lookup = {page["page_number"]: page for page in payload["pages"]}
    for page in payload["pages"]:
        page["blocks"] = []
    for element in document.elements:
        if element.page_number not in page_lookup:
            continue
        confidence = element.metadata.get("layout_confidence")
        if confidence is None:
            confidence = element.metadata.get("ocr_confidence")
        block = {
            "block_id": element.element_id,
            "type": element.element_type,
            "page": element.page_number,
            "bbox": list(element.bbox) if element.bbox else None,
            "text": element.text,
            "content": element.text,
            "confidence": confidence,
            "reading_order": element.metadata.get("reading_order", element.order),
        }
        if element.element_type == "table":
            table = element.metadata.get("table") or {}
            block.update(
                rows=table.get("rows", 0),
                columns=table.get("columns", 0),
                cells=table.get("cells", []),
                markdown=table.get("markdown", ""),
                html=table.get("html", element.metadata.get("table_html", "")),
            )
            block["content"] = table
        elif element.element_type == "image":
            block["content"] = {"asset_path": element.metadata.get("asset_path")}
        page_lookup[element.page_number]["blocks"].append(block)
    return payload


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_document(document: CanonicalDocument, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "assets").mkdir(exist_ok=True)
    atomic_text(
        directory / "document.json",
        json.dumps(structured_document_dict(document), ensure_ascii=False, indent=2),
    )
    atomic_text(directory / "document.md", document_markdown(document))


def write_summary(summary: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_text(directory / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
