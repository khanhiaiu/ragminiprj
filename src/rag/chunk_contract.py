"""Chunk contract + temporary fallback chunker.

The Step-1 owner implements the full §4.2 chunker:
  text: split by heading, ~400-600 tokens, ~10% overlap
  table: one table = one chunk + parent title, long tables split by
         row-groups repeating the header + LLM summary
  figure: VLM caption = one chunk
  every chunk carries content_hash; re-ingest skips duplicate hashes.

Until that lands, :func:`chunk_documents` below provides a minimal
spec-compatible fallback so Step 2 (embed/store) can be built and tested.
It follows the same I/O type (CanonicalDocument/dict -> list[Chunk])
so replacement is drop-in: keep function name + Chunk schema.

Fallback rules (kept intentionally simple):
- headings/paragraphs/lists concatenated up to ~2000 chars (~500 tokens)
  with ~200 chars overlap; section tracked from last heading.
- each table element = one chunk (markdown kept verbatim).
- each image element = one figure chunk; content = VLM caption if present
  in metadata (``vlm_caption``/``caption``) else ``[FIGURE: <asset>]`` stub
  plus any existing text. ``image_path`` preserved from ``asset_path``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .schemas import Chunk, utc_now_iso

# ~500 tokens ≈ 2000 chars for Vietnamese; overlap ~10%.
FALLBACK_MAX_CHARS = 2000
FALLBACK_OVERLAP_CHARS = 200

_TEXT_TYPES = {"heading", "paragraph", "list", "header", "footer", "formula", "unknown", "text"}
_TABLE_TYPES = {"table"}
_FIGURE_TYPES = {"image", "figure", "picture", "chart", "seal"}


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _as_elements(doc: Any) -> tuple[str, str, list[dict]]:
    """Return (doc_id, file_name, elements-as-dicts) from CanonicalDocument or dict."""
    if hasattr(doc, "model_dump"):
        data = doc.model_dump()
    else:
        data = dict(doc)
    doc_id = str(data.get("document_id") or data.get("doc_id") or "unknown")
    file_name = str(data.get("filename") or data.get("file_name") or "")
    raw_elements = data.get("elements", [])
    elements: list[dict] = []
    for item in raw_elements:
        elements.append(item.model_dump() if hasattr(item, "model_dump") else dict(item))
    # Canonical order is already set by Normalizer; keep stable fallback sort.
    elements.sort(key=lambda e: (e.get("order", 0)))
    return doc_id, file_name, elements


def _element_section(element: dict, current: str | None) -> str | None:
    if element.get("section_id"):
        return str(element["section_id"])
    label = str(element.get("element_type", "")).lower()
    if label == "heading" and element.get("text"):
        slug = re.sub(r"[^\w-]+", "-", element["text"].lower()).strip("-")[:80]
        return slug or current
    return current


def _image_path_of(element: dict) -> str | None:
    meta = element.get("metadata") or {}
    for key in ("asset_path", "image_path", "imagePath"):
        if meta.get(key):
            return str(meta[key])
    return None


def _figure_content(element: dict) -> str:
    meta = element.get("metadata") or {}
    for key in ("vlm_caption", "caption", "description"):
        if meta.get(key):
            return str(meta[key])
    text = str(element.get("text") or "").strip()
    asset = _image_path_of(element)
    if text:
        return text
    return f"[FIGURE: {asset or 'embedded image; caption pending'}]"


def chunk_documents(
    doc: Any,
    *,
    max_chars: int = FALLBACK_MAX_CHARS,
    overlap_chars: int = FALLBACK_OVERLAP_CHARS,
    ingested_at: str | None = None,
) -> list[Chunk]:
    """Temporary fallback chunker. See module docstring.

    The Step-1 owner may replace the body with the full §4.2 implementation
    as long as the signature and :class:`Chunk` return type are kept.
    """
    doc_id, file_name, elements = _as_elements(doc)
    stamped = ingested_at or utc_now_iso()
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    buf_page: int | None = None
    buf_section: str | None = None
    section: str | None = None
    index = 0

    def flush_text() -> None:
        nonlocal buf, buf_len, buf_page, buf_section, index
        if not buf:
            return
        content = "\n\n".join(buf).strip()
        if content:
            index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-c{index:04d}",
                    doc_id=doc_id,
                    file_name=file_name,
                    page=buf_page,
                    section=buf_section,
                    type="text",
                    content=content,
                    image_path=None,
                    content_hash=_content_hash(content),
                    ingested_at=stamped,
                    metadata={"chunker": "fallback-v1"},
                )
            )
        # Overlap: keep tail of previous buffer.
        tail = content[-overlap_chars:] if overlap_chars > 0 else ""
        buf = [tail] if tail else []
        buf_len = len(tail)
        buf_page = None
        buf_section = section

    for element in elements:
        label = str(element.get("element_type", "unknown")).lower()
        section = _element_section(element, section)
        page = element.get("page_number")
        page_no = int(page) if isinstance(page, int) and page >= 1 else None
        text = str(element.get("text") or "").strip()

        if label in _TABLE_TYPES:
            flush_text()
            buf = []
            buf_len = 0
            content = text or "(empty table)"
            # Keep parent/section title with the table per §4.2.
            if section:
                content = f"[{section}]\n{content}"
            index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-c{index:04d}",
                    doc_id=doc_id,
                    file_name=file_name,
                    page=page_no,
                    section=section,
                    type="table",
                    content=content,
                    image_path=None,
                    content_hash=_content_hash(content),
                    ingested_at=stamped,
                    metadata={"chunker": "fallback-v1", "element_id": element.get("element_id")},
                )
            )
            continue
        if label in _FIGURE_TYPES:
            flush_text()
            buf = []
            buf_len = 0
            content = _figure_content(element)
            if section and not content.startswith("["):
                content = f"[{section}] {content}"
            index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-c{index:04d}",
                    doc_id=doc_id,
                    file_name=file_name,
                    page=page_no,
                    section=section,
                    type="figure",
                    content=content,
                    image_path=_image_path_of(element),
                    content_hash=_content_hash(content),
                    ingested_at=stamped,
                    metadata={"chunker": "fallback-v1", "element_id": element.get("element_id")},
                )
            )
            continue
        if label in _TEXT_TYPES:
            if not text:
                continue
            if buf_page is None:
                buf_page = page_no
                buf_section = section
            # Start a new chunk when the buffer would exceed max_chars.
            if buf_len + len(text) + 2 > max_chars and buf:
                flush_text()
                if buf_page is None:
                    buf_page = page_no
                    buf_section = section
            buf.append(text)
            buf_len += len(text) + 2
            continue
        # Unknown labels: keep non-empty text as text chunk.
        if text:
            buf.append(text)
            buf_len += len(text) + 2
            if buf_page is None:
                buf_page = page_no
                buf_section = section
    flush_text()
    return chunks


# Backwards-compatible alias for the Step-1 owner / ingest script.
fallback_chunks_from_canonical = chunk_documents
