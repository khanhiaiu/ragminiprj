"""Retrieved source metadata and image paths confined to the configured asset root."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from project_settings import configured_path, setting


IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def serialize_sources(chunks) -> list[dict]:
    return [{"chunk_id": c.chunk_id, "citation": c.citation, "fused_score": c.fused_score,
             "dense_score": c.dense_score, "rerank_score": c.rerank_score,
             "doc_id": c.payload.get("doc_id"), "file_name": c.payload.get("file_name"),
             "type": c.payload.get("type"), "page": c.payload.get("page"),
             "image": c.payload.get("image", []), "image_path": c.payload.get("image_path"),
             "image_hash": c.payload.get("image_hash"),
             "content": str(c.payload.get("content", ""))[:setting("api.chunk_preview_chars")]}
            for c in chunks]


def source_images(source: dict) -> list[dict]:
    images = [image for image in (source.get("image") or []) if isinstance(image, dict)]
    path = source.get("image_path")
    if path and not any(image.get("path") == path for image in images):
        images = [*images, {"path": path, "page": source.get("page"),
                           "image_hash": source.get("image_hash")}]
    return images


def resolve_source_image(image: dict) -> Path | None:
    """Map old ingest paths by corpus suffix, never by an ambiguous basename."""
    raw = image.get("path")
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return None
    path = Path(raw)
    if ".." in path.parts:
        return None
    root = configured_path("api.source_assets_dir")
    candidates = [path] if path.is_absolute() else [root / path]
    # The deployed index may have been built under /workspace/... on another host.
    for marker in dict.fromkeys([root.name, "all_documents_gpu"]):
        if marker in path.parts:
            suffix = path.parts[path.parts.index(marker) + 1:]
            if suffix:
                candidates.append(root.joinpath(*suffix))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if (resolved.is_relative_to(root) and resolved.suffix.lower() in IMAGE_TYPES
                    and resolved.is_file()):
                return resolved
        except (OSError, ValueError, RuntimeError):
            continue
    return None


def attach_image_links(sources: list[dict], workspace_id: str, session_id: str,
                       message_id: int) -> list[dict]:
    enriched = []
    for source_index, source in enumerate(sources):
        images = []
        for image_index, image in enumerate(source_images(source)):
            page = image.get("page") or source.get("page")
            name = source.get("file_name") or "Tài liệu"
            citation = f"[{name}, trang {page}]" if page else f"[{name}]"
            available = resolve_source_image(image) is not None
            url = (f"/sessions/{session_id}/messages/{message_id}/sources/{source_index}"
                   f"/images/{image_index}?{urlencode({'workspace_id': workspace_id})}")
            images.append({**image, "page": page, "citation": citation,
                           "available": available, "url": url if available else None})
        enriched.append({**source, "images": images})
    return enriched
