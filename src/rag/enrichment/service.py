"""Caption orchestration, preprocessing, caching and recorded dispositions."""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

from PIL import Image

from .cache import CaptionCache, caption_cache_key
from .caption import (
    PREPROCESSING_VERSION,
    PROMPT_VERSION,
    CaptionRecord,
    CaptionStatus,
    validate_caption,
)
from .context import ImageContext
from .vlm import EgressPolicy, mime_type_for


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_api_copy(path: Path, *, max_dimension: int = 4096) -> tuple[bytes, str]:
    """Preserve original bytes unless safely downscaling a very large raster."""
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        if max(width, height) <= max_dimension:
            return raw, mime_type_for(path)
        scale = max_dimension / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        # PNG avoids new JPEG artifacts around small document text.
        resized.save(output, format="PNG", optimize=True)
        return output.getvalue(), "image/png"


class CaptionService:
    def __init__(
        self,
        client,
        cache: CaptionCache,
        egress_policy: EgressPolicy,
    ) -> None:
        self.client = client
        self.cache = cache
        self.egress_policy = egress_policy

    def process(
        self,
        *,
        document_id: str,
        image_element_id: str,
        asset_path: Path,
        context: ImageContext,
        original_caption: str | None = None,
        resume: bool = True,
    ) -> CaptionRecord:
        digest = image_hash(asset_path)
        key = caption_cache_key(
            image_hash=digest,
            provider=self.client.provider,
            model=self.client.model,
            prompt_version=PROMPT_VERSION,
            context_hash=context.context_hash,
            preprocessing_version=PREPROCESSING_VERSION,
        )
        reusable_statuses = {
            CaptionStatus.completed, CaptionStatus.needs_review, CaptionStatus.excluded,
        }
        if (
            resume
            and (cached := self.cache.get(key)) is not None
            and cached.status in reusable_statuses
        ):
            return cached
        record = CaptionRecord(
            document_id=document_id,
            image_element_id=image_element_id,
            asset_path=str(asset_path),
            image_hash=digest,
            context_hash=context.context_hash,
            cache_key=key,
            provider=self.client.provider,
            model=self.client.model,
            context_metadata=context.model_dump(),
            source_references={"source_spans": [s.model_dump() for s in context.source_spans]},
        )
        self.cache.put(record)  # checkpoint pending before external work
        try:
            self.egress_policy.require()
            api_bytes, mime_type = prepare_api_copy(asset_path)
            started = time.monotonic()
            caption, metadata = self.client.caption(
                image_bytes=api_bytes,
                mime_type=mime_type,
                context=context,
                original_caption=original_caption,
            )
            reasons = validate_caption(
                caption, context.before_context + "\n" + context.after_context
            )
            decorative = caption.image_type in {"logo", "seal"} and not caption.retrieval_useful
            record.caption = caption
            record.status = (
                CaptionStatus.excluded
                if decorative
                else CaptionStatus.needs_review
                if reasons or context.needs_review
                else CaptionStatus.completed
            )
            record.review_reasons = [*context.image_anchor.review_reasons, *reasons]
            record.model_version = metadata.get("model_version")
            record.attempts = int(metadata.get("attempts", 1))
            record.latency_ms = round((time.monotonic() - started) * 1000)
        except Exception as exc:
            record.status = CaptionStatus.error
            record.error = f"{exc.__class__.__name__}: {exc}"
        self.cache.put(record)
        return record
