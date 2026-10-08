"""Deterministic neighboring-text extraction for canonical image elements."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from document_parser.normalization.schema import CanonicalDocument, Element

from .tokenizer import BGE_M3_MODEL, BGE_M3_REVISION, TokenizerLike, tokenizer_fingerprint

_PLACEHOLDER = re.compile(
    r"^\s*(?:\[\s*(?:image|figure|ảnh|hình)[^]]*]|<(?:image|figure)[^>]*>)\s*$",
    re.IGNORECASE,
)
_ELIGIBLE_TYPES = {"heading", "paragraph", "list", "table"}
_GENERATED_KEYS = {"vlm_caption", "ai_caption", "generated_caption", "caption_generated"}


class ContextSourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element_id: str
    side: Literal["before", "after"]
    element_order: int
    page_number: int | None = None
    section_id: str | None = None
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)
    token_count: int = Field(ge=0)


class ImageAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element_id: str
    order: int
    page_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    asset_path: str | None = None
    reliable: bool = True
    review_reasons: list[str] = Field(default_factory=list)


class ImageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    image_element_id: str
    before_context: str
    after_context: str
    before_token_count: int
    after_token_count: int
    total_token_count: int
    source_spans: list[ContextSourceSpan]
    image_anchor: ImageAnchor
    heading_path: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    tokenizer_model: str = BGE_M3_MODEL
    tokenizer_revision: str = BGE_M3_REVISION
    tokenizer_fingerprint: str
    context_hash: str
    needs_review: bool = False


class _Candidate:
    def __init__(self, element: Element, text: str, tokens: list[int]):
        self.element = element
        self.text = text
        self.tokens = tokens


class ImageContextExtractor:
    """Extract at most 200 tokens around every image in canonical source order.

    The initial allocation is 100/100. Any unused quota transfers to the other
    side. Token slices are decoded by the same pinned tokenizer used to count
    them, so UTF-8 character boundaries are preserved.
    """

    version = "image-context-v1"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        max_tokens: int = 200,
        preferred_each_side: int = 100,
        model_name: str = BGE_M3_MODEL,
        revision: str = BGE_M3_REVISION,
    ) -> None:
        if max_tokens != 200:
            raise ValueError("Image neighboring context budget is fixed at exactly 200 tokens")
        if preferred_each_side != 100:
            raise ValueError("Image neighboring context allocation is fixed at 100/100")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.preferred_each_side = preferred_each_side
        self.model_name = model_name
        self.revision = revision
        self.fingerprint = tokenizer_fingerprint(
            tokenizer, model_name=model_name, revision=revision
        )

    @staticmethod
    def _ordered(document: CanonicalDocument) -> list[Element]:
        # Stable input position breaks ties without introducing filesystem order.
        return [pair[1] for pair in sorted(enumerate(document.elements), key=lambda p: (p[1].order, p[0]))]

    @staticmethod
    def _repeating_noise(elements: list[Element]) -> set[str]:
        pages_by_text: dict[str, set[int]] = {}
        for element in elements:
            if element.element_type not in {"header", "footer"} or not element.text.strip():
                continue
            key = " ".join(element.text.split()).casefold()
            if element.page_number is not None:
                pages_by_text.setdefault(key, set()).add(element.page_number)
        return {text for text, pages in pages_by_text.items() if len(pages) >= 2}

    @staticmethod
    def _eligible(element: Element, repeating: set[str]) -> bool:
        meta = element.metadata
        if element.element_type not in _ELIGIBLE_TYPES:
            return False
        if meta.get("exclude_from_content") or meta.get("rejected"):
            return False
        if meta.get("ocr_status") in {"rejected", "debug"}:
            return False
        if any(meta.get(key) for key in _GENERATED_KEYS):
            return False
        text = element.text.strip()
        if not text or _PLACEHOLDER.match(text):
            return False
        return " ".join(text.split()).casefold() not in repeating

    def _candidate(self, element: Element) -> _Candidate:
        text = element.text.strip()
        tokens = list(self.tokenizer.encode(text, add_special_tokens=False))
        return _Candidate(element, text, tokens)

    @staticmethod
    def _allocate(before_available: int, after_available: int) -> tuple[int, int]:
        before = min(100, before_available)
        after = min(100, after_available)
        remaining = 200 - before - after
        if remaining and before_available > before:
            take = min(remaining, before_available - before)
            before += take
            remaining -= take
        if remaining and after_available > after:
            after += min(remaining, after_available - after)
        return before, after

    def _take_before(self, candidates: list[_Candidate], budget: int):
        chosen = []
        remaining = budget
        for candidate in reversed(candidates):
            if remaining <= 0:
                break
            take = min(remaining, len(candidate.tokens))
            start = len(candidate.tokens) - take
            chosen.append((candidate, start, len(candidate.tokens)))
            remaining -= take
        chosen.reverse()
        return chosen

    @staticmethod
    def _take_after(candidates: list[_Candidate], budget: int):
        chosen = []
        remaining = budget
        for candidate in candidates:
            if remaining <= 0:
                break
            take = min(remaining, len(candidate.tokens))
            chosen.append((candidate, 0, take))
            remaining -= take
        return chosen

    def _decode(self, chosen, side: Literal["before", "after"]):
        combined_tokens: list[int] = []
        spans: list[ContextSourceSpan] = []
        pages: list[int] = []
        for candidate, start, end in chosen:
            combined_tokens.extend(candidate.tokens[start:end])
            element = candidate.element
            spans.append(
                ContextSourceSpan(
                    element_id=element.element_id,
                    side=side,
                    element_order=element.order,
                    page_number=element.page_number,
                    section_id=element.section_id,
                    token_start=start,
                    token_end=end,
                    token_count=end - start,
                )
            )
            if element.page_number is not None and element.page_number not in pages:
                pages.append(element.page_number)
        text = self.tokenizer.decode(
            combined_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text, spans, pages

    def _select_for_token_target(
        self,
        candidates: list[_Candidate],
        target: int,
        side: Literal["before", "after"],
    ):
        """Choose nearest source tokens whose decoded text re-tokenizes to target.

        SentencePiece boundary tokens can decode to text that re-tokenizes one token
        longer or shorter. The public count must describe the actual UTF-8 context
        string sent to the VLM, so search a narrow deterministic boundary window.
        """
        available = sum(len(candidate.tokens) for candidate in candidates)
        take = self._take_before if side == "before" else self._take_after
        best = ("", [], [], 0)
        best_key = (-1, -10_000)
        maximum_source_budget = min(available, target + 24)
        for source_budget in range(maximum_source_budget + 1):
            text, spans, pages = self._decode(take(candidates, source_budget), side)
            actual = len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0
            if actual > target:
                continue
            key = (actual, -abs(source_budget - target))
            if key > best_key:
                best = (text, spans, pages, actual)
                best_key = key
            if actual == target and source_budget == target:
                break
        return best

    @staticmethod
    def _heading_paths(elements: list[Element]) -> dict[str, list[str]]:
        path: list[tuple[int, str]] = []
        result: dict[str, list[str]] = {}
        for element in elements:
            if element.element_type == "heading" and element.text.strip():
                level = element.metadata.get("heading_level")
                level = level if isinstance(level, int) and level > 0 else 1
                path = [entry for entry in path if entry[0] < level]
                path.append((level, element.text.strip()))
            result[element.element_id] = [value for _, value in path]
        return result

    def extract(self, document: CanonicalDocument, image: Element | str) -> ImageContext:
        ordered = self._ordered(document)
        image_id = image if isinstance(image, str) else image.element_id
        matches = [i for i, element in enumerate(ordered) if element.element_id == image_id]
        if not matches:
            raise ValueError(f"Image anchor {image_id!r} is absent from document {document.document_id!r}")
        index = matches[0]
        anchor_element = ordered[index]
        if anchor_element.element_type != "image":
            raise ValueError(f"Element {image_id!r} is not an image")
        repeating = self._repeating_noise(ordered)
        before_candidates = [
            self._candidate(e) for e in ordered[:index] if self._eligible(e, repeating)
        ]
        after_candidates = [
            self._candidate(e) for e in ordered[index + 1 :] if self._eligible(e, repeating)
        ]
        before_max = self._select_for_token_target(
            before_candidates, self.max_tokens, "before"
        )[3]
        after_max = self._select_for_token_target(
            after_candidates, self.max_tokens, "after"
        )[3]
        before_target, after_target = self._allocate(before_max, after_max)
        before_text, before_spans, before_pages, before_budget = self._select_for_token_target(
            before_candidates, before_target, "before"
        )
        after_text, after_spans, after_pages, after_budget = self._select_for_token_target(
            after_candidates, after_target, "after"
        )
        # A boundary may make an exact per-side target impossible. Transfer any
        # remaining actual-string budget to the other side and search once more.
        remaining = self.max_tokens - before_budget - after_budget
        if remaining and before_max > before_budget:
            current = (before_text, before_spans, before_pages, before_budget)
            for target in range(before_budget + remaining, min(before_max, before_budget + remaining + 24) + 1):
                candidate = self._select_for_token_target(before_candidates, target, "before")
                if candidate[3] + after_budget <= self.max_tokens and candidate[3] > current[3]:
                    current = candidate
                if current[3] + after_budget == self.max_tokens:
                    break
            before_text, before_spans, before_pages, before_budget = current
            remaining = self.max_tokens - before_budget - after_budget
        if remaining and after_max > after_budget:
            current = (after_text, after_spans, after_pages, after_budget)
            for target in range(after_budget + remaining, min(after_max, after_budget + remaining + 24) + 1):
                candidate = self._select_for_token_target(after_candidates, target, "after")
                if candidate[3] + before_budget <= self.max_tokens and candidate[3] > current[3]:
                    current = candidate
                if current[3] + before_budget == self.max_tokens:
                    break
            after_text, after_spans, after_pages, after_budget = current
        reasons = []
        order_counts = Counter(element.order for element in ordered)
        if order_counts[anchor_element.order] > 1:
            reasons.append("duplicate_canonical_order")
        if anchor_element.metadata.get("reading_order_unreliable"):
            reasons.append("parser_flagged_reading_order_unreliable")
        anchor = ImageAnchor(
            element_id=anchor_element.element_id,
            order=anchor_element.order,
            page_number=anchor_element.page_number,
            bbox=anchor_element.bbox,
            asset_path=anchor_element.metadata.get("asset_path"),
            reliable=not reasons,
            review_reasons=reasons,
        )
        spans = before_spans + after_spans
        pages = list(dict.fromkeys([*before_pages, *after_pages]))
        payload = {
            "version": self.version,
            "document_id": document.document_id,
            "image_element_id": image_id,
            "before_context": before_text,
            "after_context": after_text,
            "before_token_count": before_budget,
            "after_token_count": after_budget,
            "source_spans": [span.model_dump() for span in spans],
            "anchor": anchor.model_dump(),
            "tokenizer_fingerprint": self.fingerprint,
        }
        context_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ImageContext(
            document_id=document.document_id,
            image_element_id=image_id,
            before_context=before_text,
            after_context=after_text,
            before_token_count=before_budget,
            after_token_count=after_budget,
            total_token_count=before_budget + after_budget,
            source_spans=spans,
            image_anchor=anchor,
            heading_path=self._heading_paths(ordered).get(image_id, []),
            pages=pages,
            tokenizer_model=self.model_name,
            tokenizer_revision=self.revision,
            tokenizer_fingerprint=self.fingerprint,
            context_hash=context_hash,
            needs_review=bool(reasons),
        )

    def extract_all(self, document: CanonicalDocument) -> list[ImageContext]:
        return [
            self.extract(document, element)
            for element in self._ordered(document)
            if element.element_type == "image"
        ]
