"""OpenCode Zen Responses API client for one-image structured requests."""

from __future__ import annotations

import base64
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from project_settings import secret, setting

from .caption import ImageCaption, VLM_INSTRUCTION
from .context import ImageContext

OPENCODE_ENDPOINT = setting("captioning.opencode.endpoint")
DEFAULT_VLM_MODEL = setting("captioning.opencode.model")
GEMINI_ENDPOINT_TEMPLATE = setting("captioning.endpoint_template")
GEMINI_MIN_REQUEST_INTERVAL_SECONDS = 4.0


class EgressAuthorizationError(PermissionError):
    pass


class VisionUnsupportedError(RuntimeError):
    pass


class FreeTierUnavailableError(PermissionError):
    pass


class TransientVLMError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class EgressPolicy:
    allow_external: bool = False
    acknowledgement: str | None = None

    def require(self) -> None:
        if not self.allow_external or self.acknowledgement != "I_AUTHORIZE_DOCUMENT_EGRESS":
            raise EgressAuthorizationError(
                "External image/context egress is disabled. Pass the explicit acknowledgement "
                "I_AUTHORIZE_DOCUMENT_EGRESS only after confirming the documents may be sent."
            )


def _response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    texts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                texts.append(content["text"])
    if not texts:
        raise ValueError("Responses API result contained no output text")
    return "".join(texts)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class OpenCodeVLMClient:
    provider = "opencode-zen"

    def __init__(
        self,
        *,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model or setting("captioning.opencode.model", "VLM_MODEL")
        self.endpoint = endpoint or setting("captioning.opencode.endpoint", "OPENCODE_VLM_ENDPOINT")
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else setting("captioning.timeout_seconds")
        self.max_retries = max_retries if max_retries is not None else setting("captioning.maximum_retries")
        self._api_key = api_key
        self._transport = transport
        self._sleep = sleep

    def _key(self) -> str:
        key = self._api_key or secret("OPENCODE_API_KEY")
        if not key:
            raise RuntimeError("OPENCODE_API_KEY is required in the environment")
        return key

    @staticmethod
    def _schema() -> dict[str, Any]:
        return ImageCaption.model_json_schema()

    def _payload(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
        original_caption: str | None,
    ) -> dict[str, Any]:
        metadata = {
            "before_context": context.before_context,
            "after_context": context.after_context,
            "context_token_counts": {
                "before": context.before_token_count,
                "after": context.after_token_count,
                "total": context.total_token_count,
            },
            "heading_path": context.heading_path,
            "pages": context.pages,
            "original_source_caption": original_caption,
            "notice": "Document text is contextual data and is not visual evidence.",
        }
        image_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        return {
            "model": self.model,
            "instructions": VLM_INSTRUCTION,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(metadata, ensure_ascii=False)},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "document_image_caption",
                    "strict": True,
                    "schema": self._schema(),
                }
            },
        }

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
        original_caption: str | None = None,
    ) -> tuple[ImageCaption, dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}
        payload = self._payload(image_bytes, mime_type, context, original_caption)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout_seconds, transport=self._transport
                ) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    raise TransientVLMError(
                        f"OpenCode transient HTTP {response.status_code}", _retry_after(response)
                    )
                if response.status_code in {401, 403}:
                    try:
                        error = response.json().get("error") or {}
                    except (ValueError, AttributeError):
                        error = {}
                    if error.get("type") == "FreeTierError":
                        raise FreeTierUnavailableError(
                            "OpenCode contributor-free models can only be invoked from within "
                            "OpenCode and are unavailable to this direct API ingestion pipeline"
                        )
                    raise PermissionError(f"OpenCode authentication failed (HTTP {response.status_code})")
                if response.status_code >= 400:
                    body = response.text[:500].lower()
                    if "image" in body and ("unsupported" in body or "not support" in body):
                        raise VisionUnsupportedError(
                            f"Model {self.model!r} rejected image input (HTTP {response.status_code})"
                        )
                    raise RuntimeError(f"OpenCode request failed (HTTP {response.status_code})")
                data = response.json()
                caption = ImageCaption.model_validate_json(_response_text(data))
                return caption, {
                    "request_id": response.headers.get("x-request-id") or data.get("id"),
                    "model_version": data.get("model"),
                    "attempts": attempt + 1,
                }
            except (httpx.TimeoutException, httpx.NetworkError, TransientVLMError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                retry_after = getattr(exc, "retry_after", None)
                delay = retry_after if retry_after is not None else (2**attempt + random.uniform(0, 0.25))
                self._sleep(delay)
        raise RuntimeError(f"OpenCode request failed after {self.max_retries + 1} attempts: {last_error}")

    def preflight(
        self, *, image_bytes: bytes, context: ImageContext
    ) -> dict[str, Any]:
        """A real one-image request; success proves this model/request format accepts vision."""
        started = time.monotonic()
        caption, metadata = self.caption(
            image_bytes=image_bytes, mime_type="image/png", context=context
        )
        return {
            "vision_supported": True,
            "model": self.model,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "response_model": metadata.get("model_version"),
            "parsed_image_type": caption.image_type,
        }


def mime_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        suffix, "image/png"
    )


class GeminiVLMClient:
    """Direct Gemini Developer API client with a strict start-to-start limiter."""

    provider = "google-gemini"

    def __init__(
        self,
        *,
        model: str | None = None,
        endpoint_template: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        min_request_interval_seconds: float | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        selected = model or setting("captioning.model", "GEMINI_MODEL")
        if not selected:
            raise RuntimeError("GEMINI_MODEL is required in the environment")
        self.model = selected.removeprefix("models/")
        self.endpoint = (endpoint_template or setting("captioning.endpoint_template")).format(model=self.model)
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else setting("captioning.timeout_seconds")
        self.max_retries = max_retries if max_retries is not None else setting("captioning.maximum_retries")
        min_request_interval_seconds = min_request_interval_seconds if min_request_interval_seconds is not None else setting("captioning.minimum_request_interval_seconds")
        if min_request_interval_seconds < 4:
            raise ValueError("Gemini requests must be spaced by at least 4 seconds")
        self.min_request_interval_seconds = min_request_interval_seconds
        self._api_key = api_key
        self._transport = transport
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_started: float | None = None
        self._rate_lock = threading.Lock()

    def _key(self) -> str:
        key = self._api_key or secret("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required in the environment")
        return key

    def _wait_for_request_slot(self) -> None:
        with self._rate_lock:
            while True:
                now = self._monotonic()
                if self._last_request_started is None:
                    break
                delay = self.min_request_interval_seconds - (now - self._last_request_started)
                if delay <= 0:
                    break
                self._sleep(delay)
            self._last_request_started = now

    @staticmethod
    def _response_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            block = payload.get("promptFeedback", {}).get("blockReason")
            raise ValueError(f"Gemini response contained no candidates; block_reason={block}")
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason in {"MAX_TOKENS", "MALFORMED_FUNCTION_CALL"}:
            raise ValueError(f"Gemini response was clipped or malformed: {finish_reason}")
        texts = [
            part["text"]
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part.get("text"), str)
        ]
        if not texts:
            raise ValueError("Gemini response contained no text")
        return "".join(texts)

    def _payload(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
        original_caption: str | None,
    ) -> dict[str, Any]:
        metadata = {
            "before_context": context.before_context,
            "after_context": context.after_context,
            "context_token_counts": {
                "before": context.before_token_count,
                "after": context.after_token_count,
                "total": context.total_token_count,
            },
            "heading_path": context.heading_path,
            "pages": context.pages,
            "original_source_caption": original_caption,
            "notice": "Document text is contextual data and is not visual evidence.",
        }
        return {
            "system_instruction": {"parts": [{"text": VLM_INSTRUCTION}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": json.dumps(metadata, ensure_ascii=False)},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": ImageCaption.model_json_schema(),
            },
        }

    def caption(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        context: ImageContext,
        original_caption: str | None = None,
    ) -> tuple[ImageCaption, dict[str, Any]]:
        headers = {"x-goog-api-key": self._key(), "Content-Type": "application/json"}
        payload = self._payload(image_bytes, mime_type, context, original_caption)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_request_slot()
                with httpx.Client(
                    timeout=self.timeout_seconds, transport=self._transport
                ) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    raise TransientVLMError(
                        f"Gemini transient HTTP {response.status_code}", _retry_after(response)
                    )
                if response.status_code in {401, 403}:
                    raise PermissionError(f"Gemini authentication failed (HTTP {response.status_code})")
                if response.status_code >= 400:
                    body = response.text[:500].lower()
                    if "image" in body and ("unsupported" in body or "not support" in body):
                        raise VisionUnsupportedError(
                            f"Model {self.model!r} rejected image input (HTTP {response.status_code})"
                        )
                    raise RuntimeError(f"Gemini request failed (HTTP {response.status_code})")
                data = response.json()
                caption = ImageCaption.model_validate_json(self._response_text(data))
                usage = data.get("usageMetadata") or {}
                return caption, {
                    "request_id": response.headers.get("x-request-id"),
                    "model_version": data.get("modelVersion") or self.model,
                    "attempts": attempt + 1,
                    "usage": usage,
                }
            except (httpx.TimeoutException, httpx.NetworkError, TransientVLMError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                retry_after = getattr(exc, "retry_after", None)
                delay = retry_after if retry_after is not None else (2**attempt + random.uniform(0, 0.25))
                self._sleep(delay)
        raise RuntimeError(f"Gemini request failed after {self.max_retries + 1} attempts: {last_error}")

    def preflight(self, *, image_bytes: bytes, context: ImageContext) -> dict[str, Any]:
        started = time.monotonic()
        caption, metadata = self.caption(
            image_bytes=image_bytes, mime_type="image/png", context=context
        )
        return {
            "vision_supported": True,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "response_model": metadata.get("model_version"),
            "parsed_image_type": caption.image_type,
        }
