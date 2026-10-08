import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from rag.enrichment.cache import CaptionCache, caption_cache_key
from rag.enrichment.caption import ImageCaption, CaptionStatus, validate_caption
from rag.enrichment.context import ImageAnchor, ImageContext
from rag.enrichment.service import CaptionService, prepare_api_copy
from rag.enrichment.vlm import (
    EgressPolicy,
    FreeTierUnavailableError,
    GeminiVLMClient,
    OpenCodeVLMClient,
    VisionUnsupportedError,
)


def context(context_hash="ctx"):
    return ImageContext(
        document_id="doc",
        image_element_id="image",
        before_context="trước",
        after_context="sau",
        before_token_count=1,
        after_token_count=1,
        total_token_count=2,
        source_spans=[],
        image_anchor=ImageAnchor(element_id="image", order=1, asset_path="assets/x.png"),
        heading_path=["Mục I"],
        pages=[1],
        tokenizer_fingerprint="tokenizer",
        context_hash=context_hash,
    )


def response_payload(**overrides):
    body = {
        "image_type": "diagram",
        "title": "Sơ đồ",
        "caption": "Hai hộp được nối bằng một mũi tên.",
        "visible_text": ["A", "B"],
        "steps": [],
        "relationships": [{"from": "A", "to": "B"}],
        "chart_details": {},
        "uncertainties": [],
        "retrieval_useful": True,
        "contextual_relevance": None,
    }
    body.update(overrides)
    return {"id": "req", "model": "model-version", "output_text": json.dumps(body)}


def gemini_response(**overrides):
    body = json.loads(response_payload(**overrides)["output_text"])
    return {
        "modelVersion": "gemini-version",
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": json.dumps(body)}]},
            }
        ],
        "usageMetadata": {"promptTokenCount": 10},
    }


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_gemini_payload_and_four_second_request_spacing():
    clock = FakeClock()
    starts = []
    payloads = []

    def handler(request):
        starts.append(clock.now)
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=gemini_response())

    client = GeminiVLMClient(
        model="gemini-test",
        api_key="secret",
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    for _ in range(2):
        client.caption(image_bytes=b"png", mime_type="image/png", context=context())
    assert starts == [0.0, 4.0]
    assert clock.sleeps == [4.0]
    assert client.endpoint.endswith("/models/gemini-test:generateContent")
    parts = payloads[0]["contents"][0]["parts"]
    assert sum("inline_data" in part for part in parts) == 1
    assert payloads[0]["generationConfig"]["responseMimeType"] == "application/json"
    assert "secret" not in json.dumps(payloads)


def test_gemini_retry_is_also_rate_limited_to_four_seconds():
    clock = FakeClock()
    starts = []

    def handler(request):
        starts.append(clock.now)
        if len(starts) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=gemini_response())

    client = GeminiVLMClient(
        model="gemini-test",
        api_key="secret",
        transport=httpx.MockTransport(handler),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    client.caption(image_bytes=b"png", mime_type="image/png", context=context())
    assert starts == [0.0, 4.0]
    assert sum(clock.sleeps) == 4.0


def test_gemini_refuses_interval_below_four_seconds():
    with pytest.raises(ValueError, match="at least 4 seconds"):
        GeminiVLMClient(
            model="gemini-test",
            api_key="secret",
            min_request_interval_seconds=3.99,
        )


def test_gemini_clipped_response_is_rejected():
    clipped = gemini_response()
    clipped["candidates"][0]["finishReason"] = "MAX_TOKENS"
    client = GeminiVLMClient(
        model="gemini-test",
        api_key="secret",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=clipped)),
    )
    with pytest.raises(ValueError, match="clipped"):
        client.caption(image_bytes=b"png", mime_type="image/png", context=context())


def test_request_has_exactly_one_image_and_context_notice():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    client = OpenCodeVLMClient(api_key="secret", transport=httpx.MockTransport(handler))
    result, metadata = client.caption(
        image_bytes=b"png", mime_type="image/png", context=context()
    )
    content = captured["input"][0]["content"]
    assert sum(part["type"] == "input_image" for part in content) == 1
    text = next(part["text"] for part in content if part["type"] == "input_text")
    assert "not visual evidence" in text
    assert result.retrieval_useful
    assert metadata["attempts"] == 1
    assert "secret" not in json.dumps(captured)


def test_429_honors_retry_after_and_then_succeeds():
    attempts = 0
    delays = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json=response_payload())

    client = OpenCodeVLMClient(
        api_key="x",
        transport=httpx.MockTransport(handler),
        sleep=delays.append,
        max_retries=3,
    )
    client.caption(image_bytes=b"x", mime_type="image/png", context=context())
    assert attempts == 2
    assert delays == [0.25]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_not_retried(status):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(status)

    client = OpenCodeVLMClient(api_key="bad", transport=httpx.MockTransport(handler))
    with pytest.raises(PermissionError, match="authentication"):
        client.caption(image_bytes=b"x", mime_type="image/png", context=context())
    assert attempts == 1


def test_contributor_free_api_restriction_is_not_misreported_as_bad_auth():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            403,
            json={
                "error": {
                    "type": "FreeTierError",
                    "message": "OpenCode's free tier can only be used from within OpenCode",
                }
            },
        )
    )
    client = OpenCodeVLMClient(api_key="valid", transport=transport)
    with pytest.raises(FreeTierUnavailableError, match="direct API"):
        client.caption(image_bytes=b"x", mime_type="image/png", context=context())


def test_model_and_endpoint_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "configured-model")
    monkeypatch.setenv("OPENCODE_VLM_ENDPOINT", "https://example.test/v1/responses")
    client = OpenCodeVLMClient(api_key="x")
    assert client.model == "configured-model"
    assert client.endpoint == "https://example.test/v1/responses"


def test_unsupported_vision_is_explicit():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, text="image input unsupported")
    )
    client = OpenCodeVLMClient(api_key="x", transport=transport)
    with pytest.raises(VisionUnsupportedError):
        client.caption(image_bytes=b"x", mime_type="image/png", context=context())


@pytest.mark.parametrize("output", ["{clipped", json.dumps({"image_type": "photo"})])
def test_clipped_or_schema_invalid_response_fails(output):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"output_text": output})
    )
    client = OpenCodeVLMClient(api_key="x", transport=transport)
    with pytest.raises(Exception):
        client.caption(image_bytes=b"x", mime_type="image/png", context=context())


def test_data_egress_guard_records_error_without_calling_provider(tmp_path):
    image = tmp_path / "x.png"
    Image.new("RGB", (10, 10), "white").save(image)

    def must_not_call(request):
        raise AssertionError("external API called")

    service = CaptionService(
        OpenCodeVLMClient(api_key="x", transport=httpx.MockTransport(must_not_call)),
        CaptionCache(tmp_path / "cache"),
        EgressPolicy(),
    )
    record = service.process(
        document_id="doc", image_element_id="image", asset_path=image, context=context()
    )
    assert record.status == CaptionStatus.error
    assert "EgressAuthorizationError" in record.error


def test_resume_uses_cache_and_context_change_invalidates_key(tmp_path):
    image = tmp_path / "x.png"
    Image.new("RGB", (10, 10), "white").save(image)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=response_payload())

    service = CaptionService(
        OpenCodeVLMClient(api_key="x", transport=httpx.MockTransport(handler)),
        CaptionCache(tmp_path / "cache"),
        EgressPolicy(True, "I_AUTHORIZE_DOCUMENT_EGRESS"),
    )
    first = service.process(
        document_id="doc", image_element_id="image", asset_path=image, context=context("a")
    )
    resumed = service.process(
        document_id="doc", image_element_id="image", asset_path=image, context=context("a")
    )
    changed = service.process(
        document_id="doc", image_element_id="image", asset_path=image, context=context("b")
    )
    assert calls == 2
    assert first.cache_key == resumed.cache_key
    assert changed.cache_key != first.cache_key


def test_corrupted_asset_is_recorded_as_error(tmp_path):
    image = tmp_path / "broken.png"
    image.write_bytes(b"not an image")
    service = CaptionService(
        OpenCodeVLMClient(api_key="x"),
        CaptionCache(tmp_path / "cache"),
        EgressPolicy(True, "I_AUTHORIZE_DOCUMENT_EGRESS"),
    )
    record = service.process(
        document_id="doc", image_element_id="image", asset_path=image, context=context()
    )
    assert record.status == CaptionStatus.error
    assert "UnidentifiedImageError" in record.error


def test_decorative_seal_is_excluded_and_review_heuristics_work(tmp_path):
    caption = ImageCaption.model_validate_json(
        response_payload(image_type="seal", retrieval_useful=False)["output_text"]
    )
    assert validate_caption(caption) == []
    suspicious = ImageCaption.model_validate_json(
        response_payload(image_type="chart", chart_details={}, uncertainties=[])["output_text"]
    )
    assert "chart_missing_details_or_uncertainty" in validate_caption(suspicious)


def test_cache_key_never_contains_secret_material():
    key = caption_cache_key(
        image_hash="image",
        provider="provider",
        model="model",
        prompt_version="prompt",
        context_hash="context",
        preprocessing_version="pre",
    )
    assert len(key) == 64
    assert "secret" not in key
