"""TTFT is triggered by actual content, not roles, reasoning or stream headers."""

import io
import json

import pytest

from rag.llm import LLMMessage, OpenAICompatLLM


def test_stream_first_content_callback_and_request_options(monkeypatch):
    lines = [
        ': keep-alive',
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}',
        'data: {"choices":[{"delta":{"content":"Xin "}}]}',
        'data: {"choices":[{"delta":{"content":"chào"}}]}',
        'data: {"choices":[],"usage":{"completion_tokens":2}}',
        'data: [DONE]',
    ]
    callbacks, requests = [], []

    def open_response(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(('\n\n'.join(lines) + '\n\n').encode())

    monkeypatch.setattr("rag.llm.urllib.request.urlopen", open_response)
    client = OpenAICompatLLM()
    output = client.complete_stream([LLMMessage("user", "q")], temperature=.2, max_tokens=64,
                                    on_first_token=lambda: callbacks.append("first"))
    assert output == "Xin chào" and callbacks == ["first"]
    assert requests[0]["stream"] and requests[0]["max_tokens"] == 64
    assert requests[0]["temperature"] == .2


def test_empty_stream_never_reports_a_token(monkeypatch):
    monkeypatch.setattr("rag.llm.urllib.request.urlopen", lambda *a, **k: io.BytesIO(b'data: [DONE]\n'))
    observed = []
    assert OpenAICompatLLM().complete_stream([], on_first_token=lambda: observed.append(1)) == ""
    assert observed == []


@pytest.mark.parametrize("body, error", [
    (b'data: {"choices":[{"delta":{"content":"partial"}}]}\n', ConnectionError),
    (b'data: invalid-json\n', ValueError),
    (b'data: {"error":{"message":"failure"}}\n', ValueError),
])
def test_broken_stream_does_not_return_a_partial_answer(monkeypatch, body, error):
    monkeypatch.setattr("rag.llm.urllib.request.urlopen", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(error):
        OpenAICompatLLM().complete_stream([])
