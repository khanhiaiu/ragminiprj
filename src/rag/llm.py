"""LLM clients for Step 4 (§4.5). llama.cpp Qwen3.5 2B via OpenAI-compatible API.

Production: run llama-server with a Qwen3.5-2B Q4_K_M GGUF, e.g.
  llama-server -hf unsloth/Qwen3.5-2B-GGUF:Q4_K_M --no-mmproj --reasoning off --ctx-size 8192 --port 8080
then use OpenAICompatLLM(base_url="http://localhost:8080/v1").

Tests/offline: use FakeLLM (no server, deterministic).
Only stdlib is required; no llama-cpp-python dependency.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from project_settings import setting


@dataclass
class LLMMessage:
    role: str  # system | user | assistant
    content: str


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[LLMMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


class FakeLLM:
    """Deterministic stand-in. Records calls so tests can assert no-LLM-on-fallback."""

    def __init__(self, reply: str = "Trả lời dựa trên ngữ cảnh được cung cấp.") -> None:
        self.reply = reply
        self.calls: list[list[LLMMessage]] = []

    def complete(self, messages: Sequence[LLMMessage], temperature: float | None = None, max_tokens: int | None = None) -> str:
        self.calls.append(list(messages))
        _ = (temperature, max_tokens)
        return self.reply


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat client (llama-server, vLLM, Ollama...)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model or setting("llm.model", "LLM_MODEL")
        self.base_url = (base_url or setting("llm.base_url", "LLM_BASE_URL")).rstrip("/")
        self.timeout = timeout if timeout is not None else setting("llm.timeout_seconds")

    def complete(self, messages: Sequence[LLMMessage], temperature: float | None = None, max_tokens: int | None = None) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else setting("llm.temperature"),
            "max_tokens": max_tokens if max_tokens is not None else setting("llm.max_tokens"),
            "stream": False,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ConnectionError(f"LLM server unreachable at {self.base_url}: {exc}") from exc
        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected LLM response: {payload!r}") from exc

    def complete_stream(
        self, messages: Sequence[LLMMessage], temperature: float | None = None,
        max_tokens: int | None = None, on_first_token: Callable[[], None] | None = None,
    ) -> str:
        """Consume SSE and observe the first nonempty content delta, excluding reasoning.

        The API still returns a complete, guardrailed answer. TTFT measures the
        model stream at the API, not when the browser renders the full response.
        """
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature if temperature is not None else setting("llm.temperature"),
            "max_tokens": max_tokens if max_tokens is not None else setting("llm.max_tokens"),
            "stream": True,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        pieces = []
        finished = False
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        finished = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise ValueError("LLM stream returned an error")
                    choices = event.get("choices", [])
                    if not choices:
                        continue  # usage-only event
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        if not isinstance(content, str):
                            raise ValueError("Invalid LLM stream content")
                        if not pieces and on_first_token is not None:
                            on_first_token()
                        pieces.append(content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid LLM streaming response") from exc
        except Exception as exc:
            raise ConnectionError(f"LLM server unreachable at {self.base_url}: {exc}") from exc
        if not finished:
            raise ConnectionError("LLM stream ended before completion")
        return "".join(pieces)


def make_llm(name: str | None = None, **kwargs) -> LLMClient:
    key = (name or setting("llm.backend", "LLM_BACKEND")).strip().lower()
    if key in {"fake", "test", "offline", "echo"}:
        return FakeLLM(reply=str(kwargs.get("reply", "Trả lời dựa trên ngữ cảnh được cung cấp.")))
    if key in {"server", "llama-server", "llamacpp", "llama.cpp", "openai-compat", "qwen"}:
        return OpenAICompatLLM(
            model=kwargs.get("model"),
            base_url=kwargs.get("base_url"),
            timeout=kwargs.get("timeout"),
        )
    raise ValueError(f"Unknown LLM {name!r}; expected 'fake' or 'server'")
