"""LLM clients for Step 4 (§4.5). llama.cpp Qwen 4B via OpenAI-compatible API.

Production: run llama-server with a Qwen3-4B Q4_K_M GGUF, e.g.
  llama-server -hf unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M --port 8080
then use OpenAICompatLLM(base_url="http://localhost:8080/v1").

Tests/offline: use FakeLLM (no server, deterministic).
Only stdlib is required; no llama-cpp-python dependency.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass
class LLMMessage:
    role: str  # system | user | assistant
    content: str


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[LLMMessage],
        temperature: float = 0.15,
        max_tokens: int = 512,
    ) -> str: ...


class FakeLLM:
    """Deterministic stand-in. Records calls so tests can assert no-LLM-on-fallback."""

    def __init__(self, reply: str = "Trả lời dựa trên ngữ cảnh được cung cấp.") -> None:
        self.reply = reply
        self.calls: list[list[LLMMessage]] = []

    def complete(self, messages: Sequence[LLMMessage], temperature: float = 0.15, max_tokens: int = 512) -> str:
        self.calls.append(list(messages))
        _ = (temperature, max_tokens)
        return self.reply


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat client (llama-server, vLLM, Ollama...)."""

    def __init__(
        self,
        model: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
        base_url: str = "http://localhost:8080/v1",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: Sequence[LLMMessage], temperature: float = 0.15, max_tokens: int = 512) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
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


def make_llm(name: str, **kwargs) -> LLMClient:
    key = name.strip().lower()
    if key in {"fake", "test", "offline", "echo"}:
        return FakeLLM(reply=str(kwargs.get("reply", "Trả lời dựa trên ngữ cảnh được cung cấp.")))
    if key in {"server", "llama-server", "llamacpp", "llama.cpp", "openai-compat", "qwen"}:
        return OpenAICompatLLM(
            model=str(kwargs.get("model", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")),
            base_url=str(kwargs.get("base_url", "http://localhost:8080/v1")),
            timeout=int(kwargs.get("timeout", 120)),
        )
    raise ValueError(f"Unknown LLM {name!r}; expected 'fake' or 'server'")
