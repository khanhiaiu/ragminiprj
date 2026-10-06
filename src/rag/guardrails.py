"""Input/output guardrails (lightweight, local, no external API).

- Input: reject empty / oversized questions; flag prompt-injection attempts
  ("bỏ qua chỉ dẫn", "ignore instructions", ...) into strict mode instead of
  refusing outright — the system prompt already constrains the model to the
  retrieved context.
- Output: force fallback when there is no context or the model returns
  nothing; cap length; never invent citations (citations are appended from
  the retrieved chunks by answer.py, not by the model).
"""

from __future__ import annotations

import re

from .retrieve import FALLBACK_MESSAGE

MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 4000
_INJECTION_RE = re.compile(
    r"bỏ qua (mọi )?chỉ dẫn|ignore (all )?instructions|system prompt|jailbreak|do anything now",
    re.IGNORECASE,
)


def check_input(question: str) -> dict:
    text = (question or "").strip()
    if not text:
        return {"ok": False, "reason": "empty question", "strict": False}
    if len(text) > MAX_QUESTION_CHARS:
        return {"ok": False, "reason": f"question too long ({len(text)} > {MAX_QUESTION_CHARS})", "strict": False}
    return {"ok": True, "reason": "", "strict": bool(_INJECTION_RE.search(text))}


def check_output(answer: str, has_context: bool) -> dict:
    if not has_context or not (answer or "").strip():
        return {"ok": False, "answer": FALLBACK_MESSAGE, "reason": "no context or empty model output"}
    clipped = answer.strip()
    if len(clipped) > MAX_ANSWER_CHARS:
        clipped = clipped[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"
    return {"ok": True, "answer": clipped, "reason": ""}
