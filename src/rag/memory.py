"""Conversation memory (§5.6): per-session turns, rewrite, summarize.

- Keeps N recent messages per session_id within a strict character budget.
  The API compacts older messages into a bounded trace without extra LLM calls;
  direct callers may optionally request LLM summarization.
- Rewrites follow-up questions into standalone queries before retrieval,
  e.g. "còn thẻ vàng thì sao?" → "Phí thường niên thẻ vàng là bao nhiêu?".
  Uses the LLM when available, with a heuristic fallback so offline tests
  and heuristic-only deployments still work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from project_settings import setting

from .llm import LLMClient, LLMMessage

FOLLOWUP_HINTS = ("còn", "thì sao", "thế còn", "vậy", "nó", "đó", "này", "tiếp", "sao ạ", "sao?")
DEFAULT_MAX_TURNS = setting("memory.max_turns")
DEFAULT_MAX_CHARS = setting("memory.max_chars")


@dataclass
class Turn:
    role: str  # user | assistant
    content: str


@dataclass
class SessionMemory:
    max_turns: int = field(default_factory=lambda: setting("memory.max_turns"))
    max_chars: int = field(default_factory=lambda: setting("memory.max_chars"))
    sessions: dict[str, dict] = field(default_factory=dict)

    def _state(self, session_id: str) -> dict:
        return self.sessions.setdefault(session_id, {"turns": [], "summary": ""})

    def history(self, session_id: str) -> list[Turn]:
        return list(self._state(session_id)["turns"])

    def summary(self, session_id: str) -> str:
        return str(self._state(session_id)["summary"])

    def add_turn(self, session_id: str, role: str, content: str, llm: LLMClient | None = None) -> None:
        state = self._state(session_id)
        state["turns"].append(Turn(role=role, content=content))
        # Keep only N recent turns; fold the rest into the summary.
        while len(state["turns"]) > self.max_turns:
            dropped = state["turns"].pop(0)
            state["summary"] = self._merge_summary(state["summary"], dropped, llm)
        while sum(len(t.content) for t in state["turns"]) > self.max_chars and len(state["turns"]) > 1:
            dropped = state["turns"].pop(0)
            state["summary"] = self._merge_summary(state["summary"], dropped, llm)
        if state["turns"] and len(state["turns"][0].content) > self.max_chars:
            # A single oversized message must also obey the working-context budget.
            turn = state["turns"][0]
            state["summary"] = self._merge_summary(state["summary"], turn, llm)
            state["turns"][0] = Turn(role=turn.role, content=turn.content[-self.max_chars:])

    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        # Compact deterministically: no extra LLM request or latency after generation.
        self.add_turn(session_id, "user", question)
        self.add_turn(session_id, "assistant", answer)

    def snapshot(self, session_id: str) -> dict:
        return {"turns": [{"role": t.role, "content": t.content} for t in self.history(session_id)],
                "summary": self.summary(session_id)}

    def restore(self, session_id: str, context: dict) -> None:
        self.sessions[session_id] = {
            "turns": [],
            "summary": str(context.get("summary", ""))[-setting("memory.summary_chars"):],
        }
        for turn in context.get("turns", []):
            self.add_turn(session_id, turn["role"], turn["content"])

    def _merge_summary(self, summary: str, dropped: Turn, llm: LLMClient | None) -> str:
        if llm is None:
            # Heuristic fallback: keep a truncated trace of dropped turns.
            snippet = dropped.content[:setting("memory.summary_snippet_chars")]
            combined = f"{summary}\n- {dropped.role}: {snippet}".strip()
            return combined[-setting("memory.summary_chars"): ]
        try:
            reply = llm.complete(
                [LLMMessage(role="system", content="Tóm tắt ngắn gọn đoạn hội thoại sau trong 2-3 câu, giữ các thực thể và số liệu."),
                 LLMMessage(role="user", content=f"Tóm tắt cũ: {summary}\nLượt mới ({dropped.role}): {dropped.content}")],
                temperature=setting("memory.summary_temperature"),
                max_tokens=setting("memory.summary_max_tokens"),
            )
            return reply.strip()[:setting("memory.summary_chars")]
        except Exception:
            return self._merge_summary(summary, dropped, llm=None)

    def rewrite(self, session_id: str, question: str, llm: LLMClient | None = None) -> str:
        """Rewrite a follow-up into a standalone question. Returns original if standalone."""
        question = question.strip()
        if not question or not self._looks_like_followup(question):
            return question
        history = self.history(session_id)
        if not history:
            return question
        if llm is None:
            return self._heuristic_rewrite(history, question)
        try:
            context = "\n".join(f"{t.role}: {t.content[:setting('memory.rewrite_turn_chars')]}" for t in history[-setting("memory.rewrite_turns"): ])
            reply = llm.complete(
                [LLMMessage(role="system", content=(
                    "Viết lại câu hỏi nối tiếp thành câu độc lập, giữ nguyên ý và thực thể. "
                    "Chỉ trả về câu hỏi đã viết lại, không giải thích.")),
                 LLMMessage(role="user", content=f"Lịch sử:\n{context}\nCâu nối tiếp: {question}")],
                temperature=setting("memory.rewrite_temperature"),
                max_tokens=setting("memory.rewrite_max_tokens"),
            ).strip()
            return reply or question
        except Exception:
            return self._heuristic_rewrite(history, question)

    @staticmethod
    def _looks_like_followup(question: str) -> bool:
        lowered = question.lower()
        # Short independent questions must not inherit an unrelated old topic.
        # Match whole words so, for example, "nó" does not match "nóng".
        return any(re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", lowered)
                   for hint in FOLLOWUP_HINTS)

    @staticmethod
    def _heuristic_rewrite(history: Sequence[Turn], question: str) -> str:
        # Carry the last user topic into short follow-ups.
        for turn in reversed(history):
            if turn.role == "user" and len(turn.content.split()) > 4:
                return f"{turn.content} ({question})"
        return question
