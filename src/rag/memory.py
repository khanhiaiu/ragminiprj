"""Conversation memory (§5.6): per-session turns, rewrite, summarize.

- Keeps N recent turns per session_id; older turns are summarized via LLM
  once the char budget is exceeded.
- Rewrites follow-up questions into standalone queries before retrieval,
  e.g. "còn thẻ vàng thì sao?" → "Phí thường niên thẻ vàng là bao nhiêu?".
  Uses the LLM when available, with a heuristic fallback so offline tests
  and heuristic-only deployments still work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .llm import LLMClient, LLMMessage

FOLLOWUP_HINTS = ("còn", "thì sao", "thế còn", "vậy", "nó", "đó", "này", "tiếp", "sao ạ", "sao?")
DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_CHARS = 6000


@dataclass
class Turn:
    role: str  # user | assistant
    content: str


@dataclass
class SessionMemory:
    max_turns: int = DEFAULT_MAX_TURNS
    max_chars: int = DEFAULT_MAX_CHARS
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
        total = sum(len(t.content) for t in state["turns"])
        if total > self.max_chars and len(state["turns"]) > 1:
            dropped = state["turns"].pop(0)
            state["summary"] = self._merge_summary(state["summary"], dropped, llm)

    def _merge_summary(self, summary: str, dropped: Turn, llm: LLMClient | None) -> str:
        if llm is None:
            # Heuristic fallback: keep a truncated trace of dropped turns.
            snippet = dropped.content[:200]
            combined = f"{summary}\n- {dropped.role}: {snippet}".strip()
            return combined[-2000:]
        try:
            reply = llm.complete(
                [LLMMessage(role="system", content="Tóm tắt ngắn gọn đoạn hội thoại sau trong 2-3 câu, giữ các thực thể và số liệu."),
                 LLMMessage(role="user", content=f"Tóm tắt cũ: {summary}\nLượt mới ({dropped.role}): {dropped.content}")],
                temperature=0.1,
                max_tokens=256,
            )
            return reply.strip()[:2000]
        except Exception:
            return summary

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
            context = "\n".join(f"{t.role}: {t.content[:300]}" for t in history[-4:])
            reply = llm.complete(
                [LLMMessage(role="system", content=(
                    "Viết lại câu hỏi nối tiếp thành câu độc lập, giữ nguyên ý và thực thể. "
                    "Chỉ trả về câu hỏi đã viết lại, không giải thích.")),
                 LLMMessage(role="user", content=f"Lịch sử:\n{context}\nCâu nối tiếp: {question}")],
                temperature=0.1,
                max_tokens=128,
            ).strip()
            return reply or question
        except Exception:
            return self._heuristic_rewrite(history, question)

    @staticmethod
    def _looks_like_followup(question: str) -> bool:
        lowered = question.lower()
        if len(question.split()) <= 6:
            return True
        return any(hint in lowered for hint in FOLLOWUP_HINTS)

    @staticmethod
    def _heuristic_rewrite(history: Sequence[Turn], question: str) -> str:
        # Carry the last user topic into short follow-ups.
        for turn in reversed(history):
            if turn.role == "user" and len(turn.content.split()) > 4:
                return f"{turn.content} ({question})"
        return question
