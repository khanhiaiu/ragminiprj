"""Answer generation (§4.5): prompt → LLM Qwen 4B → citations + fallback.

Pipeline order per SystemDocuments:
  guardrail-in → rewrite (memory) → hybrid retrieve → [fallback?] →
  prompt (system + top5 labeled context + history/summary + question) →
  LLM (temp 0.1-0.2) → guardrail-out → answer + [file, trang] citations.

Retrieval fallback short-circuits BEFORE any LLM call (spec §4.4).
"""

from __future__ import annotations

from typing import Any, Sequence

from . import guardrails
from .llm import LLMClient, LLMMessage
from .memory import SessionMemory
from .retrieve import FALLBACK_MESSAGE, HybridRetriever, RetrievedChunk

SYSTEM_PROMPT = (
    "Bạn là trợ lý RAG tiếng Việt. Chỉ trả lời từ NGỮ CẢNH được cung cấp; "
    "không dùng kiến thức ngoài. Nếu ngữ cảnh không đủ, chỉ trả lời đúng: "
    f'"{FALLBACK_MESSAGE}". Mọi ý trả lời phải bám vào đoạn ngữ cảnh, '
    "không bịa số liệu. Giữ câu trả lời ngắn gọn, tiếng Việt."
)
DEFAULT_TEMPERATURE = 0.15
DEFAULT_MAX_TOKENS = 512


def build_messages(
    chunks: Sequence[RetrievedChunk],
    history: Sequence,
    summary: str,
    question: str,
    strict: bool = False,
) -> list[LLMMessage]:
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        label = f"[S{i}] [{payload.get('file_name', '')}, trang {payload.get('page')}]"
        context_lines.append(f"{label}\n{payload.get('content', '')[:2000]}")
    context = "\n\n".join(context_lines)
    history_lines = []
    if summary:
        history_lines.append(f"Tóm tắt hội thoại cũ: {summary[:1000]}")
    for turn in list(history)[-6:]:
        role = getattr(turn, "role", "?")
        content = getattr(turn, "content", "")[:800]
        history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) or "(không có)"
    system = SYSTEM_PROMPT + (" Chế độ nghiêm ngặt: tuyệt đối chỉ dùng ngữ cảnh." if strict else "")
    user = (
        f"NGỮ CẢNH ({len(chunks)} đoạn):\n{context}\n\n"
        f"LỊCH SỬ:\n{history_block}\n\n"
        f"CÂU HỎI: {question}\n\n"
        "Trả lời tiếng Việt, và liệt kê các đoạn đã dùng ở cuối."
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]


def citations_of(chunks: Sequence[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for chunk in chunks:
        citation = chunk.citation
        if citation not in seen:
            seen.append(citation)
    return seen


def answer_question(
    retriever: HybridRetriever,
    question: str,
    llm: LLMClient,
    memory: SessionMemory | None = None,
    session_id: str = "default",
    top_k: int = 5,
    threshold: float | None = None,
    filters: dict | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    gate = guardrails.check_input(question)
    if not gate["ok"]:
        return {"answer": FALLBACK_MESSAGE, "citations": [], "fallback": True,
                "reason": gate["reason"], "chunks": [], "standalone_question": question}
    standalone = memory.rewrite(session_id, question, llm) if memory else question.strip()
    retrieval = retriever.retrieve(standalone, top_k=top_k, threshold=threshold, filters=filters)
    if retrieval["fallback"]:
        return {"answer": FALLBACK_MESSAGE, "citations": [], "fallback": True,
                "reason": retrieval.get("reason", "retrieval fallback"),
                "chunks": [], "standalone_question": standalone}
    chunks: list[RetrievedChunk] = retrieval["chunks"]
    history = memory.history(session_id) if memory else []
    summary = memory.summary(session_id) if memory else ""
    messages = build_messages(chunks, history, summary, standalone, strict=gate["strict"])
    raw = llm.complete(messages, temperature=temperature, max_tokens=max_tokens)
    checked = guardrails.check_output(raw, has_context=True)
    if not checked["ok"]:
        return {"answer": checked["answer"], "citations": [], "fallback": True,
                "reason": checked["reason"], "chunks": chunks, "standalone_question": standalone}
    citations = citations_of(chunks)
    answer = checked["answer"] + "\n\nNguồn: " + "; ".join(citations)
    if memory is not None:
        memory.add_turn(session_id, "user", question.strip(), llm)
        memory.add_turn(session_id, "assistant", answer, llm)
    return {"answer": answer, "citations": citations, "fallback": False,
            "chunks": chunks, "standalone_question": standalone, "scores": [c.fused_score for c in chunks]}
