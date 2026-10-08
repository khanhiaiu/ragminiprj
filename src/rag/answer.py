"""Answer generation (§4.5): prompt → LLM Qwen3.5 2B → citations + fallback.

Pipeline order per SystemDocuments:
  guardrail-in → rewrite (memory) → hybrid retrieve → [fallback?] →
  prompt (system + top5 labeled context + history/summary + question) →
  LLM (temp 0.1-0.2) → guardrail-out → answer + [file, trang] citations.

Retrieval fallback short-circuits BEFORE any LLM call (spec §4.4).
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

from project_settings import setting

from . import guardrails
from .llm import LLMClient, LLMMessage
from .memory import SessionMemory
from .retrieve import FALLBACK_MESSAGE, HybridRetriever, RerankerFn, RetrievedChunk

SYSTEM_PROMPT = (
    "Bạn là trợ lý RAG tiếng Việt. Chỉ trả lời từ NGỮ CẢNH được cung cấp; "
    "không dùng kiến thức ngoài. Nếu ngữ cảnh không đủ, chỉ trả lời đúng: "
    f'"{FALLBACK_MESSAGE}". Mọi ý trả lời phải bám vào đoạn ngữ cảnh, '
    "không bịa số liệu. Lịch sử chỉ giúp hiểu câu nối tiếp, không thay thế nguồn ngữ cảnh. "
    "Tập trung trả lời câu hỏi mới nhất. Giữ câu trả lời ngắn gọn, tiếng Việt."
)
DEFAULT_TEMPERATURE = setting("llm.temperature")
DEFAULT_MAX_TOKENS = setting("llm.max_tokens")


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
        context_lines.append(f"{label}\n{payload.get('content', '')}")
    context = "\n\n".join(context_lines)
    history_lines = []
    if summary:
        history_lines.append(f"Tóm tắt hội thoại cũ: {summary[:setting('prompt.summary_chars')]}")
    for turn in list(history)[-setting("prompt.history_turns"): ]:
        role = getattr(turn, "role", "?")
        content = getattr(turn, "content", "")[:setting("prompt.history_turn_chars")]
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
    top_k: int | None = None,
    threshold: float | None = None,
    filters: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reranker: RerankerFn | None = None,
    metrics: dict | None = None,
    request_started: float | None = None,
) -> dict[str, Any]:
    started = request_started if request_started is not None else perf_counter()
    metrics = metrics if metrics is not None else {}
    metrics.update(ttft_ms=None, llm_ttft_ms=None, generation_ms=0.0,
                   retrieval_ms=0.0, rerank_ms=0.0, retrieved_count=0,
                   reranked_count=0, selected_count=0)
    gate = guardrails.check_input(question)
    if not gate["ok"]:
        return {"answer": FALLBACK_MESSAGE, "citations": [], "fallback": True,
                "reason": gate["reason"], "chunks": [], "standalone_question": question}
    # Rejected questions must not call even the rewrite LLM. Use deterministic
    # conversation context for retrieval; generation still receives the history.
    standalone = memory.rewrite(session_id, question, llm=None) if memory else question.strip()
    retrieval = retriever.retrieve(
        standalone, top_k=top_k, threshold=threshold, filters=filters, reranker=reranker,
        metrics=metrics,
    )
    usable = [c for c in retrieval["chunks"] if str(c.payload.get("content") or "").strip()]
    if retrieval["fallback"] or not usable:
        return {"answer": FALLBACK_MESSAGE, "citations": [], "fallback": True,
                "reason": retrieval.get("reason", "retrieval fallback"),
                "chunks": [], "standalone_question": standalone}
    chunks: list[RetrievedChunk] = usable
    history = memory.history(session_id) if memory else []
    summary = memory.summary(session_id) if memory else ""
    # Expand the query for search, but ask the model the user's latest question.
    messages = build_messages(chunks, history, summary, question.strip(), strict=gate["strict"])
    generation_started = perf_counter()

    def first_token():
        now = perf_counter()
        metrics["ttft_ms"] = (now - started) * 1000
        metrics["llm_ttft_ms"] = (now - generation_started) * 1000

    options = {"temperature": temperature if temperature is not None else setting("llm.temperature"),
               "max_tokens": max_tokens if max_tokens is not None else setting("llm.max_tokens")}
    try:
        if hasattr(llm, "complete_stream"):
            raw = llm.complete_stream(messages, on_first_token=first_token, **options)
        else:
            raw = llm.complete(messages, **options)
    finally:
        metrics["generation_ms"] = (perf_counter() - generation_started) * 1000
    checked = guardrails.check_output(raw, has_context=bool(chunks))
    if not checked["ok"]:
        return {"answer": checked["answer"], "citations": [], "fallback": True,
                "reason": checked["reason"], "chunks": chunks, "standalone_question": standalone}
    citations = citations_of(chunks)
    answer = checked["answer"] + "\n\nNguồn: " + "; ".join(citations)
    if memory is not None:
        memory.add_exchange(session_id, question.strip(), answer)
    return {"answer": answer, "citations": citations, "fallback": False,
            "chunks": chunks, "standalone_question": standalone, "scores": [c.fused_score for c in chunks]}
