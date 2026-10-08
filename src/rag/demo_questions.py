"""Load grounded demo prompts; keep evaluation answers out of the chat request."""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_demo_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError("Bộ câu hỏi mẫu phải chứa danh sách questions.")
    questions = []
    seen = set()
    for item in data["questions"]:
        if not isinstance(item, dict) or not isinstance(item.get("q"), str):
            raise ValueError("Câu hỏi mẫu phải có nội dung q dạng văn bản.")
        question = item["q"].strip()
        if not question or item.get("expect_fallback") or item.get("demo") is False:
            continue
        # Only prompts with source evidence are offered in the demo picker.
        source = item.get("source")
        if not item.get("expect_contains") or not isinstance(source, dict) or not source.get("doc_id"):
            continue
        if question not in seen:
            questions.append({
                "q": question,
                "category": str(item.get("category") or "Tài liệu"),
                "source": {key: source[key] for key in ("file_name", "page", "sheet") if key in source},
            })
            seen.add(question)
    return questions


def choose_demo_question(questions: list[dict], previous: str | None = None) -> dict:
    if not questions:
        raise ValueError("Chưa có câu hỏi mẫu để chọn.")
    candidates = [item for item in questions if item["q"] != previous] or questions
    return random.choice(candidates)
