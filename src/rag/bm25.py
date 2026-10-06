"""Local BM25 index (lexical half of Step-2 hybrid search).

Pure-Python, no extra dependency, Vietnamese-friendly tokenization.
Persists to ``bm25.json`` next to vectors so retrieval (Step 3) can load
both sides without Qdrant sparse vectors.

Formula: standard BM25, k1=1.2, b=0.75.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Sequence

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize_vi(text: str) -> list[str]:
    """Lowercase Unicode word tokens. Keeps Vietnamese diacritics."""
    normalized = unicodedata.normalize("NFC", text.lower())
    return [t for t in _TOKEN_RE.findall(normalized) if t]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self._pos: dict[str, set[int]] = {}

    def __len__(self) -> int:
        return len(self.doc_ids)

    def add(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must align")
        for doc_id, text in zip(doc_ids, texts):
            tokens = tokenize_vi(text)
            idx = len(self.doc_ids)
            self.doc_ids.append(str(doc_id))
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            seen = set()
            for tok in tokens:
                if tok not in seen:
                    seen.add(tok)
                    self.df[tok] = self.df.get(tok, 0) + 1
                    self._pos.setdefault(tok, set()).add(idx)
        total = len(self.doc_ids)
        self.avgdl = sum(self.doc_len) / total if total else 0.0

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        # Robertson-Spärck Jones idf, floored at 0.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.doc_ids or top_k <= 0:
            return []
        q_tokens = tokenize_vi(query)
        if not q_tokens:
            return []
        scores: dict[int, float] = {}
        for term in set(q_tokens):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for idx in self._pos.get(term, ()):
                tf = self.doc_tokens[idx].count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / (self.avgdl or 1.0)))
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1) / (denom or 1.0))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(s)) for i, s in ranked]

    def save(self, path: Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "texts_tokens": self.doc_tokens,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1=float(data.get("k1", 1.2)), b=float(data.get("b", 0.75)))
        doc_ids = list(data["doc_ids"])
        tokens_list = list(data["texts_tokens"])
        index.add(doc_ids, [" ".join(toks) for toks in tokens_list])
        # Restore exact token lists (add() re-tokenizes; join+split is stable
        # for our tokenizer output, but keep as-is for safety).
        index.doc_tokens = tokens_list
        return index
