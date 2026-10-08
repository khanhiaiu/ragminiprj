import json
from pathlib import Path

import pytest

from rag.bm25 import BM25Index
from rag.embeddings import HashEmbedder
from rag.retrieve import FALLBACK_MESSAGE, HybridRetriever, rrf_fuse
from rag.store import LocalVectorStore


def _build_retriever():
    docs = [
        ("fee-c1", "bieu-phi", "Biểu phí thẻ tín dụng. Phí thường niên thẻ chuẩn 300.000đ.", "table"),
        ("fee-c2", "bieu-phi", "Phí thường niên thẻ vàng 500.000đ, miễn năm đầu.", "text"),
        ("flow-c1", "huong-dan", "Sơ đồ luồng mở thẻ: Bước 1 nộp hồ sơ, Bước 2 thẩm định.", "figure"),
        ("other-c1", "khac", "Chính sách nghỉ phép năm của nhân viên.", "text"),
    ]
    embedder = HashEmbedder()
    vectors = embedder.embed_texts([text for _, _, text, _ in docs])
    store = LocalVectorStore()
    payloads = [
        {"chunk_id": cid, "doc_id": doc, "file_name": f"{doc}.pdf", "page": 1,
         "section": None, "type": typ, "content": text, "content_hash": cid,
         "ingested_at": "2026-01-01T00:00:00+00:00"}
        for (cid, doc, text, typ) in docs
    ]
    store.upsert([c[0] for c in docs], vectors, payloads)
    bm25 = BM25Index()
    bm25.add([c[0] for c in docs], [c[2] for c in docs])
    return HybridRetriever(store=store, bm25=bm25, embedder=embedder)


def test_rrf_prefers_items_in_both_lists():
    fused = rrf_fuse(["a", "b"], ["b", "c"], k=60)
    assert fused["b"] > fused["a"] > fused["c"]  # in-both > rank1-only > rank2-only
    assert rrf_fuse([], [], k=60) == {}


def test_hybrid_finds_fee_table():
    retriever = _build_retriever()
    result = retriever.retrieve("Phí thường niên thẻ chuẩn là bao nhiêu?", top_k=5)
    assert not result["fallback"]
    top = result["chunks"][0]
    assert top.payload["doc_id"] == "bieu-phi"
    assert "300.000" in top.payload["content"]
    assert top.citation == "[bieu-phi.pdf, trang 1]"


def test_filters_and_threshold_fallback():
    retriever = _build_retriever()
    tables = retriever.retrieve("phí thường niên", top_k=5, filters={"type": "table"})
    assert tables["chunks"] and all(c.payload["type"] == "table" for c in tables["chunks"])
    # Absurdly high threshold forces the spec fallback path.
    fb = retriever.retrieve("phí thường niên", top_k=5, threshold=10.0)
    assert fb["fallback"] and fb["chunks"] == []
    assert fb["fallback_message"] == FALLBACK_MESSAGE
    empty = retriever.retrieve("   ")
    assert empty["fallback"]


def test_reranker_hook_and_load_roundtrip(tmp_path):
    import subprocess
    import sys

    parsed = tmp_path / "parsed" / "d1"
    parsed.mkdir(parents=True)
    parsed.joinpath("document.json").write_text(json.dumps({
        "document_id": "d1", "filename": "a.pdf", "file_type": "pdf", "source_path": "a.pdf",
        "pages": [{"page_number": 1}],
        "elements": [{"element_id": "d1-e1", "element_type": "paragraph",
                      "text": "Phí thường niên thẻ chuẩn 300.000đ.", "page_number": 1,
                      "order": 0, "metadata": {}}],
    }), encoding="utf-8")
    rag_dir = tmp_path / "rag"
    proc = subprocess.run(
        [sys.executable, "scripts/ingest.py", "--input", str(tmp_path / "parsed"),
         "--output", str(rag_dir), "--embedder", "hash", "--backend", "local"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr
    retriever = HybridRetriever.load(rag_dir, embedder_name="hash", backend="local")

    def reverse_rerank(query, candidates):
        return list(reversed(candidates))

    plain = retriever.retrieve("phí thường niên", top_k=5)
    reranked = retriever.retrieve("phí thường niên", top_k=5, reranker=reverse_rerank)
    assert [c.chunk_id for c in plain["chunks"]] == list(reversed([c.chunk_id for c in reranked["chunks"]]))

    with pytest.raises(FileNotFoundError):
        HybridRetriever.load(tmp_path / "missing", backend="local")
