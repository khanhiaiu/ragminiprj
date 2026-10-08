"""Regression checks for issue 3: evidence gating before any generation call."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

import rag.retrieve as retrieval
from rag.answer import answer_question
from rag.embeddings import HashEmbedder
from rag.guardrails import check_output
from rag.llm import FakeLLM
from rag.memory import SessionMemory
from rag.retrieve import HybridRetriever, RetrievedChunk
from rag.store import QdrantStore, ScoredHybridHit


@pytest.fixture
def evidence_store(monkeypatch):
    monkeypatch.delenv("RAG_RELEVANCE_THRESHOLD", raising=False)
    store = QdrantStore(in_memory=True, collection="relevance_test")
    store.ensure_collection()
    embedder = HashEmbedder()
    texts = ["phí thường niên thẻ chuẩn 300000 đồng", "quyền công dân"]
    values = embedder.embed_hybrid(texts)
    store.upsert_hybrid(
        ["fee-1", "law-1"], [v.dense for v in values],
        [v.sparse_indices for v in values], [v.sparse_values for v in values],
        [{"content": texts[0], "doc_id": "fees", "file_name": "fees.pdf", "type": "table"},
         {"content": texts[1], "doc_id": "law", "file_name": "law.pdf", "type": "text"}],
    )
    yield store
    store.client.close()


def test_unconfigured_threshold_fails_closed_even_with_ranked_hits(evidence_store, monkeypatch, tmp_path):
    # Keep this safety check independent of the measured demo deployment policy.
    config = tmp_path / "config.yaml"
    config.write_text("retrieval:\n  relevance_threshold: null\n")
    monkeypatch.setenv("PROJECT_CONFIG", str(config))
    retriever = HybridRetriever(store=evidence_store, embedder=HashEmbedder())
    llm = FakeLLM()
    result = answer_question(retriever, "phí thường niên", llm)
    assert result["fallback"]
    assert result["reason"] == "relevance threshold not configured"
    assert result["chunks"] == [] and result["citations"] == []
    assert llm.calls == []


def test_high_rrf_score_does_not_make_unrelated_question_relevant(evidence_store):
    embedder = HashEmbedder()
    query = "How do black holes evaporate?"
    vector = embedder.embed_hybrid([query])[0]
    raw_hits = evidence_store.search_hybrid_scored(
        vector.dense, vector.sparse_indices, vector.sparse_values
    )
    assert raw_hits and raw_hits[0].fused_score > 0
    assert all(hit.dense_score < 0.5 for hit in raw_hits)
    retriever = HybridRetriever(store=evidence_store, embedder=embedder, threshold=0.5)
    llm = FakeLLM()
    result = answer_question(retriever, query, llm)
    assert result["fallback"] and result["citations"] == []
    assert result["reason"] == "below relevance threshold"
    assert llm.calls == []


def test_only_relevant_evidence_reaches_reranker_and_generation(evidence_store):
    retriever = HybridRetriever(store=evidence_store, embedder=HashEmbedder(), threshold=0.5)
    llm = FakeLLM("300000 đồng")
    reranked = []

    def reranker(query, candidates):
        reranked.extend(candidates)
        return candidates

    result = answer_question(retriever, "phí thường niên", llm, reranker=reranker)
    assert not result["fallback"]
    assert [c.chunk_id for c in reranked] == ["fee-1"]
    assert [c.chunk_id for c in result["chunks"]] == ["fee-1"]
    assert "quyền công dân" not in llm.calls[0][-1].content


def test_request_cannot_weaken_deployment_threshold(evidence_store, monkeypatch):
    monkeypatch.setenv("RAG_RELEVANCE_THRESHOLD", "0.8")
    retriever = HybridRetriever(store=evidence_store, embedder=HashEmbedder())
    result = retriever.retrieve("phí", threshold=0.0)
    assert result["fallback"]
    assert result["threshold"] == 0.8


def test_rejected_followup_never_calls_rewrite_llm(evidence_store, monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("retrieval:\n  relevance_threshold: null\n")
    monkeypatch.setenv("PROJECT_CONFIG", str(config))
    retriever = HybridRetriever(store=evidence_store, embedder=HashEmbedder())
    memory = SessionMemory()
    memory.add_turn("s", "user", "Phí thường niên thẻ chuẩn là bao nhiêu?")
    memory.add_turn("s", "assistant", "300000 đồng")
    before = memory.history("s")
    llm = FakeLLM()
    result = answer_question(retriever, "còn thẻ vàng?", llm, memory, session_id="s")
    assert result["fallback"]
    assert llm.calls == []
    assert memory.history("s") == before


@pytest.mark.parametrize("score, content", [
    (None, "text"), (math.nan, "text"), (math.inf, "text"), (0.9, "  "),
])
def test_unusable_evidence_is_rejected_before_generation(evidence_store, monkeypatch, score, content):
    monkeypatch.setattr(evidence_store, "search_hybrid_scored", lambda *a, **k: [
        ScoredHybridHit("bad", 1.0, score, {"content": content}),
    ])
    retriever = HybridRetriever(store=evidence_store, embedder=HashEmbedder(), threshold=0.5)
    llm = FakeLLM()
    result = answer_question(retriever, "question", llm)
    assert result["fallback"] and not result["citations"]
    assert llm.calls == []


@pytest.mark.parametrize("threshold", [-0.1, math.nan, math.inf, -math.inf])
def test_invalid_thresholds_are_rejected(evidence_store, threshold):
    with pytest.raises(ValueError, match="finite and non-negative"):
        HybridRetriever(store=evidence_store, embedder=HashEmbedder(), threshold=threshold)


def test_empty_backend_response_cannot_trigger_generation():
    class EmptyRetriever:
        def retrieve(self, *args, **kwargs):
            return {"fallback": False, "chunks": []}

    llm = FakeLLM()
    result = answer_question(EmptyRetriever(), "question", llm)
    assert result["fallback"] and result["citations"] == []
    assert llm.calls == []


def test_short_independent_question_does_not_inherit_previous_topic():
    memory = SessionMemory()
    memory.add_turn("s", "user", "Phí thường niên thẻ chuẩn là bao nhiêu?")
    assert memory.rewrite("s", "Thiên văn là gì?", llm=None) == "Thiên văn là gì?"
    assert memory.rewrite("s", "Thời tiết nóng không?", llm=None) == "Thời tiết nóng không?"
    assert "Phí thường niên" in memory.rewrite("s", "còn thẻ vàng thì sao?", llm=None)


@pytest.mark.parametrize("reply", [
    "không đủ thông tin", "Không đủ thông tin.", '"không đủ thông tin".',
    "“không đủ thông tin”",
])
def test_model_fallback_variations_remain_fallback(reply):
    checked = check_output(reply, has_context=True)
    assert not checked["ok"]
    assert checked["answer"] == "không đủ thông tin"


def eval_script():
    spec = importlib.util.spec_from_file_location(
        "test_calibrate_retrieval",
        Path(__file__).resolve().parents[1] / "scripts/eval_retrieval.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_calibration_separates_labels_using_expected_source_evidence():
    script = eval_script()
    items = [{"q": "fees", "expect_contains": "300000", "expect_type": "table"},
             {"q": "unrelated", "expect_fallback": True}]
    results = [
        {"chunks": [RetrievedChunk("fee", 1.0, dense_score=0.8,
                                   payload={"content": "300000", "type": "table"})]},
        {"chunks": [], "best_score": 0.3},
    ]
    measured = script.calibrate_threshold(items, results)
    assert 0.3 < measured["threshold"] < 0.8
    assert measured["score_metric"] == "dense_cosine"
    results[1]["best_score"] = -0.9
    assert 0 < script.calibrate_threshold(items, results)["threshold"] < 0.8
    results[1]["best_score"] = 0.9
    with pytest.raises(ValueError, match="No cosine threshold separates"):
        script.calibrate_threshold(items, results)
    with pytest.raises(ValueError, match="supported and unsupported"):
        script.calibrate_threshold(items[:1], results[:1])
    results[0]["chunks"] = []
    with pytest.raises(ValueError, match="No expected source evidence"):
        script.calibrate_threshold(items, results)


def test_calibration_command_then_deployment_gate(tmp_path, monkeypatch, evidence_store):
    script = eval_script()
    monkeypatch.setattr(retrieval, "get_store", lambda *a, **k: evidence_store)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [
        {"q": "phí thường niên thẻ chuẩn 300000 đồng", "expect_contains": "300000",
         "expect_type": "table"},
        {"q": "đồng", "expect_fallback": True},
    ]}), encoding="utf-8")
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(sys, "argv", [
        "eval_retrieval.py", "--questions", str(questions), "--embedder", "hash",
        "--calibrate-output", str(output), "--as-json",
    ])
    assert script.main() == 0
    policy = json.loads(output.read_text())
    assert policy["questions_sha256"]
    assert 0 < policy["threshold"] < 1
    monkeypatch.setenv("RAG_RELEVANCE_THRESHOLD", str(policy["threshold"]))
    retriever = HybridRetriever.load(embedder_name="hash")
    llm = FakeLLM("300000 đồng")
    rejected = answer_question(retriever, "đồng", llm)
    assert rejected["fallback"] and not llm.calls
    accepted = answer_question(retriever, "phí thường niên thẻ chuẩn 300000 đồng", llm)
    assert not accepted["fallback"] and len(llm.calls) == 1
