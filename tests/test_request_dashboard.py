"""Real request lifecycle and durable metrics without downloading models."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag.conversations import ConversationStore
from rag.embeddings import HashEmbedder
from rag.llm import FakeLLM
from rag.requests import RequestHistory
from rag.retrieve import BgeReranker, HybridRetriever
from rag.store import ScoredHybridHit


@pytest.fixture
def dashboard_api(tmp_path, monkeypatch):
    import api.main as api

    conversations = ConversationStore(tmp_path / "dashboard.sqlite3")
    monkeypatch.setattr(api, "get_conversations", lambda: conversations)
    seen = []

    class Store:
        def search_hybrid_scored(self, *args, **kwargs):
            assert kwargs["top_k"] == 10
            return [ScoredHybridHit(str(i), 1 / (60 + i), .9,
                                   {"content": f"nội dung {i}", "file_name": f"file-{i}.pdf"})
                    for i in range(1, 13)]

    reranker = BgeReranker()

    class Model:
        def predict(self, pairs, **kwargs):
            seen.extend(pairs)
            return list(range(len(pairs)))

    reranker._model = Model()
    retriever = HybridRetriever(store=Store(), embedder=HashEmbedder(), threshold=.4)
    llm = FakeLLM("Nội dung có trong tài liệu.")
    monkeypatch.setattr(api, "get_components", lambda: {
        "retriever": retriever, "llm": llm, "reranker": reranker, "error": None,
    })
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client, conversations, llm, seen


def test_retrieve_ten_rerank_five_and_persist_request(dashboard_api):
    client, conversations, llm, seen = dashboard_api
    response = client.post("/query", json={"question": "nội dung", "workspace_id": "owner"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(seen) == 10
    assert [chunk["chunk_id"] for chunk in result["chunks"]] == ["10", "9", "8", "7", "6"]
    assert [chunk["rerank_score"] for chunk in result["chunks"]] == [9, 8, 7, 6, 5]
    assert "nội dung 10" in llm.calls[0][-1].content
    assert "nội dung 11" not in llm.calls[0][-1].content
    assert response.headers["X-Request-ID"] == result["request_id"]
    reopened = RequestHistory(conversations.path)
    record = reopened.get("owner", result["request_id"])
    assert record["status"] == "success" and record["http_status"] == 200
    assert record["runtime_ms"] > 0 and record["ttft_ms"] is None  # non-streaming fake
    assert record["metrics"]["retrieved_count"] == 10
    assert record["metrics"]["reranked_count"] == 10
    assert record["metrics"]["selected_count"] == 5
    assert record["details"]["answer"] == result["answer"]
    assert client.get(f"/requests/{result['request_id']}",
                      params={"workspace_id": "other"}).status_code == 404
    assert client.get("/requests", params={"workspace_id": "other"}).json()["total"] == 0


def test_fallback_validation_and_model_failure_are_logged(dashboard_api, monkeypatch):
    client, _, llm, _ = dashboard_api
    common = {"question": "nội dung", "workspace_id": "owner"}
    fallback = client.post("/query", json={**common, "threshold": 10}).json()
    assert fallback["fallback"] and fallback["metrics"]["ttft_ms"] is None
    assert llm.calls == []
    assert client.post("/query", json={**common, "top_k": 6}).status_code == 422

    def broken(*args, **kwargs):
        raise ConnectionError("model unavailable")

    monkeypatch.setattr(llm, "complete", broken)
    assert client.post("/query", json=common).status_code == 502
    history = client.get("/requests", params={"workspace_id": "owner"}).json()
    assert history["summary"]["fallback"] == 1
    assert history["summary"]["error"] == 2
    assert history["summary"]["avg_ttft_ms"] is None
    assert all(r["runtime_ms"] >= 0 for r in history["requests"])
    filtered = client.get("/requests", params={"workspace_id": "owner", "status": "error",
                                               "limit": 1, "offset": 1}).json()
    assert filtered["total"] == 2 and len(filtered["requests"]) == 1
    assert filtered["requests"][0]["http_status"] == 422


def test_streaming_ttft_is_observed_before_generation_finishes(dashboard_api, monkeypatch):
    client, _, llm, _ = dashboard_api

    def stream(messages, on_first_token, **kwargs):
        on_first_token()
        return "Nội dung từ model."

    monkeypatch.setattr(llm, "complete_stream", stream, raising=False)
    result = client.post("/query", json={"question": "nội dung", "workspace_id": "owner"}).json()
    timing = result["metrics"]
    assert 0 <= timing["llm_ttft_ms"] <= timing["generation_ms"]
    assert timing["llm_ttft_ms"] <= timing["ttft_ms"] <= timing["runtime_ms"]
    record = client.get(f"/requests/{result['request_id']}", params={"workspace_id": "owner"}).json()
    assert record["ttft_ms"] == timing["ttft_ms"]
    assert client.get("/requests", params={"workspace_id": "owner"}).json()["summary"]["ttft_samples"] == 1


def test_unhandled_server_error_and_malformed_json_are_logged(dashboard_api, monkeypatch):
    import api.main as api

    client, _, _, _ = dashboard_api

    def failure():
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(api, "get_components", failure)
    assert client.post("/query", json={"question": "q", "workspace_id": "owner"}).status_code == 500
    history = client.get("/requests", params={"workspace_id": "owner"}).json()
    assert history["requests"][0]["status"] == "error"
    assert history["requests"][0]["http_status"] == 500
    assert client.post("/query", content="{", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.get("/requests").json()["requests"][0]["http_status"] == 422


def test_unavailable_rag_is_logged_without_breaking_stats(dashboard_api, monkeypatch):
    import api.main as api

    client, _, _, _ = dashboard_api
    monkeypatch.setattr(api, "get_components", lambda: {"retriever": None, "error": "Qdrant unavailable"})
    assert client.get("/stats").status_code == 503
    response = client.post("/query", json={"question": "q", "workspace_id": "owner"})
    assert response.status_code == 503
    request_id = response.headers["X-Request-ID"]
    record = client.get(f"/requests/{request_id}", params={"workspace_id": "owner"}).json()
    assert record["details"]["error"] == "Qdrant unavailable"
    assert record["status"] == "error" and record["ttft_ms"] is None


def test_reranker_rejects_nonfinite_or_missing_scores():
    from rag.retrieve import RetrievedChunk

    candidates = [RetrievedChunk("x", .8, payload={"content": "text"})]
    reranker = BgeReranker()
    for scores in [[], [float("nan")], [float("inf")]]:
        reranker._model = SimpleNamespace(predict=lambda *args, **kwargs: scores)
        with pytest.raises(ValueError, match="invalid scores"):
            reranker.rerank("q", candidates)
