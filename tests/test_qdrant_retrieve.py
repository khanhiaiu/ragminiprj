"""Exercise production retrieval through the real, embedded Qdrant Query API."""

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

import rag.retrieve as retrieval
from rag.answer import answer_question, build_messages
from rag.embeddings import HashEmbedder
from rag.indexing import GenerationIndexer
from rag.llm import FakeLLM
from rag.retrieve import HybridRetriever
from rag.schemas import Chunk
from rag.store import CURRENT_ALIAS, QdrantStore


@pytest.fixture
def published_store(monkeypatch):
    # Explicit fixture policy, not a production relevance default.
    monkeypatch.setenv("RAG_RELEVANCE_THRESHOLD", "0")
    monkeypatch.setattr(retrieval.BgeReranker, "_load", lambda self: SimpleNamespace(
        predict=lambda pairs, **kwargs: list(range(len(pairs), 0, -1)),
    ))
    client = QdrantClient(":memory:")
    store = QdrantStore(client=client, collection="generation_test")
    store.ensure_collection()
    embedder = HashEmbedder()
    texts = ["phí thường niên thẻ chuẩn 300000 đồng", "quyền công dân"]
    embeddings = embedder.embed_hybrid(texts)
    indexer = GenerationIndexer(store)
    indexer.upsert(
        ["fee-1", "law-1"], embeddings,
        [
            {"doc_id": "fees", "file_name": "fees.pdf", "page": 12,
             "type": "table", "content": texts[0], "image": [{"element_id": "image-1"}]},
            {"doc_id": "law", "file_name": "law.pdf", "page": 1,
             "type": "text", "content": texts[1]},
        ],
    )
    indexer.publish(indexer.verify(2, embeddings[0]))
    yield QdrantStore(client=client, collection=CURRENT_ALIAS)
    client.close()


def test_qdrant_load_without_any_local_artifacts(tmp_path, monkeypatch, published_store):
    selected = {}

    def get_store(backend, **kwargs):
        selected.update(backend=backend, **kwargs)
        return published_store

    monkeypatch.setattr(retrieval, "get_store", get_store)
    retriever = HybridRetriever.load(tmp_path / "does-not-exist", embedder_name="hash")
    assert selected["backend"] == "qdrant"
    assert selected["collection"] == CURRENT_ALIAS
    assert retriever.bm25 is None
    found = retriever.retrieve("phí thường niên")
    assert not found["fallback"]
    assert found["chunks"][0].chunk_id == "fee-1"
    assert found["chunks"][0].citation == "[fees.pdf, trang 12]"
    assert found["chunks"][0].payload["image"] == [{"element_id": "image-1"}]


def test_query_encodes_both_vectors_and_delegates_fusion_once(monkeypatch, published_store):
    embedder = HashEmbedder()
    encodings = []
    queries = []
    original_encode = embedder.embed_hybrid
    original_query = published_store.client.query_points

    def encode(texts):
        encodings.append(texts)
        return original_encode(texts)

    def query(**kwargs):
        queries.append(kwargs)
        return original_query(**kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Production must use Qdrant hybrid search, not dense-only/local fusion")

    monkeypatch.setattr(embedder, "embed_hybrid", encode)
    monkeypatch.setattr(embedder, "embed_texts", forbidden)
    monkeypatch.setattr(published_store, "search_dense", forbidden)
    monkeypatch.setattr(published_store.client, "query_points", query)
    retriever = HybridRetriever(store=published_store, embedder=embedder)
    result = retriever.retrieve(
        "phí thường niên", top_k=1, dense_top=3, sparse_top=4,
        filters={"doc_id": "fees", "type": "table"},
    )
    assert not result["fallback"]
    assert encodings == [["phí thường niên"]]
    assert len(queries) == 1
    request = queries[0]
    assert request["collection_name"] == CURRENT_ALIAS
    assert request["query"].fusion.value == "rrf"
    assert [p.using for p in request["prefetch"]] == ["dense", "sparse"]
    assert [p.limit for p in request["prefetch"]] == [3, 4]
    assert all(p.filter == request["query_filter"] for p in request["prefetch"])
    assert request["prefetch"][1].query.indices


def test_filters_apply_before_prefetch_limits(published_store):
    # Numerous better unfiltered hits must not crowd a matching document out
    # before the filter is evaluated.
    embedder = HashEmbedder()
    store = published_store
    indexer = GenerationIndexer(store)
    indexer.upsert(
        [f"other-{i}" for i in range(30)],
        embedder.embed_hybrid(["phí thường niên"] * 30),
        [{"doc_id": "other", "type": "text", "content": "phí thường niên"}] * 30,
    )
    retriever = HybridRetriever(store=store, embedder=embedder)
    result = retriever.retrieve(
        "phí thường niên", top_k=1, dense_top=1, sparse_top=1,
        filters={"doc_id": "fees", "type": "table"},
    )
    assert [c.chunk_id for c in result["chunks"]] == ["fee-1"]


def test_rerank_receives_full_candidate_pool_before_top_k(published_store):
    retriever = HybridRetriever(store=published_store, embedder=HashEmbedder())
    seen = []

    def rerank(query, candidates):
        seen.extend(candidates)
        return list(reversed(candidates))

    result = retriever.retrieve("phí quyền", top_k=1, reranker=rerank)
    assert len(seen) == 2
    assert [c.chunk_id for c in result["chunks"]] == [seen[-1].chunk_id]


@pytest.mark.parametrize("options", [
    {"filters": {"doc_id": "missing"}},
    {"threshold": 10.0},
])
def test_empty_or_below_threshold_results_skip_generation(published_store, options):
    retriever = HybridRetriever(store=published_store, embedder=HashEmbedder())
    llm = FakeLLM()
    result = answer_question(retriever, "phí thường niên", llm, **options)
    assert result["fallback"]
    assert result["citations"] == []
    assert llm.calls == []


def test_model_fallback_is_preserved_without_citations(published_store):
    retriever = HybridRetriever(store=published_store, embedder=HashEmbedder())
    result = answer_question(retriever, "phí thường niên", FakeLLM("không đủ thông tin"))
    assert result["fallback"]
    assert result["answer"] == "không đủ thông tin"
    assert result["citations"] == []


def test_full_retrieved_content_reaches_prompt(published_store):
    retriever = HybridRetriever(store=published_store, embedder=HashEmbedder())
    chunks = retriever.retrieve("phí thường niên")["chunks"]
    chunks[0].payload["content"] = "x " * 1100 + "ANSWER_AT_END"
    assert "ANSWER_AT_END" in build_messages(chunks, [], "", "question")[-1].content


def test_ingest_publish_then_load_and_answer_without_bm25(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "review_ingest", Path(__file__).resolve().parents[1] / "scripts/ingest.py"
    )
    ingest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest)
    client = QdrantClient(":memory:")

    def store_factory(backend, **kwargs):
        assert backend == "qdrant"
        return QdrantStore(client=client, collection=kwargs["collection"])

    monkeypatch.setattr(ingest, "get_store", store_factory)
    monkeypatch.setattr(retrieval, "get_store", store_factory)
    chunk = Chunk(
        chunk_id="fee-1", doc_id="fees", file_name="fees.pdf", page=12,
        type="table", content="phí thường niên 300000 đồng", content_hash="fee-hash",
    )
    chunks = tmp_path / "input.jsonl"
    chunks.write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "ingest.py", "--chunks", str(chunks), "--output", str(output),
        "--backend", "qdrant", "--embedder", "hash", "--collection", "generation_e2e",
    ])
    try:
        assert ingest.main() == 0
        assert json.loads((output / "manifest.json").read_text())["published"]
        assert not (output / "bm25.json").exists()
        retriever = HybridRetriever.load(embedder_name="hash", threshold=0.0)
        result = answer_question(retriever, "phí thường niên", FakeLLM("300000 đồng"))
        assert not result["fallback"]
        assert result["citations"] == ["[fees.pdf, trang 12]"]
    finally:
        client.close()


def test_api_queries_qdrant_without_local_files_and_recovers(monkeypatch, tmp_path, published_store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api.main as api
    from rag.conversations import ConversationStore

    conversations = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setattr(api, "get_conversations", lambda: conversations)
    monkeypatch.setenv("RAG_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("RAG_BACKEND", "qdrant")
    monkeypatch.setenv("RAG_EMBEDDER", "hash")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_alias")
    monkeypatch.setenv("LLM_BACKEND", "fake")
    monkeypatch.setattr(api, "_state", {"retriever": None, "error": None})
    attempts = []

    def factory(backend, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ConnectionError("temporary Qdrant outage")
        assert kwargs["collection"] == "custom_alias"
        return published_store

    monkeypatch.setattr(retrieval, "get_store", factory)
    with TestClient(api.app) as client:
        assert not client.get("/health").json()["rag_loaded"]
        health = client.get("/health").json()
        assert health["rag_loaded"] and health["error"] is None
        stats = client.get("/stats").json()
        assert stats["points"] == 2 and stats["bm25_docs"] == 0
        response = client.post("/query", json={
            "question": "phí thường niên", "doc_id": "fees", "chunk_type": "table",
        })
        assert response.status_code == 200
        result = response.json()
        assert not result["fallback"]
        assert result["citations"] == ["[fees.pdf, trang 12]"]
        assert result["chunks"][0]["image"] == [{"element_id": "image-1"}]
        assert result["chunks"][0]["dense_score"] > 0
        api._state["retriever"].threshold = 0.8
        api._state["llm"].calls.clear()
        rejected = client.post("/query", json={
            "question": "black holes evaporate", "threshold": 0.0, "session_id": "unrelated",
        }).json()
        assert rejected["fallback"] and rejected["citations"] == []
        assert api._state["llm"].calls == []
        invalid = client.post("/query", json={"question": "fees", "threshold": "nan"})
        assert invalid.status_code == 422


def test_read_only_auto_load_does_not_create_collections(monkeypatch, published_store):
    selected = []

    def factory(backend, **kwargs):
        selected.append(backend)
        return published_store

    monkeypatch.setattr(retrieval, "get_store", factory)
    HybridRetriever.load(embedder_name="hash", backend="auto")
    assert selected == ["qdrant"]


@pytest.mark.parametrize("script_name", ["query", "chat", "eval_retrieval"])
def test_cli_defaults_query_published_hybrid_index(
    script_name, monkeypatch, tmp_path, published_store, capsys
):
    spec = importlib.util.spec_from_file_location(
        f"test_cli_{script_name}",
        Path(__file__).resolve().parents[1] / f"scripts/{script_name}.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    selected = {}

    def store_factory(backend, **kwargs):
        selected.update(backend=backend, **kwargs)
        return published_store

    def embedder_factory(name, **kwargs):
        assert name == "bge-m3"
        return HashEmbedder()

    monkeypatch.setattr(retrieval, "get_store", store_factory)
    monkeypatch.setattr(retrieval, "make_embedder", embedder_factory)
    if script_name == "eval_retrieval":
        questions = tmp_path / "questions.json"
        questions.write_text(json.dumps({"questions": [{
            "q": "phí thường niên", "expect_contains": "300000", "expect_type": "table",
        }]}), encoding="utf-8")
        args = ["--questions", str(questions), "--as-json"]
    else:
        args = ["--query", "phí thường niên"]
    if script_name == "chat":
        def llm_factory(backend, **kwargs):
            assert backend == "server"
            return FakeLLM("300000 đồng")

        monkeypatch.setattr(script, "make_llm", llm_factory)
    monkeypatch.setattr(sys, "argv", [f"{script_name}.py", *args])
    assert script.main() == 0
    assert selected["backend"] == "qdrant"
    assert selected["collection"] == CURRENT_ALIAS
    assert "fees.pdf" in capsys.readouterr().out
