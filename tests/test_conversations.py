"""Durable history, isolated context, compaction and concurrent update protection."""

import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from rag.conversations import ConversationConflict, ConversationStore
from rag.llm import FakeLLM
from rag.memory import SessionMemory
from rag.retrieve import RetrievedChunk


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    import api.main as api

    store = ConversationStore(tmp_path / "chat.sqlite3")
    llm = FakeLLM("Có thể nộp hồ sơ trực tuyến.")

    class Retriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, question, **kwargs):
            self.queries.append(question)
            if "ngoài tài liệu" in question:
                return {"fallback": True, "chunks": [], "reason": "below relevance threshold"}
            return {"fallback": False, "chunks": [RetrievedChunk(
                "source", 0.7, payload={"content": "Người dân có thể nộp hồ sơ trực tuyến.",
                                "file_name": "Luật hộ tịch.pdf", "page": 6, "type": "text"},
                dense_score=0.8,
            )]}

    retriever = Retriever()
    monkeypatch.setattr(api, "get_conversations", lambda: store)
    monkeypatch.setattr(api, "get_components", lambda: {
        "retriever": retriever, "llm": llm,
        "reranker": SimpleNamespace(rerank=lambda query, chunks: chunks),
    })
    with TestClient(api.app) as client:
        yield client, store, llm, retriever


def create(client, workspace="workspace-a"):
    response = client.post("/sessions", json={"workspace_id": workspace})
    assert response.status_code == 200
    return response.json()["id"]


def ask(client, session, question, workspace="workspace-a"):
    return client.post("/query", json={"session_id": session, "workspace_id": workspace,
                                       "question": question})


def read(client, session, workspace="workspace-a"):
    return client.get(f"/sessions/{session}", params={"workspace_id": workspace})


def test_history_and_followup_context_survive_api_store_restart(api_client, monkeypatch):
    import api.main as api

    client, store, llm, retriever = api_client
    session = create(client)
    first = "Người dân nộp hồ sơ hộ tịch bằng cách nào?"
    assert ask(client, session, first).status_code == 200
    # Re-open SQLite with no process-local conversation memory.
    reopened = ConversationStore(store.path)
    monkeypatch.setattr(api, "get_conversations", lambda: reopened)
    detail = read(client, session).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"] == ["[Luật hộ tịch.pdf, trang 6]"]
    assert detail["context"]["recent_messages"] == 2
    assert ask(client, session, "Còn gửi qua bưu chính thì sao?").status_code == 200
    assert first in retriever.queries[-1]
    assert "Người dân nộp hồ sơ" in llm.calls[-1][1].content
    assert "CÂU HỎI: Còn gửi qua bưu chính thì sao?\n" in llm.calls[-1][1].content
    assert f"CÂU HỎI: {first}" not in llm.calls[-1][1].content
    assert len(read(client, session).json()["messages"]) == 4


def test_sessions_and_workspaces_do_not_share_context(api_client):
    client, _, llm, retriever = api_client
    first, second = create(client), create(client)
    assert ask(client, first, "Hồ sơ đăng ký hộ tịch nộp bằng cách nào?").status_code == 200
    assert ask(client, second, "Còn bưu chính thì sao?").status_code == 200
    assert retriever.queries[-1] == "Còn bưu chính thì sao?"
    assert "(không có)" in llm.calls[-1][1].content
    assert client.get("/sessions", params={"workspace_id": "workspace-b"}).json() == {"sessions": []}
    assert read(client, first, "workspace-b").status_code == 404
    assert ask(client, first, "test", "workspace-b").status_code == 404
    assert client.post(f"/sessions/{first}/context/reset",
                       json={"workspace_id": "workspace-b"}).status_code == 404


def test_reset_preserves_history_and_starts_independent_context(api_client):
    client, _, llm, retriever = api_client
    session = create(client)
    ask(client, session, "Hồ sơ hộ tịch được nộp bằng cách nào?")
    response = client.post(f"/sessions/{session}/context/reset", json={"workspace_id": "workspace-a"})
    detail = response.json()
    assert response.status_code == 200
    assert len(detail["messages"]) == 2
    assert detail["context"]["recent_messages"] == 0 and detail["context"]["summary"] == ""
    ask(client, session, "Còn bưu chính thì sao?")
    assert retriever.queries[-1] == "Còn bưu chính thì sao?"
    assert "(không có)" in llm.calls[-1][1].content
    assert len(read(client, session).json()["messages"]) == 4


def test_fallback_is_saved_but_does_not_pollute_working_context(api_client):
    client, store, llm, _ = api_client
    session = create(client)
    ask(client, session, "Hồ sơ hộ tịch nộp bằng cách nào?")
    before = store.get("workspace-a", session)["context"]
    llm.calls.clear()
    result = ask(client, session, "Câu hỏi ngoài tài liệu").json()
    assert result["fallback"] and llm.calls == []
    saved = store.get("workspace-a", session)
    assert saved["context"] == before
    assert len(saved["messages"]) == 4 and saved["messages"][-1]["fallback"]


def test_generation_error_does_not_save_partial_exchange(api_client, monkeypatch):
    client, store, llm, _ = api_client
    session = create(client)

    def fail(*args, **kwargs):
        raise ConnectionError("model offline")

    monkeypatch.setattr(llm, "complete", fail)
    assert ask(client, session, "Nộp hồ sơ bằng cách nào?").status_code == 502
    saved = store.get("workspace-a", session)
    assert saved["messages"] == [] and saved["context"]["turns"] == []


def test_compaction_obeys_both_budgets_without_losing_full_history(api_client, monkeypatch):
    import project_settings

    client, store, llm, _ = api_client
    original = project_settings.setting
    monkeypatch.setattr("rag.memory.setting", lambda name: 60 if name == "memory.max_chars" else
                        2 if name == "memory.max_turns" else original(name))
    session = create(client)
    for i in range(5):
        assert ask(client, session, f"Nộp hồ sơ hộ tịch lần {i} bằng cách nào?").status_code == 200
    saved = store.get("workspace-a", session)
    assert len(saved["messages"]) == 10
    assert len(saved["context"]["turns"]) <= 2
    assert sum(len(t["content"]) for t in saved["context"]["turns"]) <= 60
    assert saved["context"]["summary"]
    assert len(llm.calls) == 5  # compaction never triggers another model call
    memory = SessionMemory(max_turns=2, max_chars=10)
    memory.add_turn("large", "user", "x" * 100)
    assert len(memory.history("large")[0].content) <= 10


def test_stale_worker_cannot_overwrite_context_reset(tmp_path):
    store = ConversationStore(tmp_path / "chat.sqlite3")
    session = store.create("owner")
    other_worker = ConversationStore(store.path)
    other_worker.reset_context("owner", session["id"])
    with pytest.raises(ConversationConflict):
        store.record_exchange("owner", session["id"], "old question",
                              {"answer": "old answer"}, {"turns": [], "summary": "old"},
                              session["revision"])
    saved = store.get("owner", session["id"])
    assert saved["messages"] == [] and saved["context"]["summary"] == ""


def test_delete_is_workspace_scoped_cascades_messages_and_keeps_request_metrics(api_client):
    from contextlib import closing

    client, store, _, _ = api_client
    session = create(client)
    result = ask(client, session, "Hồ sơ đăng ký hộ tịch nộp bằng cách nào?").json()
    assert client.delete(f"/sessions/{session}", params={"workspace_id": "workspace-b"}).status_code == 404
    assert len(read(client, session).json()["messages"]) == 2
    assert client.delete(f"/sessions/{session}", params={"workspace_id": "workspace-a"}).status_code == 200
    assert read(client, session).status_code == 404
    assert client.get("/sessions", params={"workspace_id": "workspace-a"}).json() == {"sessions": []}
    with closing(store._connect()) as db:
        assert db.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session,)).fetchone()[0] == 0
    assert client.get(f"/requests/{result['request_id']}", params={"workspace_id": "workspace-a"}).status_code == 200


def test_source_images_keep_exact_page_citation_and_survive_reload(api_client, monkeypatch, tmp_path):
    import api.main as api
    import rag.source_assets as assets
    from PIL import Image

    client, store, _, retriever = api_client
    root = tmp_path / "corpus"
    image_path = root / "Tài liệu" / "assets" / "p0003_region_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(image_path)
    monkeypatch.setattr(assets, "configured_path", lambda name: root)
    monkeypatch.setattr(retriever, "retrieve", lambda *a, **k: {
        "fallback": False, "chunks": [RetrievedChunk("figure", 0.8, dense_score=0.9, payload={
            "doc_id": "doc", "file_name": "Tài liệu.pdf", "type": "figure", "page": 2,
            "content": "Nộp hồ sơ trực tuyến.", "image": [{
                "path": "/workspace/old/parsed_test_document/all_documents_gpu/Tài liệu/assets/p0003_region_001.png",
                "page": 3, "element_id": "figure-3", "caption": "Sơ đồ nộp hồ sơ"}],
        })]})
    session = create(client)
    result = ask(client, session, "Nộp hồ sơ trực tuyến bằng cách nào?").json()
    image = result["chunks"][0]["images"][0]
    assert result["chunks"][0]["citation"] == "[Tài liệu.pdf, trang 2]"
    assert image["citation"] == "[Tài liệu.pdf, trang 3]" and image["available"]
    served = client.get(image["url"])
    assert served.status_code == 200 and served.content == image_path.read_bytes()
    assert served.headers["content-type"] == "image/png"
    assert client.get(image["url"].replace("workspace-a", "workspace-b")).status_code == 404
    assert client.get(image["url"].replace("/images/0", "/images/-1")).status_code == 404
    assert client.get(image["url"].replace("/sources/0", "/sources/9")).status_code == 404
    monkeypatch.setattr(api, "get_conversations", lambda: ConversationStore(store.path))
    restored = read(client, session).json()["messages"][-1]["sources"][0]["images"][0]
    assert restored == image
    image_path.unlink()
    unavailable = read(client, session).json()["messages"][-1]["sources"][0]["images"][0]
    assert not unavailable["available"] and unavailable["url"] is None
    assert client.get(image["url"]).status_code == 404
    Image.new("RGB", (8, 8), "blue").save(image_path)
    assert read(client, session).json()["messages"][-1]["sources"][0]["images"][0]["available"]
    client.delete(f"/sessions/{session}", params={"workspace_id": "workspace-a"})
    assert client.get(image["url"]).status_code == 404
    assert image_path.is_file()


def test_asset_resolver_rejects_traversal_symlinks_and_non_images(tmp_path, monkeypatch):
    import rag.source_assets as assets

    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "private.png"
    outside.write_bytes(b"private")
    (root / "escape.png").symlink_to(outside)
    (root / "secret.txt").write_text("private")
    monkeypatch.setattr(assets, "configured_path", lambda name: root)
    for path in [str(outside), "../private.png", "escape.png", "secret.txt", "bad\x00.png"]:
        assert assets.resolve_source_image({"path": path}) is None
    assert assets.source_images({"image_path": "diagram.png", "page": 5}) == [
        {"path": "diagram.png", "page": 5, "image_hash": None}]


def test_existing_sqlite_transcripts_migrate_without_losing_messages(tmp_path):
    import sqlite3
    from contextlib import closing

    path = tmp_path / "legacy.sqlite3"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, citations TEXT NOT NULL DEFAULT '[]',
            fallback INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)""")
        db.execute("INSERT INTO messages(session_id, role, content, created_at) VALUES ('legacy', 'user', 'Câu cũ', '2026')")
    store = ConversationStore(path)
    session = store.create("owner", "legacy")
    assert session["messages"][0]["content"] == "Câu cũ"
    assert session["messages"][0]["sources"] == []
    assert ConversationStore(path).get("owner", "legacy")["messages"] == session["messages"]
