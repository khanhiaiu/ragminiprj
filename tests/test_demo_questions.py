"""Demo questions use grounded prompts and submit through the real chat flow."""

import importlib.util
import json
from pathlib import Path

import pytest

from rag.demo_questions import choose_demo_question, load_demo_questions
from rag.retrieve import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]


def test_demo_pool_excludes_fallbacks_answers_and_duplicate_prompts(tmp_path):
    path = tmp_path / "questions.json"
    question = {"q": "Một câu hỏi?", "expect_contains": "answer", "source": {"doc_id": "doc", "file_name": "doc.pdf"}}
    path.write_text(json.dumps({"questions": [
        question, question, {**question, "q": "Ngoài tài liệu?", "expect_fallback": True},
        {"q": "Không có nguồn?", "expect_contains": "answer"},
    ]}))
    pool = load_demo_questions(path)
    assert pool == [{"q": "Một câu hỏi?", "category": "Tài liệu", "source": {"file_name": "doc.pdf"}}]
    assert choose_demo_question(pool, previous=pool[0]["q"]) == pool[0]
    with pytest.raises(ValueError, match="Chưa có"):
        choose_demo_question([])


def test_random_selection_does_not_repeat_previous_question():
    pool = [{"q": "A"}, {"q": "B"}]
    assert choose_demo_question(pool, previous="A")["q"] == "B"


def test_curated_questions_cover_text_tables_figures_and_keep_fictional_scope():
    data = json.loads((ROOT / "eval/questions.json").read_text())
    supported = [q for q in data["questions"] if not q.get("expect_fallback")]
    assert len(load_demo_questions(ROOT / "eval/questions.json")) == len(supported) == 20
    assert {q["expect_type"] for q in supported} == {"text", "table", "figure"}
    assert all(q["source"]["chunk_id"] and q["source"]["content_sha256"] for q in supported)
    assert all("giả định" in q["q"] for q in supported if q["source"]["file_name"].endswith(".xlsx"))


def test_evaluation_requires_answer_to_come_from_expected_document():
    spec = importlib.util.spec_from_file_location("demo_eval", ROOT / "scripts/eval_retrieval.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    item = {"q": "fee", "expect_contains": "300", "expect_type": "table", "source": {"doc_id": "correct"}}
    chunk = RetrievedChunk(chunk_id="other", fused_score=1, dense_score=0.9,
                           payload={"content": "300", "type": "table", "doc_id": "wrong"})
    result = {"chunks": [chunk], "fallback": False}
    assert not evaluation.grade(item, result, 5)["pass"]
    with pytest.raises(ValueError, match="No expected source evidence"):
        evaluation.calibrate_threshold([item, {"q": "outside", "expect_fallback": True}],
                                       [result, {"chunks": [], "best_score": 0.1}])


@pytest.fixture
def ui_app(monkeypatch, tmp_path):
    testing = pytest.importorskip("streamlit.testing.v1")
    import httpx

    from rag.conversations import ConversationStore
    from rag.memory import SessionMemory
    from rag.requests import RequestHistory
    from api.main import session_detail, context_info

    calls = []
    store = ConversationStore(tmp_path / "ui.sqlite3")
    requests = RequestHistory(store.path)

    def response(method, url, data):
        return httpx.Response(200, json=data, request=httpx.Request(method, url))

    def get(client, url, **kwargs):
        route = url.split("8000")[-1]
        workspace = (kwargs.get("params") or {}).get("workspace_id", "local")
        if route == "/health":
            data = {"status": "ok", "rag_loaded": True}
        elif route == "/sessions":
            data = {"sessions": store.list(workspace)}
        elif route == "/requests":
            params = kwargs.get("params") or {}
            data = requests.list(workspace, limit=params.get("limit", 50), offset=params.get("offset", 0),
                                 status=params.get("status"), session_id=params.get("session_id"))
        elif route.startswith("/requests/"):
            data = requests.get(workspace, route.rsplit("/", 1)[-1])
        else:
            data = session_detail(store.get(workspace, route.rsplit("/", 1)[-1]))
        return response("GET", url, data)

    def post(client, url, **kwargs):
        route = url.split("8000")[-1]
        payload = kwargs["json"]
        workspace = payload["workspace_id"]
        if route == "/sessions":
            data = session_detail(store.create(workspace))
        elif route.endswith("/context/reset"):
            data = session_detail(store.reset_context(workspace, route.split("/")[2]))
        else:
            calls.append(payload)
            session_id = payload["session_id"]
            session = store.get(workspace, session_id)
            memory = SessionMemory()
            memory.restore(session_id, session["context"])
            data = {"answer": "Đáp án từ tài liệu.\n\nNguồn: [doc.pdf, trang 1]",
                    "citations": ["[doc.pdf, trang 1]"], "fallback": False}
            memory.add_exchange(session_id, payload["question"], data["answer"])
            store.record_exchange(workspace, session_id, payload["question"], data,
                                  memory.snapshot(session_id), session["revision"])
            data["context"] = context_info(memory.snapshot(session_id))
            request_id = requests.begin(workspace_id=workspace, session_id=session_id,
                                        question=payload["question"], model="test-model", rerank=True)
            requests.finish(request_id, status="success", http_status=200,
                            metrics={"runtime_ms": 2000, "ttft_ms": 1500,
                                     "retrieved_count": 10, "reranked_count": 10, "selected_count": 5},
                            details=data)
        return response("POST", url, data)

    def delete(client, url, **kwargs):
        store.delete(kwargs["params"]["workspace_id"], url.rsplit("/", 1)[-1])
        return response("DELETE", url, {"deleted": True})

    monkeypatch.setattr(httpx.Client, "get", get)
    monkeypatch.setattr(httpx.Client, "post", post)
    monkeypatch.setattr(httpx.Client, "delete", delete)
    return testing.AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=15), calls, store


def test_landing_random_button_previews_then_submits_once(ui_app):
    app, calls, store = ui_app
    app.run()
    assert not app.exception
    first = app.session_state["demo_question"]["q"]
    app.button(key="random_demo_question").click().run()
    selected = app.session_state["demo_question"]["q"]
    assert selected != first
    assert calls == []  # rolling a question does not run retrieval or generation
    app.button(key="ask_demo_question").click().run()
    assert not app.exception
    assert calls == [{"session_id": app.session_state["session_id"],
                      "workspace_id": app.session_state["loaded_workspace"], "question": selected}]
    assert len(app.chat_message) == 2
    app.run()
    assert len(calls) == 1  # rerendering never resubmits the sample
    assert app.session_state["messages"][-1]["content"].count("[doc.pdf, trang 1]") == 1


def test_new_conversation_clears_demo_chat_history(ui_app):
    app, calls, store = ui_app
    app.run()
    old_session = app.session_state["session_id"]
    app.button(key="ask_demo_question").click().run()
    app.button(key="new_session").click().run()
    assert not app.exception
    assert app.session_state["messages"] == []
    assert app.session_state["session_id"] != old_session
    assert len(calls) == 1


def test_missing_sample_file_still_allows_manual_chat(ui_app, monkeypatch, tmp_path):
    import project_settings

    original = project_settings.configured_path
    monkeypatch.setattr(project_settings, "configured_path", lambda name, env=None:
                        tmp_path / "missing.json" if name == "ui.demo_questions" else original(name, env))
    app, calls, store = ui_app
    app.run()
    assert not app.exception
    assert not any(button.key == "ask_demo_question" for button in app.button)
    app.chat_input[0].set_value("Câu hỏi tự nhập").run()
    assert calls[0]["question"] == "Câu hỏi tự nhập"


def test_saved_conversation_can_be_reopened_and_restored_after_page_reload(ui_app):
    testing = pytest.importorskip("streamlit.testing.v1")
    app, calls, _ = ui_app
    app.run()
    first_session = app.session_state["session_id"]
    app.button(key="ask_demo_question").click().run()
    first_question = calls[0]["question"]
    app.button(key="new_session").click().run()
    second_session = app.session_state["session_id"]
    assert second_session != first_session and app.session_state["messages"] == []
    app.button(key=f"session_{first_session}").click().run()
    assert not app.exception
    assert app.session_state["session_id"] == first_session
    assert app.session_state["messages"][0]["content"] == first_question
    assert app.session_state["context"]["recent_messages"] == 2
    reloaded = testing.AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=15)
    reloaded.query_params.update(app.query_params)
    reloaded.run()
    assert not reloaded.exception
    assert reloaded.session_state["session_id"] == first_session
    assert len(reloaded.chat_message) == 2
    assert len(calls) == 1


def test_reset_context_in_sidebar_keeps_chat_transcript(ui_app):
    app, calls, store = ui_app
    app.run()
    app.button(key="ask_demo_question").click().run()
    assert app.session_state["context"]["recent_messages"] == 2
    app.button(key="reset_context").click().run()
    assert not app.exception
    assert app.session_state["context"]["recent_messages"] == 0
    assert len(app.chat_message) == 2 and len(calls) == 1
    saved = store.get(calls[0]["workspace_id"], calls[0]["session_id"])
    assert len(saved["messages"]) == 2 and saved["context"]["turns"] == []


def test_samples_remain_in_icon_popover_and_history_errors_are_recoverable(ui_app, monkeypatch):
    import httpx

    app, _, _ = ui_app
    app.run()
    assert len(app.get("popover")) == 1
    assert app.get("popover")[0].proto.popover.label == "💡"
    assert not any(e.label == "Câu hỏi mẫu" for e in app.expander)
    app.button(key="ask_demo_question").click().run()
    assert len(app.get("popover")) == 1
    with monkeypatch.context() as broken:
        broken.setattr(httpx.Client, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("offline")))
        app.run()
        assert not app.exception and app.chat_input[0].disabled
    app.run()
    assert not app.chat_input[0].disabled


def test_dashboard_shows_saved_request_metrics_and_filters(ui_app):
    app, calls, _ = ui_app
    app.run()
    app.button(key="ask_demo_question").click().run()
    app.radio(key="workspace_view").set_value("Request dashboard").run()
    assert not app.exception
    assert app.title[0].value == "Request dashboard"
    assert [metric.value for metric in app.metric] == ["1", "2.00 s", "1.50 s", "0"]
    assert len(app.dataframe) == 1 and not app.chat_input
    assert app.dataframe[0].value.iloc[0]["Câu hỏi"] == calls[0]["question"]
    status = next(widget for widget in app.selectbox if widget.label == "Trạng thái")
    status.select("error").run()
    assert not app.exception and app.metric[0].value == "0"
    assert len(calls) == 1  # navigation, filters and refresh do not submit queries


@pytest.mark.parametrize("use_sample", [False, True])
def test_first_question_replaces_landing_and_keeps_composer_enabled(ui_app, use_sample):
    app, calls, _ = ui_app
    app.run()
    assert app.title[0].value.startswith("Tài liệu của bạn.")
    if use_sample:
        app.button(key="ask_demo_question").click().run()
    else:
        app.chat_input[0].set_value("Câu hỏi đầu tiên").run()
    assert not app.exception and len(calls) == 1
    assert not app.title
    assert not any("Bắt đầu từ điều" in element.value for element in app.markdown)
    assert len(app.chat_message) == 2 and not app.chat_input[0].disabled
    assert not any(widget.key == "session_picker" for widget in app.selectbox)


def test_conversation_list_deletes_inactive_active_and_last_session(ui_app):
    app, _, store = ui_app
    app.run()
    first = app.session_state["session_id"]
    app.button(key="ask_demo_question").click().run()
    app.button(key="new_session").click().run()
    second = app.session_state["session_id"]
    app.button(key=f"delete_{first}").click().run()
    assert not app.exception and app.session_state["session_id"] == second
    workspace = app.session_state["loaded_workspace"]
    with pytest.raises(LookupError):
        store.get(workspace, first)
    app.button(key="new_session").click().run()
    third = app.session_state["session_id"]
    app.button(key=f"delete_{third}").click().run()
    assert app.session_state["session_id"] == second
    app.button(key=f"delete_{second}").click().run()
    replacement = app.session_state["session_id"]
    assert replacement not in {first, second, third}
    assert app.query_params["session"] == replacement
    assert len(store.list(workspace)) == 1 and app.session_state["messages"] == []


def test_query_error_keeps_question_visible_and_allows_retry(ui_app, monkeypatch):
    import httpx

    app, calls, _ = ui_app
    app.run()
    original = httpx.Client.post

    def fail_query(client, url, **kwargs):
        if url.endswith("/query"):
            raise httpx.ConnectError("offline")
        return original(client, url, **kwargs)

    with monkeypatch.context() as failed:
        failed.setattr(httpx.Client, "post", fail_query)
        app.chat_input[0].set_value("Câu hỏi bị lỗi").run()
        assert not app.exception and not app.title
        assert app.chat_message[0].markdown[0].value == "Câu hỏi bị lỗi"
        assert app.warning and not app.chat_input[0].disabled
    app.chat_input[0].set_value("Thử lại").run()
    assert not app.exception and len(calls) == 1 and len(app.chat_message) == 2


def test_reopened_answer_displays_deduplicated_images_with_page_citations(ui_app, monkeypatch):
    import httpx
    import rag.source_assets as assets
    from PIL import Image

    app, _, store = ui_app
    app.run()
    workspace, session_id = app.session_state["loaded_workspace"], app.session_state["session_id"]
    root = store.path.parent / "corpus"
    root.mkdir()
    path = root / "figure.png"
    Image.new("RGB", (8, 8), "green").save(path)
    monkeypatch.setattr(assets, "configured_path", lambda name: root)
    original_get = httpx.Client.get

    def get_image(client, url, **kwargs):
        if "/images/" in url:
            return httpx.Response(200, content=path.read_bytes(), request=httpx.Request("GET", url))
        return original_get(client, url, **kwargs)

    monkeypatch.setattr(httpx.Client, "get", get_image)
    source = {"citation": "[doc.pdf, trang 2]", "file_name": "doc.pdf", "doc_id": "doc",
              "page": 2, "content": "Sơ đồ nộp hồ sơ", "image": [{"path": str(path), "page": 3}]}
    store.record_exchange(workspace, session_id, "Sơ đồ nào?", {
        "answer": "Theo sơ đồ đính kèm.", "sources": [source, source]},
        {"turns": [], "summary": ""}, store.get(workspace, session_id)["revision"])
    app.run()
    assert not app.exception and len(app.get("image")) == 1
    assert app.get("image")[0].captions == ["[doc.pdf, trang 3]"]
    path.unlink()
    app.session_state["session_updated_at"] = "force-reload"
    app.run()
    assert not app.exception and not app.get("image")
    assert any("[doc.pdf, trang 3] · Ảnh nguồn chưa có" in caption.value for caption in app.caption)
