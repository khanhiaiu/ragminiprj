"""Document chat, an icon-only sample picker and a persistent request dashboard.

Run the API first, then:
  streamlit run ui/app.py
Env: RAG_API_URL (default http://localhost:8000)
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_settings import configured_path, setting
from rag.demo_questions import choose_demo_question, load_demo_questions
from ui.dashboard import render_dashboard
from ui.sources import render_sources

API_URL = setting("ui.api_url", "RAG_API_URL").rstrip("/")


def api_post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=setting("ui.query_timeout_seconds")) as client:
        response = client.post(API_URL + path, json=payload)
        response.raise_for_status()
        return response.json()


def api_get(path: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=setting("ui.status_timeout_seconds")) as client:
        response = client.get(API_URL + path, params=params)
        response.raise_for_status()
        return response.json()


def api_delete(path: str) -> None:
    with httpx.Client(timeout=setting("ui.query_timeout_seconds")) as client:
        response = client.delete(API_URL + path, params={"workspace_id": workspace_id})
        response.raise_for_status()


def queue_question(question: str | None = None) -> None:
    question = question if question is not None else st.session_state.get("question_input")
    if isinstance(question, str) and question.strip():
        st.session_state.pending_question = question.strip()
        st.session_state.pop("query_error", None)


st.set_page_config(
    page_title="TP · Document workspace", page_icon="📚", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], [data-testid="stApp"] { font-family: 'Be Vietnam Pro', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    .block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { letter-spacing: -1.4px; color: #142440; font-weight: 800 !important; }
    h2, h3, h4 { color: #142440; letter-spacing: -.4px; }
    [data-testid="stSidebar"] { background: #eef2f8; border-right: 1px solid #dfe6f0; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }
    [data-testid="stChatMessage"] { background: #fff; border: 1px solid #e4eaf2;
        border-radius: 18px; margin-bottom: 1rem; padding: 1.2rem; }
    [data-testid="stButton"] button { border-radius: 10px; min-height: 2.5rem; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e4eaf2;
        border-radius: 16px; padding: 1rem; }
    [data-testid="stMetricValue"] { color: #183b68; }
    .st-key-composer { position: sticky; bottom: 1rem; z-index: 10; background: #fff;
        border-radius: 18px !important; box-shadow: 0 10px 35px #1b335918; padding: .55rem !important; }
    .st-key-composer [data-testid="stPopover"] button,
    .st-key-composer .st-key-samples_busy button { height: 48px; width: 48px;
        border: 0; border-radius: 12px; background: #edf3ff; font-size: 22px; }
    .st-key-composer [data-testid="stPopover"] button [data-testid="stIconMaterial"] { display: none; }
    .st-key-composer [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .5rem; }
    .st-key-composer [data-testid="stColumn"]:first-child { flex: 0 0 48px !important; min-width: 48px !important; }
    .st-key-composer [data-testid="stColumn"]:last-child { flex: 1 1 auto !important; min-width: 0 !important; }
    .st-key-composer [data-testid="stChatInput"] { border-radius: 12px; }
    .st-key-transcript, [data-testid="stLayoutWrapper"]:has(> .st-key-transcript) {
        height: clamp(180px, calc(100dvh - 370px), 760px) !important;
        flex: 0 0 clamp(180px, calc(100dvh - 370px), 760px) !important;
        min-height: 180px !important; max-height: 760px !important; }
    .st-key-session_list [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .25rem; }
    .st-key-session_list [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; min-width: 0 !important; }
    .st-key-session_list [data-testid="stColumn"]:last-child { flex: 0 0 36px !important; min-width: 36px !important; }
    .st-key-session_list button { text-align: left; }
    .workspace-brand { color: #183b68; font-size: 24px; font-weight: 800; letter-spacing: -1px; }
    .workspace-brand span { background: #2563eb; color: white; border-radius: 9px;
        padding: 6px 9px; margin-right: 8px; font-size: 17px; letter-spacing: 0; }
    .empty-note { color: #64748b; padding: 2rem 0 3rem; line-height: 1.8; }
    </style>""", unsafe_allow_html=True,
)

def activate_session(session: dict) -> None:
    st.session_state.session_id = session["id"]
    st.session_state.messages = session.get("messages", [])
    st.session_state.context = session.get("context", {})
    st.session_state.session_updated_at = session["updated_at"]
    st.session_state.loaded_workspace = workspace_id
    st.query_params["session"] = session["id"]
    st.session_state.pop("demo_question", None)
    st.session_state.pop("session_picker", None)
    st.session_state.pop("pending_question", None)
    st.session_state.pop("query_error", None)


# Keep the local demo workspace and active session in the URL across reloads.
workspace_id = st.query_params.get("workspace", "")
if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", workspace_id):
    workspace_id = uuid.uuid4().hex
    st.query_params["workspace"] = workspace_id
if st.session_state.get("loaded_workspace") != workspace_id:
    for key in ("session_id", "messages", "context", "session_picker"):
        st.session_state.pop(key, None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("context", {})
sessions_ready = False

with st.sidebar:
    st.markdown('<div class="workspace-brand"><span>TP</span> Workspace</div>', unsafe_allow_html=True)
    st.caption("TÀI LIỆU · HỘI THOẠI · HIỆU NĂNG")
    view = st.radio("Không gian làm việc", ["Hỏi đáp", "Request dashboard"],
                    label_visibility="collapsed", key="workspace_view")
    st.divider()
    st.subheader("Cuộc trò chuyện")
    try:
        sessions = api_get("/sessions", {"workspace_id": workspace_id})["sessions"]
        requested_id = st.query_params.get("session") or st.session_state.get("session_id")
        if requested_id not in {s["id"] for s in sessions}:
            requested_id = sessions[0]["id"] if sessions else None
        metadata = next((s for s in sessions if s["id"] == requested_id), None)
        if (requested_id != st.session_state.get("session_id") or "session_id" not in st.session_state
                or (metadata and metadata["updated_at"] != st.session_state.get("session_updated_at"))):
            session = (api_get(f"/sessions/{requested_id}", {"workspace_id": workspace_id})
                       if requested_id else api_post("/sessions", {"workspace_id": workspace_id}))
            activate_session(session)
            if not sessions:
                sessions = [session]
        sessions_ready = True
        if st.button("＋ Cuộc trò chuyện mới", key="new_session", use_container_width=True):
            activate_session(api_post("/sessions", {"workspace_id": workspace_id}))
            st.rerun()
        with st.container(key="session_list", height=min(320, max(70, len(sessions) * 55)), border=False):
            for saved in sessions:
                title_column, delete_column = st.columns([7, 1], gap="small")
                with title_column:
                    if st.button(saved["title"], key=f"session_{saved['id']}", width="stretch",
                                 type="primary" if saved["id"] == st.session_state.session_id else "tertiary",
                                 help=saved["title"]):
                        activate_session(api_get(f"/sessions/{saved['id']}", {"workspace_id": workspace_id}))
                        st.rerun()
                with delete_column:
                    if st.button("", icon=":material/delete:", key=f"delete_{saved['id']}",
                                 help=f"Xóa cuộc trò chuyện: {saved['title']}", type="tertiary"):
                        api_delete(f"/sessions/{saved['id']}")
                        if saved["id"] == st.session_state.session_id:
                            remaining = [s for s in sessions if s["id"] != saved["id"]]
                            replacement = (api_get(f"/sessions/{remaining[0]['id']}", {"workspace_id": workspace_id})
                                           if remaining else api_post("/sessions", {"workspace_id": workspace_id}))
                            activate_session(replacement)
                        st.rerun()
        st.caption(f"{len(sessions)} hội thoại · Tự động lưu")
        st.caption("Lưu lại liên kết trên thanh địa chỉ để tiếp tục các hội thoại này.")
        with st.expander("Ngữ cảnh phiên này"):
            context = st.session_state.context
            st.caption(
                f"{context.get('recent_messages', 0)}/{context.get('max_messages', setting('memory.max_turns'))} "
                f"tin nhắn gần nhất · {context.get('recent_chars', 0)}/"
                f"{context.get('max_chars', setting('memory.max_chars'))} ký tự"
            )
            if context.get("summary"):
                st.write("Ghi nhớ từ các lượt trước")
                st.text(context["summary"])
            else:
                st.caption("Chưa có ghi nhớ rút gọn từ các lượt cũ.")
            st.caption("Làm mới ngữ cảnh để đổi chủ đề. Lịch sử trò chuyện vẫn được giữ.")
            if st.button("Làm mới ngữ cảnh", key="reset_context", use_container_width=True,
                         disabled=not (context.get("recent_messages") or context.get("summary"))):
                session = api_post(f"/sessions/{st.session_state.session_id}/context/reset",
                                   {"workspace_id": workspace_id})
                st.session_state.context = session["context"]
                st.rerun()
    except httpx.HTTPError:
        sessions_ready = False
        st.warning("Chưa tải được lịch sử hội thoại. Vui lòng tải lại trang sau ít phút.")
    try:
        health = api_get("/health")
        if health.get("status") == "ok":
            st.caption("● Kho tài liệu sẵn sàng")
        elif health.get("rag_loaded"):
            st.info("Đã kết nối tài liệu. Phần trả lời đang được chuẩn bị.")
        else:
            st.warning("Kho tài liệu chưa sẵn sàng.")
    except httpx.HTTPError:
        st.warning("Chưa kết nối được dịch vụ trả lời. Vui lòng thử lại sau.")

if view == "Request dashboard":
    try:
        render_dashboard(api_get, workspace_id, st.session_state.get("session_id"))
    except httpx.HTTPError:
        st.error("Chưa tải được lịch sử request. Hãy kiểm tra dịch vụ API rồi bấm làm mới.")
    st.stop()

question = st.session_state.pop("pending_question", None)
chat_active = bool(st.session_state.messages or question or st.session_state.get("query_error"))
st.caption("KHÔNG GIAN HỎI ĐÁP")
title_slot = st.empty()
description_slot = st.empty()
if chat_active:
    title_slot.subheader("Cuộc trò chuyện", anchor=False)
else:
    title_slot.title("Tài liệu của bạn. Câu trả lời rõ ràng.")
    description_slot.write("Đặt câu hỏi, đối chiếu thông tin và tiếp tục khám phá — cùng nguồn trích dẫn.")

try:
    demo_questions = load_demo_questions(configured_path("ui.demo_questions"))
except (OSError, ValueError):
    demo_questions = []

# Keep separate placeholders: the landing receives a clearing delta that is
# not reused by the transcript while a slow query is still running.
landing_slot = st.empty()
if not chat_active:
    with landing_slot.container():
        with st.container(border=True):
            st.markdown("#### Bắt đầu từ điều bạn muốn tìm hiểu")
            st.write("Hỏi về quy định, tìm một con số hoặc đối chiếu nội dung trong tài liệu.")
            a, b, c = st.columns(3)
            a.caption("01 · Tìm nội dung liên quan")
            b.caption("02 · Đối chiếu nguồn tài liệu")
            c.caption("03 · Tiếp tục cùng ngữ cảnh")
        st.markdown('<div class="empty-note">Câu trả lời sẽ xuất hiện tại đây.<br>'
                    'Nhấn biểu tượng 💡 cạnh ô nhập để mở câu hỏi mẫu.</div>', unsafe_allow_html=True)
transcript_slot = st.empty()
if chat_active:
    with transcript_slot.container():
        transcript = st.container(key="transcript", height=520, border=False)
        with transcript:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📚"):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        render_sources(message.get("sources", []), API_URL)
            if question:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(question)
            elif st.session_state.get("query_error"):
                failed_question, error_body = st.session_state.query_error
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(failed_question)
                with st.chat_message("assistant", avatar="📚"):
                    st.warning(error_body)

# A chat_input inside columns stays inline, allowing the sample icon on its left.
with st.container(key="composer", border=True):
    sample_column, input_column = st.columns([1, 14], vertical_alignment="center", gap="small")
    with sample_column:
        # Unmount the open popover before generation so it cannot cover the chat.
        picker = None
        if question:
            st.button("💡", key="samples_busy", disabled=True, help="Đang xử lý câu hỏi")
        else:
            picker = st.popover("💡", help="Mở câu hỏi mẫu", disabled=not demo_questions)
        if picker is not None:
            with picker:
                st.markdown("#### Câu hỏi mẫu")
                st.caption("Chọn một chủ đề hoặc đổi câu để tìm gợi ý phù hợp.")
                categories = ["Tất cả", *sorted({q["category"] for q in demo_questions})]
                category = st.selectbox("Chủ đề", categories, key="demo_category")
                pool = [q for q in demo_questions if category == "Tất cả" or q["category"] == category]
                if pool:
                    selected = st.session_state.get("demo_question")
                    if not selected or selected["q"] not in {item["q"] for item in pool}:
                        selected = choose_demo_question(pool)
                    if st.button("Đổi câu", icon=":material/shuffle:", key="random_demo_question", width="stretch"):
                        selected = choose_demo_question(pool, previous=selected["q"])
                    st.session_state.demo_question = selected
                    st.write(selected["q"])
                    source = selected["source"]
                    location = f" · trang {source['page']}" if source.get("page") else ""
                    if source.get("sheet"):
                        location += f" · sheet {source['sheet']}"
                    st.caption(f"{source.get('file_name', '')}{location}")
                    st.button("Hỏi câu này", icon=":material/arrow_forward:", key="ask_demo_question",
                              type="primary", width="stretch", disabled=not sessions_ready,
                              on_click=queue_question, args=(selected["q"],))
                else:
                    st.caption("Chưa có câu hỏi mẫu.")
    with input_column:
        st.chat_input("Nhập câu hỏi của bạn về tài liệu…", key="question_input", on_submit=queue_question,
                                       max_chars=setting("guardrails.max_question_chars"),
                                       disabled=not sessions_ready or bool(question))

st.caption("Mỗi câu trả lời đi kèm nguồn để bạn đối chiếu.")
if question:
    try:
        with transcript, st.chat_message("assistant", avatar="📚"), st.spinner("Đang tìm tài liệu, xếp hạng và soạn câu trả lời…"):
            result = api_post("/query", {
                "session_id": st.session_state.session_id,
                "workspace_id": workspace_id, "question": question,
            })
        session = api_get(f"/sessions/{st.session_state.session_id}", {"workspace_id": workspace_id})
        activate_session(session)
        st.rerun()
    except httpx.TimeoutException:
        body = "Request vẫn có thể đang xử lý. Xem Request dashboard trước khi gửi lại câu hỏi."
    except httpx.HTTPStatusError as exc:
        body = ("Phiên đã thay đổi. Vui lòng tải lại trang rồi gửi lại câu hỏi."
                if exc.response.status_code == 409 else
                "Chưa xử lý được câu hỏi. Xem Request dashboard để kiểm tra request.")
    except httpx.HTTPError:
        body = "Chưa kết nối được dịch vụ trả lời. Vui lòng thử lại sau."
    st.session_state.query_error = (question, body)
    st.rerun()
