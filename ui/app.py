"""Minimal Streamlit chat UI for the local RAG MVP.

Run the API first, then:
  streamlit run ui/app.py
Env: RAG_API_URL (default http://localhost:8000)
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")


def api_post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=120) as client:
        response = client.post(API_URL + path, json=payload)
        response.raise_for_status()
        return response.json()


def api_get(path: str) -> dict:
    with httpx.Client(timeout=10) as client:
        response = client.get(API_URL + path)
        response.raise_for_status()
        return response.json()


st.set_page_config(page_title="TP RAG Chat (local)", layout="wide")
st.title("TP RAG Chat — local (Qwen 4B via llama.cpp)")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"s-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption(f"API: {API_URL}")
    st.caption(f"Session: `{st.session_state.session_id}`")
    try:
        health = api_get("/health")
        st.write("RAG loaded:", health.get("rag_loaded"))
        st.write("LLM:", health.get("llm_backend"))
    except Exception as exc:
        st.error(f"API unreachable: {exc}")
    if st.button("New session"):
        st.session_state.session_id = f"s-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Hỏi về tài liệu (có trích dẫn [file, trang])…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        result = api_post("/query", {"session_id": st.session_state.session_id, "question": question})
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        body = answer + (f"\n\n**Trích dẫn:** {'; '.join(citations)}" if citations else "")
    except Exception as exc:
        body = f"Lỗi gọi API: {exc}"
    st.session_state.messages.append({"role": "assistant", "content": body})
    with st.chat_message("assistant"):
        st.markdown(body)
