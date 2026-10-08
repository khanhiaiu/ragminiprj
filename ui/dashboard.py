"""Request dashboard backed by the API's persistent request history."""

from __future__ import annotations

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from project_settings import setting
from ui.sources import render_source_images


STATUS_LABELS = {"running": "Đang xử lý", "success": "Thành công",
                 "fallback": "Thiếu thông tin", "error": "Lỗi"}


def seconds(value):
    return f"{value / 1000:,.2f} s" if value is not None else "—"


def render_dashboard(api_get, workspace_id: str, session_id: str | None):
    st.caption("QUAN SÁT & HIỆU NĂNG")
    st.title("Request dashboard")
    st.write("Theo dõi từng câu hỏi, thời gian xử lý và hành trình tìm câu trả lời.")
    status_col, session_col, refresh_col = st.columns([3, 3, 1], vertical_alignment="bottom")
    with status_col:
        status = st.selectbox("Trạng thái", ["all", *STATUS_LABELS],
                              format_func=lambda v: "Tất cả trạng thái" if v == "all" else STATUS_LABELS[v])
    with session_col:
        current_only = st.toggle("Chỉ hội thoại hiện tại", disabled=not session_id)
    with refresh_col:
        st.button("Làm mới", icon=":material/refresh:", width="stretch")
    filters = (status, session_id if current_only else None)
    if st.session_state.get("dashboard_filters") != filters:
        st.session_state.dashboard_filters = filters
        st.session_state.dashboard_page = 1

    @st.fragment(run_every=5)
    def request_history():
        params = {"workspace_id": workspace_id, "limit": 50,
                  "offset": (st.session_state.get("dashboard_page", 1) - 1) * 50}
        if status != "all":
            params["status"] = status
        if current_only and session_id:
            params["session_id"] = session_id
        try:
            data = api_get("/requests", params)
        except httpx.HTTPError:
            st.error("Chưa kết nối được lịch sử request. Dashboard sẽ thử lại sau vài giây.")
            return
        summary = data["summary"]
        a, b, c, d = st.columns(4)
        a.metric("Tổng request", f"{summary['total']:,}", help="Theo bộ lọc, trên toàn bộ lịch sử đã lưu.")
        b.metric("Runtime trung bình", seconds(summary["avg_runtime_ms"]),
                 help="Từ lúc API nhận câu hỏi đến khi xử lý xong, gồm cả thời gian chờ.")
        c.metric("TTFT trung bình", seconds(summary["avg_ttft_ms"]),
                 help="Từ lúc API nhận câu hỏi đến nội dung đầu tiên từ model. Không phải thời điểm trình duyệt hiển thị.")
        d.metric("Request lỗi", summary["error"],
                 help=f"{summary['fallback']} request thiếu thông tin · {summary['running']} đang xử lý.")
        st.caption(f"Tự làm mới mỗi 5 giây · {summary['ttft_samples']} mẫu TTFT · "
                   "Request không sinh token hiển thị —, không tính là 0.")
        rows = data["requests"]
        if not rows:
            with st.container(border=True):
                st.markdown("### Chưa có request trong bộ lọc này")
                st.write("Gửi một câu hỏi ở mục Hỏi đáp để bắt đầu theo dõi.")
            return
        frame = pd.DataFrame([
            {"id": row["id"], "Thời gian": row["started_at"], "Câu hỏi": row["question"],
             "Trạng thái": STATUS_LABELS[row["status"]],
             "Runtime (s)": row["runtime_ms"] / 1000 if row["runtime_ms"] is not None else None,
             "TTFT (s)": row["ttft_ms"] / 1000 if row["ttft_ms"] is not None else None,
             "Retrieve": row["metrics"].get("retrieved_count", 0),
             "Rerank": row["metrics"].get("reranked_count", 0),
             "Top": row["metrics"].get("selected_count", 0)} for row in rows
        ])
        frame["Thời gian"] = pd.to_datetime(frame["Thời gian"], utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
        with st.container(border=True):
            st.markdown("#### Thời gian phản hồi")
            st.caption("Runtime và TTFT của các request trên trang hiện tại, đơn vị giây.")
            chart_data = frame[["Thời gian", "Runtime (s)", "TTFT (s)"]].melt(
                id_vars="Thời gian", var_name="Chỉ số", value_name="Giây")
            chart = alt.Chart(chart_data).mark_line(
                point=alt.OverlayMarkDef(filled=True, size=60),
            ).encode(
                x=alt.X("Thời gian:T", axis=alt.Axis(format="%H:%M:%S"), title=None),
                y=alt.Y("Giây:Q", title="Giây", scale=alt.Scale(zero=True)),
                color=alt.Color("Chỉ số:N", scale=alt.Scale(
                    domain=["Runtime (s)", "TTFT (s)"], range=["#2563eb", "#14b8a6"]),
                    legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[alt.Tooltip("Thời gian:T", format="%d/%m %H:%M:%S"),
                         "Chỉ số:N", alt.Tooltip("Giây:Q", format=".2f")],
            ).properties(height=220).interactive()
            st.altair_chart(chart, width="stretch")
        st.markdown("#### Lịch sử request")
        st.caption("Chọn một dòng để xem câu trả lời, các bước xử lý và tài liệu đã dùng.")
        selection = st.dataframe(
            frame.drop(columns=["id"]), hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row",
            key=f"requests_table_{status}_{current_only}_{params['offset']}",
            column_config={
                "Thời gian": st.column_config.DatetimeColumn(format="DD/MM HH:mm:ss"),
                "Câu hỏi": st.column_config.TextColumn(width="large"),
                "Runtime (s)": st.column_config.NumberColumn(format="%.2f"),
                "TTFT (s)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        pages = max(1, (data["total"] + 49) // 50)
        st.number_input("Trang", min_value=1, max_value=pages, step=1, key="dashboard_page")
        st.caption(f"{data['total']} request · 50 request mỗi trang · Lưu cùng cơ sở dữ liệu hội thoại.")
        selected_rows = selection.selection.rows
        if selected_rows and selected_rows[0] < len(rows):
            record = api_get(f"/requests/{rows[selected_rows[0]]['id']}", {"workspace_id": workspace_id})
            render_detail(record)

    request_history()


def render_detail(record):
    with st.container(border=True):
        st.markdown("#### Chi tiết request")
        st.write(record["question"])
        st.caption(f"{record['id']} · {STATUS_LABELS[record['status']]} · HTTP {record['http_status'] or '—'}")
        a, b, c = st.columns(3)
        a.metric("Runtime", seconds(record["runtime_ms"]))
        b.metric("TTFT toàn luồng", seconds(record["ttft_ms"]))
        c.metric("TTFT riêng model", seconds(record["metrics"].get("llm_ttft_ms")))
        timings = record["metrics"]
        st.dataframe([
            {"Bước": label, "Thời gian (s)": timings.get(key, 0) / 1000}
            for label, key in [("Truy xuất", "retrieval_ms"), ("Rerank", "rerank_ms"),
                               ("Sinh câu trả lời", "generation_ms")]
        ], hide_index=True, width="stretch")
        st.caption(f"Model: {record['model']} · Rerank: {'Bật' if record['rerank'] else 'Tắt'}")
        details = record["details"]
        if details.get("error"):
            st.error(details["error"])
        if details.get("answer"):
            st.markdown(details["answer"])
        if details.get("reason"):
            st.caption(f"Lý do: {details['reason']}")
        for i, chunk in enumerate(details.get("chunks", []), 1):
            with st.expander(f"{i}. {chunk['citation']}"):
                score = chunk.get("rerank_score")
                st.caption(f"Điểm rerank: {score:.4f}" if score is not None else "Chưa có điểm rerank")
                st.write(chunk.get("content", ""))
                render_source_images([chunk], setting("ui.api_url", "RAG_API_URL"))
