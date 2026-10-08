"""Display retrieved images together with their exact document/page citation."""

from __future__ import annotations

import httpx
import streamlit as st

from project_settings import setting


def render_source_images(sources: list[dict], api_url: str) -> None:
    seen = set()
    for source in sources:
        for image in source.get("images", []):
            identity = (source.get("doc_id") or source.get("file_name"), image.get("page"),
                        image.get("image_hash") or image.get("path") or image.get("element_id"))
            if identity in seen:
                continue
            seen.add(identity)
            citation = image.get("citation") or source.get("citation", "Nguồn tài liệu")
            if not image.get("available") or not image.get("url"):
                st.caption(f"{citation} · Ảnh nguồn chưa có trên máy chủ.")
                continue
            try:
                # Fetch on the UI server: API_URL may be private or localhost.
                with httpx.Client(timeout=setting("ui.status_timeout_seconds")) as client:
                    response = client.get(api_url.rstrip("/") + image["url"])
                    response.raise_for_status()
                st.image(response.content, caption=citation, width="stretch")
                if image.get("caption"):
                    st.caption(str(image["caption"])[:300])
            except (httpx.HTTPError, OSError, ValueError):
                st.caption(f"{citation} · Chưa tải được ảnh nguồn. Hãy thử mở lại hội thoại.")


def render_sources(sources: list[dict], api_url: str) -> None:
    if not sources:
        return
    st.markdown("**Nguồn đối chiếu**")
    render_source_images(sources, api_url)
    for index, source in enumerate(sources, 1):
        with st.expander(f"{index}. {source.get('citation', 'Tài liệu')}"):
            st.write(source.get("content", ""))
