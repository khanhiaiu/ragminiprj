"""Opt-in checks for dependencies that require network/services/credentials."""

import io
import os

import pytest
from PIL import Image

from rag.embeddings import BgeM3Embedder
from rag.enrichment.context import ImageAnchor, ImageContext
from rag.enrichment.tokenizer import load_bge_m3_tokenizer
from rag.enrichment.vlm import GeminiVLMClient
from rag.store import QdrantStore

pytestmark = pytest.mark.integration


def _enabled(name):
    if os.environ.get(name) != "1":
        pytest.skip(f"set {name}=1 to run this external integration test")


def test_real_bge_m3_dense_sparse_and_vietnamese_tokenizer():
    _enabled("RUN_BGE_M3_INTEGRATION")
    tokenizer = load_bge_m3_tokenizer()
    text = "Điều 15. Quyền và nghĩa vụ của công dân Việt Nam."
    assert tokenizer.encode(text, add_special_tokens=False)
    result = BgeM3Embedder(device=os.environ.get("BGE_DEVICE", "cpu")).embed_hybrid([text])[0]
    assert len(result.dense) == 1024
    assert result.sparse_indices and result.sparse_values


def test_real_gemini_synthetic_vision_preflight_only():
    _enabled("RUN_GEMINI_VISION_PREFLIGHT")
    if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("GEMINI_MODEL"):
        pytest.skip("GEMINI_API_KEY or GEMINI_MODEL is unset")
    image = Image.new("RGB", (80, 60), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    context = ImageContext(
        document_id="synthetic",
        image_element_id="synthetic",
        before_context="Synthetic test only.",
        after_context="",
        before_token_count=3,
        after_token_count=0,
        total_token_count=3,
        source_spans=[],
        image_anchor=ImageAnchor(element_id="synthetic", order=0),
        tokenizer_fingerprint="synthetic",
        context_hash="synthetic",
    )
    result = GeminiVLMClient().preflight(image_bytes=output.getvalue(), context=context)
    assert result["vision_supported"] is True


def test_real_qdrant_connection():
    _enabled("RUN_QDRANT_INTEGRATION")
    store = QdrantStore(url=os.environ.get("QDRANT_URL", "http://localhost:6333"), collection="test_rag_ingestion_external")
    store.ensure_collection()
    assert store.count() >= 0
