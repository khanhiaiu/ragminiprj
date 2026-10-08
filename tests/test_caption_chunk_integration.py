import json
from copy import deepcopy

import pytest

from document_parser.normalization.schema import CanonicalDocument
from document_parser.retrieval.models import RetrievalDocument, RetrievalElement
from document_parser.retrieval.preprocessor import RetrievalPreprocessor
from rag.chunking.chunker import ChunkingConfig, TypeAwareChunker
from rag.schemas import Chunk
from scripts.ingest import WhitespaceTokenizer, load_captions, payload


def test_caption_merged_at_image_position_survives_chunk_json_and_payload(tmp_path):
    doc = CanonicalDocument.model_validate({
        "document_id": "doc", "filename": "source.pdf", "file_type": "pdf",
        "source_path": "source.pdf", "elements": [
            {"element_id": "after", "element_type": "paragraph", "text": "Đoạn sau.", "order": 3},
            {"element_id": "figure", "element_type": "image", "text": "OCR cũ", "order": 2,
             "page_number": 2, "metadata": {"asset_path": "assets/figure.png"}},
            {"element_id": "heading", "element_type": "heading", "text": "Quy trình", "order": 0},
            {"element_id": "before", "element_type": "paragraph", "text": "Đoạn trước.", "order": 1},
        ],
    })
    original = deepcopy(doc.model_dump())
    captions = {"figure": {
        "document_id": "doc", "status": "completed", "image_hash": "sha256-image",
        "caption": {"caption": "Sơ đồ có bước duyệt.", "title": "Luồng phê duyệt",
                    "visible_text": ["Tiếp nhận", "Phê duyệt"], "retrieval_useful": True},
    }}
    derived = RetrievalPreprocessor().process(doc, captions, asset_base=tmp_path)
    assert [e.element_id for e in derived.elements] == ["heading", "before", "figure", "after"]
    figure = derived.elements[2]
    assert "Sơ đồ có bước duyệt." in figure.text
    assert "Phê duyệt" in figure.text
    assert figure.heading_path == ["Quy trình"]
    assert doc.model_dump() == original
    chunks = TypeAwareChunker(
        WhitespaceTokenizer(), ChunkingConfig(merge_image_captions=True)
    ).chunk(derived)
    assert len(chunks) == 1
    assert "Đoạn trước." in chunks[0].content
    assert "Sơ đồ có bước duyệt." in chunks[0].content
    assert "Đoạn sau." in chunks[0].content
    restored = Chunk.model_validate_json(chunks[0].model_dump_json())
    assert len(restored.image) == 1
    image = restored.image[0]
    assert image.element_id == "figure"
    assert image.path == str(tmp_path / "assets/figure.png")
    assert image.page == 2
    assert image.image_hash == "sha256-image"
    assert image.visible_text == ["Tiếp nhận", "Phê duyệt"]
    assert payload(restored)["image"] == [image.model_dump(mode="json")]


def test_dry_run_captions_cannot_silently_produce_another_incomplete_index(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text(json.dumps({"image_element_id": "image", "status": "pending", "caption": None}) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        load_captions(path)


def test_caption_with_visible_text_only_is_kept():
    record = {"caption": {"caption": "", "visible_text": ["A → B"], "retrieval_useful": True}}
    content, embedding = RetrievalPreprocessor._figure_text(record, ["Quy trình"])
    assert "A → B" in content
    assert "Quy trình" in embedding


def test_image_links_follow_only_the_caption_tokens_in_split_and_overlap():
    elements = [
        RetrievalElement(
            element_id=name, element_type=kind, document_id="doc", order=order,
            text=text, text_for_embedding=text, source_element_ids=[name],
            asset_path="assets/figure.png" if kind == "image" else None,
        )
        for order, (name, kind, text) in enumerate([
            ("before", "paragraph", " ".join(f"before{i}" for i in range(6))),
            ("image", "image", "visual0 visual1 visual2"),
            ("after", "paragraph", " ".join(f"after{i}" for i in range(12))),
        ])
    ]
    doc = RetrievalDocument(document_id="doc", filename="source.pdf", file_type="pdf",
                            source_path="source.pdf", elements=elements)
    config = ChunkingConfig(target_tokens=6, preferred_min_tokens=5,
                            preferred_max_tokens=6, hard_max_tokens=8,
                            overlap_tokens=2, merge_image_captions=True)
    chunks = TypeAwareChunker(WhitespaceTokenizer(), config).chunk(doc)
    # A fitting caption stays whole and is never copied as partial overlap.
    assert sum(bool(c.image) for c in chunks) == 1
    assert "visual0 visual1 visual2" in next(c.content for c in chunks if c.image)
    for chunk in chunks:
        assert bool(chunk.image) == ("visual" in chunk.content)


def test_structured_caption_facts_reach_embedding_with_their_relationships():
    record = {"caption": {
        "title": "Biểu đồ", "caption": "So sánh hai quý.", "visible_text": ["2025"],
        "steps": [{"from": "Tiếp nhận", "to": "Duyệt"}],
        "relationships": [{"source": "A", "target": "B", "condition": "Đồng ý"}],
        "chart_details": {"series": [{"name": "Quý I", "value": 3214}]},
    }}
    content, embedding = RetrievalPreprocessor._figure_text(record, ["Báo cáo"])
    for value in ('"from": "Tiếp nhận"', '"condition": "Đồng ý"', '"value": 3214', '"name": "Quý I"'):
        assert value in content and value in embedding
