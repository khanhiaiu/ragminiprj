from __future__ import annotations

import re

import pytest

from document_parser.normalization.schema import CanonicalDocument
from rag.enrichment.context import ImageContextExtractor


class WordTokenizer:
    """Reversible test tokenizer; production uses pinned BGE-M3 AutoTokenizer."""

    name_or_path = "test/word-tokenizer"

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        for token in re.findall(r"\S+", text):
            if token not in self._token_to_id:
                value = len(self._token_to_id) + 1
                self._token_to_id[token] = value
                self._id_to_token[value] = token
            result.append(self._token_to_id[token])
        return result

    def decode(self, token_ids, **kwargs):
        return " ".join(self._id_to_token[value] for value in token_ids)


def words(prefix, count):
    return " ".join(f"{prefix}{i}" for i in range(count))


def document(before, after, *, extra=None, image_order=1):
    elements = [
        {
            "element_id": "before",
            "element_type": "paragraph",
            "text": before,
            "page_number": 1,
            "order": 0,
            "metadata": {},
        },
        {
            "element_id": "image",
            "element_type": "image",
            "text": "",
            "page_number": 2,
            "order": image_order,
            "metadata": {"asset_path": "assets/image.png"},
        },
        {
            "element_id": "after",
            "element_type": "paragraph",
            "text": after,
            "page_number": 3,
            "order": 2,
            "metadata": {},
        },
    ]
    if extra:
        elements.extend(extra)
    return CanonicalDocument.model_validate(
        {
            "document_id": "doc",
            "filename": "doc.pdf",
            "file_type": "pdf",
            "source_path": "doc.pdf",
            "pages": [
                {"page_number": 1},
                {"page_number": 2},
                {"page_number": 3},
            ],
            "elements": elements,
        }
    )


@pytest.mark.parametrize(
    "before,after,expected",
    [
        (150, 150, (100, 100)),
        (30, 200, (30, 170)),
        (250, 0, (200, 0)),
        (12, 17, (12, 17)),
    ],
)
def test_exact_context_budget_allocation(before, after, expected):
    context = ImageContextExtractor(WordTokenizer()).extract(
        document(words("b", before), words("a", after)), "image"
    )
    assert (context.before_token_count, context.after_token_count) == expected
    assert context.total_token_count == sum(expected)
    assert len(context.before_context.split()) == expected[0]
    assert len(context.after_context.split()) == expected[1]


def test_nearest_before_tokens_and_page_boundary_are_preserved():
    context = ImageContextExtractor(WordTokenizer()).extract(
        document(words("b", 130), words("a", 100)), "image"
    )
    assert context.before_context.split()[0] == "b30"
    assert context.before_context.split()[-1] == "b129"
    assert context.pages == [1, 3]
    assert {span.page_number for span in context.source_spans} == {1, 3}


def test_section_boundary_and_heading_path_are_metadata_not_extra_tokens():
    doc = document(words("b", 10), words("a", 10))
    doc.elements.insert(
        1,
        doc.elements[0].model_copy(
            update={
                "element_id": "heading",
                "element_type": "heading",
                "text": "CHƯƠNG II",
                "order": 1,
                "section_id": "chapter-2",
                "metadata": {"heading_level": 1},
            }
        ),
    )
    doc.elements[2] = doc.elements[2].model_copy(update={"order": 2})
    doc.elements[3] = doc.elements[3].model_copy(update={"order": 3})
    context = ImageContextExtractor(WordTokenizer()).extract(doc, "image")
    assert context.heading_path == ["CHƯƠNG II"]
    assert context.before_token_count == 12
    assert context.total_token_count == 22
    assert any(span.section_id == "chapter-2" for span in context.source_spans)


def test_canonical_order_wins_over_page_and_input_order_for_multicolumn_content():
    doc = document(words("left", 3), words("right", 3))
    # Deliberately reverse physical list order. Canonical `order` remains authoritative.
    doc.elements = list(reversed(doc.elements))
    context = ImageContextExtractor(WordTokenizer()).extract(doc, "image")
    assert context.before_context == "left0 left1 left2"
    assert context.after_context == "right0 right1 right2"


def test_vietnamese_tokenization_is_counted_by_the_configured_tokenizer():
    text = "Điều 15 quyền và nghĩa vụ của công dân Việt Nam"
    tokenizer = WordTokenizer()
    context = ImageContextExtractor(tokenizer).extract(document(text, ""), "image")
    assert context.before_token_count == len(tokenizer.encode(text, add_special_tokens=False))
    assert "Việt Nam" in context.before_context


def test_headers_excluded_ocr_and_placeholders_are_not_context():
    extra = [
        {
            "element_id": "header",
            "element_type": "header",
            "text": "CƠ QUAN LẶP LẠI",
            "page_number": 1,
            "order": 0,
            "metadata": {},
        },
        {
            "element_id": "excluded",
            "element_type": "paragraph",
            "text": "DEBUG OCR REJECTED",
            "page_number": 1,
            "order": 0,
            "metadata": {"exclude_from_content": True},
        },
        {
            "element_id": "placeholder",
            "element_type": "paragraph",
            "text": "[IMAGE: x.png]",
            "page_number": 1,
            "order": 0,
            "metadata": {},
        },
    ]
    context = ImageContextExtractor(WordTokenizer()).extract(
        document("legitimate before", "legitimate after", extra=extra), "image"
    )
    combined = context.before_context + context.after_context
    assert "CƠ QUAN" not in combined
    assert "DEBUG" not in combined
    assert "IMAGE" not in combined
    assert context.total_token_count == 4


def test_missing_anchor_fails_instead_of_inventing_position():
    with pytest.raises(ValueError, match="absent"):
        ImageContextExtractor(WordTokenizer()).extract(document("a", "b"), "missing")


def test_duplicate_order_is_flagged_for_review_and_hash_changes_with_context():
    extractor = ImageContextExtractor(WordTokenizer())
    first = extractor.extract(document("one", "two", image_order=0), "image")
    second = extractor.extract(document("changed", "two", image_order=0), "image")
    assert first.needs_review
    assert "duplicate_canonical_order" in first.image_anchor.review_reasons
    assert first.context_hash != second.context_hash
