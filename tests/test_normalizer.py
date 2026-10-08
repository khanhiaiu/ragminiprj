import json

import pytest

from document_parser.config import ParserConfig
from document_parser.normalization.normalizer import Normalizer
from document_parser.normalization.ocr_postprocessing import (
    clean_text,
    prepare_blocks,
    process_page,
    review_document,
)
from document_parser.normalization.schema import CanonicalDocument
from document_parser.utils.serialization import document_markdown


@pytest.mark.parametrize(
    "label,expected",
    [
        ("text", "paragraph"),
        ("section_header", "heading"),
        ("picture", "image"),
        ("table", "table"),
        ("future-label", "unknown"),
    ],
)
def test_plain_parser_outputs(label, expected):
    doc = Normalizer().normalize(
        {
            "document_id": "doc",
            "filename": "x",
            "file_type": "pdf",
            "source_path": "/x",
            "elements": [
                {"element_type": label, "text": " Text ", "metadata": {"parser": "example"}}
            ],
        }
    )
    assert isinstance(doc, CanonicalDocument)
    assert doc.elements[0].element_type == expected
    assert json.loads(doc.model_dump_json())["elements"][0]["text"] == "Text"
    assert Normalizer().normalize(doc) == doc


def test_section_and_ids():
    doc = Normalizer().normalize(
        {
            "document_id": "doc",
            "filename": "x",
            "file_type": "pdf",
            "source_path": "/x",
            "elements": [
                {"element_type": "heading", "text": "Title"},
                {"element_type": "paragraph", "text": "Body"},
            ],
        }
    )
    assert doc.elements[0].section_id == doc.elements[1].section_id
    assert len({e.element_id for e in doc.elements}) == 2


def test_punctuation_preserves_numbers_and_repeated_characters():
    assert (
        clean_text("  cần cù,sáng tạo ;  1,234.56 và 19 30, Điều 111  ")
        == "cần cù, sáng tạo; 1,234.56 và 19 30, Điều 111"
    )
    assert clean_text("e\u0301  và  tổ chức") == "é và tổ chức"
    links = "https://example.com/a,b?key=value contact@example.com TP.HCM example.org"
    assert clean_text(links) == links


def test_reading_order_corrects_heading_and_keeps_sections():
    elements = [
        {
            "element_type": "paragraph_title",
            "text": "Điều 1",
            "bbox": [100, 300, 160, 320],
            "metadata": {},
        },
        {
            "element_type": "paragraph_title",
            "text": "CHƯƠNG I",
            "bbox": [200, 250, 500, 280],
            "metadata": {},
        },
    ]
    processed = process_page(elements, 1000, 1000, ParserConfig().postprocessing)
    assert [e["text"] for e in processed] == ["CHƯƠNG I", "Điều 1"]


def test_reading_order_auto_preserves_two_columns():
    elements = [
        {"element_type": "text", "text": "Left first", "bbox": [50, 100, 400, 300], "metadata": {}},
        {
            "element_type": "text",
            "text": "Left second",
            "bbox": [50, 400, 400, 600],
            "metadata": {},
        },
        {
            "element_type": "text",
            "text": "Right first",
            "bbox": [600, 100, 950, 300],
            "metadata": {},
        },
    ]
    assert [
        e["text"] for e in process_page(elements, 1000, 1000, ParserConfig().postprocessing)
    ] == ["Left first", "Left second", "Right first"]


def test_line_reconstruction_does_not_drop_words():
    elements = [
        {
            "element_type": "text",
            "text": "Nhân dân làm chủ",
            "bbox": [0, 0, 400, 40],
            "metadata": {},
        }
    ]
    ocr = {"rec_texts": ["Nhân dân chủ"], "rec_boxes": [[0, 0, 400, 40]], "rec_scores": [0.9]}
    prepare_blocks(elements, ocr, ParserConfig().postprocessing)
    assert elements[0]["text"] == "Nhân dân làm chủ"
    assert "text_changes" not in elements[0]["metadata"]


def ocr_document(texts):
    return Normalizer().normalize(
        {
            "document_id": "doc",
            "filename": "scan.pdf",
            "file_type": "pdf",
            "source_path": "/scan.pdf",
            "pages": [{"page_number": 1}],
            "elements": [
                {
                    "element_type": "paragraph_title"
                    if text.startswith(("Điều", "CHƯƠNG"))
                    else "text",
                    "text": text,
                    "page_number": 1,
                    "metadata": {"parser": "paddleocr"},
                }
                for text in texts
            ],
        }
    )


def test_year_and_spelling_are_suggestions_only():
    doc = ocr_document(["năm 19 30, thưực hiện, phuục vụ và 111 đồng"])
    review_document(doc, ParserConfig().postprocessing)
    assert doc.elements[0].text == "năm 19 30, thưực hiện, phuục vụ và 111 đồng"
    issues = doc.elements[0].metadata["ocr_review"]
    assert any(e.get("suggested") == "năm 1930" for e in issues)
    assert any(e.get("suggestions") == ["thực"] for e in issues)


@pytest.mark.parametrize(
    "confidence,count,corrected", [(0.9, 2, True), (0.9, 1, False), (0.5, 2, False)]
)
def test_numbering_requires_agreeing_crop_ocr(confidence, count, corrected):
    doc = ocr_document(["Điều 10", "Điều 111", "Điều 12"])

    def retry(element, match, scales):
        return [
            {"scale": scales[i], "text": "Điều 11", "confidence": confidence} for i in range(count)
        ]

    review_document(doc, ParserConfig().postprocessing, retry)
    assert doc.elements[1].text == ("Điều 11" if corrected else "Điều 111")
    assert "ocr_review" not in doc.elements[2].metadata
    if corrected:
        assert doc.elements[1].metadata["text_original"] == "Điều 111"


def test_legitimate_numbers_are_never_collapsed():
    doc = ocr_document(["Điều 110", "Điều 111", "Điều 112", "CHƯƠNG II", "CHƯƠNG III"])
    review_document(doc, ParserConfig().postprocessing)
    assert [e.text for e in doc.elements] == [
        "Điều 110",
        "Điều 111",
        "Điều 112",
        "CHƯƠNG II",
        "CHƯƠNG III",
    ]
    assert not any(e.metadata.get("ocr_review") for e in doc.elements)


def test_page_number_retained_in_json_and_excluded_from_markdown():
    doc = ocr_document(["2", "Body text"])
    doc.elements[0].element_type = "footer"
    doc.elements[0].metadata.update(exclude_from_content=True, exclusion_reason="page_number")
    markdown = document_markdown(doc)
    assert "Body text" in markdown and "\n\n2\n" not in markdown
    assert doc.model_dump()["elements"][0]["text"] == "2"
