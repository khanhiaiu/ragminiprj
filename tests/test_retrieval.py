from copy import deepcopy

from document_parser.normalization.normalizer import Normalizer
from document_parser.retrieval.preprocessor import RetrievalPreprocessor, clean_retrieval_text


def canonical_document():
    return Normalizer().normalize(
        {
            "document_id": "legal-doc",
            "filename": "law.pdf",
            "file_type": "pdf",
            "source_path": "/law.pdf",
            "metadata": {"parser": "page_routed_pdf"},
            "pages": [{"page_number": 1}, {"page_number": 2}],
            "elements": [
                {
                    "element_type": "heading",
                    "text": "CHƯƠNG II",
                    "page_number": 1,
                    "metadata": {"parser": "paddleocr"},
                },
                {
                    "element_type": "heading",
                    "text": "QUYỀN CON NGƯỜI,\nQUYỀN VÀ NGHĨA VỤ CƠ BẢN CỦA CÔNG DÂN",
                    "page_number": 1,
                    "metadata": {"parser": "paddleocr"},
                },
                {
                    "element_type": "heading",
                    "text": "Điều 15",
                    "page_number": 1,
                    "metadata": {"parser": "paddleocr"},
                },
                {
                    "element_type": "paragraph",
                    "text": "1. Quyền công dân không tách rời nghĩa vụ công dân.\n"
                    "Nhà nước bảo đảm quyền, lợi ích và 100% giá trị.",
                    "page_number": 2,
                    "section_id": "dieu-15",
                    "metadata": {
                        "parser": "paddleocr",
                        "ocr_lines": [{"text": "large debug payload"}],
                        "text_correction": {
                            "status": "needs_review",
                            "candidate": "Rejected rewrite",
                        },
                    },
                },
                {
                    "element_type": "footer",
                    "text": "2",
                    "page_number": 2,
                    "metadata": {
                        "parser": "paddleocr",
                        "exclude_from_content": True,
                        "exclusion_reason": "page_number",
                    },
                },
                {
                    "element_type": "paragraph",
                    "text": "Không đưa nội dung này vào retrieval",
                    "page_number": 2,
                    "metadata": {"exclude_from_content": True},
                },
                {
                    "element_type": "image",
                    "text": "[IMAGE: p0001.png]",
                    "page_number": 1,
                    "metadata": {"asset_path": "assets/p0001.png"},
                },
            ],
        }
    )


def test_retrieval_preserves_legal_text_context_and_traceability():
    document = canonical_document()
    retrieval = RetrievalPreprocessor().process(document)
    paragraph = next(
        element for element in retrieval.elements if element.element_type == "paragraph"
    )
    assert "Điều 15" in paragraph.text_for_embedding
    assert "CHƯƠNG II" in paragraph.text_for_embedding
    assert "QUYỀN CON NGƯỜI" in paragraph.text_for_embedding
    assert "1. Quyền công dân" in paragraph.text
    assert "100%" in paragraph.text
    assert "quyền, lợi ích" in paragraph.text
    assert paragraph.page_number == 2
    assert paragraph.section_id == "dieu-15"
    assert paragraph.source_element_ids == [paragraph.element_id]
    assert "ocr_lines" not in paragraph.metadata
    assert "text_correction" not in paragraph.metadata
    assert "Rejected rewrite" not in paragraph.text_for_embedding


def test_retrieval_excludes_footer_explicit_exclusions_and_image_placeholders():
    retrieval = RetrievalPreprocessor().process(canonical_document())
    combined = "\n".join(element.text for element in retrieval.elements)
    assert "Không đưa nội dung" not in combined
    assert "[IMAGE:" not in combined
    assert not any(element.element_type == "footer" for element in retrieval.elements)


def test_retrieval_preprocessor_does_not_mutate_canonical_document():
    document = canonical_document()
    before = deepcopy(document.model_dump())
    RetrievalPreprocessor().process(document)
    assert document.model_dump() == before


def test_retrieval_cleaning_is_conservative():
    text = "  Điều 15\nquy-\nền   con người,  số  123.  \x00\n\n\n"
    cleaned = clean_retrieval_text(text)
    assert "Điều 15" in cleaned
    assert "quyền con người" in cleaned
    assert "123" in cleaned
    assert "," in cleaned and "." in cleaned
    assert "\x00" not in cleaned
    assert "\n\n\n" not in cleaned


def test_table_structure_is_preserved_for_future_table_chunker():
    document = Normalizer().normalize(
        {
            "document_id": "table-doc",
            "filename": "table.xlsx",
            "file_type": "xlsx",
            "source_path": "/table.xlsx",
            "elements": [
                {
                    "element_type": "table",
                    "text": "| Cột A | Cột B |\n| --- | --- |\n| Một | 2 |",
                    "metadata": {"parser": "openpyxl", "sheet": "Sheet1", "values": [[1, 2]]},
                }
            ],
        }
    )
    table = RetrievalPreprocessor().process(document).elements[0]
    assert table.element_type == "table"
    assert table.text == document.elements[0].text
    assert table.metadata == {"parser": "openpyxl", "sheet": "Sheet1"}
