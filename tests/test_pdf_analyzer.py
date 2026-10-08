import pytest

from document_parser.router.pdf_analyzer import PDFAnalyzer, rectangle_union_area


@pytest.mark.parametrize(
    "text,images,coverage,expected",
    [
        ("Readable native text. " * 100, 0, 0, "digital"),
        ("", 1, 0.98, "scanned"),
        ("Readable native text. " * 100, 2, 0.25, "hybrid"),
        ("\ufffd" * 500, 0, 0, "scanned"),
        ("Short title", 0, 0, "digital"),
        ("page 1", 1, 0.98, "scanned"),
        ("", 0, 0, "digital"),
    ],
)
def test_classification(text, images, coverage, expected):
    result = PDFAnalyzer().classify(
        text=text,
        text_block_count=1 if text else 0,
        image_count=images,
        image_coverage=coverage,
        page_area=500000,
    )
    assert result.page_type == expected


def test_overlap_not_double_counted():
    assert rectangle_union_area([(0, 0, 10, 10), (0, 0, 10, 10), (5, 0, 15, 10)]) == 150
