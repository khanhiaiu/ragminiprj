import json
from types import SimpleNamespace

from PIL import Image

from document_parser.ocr import LayoutDetector, OCRDetector, TableRecognizer
from document_parser.ocr.models import LayoutBlock, RecognizedText
from document_parser.parsers.image.paddle_image_parser import PaddleEngine, PaddleImageParser
from document_parser.utils.serialization import write_document


def test_detector_exposes_geometry_without_recognition_text():
    boxes = OCRDetector().detect(
        {
            "dt_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
            "dt_scores": [0.91],
            "rec_texts": ["must not leak into detector"],
        }
    )
    assert boxes[0].bbox == (0, 0, 20, 10)
    assert boxes[0].confidence == 0.91
    assert not hasattr(boxes[0], "text")


def test_layout_detector_keeps_reading_order_and_layout_confidence():
    blocks = LayoutDetector().detect(
        {
            "parsing_res_list": [
                {
                    "block_label": "title",
                    "block_bbox": [10, 10, 190, 40],
                    "block_order": 3,
                }
            ],
            "layout_det_res": {
                "boxes": [
                    {"label": "title", "score": 0.94, "coordinate": [10, 10, 190, 40]}
                ]
            },
        }
    )
    assert blocks[0].reading_order == 3
    assert blocks[0].confidence == 0.94


def test_table_recognizer_maps_vietnamese_ocr_back_to_cells():
    block = LayoutBlock("table", (100, 50, 400, 150), 0.95, 2)
    result = {
        "pred_html": (
            "<table><tr><th>Họ tên</th><th>Tuổi</th><th>Điểm</th></tr>"
            "<tr><td>Khánh</td><td>22</td><td>9.0</td></tr></table>"
        ),
        # Coordinates are relative to the table crop and must be restored.
        "cell_box_list": [
            [0, 0, 100, 50],
            [100, 0, 200, 50],
            [200, 0, 300, 50],
            [0, 50, 100, 100],
            [100, 50, 200, 100],
            [200, 50, 300, 100],
        ],
        "table_ocr_pred": {
            "rec_boxes": [
                [5, 10, 90, 30],
                [110, 10, 180, 30],
                [210, 10, 285, 30],
                [5, 60, 90, 80],
                [110, 60, 180, 80],
                [210, 60, 285, 80],
            ],
            "rec_texts": ["Họ tên", "Tuổi", "Điểm", "Khánh", "22", "9.0"],
            "rec_scores": [0.99, 0.99, 0.99, 0.98, 0.97, 0.96],
        },
    }
    table = TableRecognizer().process(block, result)
    assert table["rows"] == 2
    assert table["columns"] == 3
    assert table["cells"][3] == {
        "row_index": 1,
        "column_index": 0,
        "rowspan": 1,
        "colspan": 1,
        "bbox": [100.0, 100.0, 200.0, 150.0],
        "text": "Khánh",
        "confidence": 0.98,
        "is_header": False,
    }
    assert "| Khánh | 22 | 9.0 |" in table["markdown"]
    assert table["cell_coordinates_shifted_from_crop"]


def test_merged_table_keeps_html_instead_of_lossy_markdown():
    block = LayoutBlock("table", (0, 0, 200, 100), 0.9, 0)
    table = TableRecognizer().process(
        block,
        {
            "pred_html": (
                '<table><tr><td rowspan="2">Tên</td><td>Điểm</td></tr>'
                '<tr><td colspan="2">9.0</td></tr></table>'
            )
        },
    )
    assert table["has_merged_cells"]
    assert table["markdown"] == ""
    assert table["cells"][0]["rowspan"] == 2
    assert table["cells"][2]["column_index"] == 1
    assert "rowspan" in table["html"]


def test_table_results_match_layout_by_geometry_not_result_order():
    blocks = [
        LayoutBlock("table", (0, 0, 100, 40), 0.9, 0, block_id=10),
        LayoutBlock("table", (0, 50, 100, 90), 0.9, 1, block_id=11),
        LayoutBlock("table", (0, 100, 100, 140), 0.9, 2, block_id=12),
    ]
    results = [
        {"pred_html": "third", "cell_box_list": [[0, 100, 100, 140]]},
        {"pred_html": "first", "cell_box_list": [[0, 0, 100, 40]]},
        {"pred_html": "second", "cell_box_list": [[0, 50, 100, 90]]},
    ]
    matched = TableRecognizer().match_results(blocks, results)
    assert [item["pred_html"] for item in matched] == ["first", "second", "third"]


def test_simple_table_normalizes_reversed_physical_row_and_column_order():
    block = LayoutBlock("table", (0, 0, 200, 100), 0.9, 0)
    result = {
        "pred_html": (
            "<table><tr><td>bottom right</td><td>bottom left</td></tr>"
            "<tr><td>top right</td><td>top left</td></tr></table>"
        ),
        "cell_box_list": [
            [100, 50, 200, 100],
            [0, 50, 100, 100],
            [100, 0, 200, 50],
            [0, 0, 100, 50],
        ],
        "table_ocr_pred": {
            # Deliberately unrelated geometry exercises confidence fallback by text.
            "rec_boxes": [[300, 300, 350, 320]],
            "rec_texts": ["top left"],
            "rec_scores": [0.88],
        },
    }
    table = TableRecognizer().process(block, result)
    assert table["geometry_order_normalized"]
    assert [cell["text"] for cell in table["cells"]] == [
        "top left",
        "top right",
        "bottom left",
        "bottom right",
    ]
    assert table["cells"][0]["confidence"] == 0.88
    assert table["markdown"].splitlines()[0] == "| top left | top right |"
    assert table["html"].startswith("<table><tr><td>top left</td><td>top right</td>")


def test_table_prefers_overall_ocr_when_table_ocr_geometry_is_wrong():
    block = LayoutBlock("table", (100, 100, 400, 150), 0.9, 0)
    result = {
        "pred_html": (
            "<table><tr><td>wrong A</td><td>wrong B</td>"
            "<td>spurious HTML</td></tr></table>"
        ),
        "cell_box_list": [
            [100, 100, 200, 150],
            [200, 100, 300, 150],
            [300, 100, 400, 150],
        ],
        "table_ocr_pred": {
            "rec_boxes": [[500, 500, 550, 520]],
            "rec_texts": ["unrelated page heading"],
            "rec_scores": [0.99],
        },
    }
    overall = [
        RecognizedText((110, 110, 190, 140), "correct A", 0.91),
        RecognizedText((210, 110, 290, 140), "correct B", 0.92),
    ]
    table = TableRecognizer().process(block, result, overall)
    assert table["cell_text_source"] == "overall_ocr_res"
    assert not table["pred_html_fallback_allowed"]
    assert [cell["text"] for cell in table["cells"]] == ["correct A", "correct B", ""]
    assert [cell["confidence"] for cell in table["cells"]] == [0.91, 0.92, None]


def test_ppstructure_table_is_structured_in_markdown_and_page_json(tmp_path):
    source = tmp_path / "table.png"
    Image.new("RGB", (300, 100), "white").save(source)

    class Backend:
        def predict(self, path):
            yield SimpleNamespace(
                json={
                    "res": {
                        "parsing_res_list": [
                            {
                                "block_label": "table",
                                "block_bbox": [0, 0, 300, 100],
                                "block_order": 0,
                            }
                        ],
                        "table_res_list": [
                            {
                                "pred_html": (
                                    "<table><tr><td>Tên</td><td>Tuổi</td></tr>"
                                    "<tr><td>Nam</td><td>23</td></tr></table>"
                                ),
                                "cell_box_list": [
                                    [0, 0, 150, 50],
                                    [150, 0, 300, 50],
                                    [0, 50, 150, 100],
                                    [150, 50, 300, 100],
                                ],
                                "table_ocr_pred": {
                                    "rec_boxes": [
                                        [10, 10, 100, 30],
                                        [160, 10, 250, 30],
                                        [10, 60, 100, 80],
                                        [160, 60, 250, 80],
                                    ],
                                    "rec_texts": ["Tên", "Tuổi", "Nam", "23"],
                                    "rec_scores": [0.9, 0.9, 0.9, 0.9],
                                },
                            }
                        ],
                    }
                }
            )

    document = PaddleImageParser(engine=PaddleEngine(backend=Backend())).parse(source)
    table = next(element for element in document.elements if element.element_type == "table")
    assert "| Nam | 23 |" in table.text
    assert table.metadata["table"]["page"] == 1
    output = tmp_path / "output"
    write_document(document, output)
    serialized = json.loads((output / "document.json").read_text())
    block = next(item for item in serialized["pages"][0]["blocks"] if item["type"] == "table")
    assert block["rows"] == 2 and block["columns"] == 2
    assert block["cells"][3]["text"] == "23"
    assert "| Nam | 23 |" in (output / "document.md").read_text()


def test_failed_table_structure_does_not_fall_back_to_flattened_lines(tmp_path):
    source = tmp_path / "table.png"
    Image.new("RGB", (300, 100), "white").save(source)

    class Backend:
        def predict(self, path):
            yield SimpleNamespace(
                json={
                    "res": {
                        "parsing_res_list": [
                            {"block_label": "table", "block_bbox": [0, 0, 300, 100]}
                        ],
                        "table_res_list": [{}],
                        "overall_ocr_res": {
                            "rec_boxes": [[10, 10, 60, 30], [100, 10, 150, 30]],
                            "rec_texts": ["A", "B"],
                            "rec_scores": [0.9, 0.9],
                        },
                    }
                }
            )

    parsed = PaddleEngine(backend=Backend()).predict(source)
    assert len(parsed) == 1
    assert parsed[0]["element_type"] == "table"
    assert not any(item["metadata"].get("layout_fallback") for item in parsed)
