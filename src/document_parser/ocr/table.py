"""Table structure reconstruction independent from text recognition models."""

import html
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from statistics import median
from typing import Any

from ..utils.file_utils import table_markdown
from .models import BBox, LayoutBlock, RecognizedText, bbox_from_geometry


@dataclass
class _HTMLCell:
    row_index: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    header: bool = False
    column_index: int = 0


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.table_depth = 0
        self.row_index = -1
        self.cells: list[_HTMLCell] = []
        self.current: _HTMLCell | None = None
        self.nested = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            if self.table_depth:
                self.nested = True
            self.table_depth += 1
            return
        if self.table_depth != 1:
            return
        if tag == "tr":
            self.row_index += 1
        elif tag in {"td", "th"}:
            values = dict(attrs)
            self.current = _HTMLCell(
                row_index=max(0, self.row_index),
                text="",
                rowspan=_positive_span(values.get("rowspan")),
                colspan=_positive_span(values.get("colspan")),
                header=tag == "th",
            )
        elif tag == "br" and self.current:
            self.current.text += "\n"

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "table":
            self.table_depth = max(0, self.table_depth - 1)
        elif self.table_depth == 1 and tag in {"td", "th"} and self.current:
            self.current.text = _clean_cell_text(self.current.text)
            self.cells.append(self.current)
            self.current = None

    def handle_data(self, data: str) -> None:
        if self.current:
            self.current.text += data


def _positive_span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _clean_cell_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _place_cells(cells: list[_HTMLCell]) -> tuple[int, int]:
    occupied: set[tuple[int, int]] = set()
    rows = columns = 0
    for cell in cells:
        column = 0
        while (cell.row_index, column) in occupied:
            column += 1
        cell.column_index = column
        for row in range(cell.row_index, cell.row_index + cell.rowspan):
            for col in range(column, column + cell.colspan):
                occupied.add((row, col))
        rows = max(rows, cell.row_index + cell.rowspan)
        columns = max(columns, column + cell.colspan)
    return rows, columns


def _intersection_ratio(inner: BBox, outer: BBox) -> float:
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if not area:
        return 0.0
    intersection = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(
        0.0, min(inner[3], outer[3]) - max(inner[1], outer[1])
    )
    return intersection / area


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _geometry_extent(values: Any) -> BBox | None:
    if values is None:
        return None
    boxes = [bbox_from_geometry(value) for value in values]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _text_key(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _reindex_simple_cells(cells: list[dict]) -> bool:
    """Normalize reversed Paddle HTML rows/columns using physical cell positions."""
    if not cells or any(cell.get("bbox") is None for cell in cells):
        return False
    row_centers: dict[int, list[float]] = {}
    column_centers: dict[int, list[float]] = {}
    for cell in cells:
        box = cell["bbox"]
        row_centers.setdefault(cell["row_index"], []).append((box[1] + box[3]) / 2)
        column_centers.setdefault(cell["column_index"], []).append((box[0] + box[2]) / 2)
    row_order = {
        old: new
        for new, old in enumerate(
            sorted(row_centers, key=lambda index: median(row_centers[index]))
        )
    }
    column_order = {
        old: new
        for new, old in enumerate(
            sorted(column_centers, key=lambda index: median(column_centers[index]))
        )
    }
    changed = any(
        row_order[cell["row_index"]] != cell["row_index"]
        or column_order[cell["column_index"]] != cell["column_index"]
        for cell in cells
    )
    for cell in cells:
        cell["row_index"] = row_order[cell["row_index"]]
        cell["column_index"] = column_order[cell["column_index"]]
    cells.sort(key=lambda cell: (cell["row_index"], cell["column_index"]))
    return changed


def _shift_if_crop_local(boxes: list[BBox], table_bbox: BBox) -> tuple[list[BBox], bool]:
    if not boxes:
        return boxes, False
    width, height = table_bbox[2] - table_bbox[0], table_bbox[3] - table_bbox[1]
    local = all(
        box[0] >= -width * 0.1
        and box[1] >= -height * 0.1
        and box[2] <= width * 1.1
        and box[3] <= height * 1.1
        for box in boxes
    )
    if not local:
        return boxes, False
    x, y = table_bbox[:2]
    shifted = [(a + x, b + y, c + x, d + y) for a, b, c, d in boxes]
    original_score = sum(_intersection_ratio(box, table_bbox) for box in boxes)
    shifted_score = sum(_intersection_ratio(box, table_bbox) for box in shifted)
    if shifted_score > original_score + 1e-6:
        return shifted, True
    return boxes, False


def _rows_from_geometry(boxes: list[BBox]) -> list[list[int]]:
    rows: list[dict[str, Any]] = []
    order = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    for index in order:
        box = boxes[index]
        center = (box[1] + box[3]) / 2
        height = max(1.0, box[3] - box[1])
        target = next(
            (
                row
                for row in rows
                if abs(center - row["center"]) <= min(height, row["height"]) * 0.5
            ),
            None,
        )
        if target is None:
            rows.append({"center": center, "height": height, "indices": [index]})
        else:
            target["indices"].append(index)
    return [sorted(row["indices"], key=lambda i: boxes[i][0]) for row in rows]


class TableRecognizer:
    """Restore rows/cells from PP table structure and attach OCR text to each cell."""

    def match_results(
        self, blocks: list[LayoutBlock], table_results: list[dict]
    ) -> list[dict | None]:
        """Match PP table results to layout blocks without trusting list order."""
        assignments: list[dict | None] = [None] * len(blocks)
        used_results: set[int] = set()

        # Prefer explicit region IDs when a Paddle version exposes them.
        for block_index, block in enumerate(blocks):
            if block.block_id is None:
                continue
            for result_index, result in enumerate(table_results):
                region_id = next(
                    (
                        result[name]
                        for name in ("table_region_id", "region_id", "block_id")
                        if result.get(name) is not None
                    ),
                    None,
                )
                if region_id == block.block_id and result_index not in used_results:
                    assignments[block_index] = result
                    used_results.add(result_index)
                    break

        # Paddle 3.7 may return table_res_list in a different order from
        # parsing_res_list. Cell extents remain in page coordinates, so use a
        # one-to-one maximum-overlap assignment.
        candidates = []
        for block_index, block in enumerate(blocks):
            if assignments[block_index] is not None:
                continue
            for result_index, result in enumerate(table_results):
                if result_index in used_results:
                    continue
                result_bbox = self.result_bbox(result)
                if result_bbox is None:
                    continue
                score = _bbox_iou(block.bbox, result_bbox)
                if score > 0:
                    candidates.append((score, block_index, result_index))
        for _, block_index, result_index in sorted(candidates, reverse=True):
            if assignments[block_index] is None and result_index not in used_results:
                assignments[block_index] = table_results[result_index]
                used_results.add(result_index)

        # Compatibility fallback for older/injected payloads without geometry.
        unmatched_blocks = [i for i, value in enumerate(assignments) if value is None]
        unmatched_results = [i for i in range(len(table_results)) if i not in used_results]
        for block_index, result_index in zip(unmatched_blocks, unmatched_results):
            assignments[block_index] = table_results[result_index]
        return assignments

    @staticmethod
    def result_bbox(result: dict) -> BBox | None:
        for name in ("table_bbox", "bbox", "coordinate"):
            if result.get(name) is not None:
                return bbox_from_geometry(result[name])
        return _geometry_extent(result.get("cell_box_list"))

    def process(
        self,
        block: LayoutBlock,
        table_result: dict | None,
        fallback_lines: list[RecognizedText] | None = None,
    ) -> dict[str, Any]:
        result = table_result or {}
        html_source = str(result.get("pred_html") or block.content or "")
        parser = _TableHTMLParser()
        if "<table" in html_source.casefold():
            parser.feed(html_source)
            parser.close()
        cells = parser.cells
        rows, columns = _place_cells(cells)

        raw_boxes = result.get("cell_box_list")
        if raw_boxes is None:
            raw_boxes = []
        cell_boxes = [bbox_from_geometry(box) for box in raw_boxes]
        cell_boxes, boxes_shifted = _shift_if_crop_local(cell_boxes, block.bbox)

        table_ocr = result.get("table_ocr_pred") or {}
        table_ocr_lines = self._table_ocr_lines(table_ocr, block.bbox)
        overall_ocr_lines = [
            line
            for line in (fallback_lines or [])
            if _intersection_ratio(line.bbox, block.bbox) >= 0.5
        ]
        ocr_lines, ocr_source, allow_html_fallback = self._best_ocr_source(
            cell_boxes, table_ocr_lines, overall_ocr_lines
        )

        if not cells and cell_boxes:
            geometry_rows = _rows_from_geometry(cell_boxes)
            cell_boxes = [cell_boxes[index] for row in geometry_rows for index in row]
            for row_index, row in enumerate(geometry_rows):
                for column_index, _ in enumerate(row):
                    cells.append(
                        _HTMLCell(
                            row_index=row_index,
                            column_index=column_index,
                            text="",
                        )
                    )
            rows = max((cell.row_index for cell in cells), default=-1) + 1
            columns = max((cell.column_index for cell in cells), default=-1) + 1

        serialized_cells = []
        cell_confidences = []
        used_text_matches: set[int] = set()
        for index, cell in enumerate(cells):
            bbox = cell_boxes[index] if index < len(cell_boxes) else None
            assigned = self._lines_for_cell(ocr_lines, bbox) if bbox else []
            text = (
                self._join_lines(assigned)
                if assigned
                else (cell.text if allow_html_fallback else "")
            )
            scores = [line.confidence for line in assigned if line.confidence is not None]
            if not scores and cell.text:
                match_index = next(
                    (
                        line_index
                        for line_index, line in enumerate(ocr_lines)
                        if line_index not in used_text_matches
                        and _text_key(line.text) == _text_key(cell.text)
                    ),
                    None,
                )
                if match_index is not None:
                    used_text_matches.add(match_index)
                    if ocr_lines[match_index].confidence is not None:
                        scores = [ocr_lines[match_index].confidence]
            confidence = sum(scores) / len(scores) if scores else None
            if confidence is not None:
                cell_confidences.append(confidence)
            serialized_cells.append(
                {
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "rowspan": cell.rowspan,
                    "colspan": cell.colspan,
                    "bbox": list(bbox) if bbox else None,
                    "text": text,
                    "confidence": confidence,
                    "is_header": cell.header,
                }
            )

        has_merged = any(cell.rowspan > 1 or cell.colspan > 1 for cell in cells)
        complex_table = has_merged or parser.nested
        geometry_reordered = False
        if not complex_table:
            geometry_reordered = _reindex_simple_cells(serialized_cells)
        grid = [["" for _ in range(columns)] for _ in range(rows)]
        for cell in serialized_cells:
            if cell["row_index"] < rows and cell["column_index"] < columns:
                grid[cell["row_index"]][cell["column_index"]] = cell["text"]
        markdown = "" if complex_table else table_markdown(grid)
        if not complex_table and serialized_cells:
            html_source = self._html_from_cells(serialized_cells, rows)
        elif not html_source and cells:
            html_source = self._html_from_cells(serialized_cells, rows)
        confidence = (
            sum(cell_confidences) / len(cell_confidences)
            if cell_confidences
            else block.confidence
        )
        return {
            "type": "table",
            "page": None,
            "bbox": list(block.bbox),
            "rows": rows,
            "columns": columns,
            "cells": serialized_cells,
            "markdown": markdown,
            "html": html_source,
            "confidence": confidence,
            "has_merged_cells": has_merged,
            "has_nested_table": parser.nested,
            "geometry_order_normalized": geometry_reordered,
            "cell_text_source": ocr_source,
            "pred_html_fallback_allowed": allow_html_fallback,
            "cell_coordinates_shifted_from_crop": boxes_shifted,
        }

    @staticmethod
    def _best_ocr_source(
        cell_boxes: list[BBox],
        table_lines: list[RecognizedText],
        overall_lines: list[RecognizedText],
    ) -> tuple[list[RecognizedText], str, bool]:
        if not cell_boxes:
            if table_lines:
                return table_lines, "table_ocr_pred", True
            if overall_lines:
                return overall_lines, "overall_ocr_res", True
            return [], "pred_html", True

        def score(lines: list[RecognizedText]) -> tuple[int, float]:
            matched = [
                line
                for line in lines
                if any(_intersection_ratio(line.bbox, box) >= 0.5 for box in cell_boxes)
            ]
            confidences = [line.confidence for line in matched if line.confidence is not None]
            return len(matched), sum(confidences) / len(confidences) if confidences else 0.0

        table_score = score(table_lines)
        overall_score = score(overall_lines)
        if overall_score >= table_score and overall_score[0] > 0:
            # A strict win means table_ocr_pred is geometrically unreliable;
            # do not reintroduce its unverified pred_html text into empty cells.
            return overall_lines, "overall_ocr_res", overall_score == table_score
        if table_score[0] > 0:
            return table_lines, "table_ocr_pred", True
        if overall_lines:
            return overall_lines, "overall_ocr_res_unmatched", False
        if table_lines:
            return table_lines, "table_ocr_pred_unmatched", True
        return [], "pred_html", True

    @staticmethod
    def _table_ocr_lines(payload: dict, table_bbox: BBox) -> list[RecognizedText]:
        texts = payload.get("rec_texts")
        if texts is None:
            texts = []
        scores = payload.get("rec_scores")
        if scores is None:
            scores = []
        geometries = payload.get("rec_boxes")
        if geometries is None:
            geometries = payload.get("rec_polys")
        if geometries is None:
            geometries = []
        boxes = [bbox_from_geometry(value) for value in geometries]
        boxes, _ = _shift_if_crop_local(boxes, table_bbox)
        return [
            RecognizedText(
                bbox=boxes[index],
                text=str(text),
                confidence=float(scores[index]) if index < len(scores) else None,
            )
            for index, text in enumerate(texts)
            if index < len(boxes) and str(text).strip()
        ]

    @staticmethod
    def _lines_for_cell(lines: list[RecognizedText], bbox: BBox) -> list[RecognizedText]:
        return [line for line in lines if _intersection_ratio(line.bbox, bbox) >= 0.5]

    @staticmethod
    def _join_lines(lines: list[RecognizedText]) -> str:
        ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        return " ".join(line.text.strip() for line in ordered if line.text.strip())

    @staticmethod
    def _html_from_cells(cells: list[dict], rows: int) -> str:
        grouped = [[] for _ in range(rows)]
        for cell in cells:
            grouped[cell["row_index"]].append(cell)
        output = ["<table>"]
        for row in grouped:
            output.append("<tr>")
            for cell in sorted(row, key=lambda value: value["column_index"]):
                attrs = ""
                if cell["rowspan"] > 1:
                    attrs += f' rowspan="{cell["rowspan"]}"'
                if cell["colspan"] > 1:
                    attrs += f' colspan="{cell["colspan"]}"'
                tag = "th" if cell.get("is_header") else "td"
                output.append(f"<{tag}{attrs}>{html.escape(cell['text'])}</{tag}>")
            output.append("</tr>")
        output.append("</table>")
        return "".join(output)
