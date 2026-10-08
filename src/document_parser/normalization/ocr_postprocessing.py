"""Conservative OCR cleanup grounded in line geometry; ambiguous edits need review."""

import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher


def compact(text):
    return "".join(unicodedata.normalize("NFC", text).split())


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / area_a if area_a else 0, intersection / min(area_a, area_b) if min(
        area_a, area_b
    ) else 0


def record_text_change(element, text, reason):
    if text == element["text"]:
        return
    metadata = element["metadata"]
    metadata.setdefault("text_original", element["text"])
    changes = metadata.setdefault("text_changes", [])
    changes.append({"reason": reason, "before": element["text"], "after": text})
    element["text"] = text
    metadata["text_corrected"] = text


def clean_text(text, punctuation=True):
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text).strip()
    if punctuation:
        # Leave URLs, email addresses, domains and dotted abbreviations intact.
        parts = re.split(
            r"((?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.\w+|"
            r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b)",
            text,
        )
        for index in range(0, len(parts), 2):
            part = re.sub(r" +([,;:.!?])", r"\1", parts[index])
            # Following letters distinguish punctuation from decimal/group separators.
            part = re.sub(r"([,;:!?])(?=[^\W\d_])", r"\1 ", part)
            parts[index] = re.sub(r"\.(?=[^\W\d_])", ". ", part)
        text = "".join(parts)
    return text


def collect_lines(ocr):
    lines = []
    for index, text in enumerate(ocr.get("rec_texts", [])):
        if not text.strip():
            continue
        boxes = ocr.get("rec_boxes", [])
        polygons = ocr.get("rec_polys", [])
        if index < len(boxes):
            box = [float(v) for v in boxes[index]]
        elif index < len(polygons) and len(polygons[index]):
            polygon = polygons[index]
            box = [
                min(p[0] for p in polygon),
                min(p[1] for p in polygon),
                max(p[0] for p in polygon),
                max(p[1] for p in polygon),
            ]
        else:
            continue
        scores = ocr.get("rec_scores", [])
        lines.append(
            {
                "text": text,
                "bbox": box,
                "coordinate_unit": "px",
                "confidence": float(scores[index]) if index < len(scores) else None,
            }
        )
    return lines


def join_lines(lines, row_tolerance):
    rows = []
    for line in sorted(
        lines, key=lambda line: ((line["bbox"][1] + line["bbox"][3]) / 2, line["bbox"][0])
    ):
        box = line["bbox"]
        center = (box[1] + box[3]) / 2
        row = rows[-1] if rows else None
        if (
            row
            and abs(center - row["center"]) <= min(box[3] - box[1], row["height"]) * row_tolerance
        ):
            row["lines"].append(line)
        else:
            rows.append({"center": center, "height": box[3] - box[1], "lines": [line]})
    return " ".join(
        line["text"].strip()
        for row in rows
        for line in sorted(row["lines"], key=lambda line: line["bbox"][0])
    )


def prepare_blocks(elements, ocr, options):
    """Attach each raw OCR line to one block; reconstruct only text-like blocks."""
    if not options["enabled"]:
        return elements
    for line in collect_lines(ocr):
        candidates = [(overlap(line["bbox"], e["bbox"])[0], i) for i, e in enumerate(elements)]
        if not candidates:
            continue
        coverage, index = max(candidates)
        if coverage >= options["line_assignment_ratio"]:
            elements[index]["metadata"].setdefault("ocr_lines", []).append(line)
    for element in elements:
        lines = element["metadata"].get("ocr_lines", [])
        if not lines:
            continue
        scores = [line["confidence"] for line in lines if line["confidence"] is not None]
        if scores:
            element["metadata"]["ocr_confidence"] = sum(scores) / len(scores)
        if options["reconstruct_lines"] and element["element_type"] in {
            "text",
            "paragraph",
            "paragraph_title",
            "doc_title",
            "header",
            "footer",
            "number",
        }:
            joined = join_lines(lines, options["line_row_tolerance"])
            similarity = SequenceMatcher(None, compact(element["text"]), compact(joined)).ratio()
            if similarity >= options["line_match_ratio"]:
                record_text_change(element, joined, "join_ocr_lines_with_spaces")
    return elements


def single_column(elements, width, options):
    blocks = [
        e
        for e in elements
        if e["element_type"] in {"text", "paragraph"}
        and e["bbox"][2] - e["bbox"][0] >= width * options["column_min_width_ratio"]
    ]
    for i, left in enumerate(blocks):
        for right in blocks[i + 1 :]:
            a, b = left["bbox"], right["bbox"]
            gap = max(b[0] - a[2], a[0] - b[2])
            vertical = min(a[3], b[3]) - max(a[1], b[1])
            height = min(a[3] - a[1], b[3] - b[1])
            if (
                height > 0
                and gap > width * options["column_gap_ratio"]
                and vertical / height > options["column_overlap_ratio"]
            ):
                return False
    return True


def process_page(elements, width, height, options):
    if not options["enabled"]:
        return elements
    unique = []
    for element in elements:
        duplicate = next(
            (
                e
                for e in unique
                if compact(e["text"])
                and compact(e["text"]) == compact(element["text"])
                and overlap(e["bbox"], element["bbox"])[1] >= options["duplicate_overlap"]
            ),
            None,
        )
        if duplicate:
            duplicate["metadata"].setdefault("duplicate_sources", []).append(deepcopy(element))
            continue
        unique.append(element)
    for element in unique:
        box = element["bbox"]
        in_margin = box[3] <= height * options["page_number_margin_ratio"] or box[1] >= height * (
            1 - options["page_number_margin_ratio"]
        )
        centered = (
            abs((box[0] + box[2]) / 2 - width / 2)
            <= width * options["page_number_center_tolerance"]
        )
        text = element["text"].strip()
        numeric = bool(re.fullmatch(r"[\d\s()\[\]C]{1,5}", text) and re.search(r"\d", text))
        if in_margin and centered and numeric:
            element["metadata"].update(
                exclude_from_content=True,
                exclusion_reason="page_number",
                original_element_type=element["element_type"],
            )
            element["element_type"] = "footer"
        record_text_change(
            element,
            clean_text(element["text"], options["normalize_punctuation"]),
            "normalize_whitespace_and_punctuation",
        )
    mode = options["reading_order"]
    if mode == "single_column" or (mode == "auto" and single_column(unique, width, options)):
        for index, element in enumerate(unique):
            element["metadata"]["order_before_postprocessing"] = index
        unique.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    return unique


def roman_value(text):
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(
        -values[c] if i + 1 < len(text) and values[c] < values[text[i + 1]] else values[c]
        for i, c in enumerate(text)
    )


def roman_text(number):
    result = ""
    for value, token in [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]:
        count, number = divmod(number, value)
        result += token * count
    return result


def base_letter(character):
    return unicodedata.normalize("NFD", character.casefold())[0]


def review_document(document, options, retry=None):
    """Flag ambiguous text; numbering edits require agreeing OCR readings of the source crop."""
    if not options["enabled"]:
        return document
    previous = {}
    vocabulary = {word.casefold() for word in options["spelling_vocabulary"]}
    issue_count = 0
    for element in document.elements:
        if element.metadata.get("parser") != "paddleocr" or element.metadata.get(
            "exclude_from_content"
        ):
            continue
        issues = []
        heading = element.element_type == "heading"
        match = (
            re.match(r"^(Điều)\s+(\d+)\b|^(CHƯƠNG)\s+([IVXLCDM]+)\b", element.text, re.IGNORECASE)
            if heading
            else None
        )
        if match:
            kind = "article" if match.group(1) else "chapter"
            token = match.group(2) or match.group(4)
            number = int(token) if kind == "article" else roman_value(token.upper())
            if kind in previous and number != previous[kind] + 1:
                expected = previous[kind] + 1
                candidate = str(expected) if kind == "article" else roman_text(expected)
                issue = {
                    "reason": "numbering_sequence",
                    "observed": token,
                    "suggested": candidate,
                    "status": "needs_review",
                }
                readings = []
                if options["retry_numbering"] and retry:
                    try:
                        readings = retry(element, match, options["retry_scales"])
                    except Exception as exc:
                        issue["retry_error"] = str(exc)
                issue["retry_readings"] = readings
                confirmations = []
                for reading in readings:
                    found = re.match(
                        r"^Điều\s+(\d+)\b" if kind == "article" else r"^CHƯƠNG\s+([IVXLCDM]+)\b",
                        reading["text"],
                        re.IGNORECASE,
                    )
                    if (
                        found
                        and found.group(1).upper() == candidate
                        and reading["confidence"] >= options["retry_min_confidence"]
                    ):
                        confirmations.append(reading["scale"])
                if len(set(confirmations)) >= options["retry_min_agreement"]:
                    start, end = match.span(2 if kind == "article" else 4)
                    raw = {"text": element.text, "metadata": element.metadata}
                    record_text_change(
                        raw,
                        element.text[:start] + candidate + element.text[end:],
                        "numbering_verified_by_crop_ocr",
                    )
                    element.text = raw["text"]
                    issue["status"] = "corrected_by_crop_ocr"
                    number = expected
                issues.append(issue)
                # A suspicious label must not create a cascade of false sequence errors.
                previous[kind] = expected if number != expected else number
            else:
                previous[kind] = number
        if options["flag_years"]:
            for year in re.finditer(r"\bnăm\s+([12]\d)\s+(\d{2})\b", element.text, re.IGNORECASE):
                issues.append(
                    {
                        "reason": "split_year",
                        "observed": year.group(0),
                        "suggested": "năm " + year.group(1) + year.group(2),
                        "status": "needs_review",
                    }
                )
        if options["flag_spelling"]:
            for token in dict.fromkeys(re.findall(r"[^\W\d_]+", element.text)):
                if token.casefold() in vocabulary:
                    continue
                candidates = set()
                for index in range(len(token) - 1):
                    if base_letter(token[index]) == base_letter(token[index + 1]):
                        for offset in (index, index + 1):
                            candidate = token[:offset] + token[offset + 1 :]
                            if candidate.casefold() in vocabulary:
                                candidates.add(candidate)
                if candidates:
                    issues.append(
                        {
                            "reason": "repeated_letter",
                            "observed": token,
                            "suggestions": sorted(candidates),
                            "status": "needs_review",
                        }
                    )
        if issues:
            element.metadata["ocr_review"] = issues
            issue_count += sum(issue["status"] == "needs_review" for issue in issues)
    document.metadata["ocr_postprocessing"] = {
        "enabled": True,
        "changed_elements": sum(bool(e.metadata.get("text_changes")) for e in document.elements),
        "excluded_page_numbers": sum(
            e.metadata.get("exclusion_reason") == "page_number" for e in document.elements
        ),
        "review_issue_count": issue_count,
    }
    return document
