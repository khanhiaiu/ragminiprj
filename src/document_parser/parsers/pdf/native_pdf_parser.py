import logging
import statistics
from pathlib import Path

import pymupdf as fitz

from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...utils.file_utils import document_data, table_markdown

logger = logging.getLogger(__name__)


class NativePDFParser:
    def __init__(self, assets_dir: Path | None = None):
        self.assets_dir = assets_dir

    def parse_page(self, page) -> list[dict]:
        number = page.number + 1
        elements = []
        table_rects = []
        try:
            for table in page.find_tables().tables:
                rows = table.extract()
                if rows:
                    table_rects.append(fitz.Rect(table.bbox))
                    elements.append({"element_type": "table", "text": table_markdown(rows), "page_number": number,
                                     "bbox": tuple(table.bbox), "metadata": {"parser": "pymupdf", "values": rows}})
        except Exception as exc:
            logger.warning("Table detection failed on page %d: %s", number, exc)
        # DICT without image bytes; extract images separately with their soft masks.
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES, sort=True)["blocks"]
        sizes = [s["size"] for b in blocks if b["type"] == 0 for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        body_size = statistics.median(sizes) if sizes else 12.0
        for block in blocks:
            if block["type"] != 0:
                continue
            rect = fitz.Rect(block["bbox"])
            if any((rect & t).get_area() / max(rect.get_area(), 1) > 0.7 for t in table_rects):
                continue
            lines = block["lines"]
            text = "\n".join("".join(span["text"] for span in line["spans"]) for line in lines).strip()
            if not text:
                continue
            max_size = max(s["size"] for line in lines for s in line["spans"])
            kind = "heading" if max_size >= body_size * 1.25 and len(text) < 250 else "paragraph"
            if text.lstrip().startswith(("•", "●", "▪")):
                kind = "list"
            elements.append({"element_type": kind, "text": text, "page_number": number, "bbox": tuple(rect),
                "metadata": {"parser": "pymupdf", "coordinate_unit": "pt", "font_size": max_size,
                             "block_number": block.get("number"), "reading_order_method": "geometric_sort"}})
        for index, image in enumerate(page.get_image_info(xrefs=True), start=1):
            metadata = {"parser": "pymupdf", "xref": image.get("xref"), "coordinate_unit": "pt",
                        "enrichment_pending": True}
            if self.assets_dir:
                self.assets_dir.mkdir(parents=True, exist_ok=True)
                filename = f"p{number:04d}_image_{index:03d}.png"
                destination = self.assets_dir / filename
                try:
                    xref = image.get("xref", 0)
                    if xref:
                        pix = fitz.Pixmap(page.parent, xref)
                        smask = page.parent.extract_image(xref).get("smask", 0)
                        if smask:
                            pix = fitz.Pixmap(pix, fitz.Pixmap(page.parent, smask))
                        if pix.colorspace and pix.colorspace.n > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                    else:
                        pix = page.get_pixmap(clip=fitz.Rect(image["bbox"]), alpha=False)
                    pix.save(str(destination))
                    metadata["asset_path"] = f"assets/{filename}"
                except Exception as exc:
                    logger.warning("Image extraction failed on page %d: %s", number, exc)
                    metadata["asset_error"] = str(exc)
            elements.append({"element_type": "image", "text": "", "page_number": number,
                             "bbox": tuple(image["bbox"]), "metadata": metadata})
        return sorted(elements, key=lambda e: (round(e["bbox"][1], 1), e["bbox"][0]))

    def parse(self, path: Path) -> CanonicalDocument:
        data = document_data(path, "pymupdf")
        with fitz.open(path) as pdf:
            for page in pdf:
                record = {"page_number": page.number + 1, "width": page.rect.width, "height": page.rect.height}
                try:
                    data["elements"].extend(self.parse_page(page))
                except Exception as exc:
                    logger.exception("Native extraction failed on page %d", page.number + 1)
                    record.update(status="failed", metadata={"error": str(exc)})
                data["pages"].append(record)
        data["metadata"]["failed_pages"] = [p["page_number"] for p in data["pages"] if p.get("status") == "failed"]
        return Normalizer().normalize(data)
