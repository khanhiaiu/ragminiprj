import re
from typing import Any

from .schema import CanonicalDocument, Element

LABELS = {
    "text": "paragraph", "title": "heading", "doc_title": "heading",
    "paragraph_title": "heading", "section_header": "heading", "list_item": "list",
    "picture": "image", "figure": "image", "chart": "image", "seal": "image",
    "display_formula": "formula", "equation": "formula", "page_header": "header",
    "page_footer": "footer", "page_number": "footer", "caption": "paragraph",
    "figure_title": "paragraph", "table_title": "paragraph", "reference": "paragraph",
    "figure_caption": "paragraph", "table_caption": "paragraph", "figure_table_chart_title": "paragraph",
    "number": "footer", "footnote": "footer", "algorithm": "paragraph", "code": "paragraph",
    "abstract": "paragraph", "content": "paragraph", "reference_content": "paragraph",
}
TYPES = {"heading", "paragraph", "list", "table", "image", "formula", "header", "footer", "unknown"}


class Normalizer:
    """Consume plain parser dictionaries, never third-party document objects."""

    def normalize(self, raw: dict[str, Any] | CanonicalDocument) -> CanonicalDocument:
        data = raw.model_dump() if isinstance(raw, CanonicalDocument) else dict(raw)
        doc_id = data["document_id"]
        elements = []
        section = None
        for order, item in enumerate(data.get("elements", [])):
            item = item.model_dump() if isinstance(item, Element) else dict(item)
            label = str(item.get("element_type", "unknown")).lower()
            label = LABELS.get(label, label)
            item["element_type"] = label if label in TYPES else "unknown"
            item["element_id"] = item.get("element_id") or f"{doc_id}-e{order + 1}"
            item["order"] = order
            item["text"] = str(item.get("text") or "").replace("\x00", "").strip()
            if label == "heading":
                slug = re.sub(r"[^\w-]+", "-", item["text"].lower()).strip("-")[:80]
                section = f"{slug or 'section'}-{order + 1}"
            item["section_id"] = item.get("section_id") or section
            elements.append(Element.model_validate(item))
        data["elements"] = elements
        doc = CanonicalDocument.model_validate(data)
        # This also rejects non-serializable objects leaking into metadata.
        doc.model_dump_json()
        return doc
