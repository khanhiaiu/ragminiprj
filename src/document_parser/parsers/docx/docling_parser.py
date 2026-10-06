import logging
from pathlib import Path

from ...errors import DependencyUnavailableError, DocumentParseError
from ...normalization.normalizer import Normalizer
from ...normalization.schema import CanonicalDocument
from ...utils.file_utils import document_data

logger = logging.getLogger(__name__)


class DoclingDOCXParser:
    def __init__(self, assets_dir: Path | None = None, converter=None):
        self.assets_dir = assets_dir
        self._converter = converter

    def parse(self, path: Path) -> CanonicalDocument:
        if self._converter is None:
            try:
                from docling.backend.msword_backend import MsWordDocumentBackend
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.document import InputDocument
            except ImportError as exc:
                raise DependencyUnavailableError("Install the DOCX extra: pip install '.[docx]'") from exc
            # This is the same declarative Word backend used by SimplePipeline.
            # DocumentConverter imports unrelated OCR pipelines in SDK 2.133,
            # so invoke the Word backend without requiring scipy / torch.
            input_doc = InputDocument(path_or_stream=path, format=InputFormat.DOCX, backend=MsWordDocumentBackend)
            backend = getattr(input_doc, "_backend", None)
            try:
                if not input_doc.valid or backend is None or not backend.is_valid():
                    raise DocumentParseError(f"Docling cannot load Word document: {path}")
                doc = backend.convert()
            finally:
                if backend is not None:
                    backend.unload()
            status, errors = "success", []
        else:
            result = self._converter.convert(path)
            status = str(result.status.value)
            if status not in {"success", "partial_success"}:
                raise DocumentParseError(f"Docling conversion status: {status}")
            doc, errors = result.document, [str(e) for e in result.errors]
        data = document_data(path, "docling")
        data["metadata"]["backend"] = "MsWordDocumentBackend" if self._converter is None else "DocumentConverter"
        data["metadata"]["conversion_status"] = status
        data["metadata"]["conversion_errors"] = errors
        for number, page in doc.pages.items():
            data["pages"].append({"page_number": number, "width": page.size.width, "height": page.size.height})
        references = {}
        parent_references = []
        for item, level in doc.iterate_items():
            label = str(item.label.value)
            metadata = {"parser": "docling", "original_label": label, "structural_level": level,
                        "docling_ref": item.self_ref}
            element = {"element_type": label, "text": getattr(item, "text", ""), "metadata": metadata}
            if label == "table":
                element["text"] = item.export_to_markdown(doc=doc)
                metadata["table_html"] = item.export_to_html(doc=doc)
                metadata["table_data"] = item.data.model_dump(mode="json")
            if label == "picture":
                if self.assets_dir:
                    image = item.get_image(doc)
                    if image is not None:
                        self.assets_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"image_{len(data['elements']) + 1:03d}.png"
                        image.save(self.assets_dir / filename)
                        metadata["asset_path"] = f"assets/{filename}"
                    else:
                        metadata["asset_warning"] = "Docling did not expose image bytes"
                metadata["captions"] = [c.cref for c in item.captions]
            if hasattr(item, "level"):
                metadata["heading_level"] = item.level
            if hasattr(item, "marker"):
                metadata["marker"] = item.marker
                metadata["enumerated"] = item.enumerated
            if item.prov:
                prov = item.prov[0]
                element["page_number"] = prov.page_no
                box = prov.bbox
                page = doc.pages.get(prov.page_no)
                if page:
                    box = box.to_top_left_origin(page_height=page.size.height)
                element["bbox"] = (box.l, box.t, box.r, box.b)
                metadata["provenance"] = [p.model_dump(mode="json") for p in item.prov]
            element["element_id"] = f"{data['document_id']}-e{len(data['elements']) + 1}"
            references[item.self_ref] = element["element_id"]
            parent_references.append(item.parent.cref if item.parent else None)
            data["elements"].append(element)
        for element, parent_ref in zip(data["elements"], parent_references):
            element["parent_id"] = references.get(parent_ref)
            element["metadata"]["structural_parent_ref"] = parent_ref
        return Normalizer().normalize(data)
