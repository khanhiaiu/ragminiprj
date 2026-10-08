import hashlib
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .config import ParserConfig
from .correction.base import TextCorrector
from .correction.factory import CorrectorFactory
from .correction.service import TextCorrectionService
from .loader.file_loader import FileLoader
from .normalization.normalizer import Normalizer
from .normalization.ocr_postprocessing import review_document
from .normalization.schema import CanonicalDocument
from .parsers.docx.docling_parser import DoclingDOCXParser
from .parsers.excel.excel_parser import ExcelParser
from .parsers.image.paddle_image_parser import PaddleEngine, PaddleImageParser
from .parsers.pdf.pdf_parser import PDFParser
from .quality.validator import ParseQualityValidator
from .router.document_router import DocumentRouter
from .utils.file_utils import file_digest
from .utils.serialization import write_document, write_summary

logger = logging.getLogger(__name__)


class DocumentPipeline:
    def __init__(
        self,
        config: ParserConfig | None = None,
        ocr_engine: PaddleEngine | None = None,
        text_corrector: TextCorrector | None = None,
    ):
        self.config = config or ParserConfig()
        self.loader = FileLoader()
        self.normalizer = Normalizer()
        self.validator = ParseQualityValidator(self.config.pdf)
        self.ocr_engine = ocr_engine or PaddleEngine(self.config)
        self.text_corrector = CorrectorFactory.create(
            self.config.text_correction, override=text_corrector
        )
        self.text_correction = TextCorrectionService(
            self.config.text_correction, self.text_corrector
        )

    def parse(self, path: Path | str, output_dir: Path | str | None = None) -> CanonicalDocument:
        path = self.loader.load(path)
        directory = (
            Path(output_dir).resolve() if output_dir else Path(self.config.output_dir) / path.stem
        )
        # Refuse writes into the source folder, including nested subfolders.
        if directory == path.parent or path.parent in directory.parents:
            raise ValueError("Output must be outside the source document directory")
        assets = directory / "assets"
        router = DocumentRouter(
            {
                "pdf": lambda: PDFParser(assets, self.config, self.ocr_engine),
                "docx": lambda: DoclingDOCXParser(assets),
                "excel": lambda: ExcelParser(self.config),
                "image": lambda: PaddleImageParser(assets, self.ocr_engine),
            }
        )
        document = self.normalizer.normalize(router.route(path).parse(path))
        document = review_document(
            document, self.config.postprocessing, getattr(self.ocr_engine, "retry_heading", None)
        )
        document = self.text_correction.process(document)
        if self.config.postprocessing["enabled"]:
            for element in document.elements:
                if element.metadata.get("parser") == "paddleocr":
                    element.section_id = None
            document = self.normalizer.normalize(document)
        document.metadata["parser_config"] = asdict(self.config)
        document.metadata["quality"] = self.validator.validate(document)
        write_document(document, directory)
        return document

    def parse_all(
        self, input_dir: Path | str | None = None, output_dir: Path | str | None = None
    ) -> dict:
        input_dir = input_dir if input_dir is not None else self.config.input_dir
        output_dir = output_dir if output_dir is not None else self.config.output_dir
        files = self.loader.discover(input_dir)
        input_root, output_root = Path(input_dir).resolve(), Path(output_dir).resolve()
        if output_root == input_root or input_root in output_root.parents:
            raise ValueError("Output must be outside the input directory")
        # Resolve same stems across extensions / nested directories before writing.
        counts = Counter(path.stem.casefold() for path in files)
        summary = {"input_dir": str(input_root), "output_dir": str(output_root), "files": []}
        allocated = set()
        for path in files:
            name = (
                path.stem
                if counts[path.stem.casefold()] == 1
                else f"{path.stem}-{path.suffix[1:]}-{file_digest(path)[:8]}"
            )
            if name.casefold() in allocated:
                name += "-" + hashlib.sha256(str(path).encode()).hexdigest()[:8]
            allocated.add(name.casefold())
            directory = output_root / name
            record = {"filename": path.name, "source_path": str(path), "output_dir": str(directory)}
            try:
                document = self.parse(path, directory)
                record.update(
                    status=document.metadata["quality"]["status"],
                    parser=document.metadata["parser"],
                    quality=document.metadata["quality"],
                    page_parsers=[
                        {
                            "page": p.page_number,
                            "type": p.page_type,
                            "parser": p.metadata.get("parser"),
                            "status": p.status,
                        }
                        for p in document.pages
                    ],
                )
            except Exception as exc:
                logger.exception("Document failed: %s", path)
                record.update(status="failed", error=str(exc))
                directory.mkdir(parents=True, exist_ok=True)
                write_summary({"source_path": str(path), "error": str(exc)}, directory)
            summary["files"].append(record)
        summary["counts"] = {
            status: sum(f["status"] == status for f in summary["files"])
            for status in ("ok", "partial", "failed")
        }
        write_summary(summary, output_root)
        return summary
