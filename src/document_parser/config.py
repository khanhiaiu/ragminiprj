from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PDFAnalyzerConfig:
    minimum_text_chars: int = 40
    minimum_text_blocks: int = 1
    minimum_printable_ratio: float = 0.95
    max_invalid_char_ratio: float = 0.03
    minimum_text_quality: float = 0.85
    minimum_text_density: float = 0.00005
    scan_image_coverage_threshold: float = 0.70
    hybrid_image_coverage_threshold: float = 0.08


@dataclass(frozen=True)
class ParserConfig:
    pdf: PDFAnalyzerConfig = field(default_factory=PDFAnalyzerConfig)
    ocr_dpi: int = 150
    # PPStructureV3 constructor arguments; models load only when OCR is needed.
    ocr_options: dict[str, Any] = field(default_factory=lambda: {
        "device": "cpu", "cpu_threads": 4, "enable_mkldnn": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False, "use_textline_orientation": False,
        "use_formula_recognition": False,
    })
    max_excel_region_cells: int = 1_000_000
