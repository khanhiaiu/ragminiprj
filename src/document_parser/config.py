"""Load model choices, runtime options and paths from the shared YAML config."""

import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_settings import DEFAULT_CONFIG, config_path, configured_path, load_config, setting

PROJECT_ROOT = DEFAULT_CONFIG.parent


def normalize_device(device: str) -> str:
    if device == "cpu":
        return device
    if device == "gpu":
        return "gpu:0"
    if isinstance(device, str) and re.fullmatch(r"gpu:\d+", device):
        return device
    raise ValueError("device must be 'cpu', 'gpu' or 'gpu:N'")


def _read_settings(path: Path) -> dict[str, Any]:
    settings = load_config(path)
    allowed = {
        "device",
        "models",
        "paths",
        "ocr",
        "pdf",
        "max_excel_region_cells",
        "environment",
        "recognition_download",
        "postprocessing",
        "text_correction",
    }
    if not isinstance(settings, dict) or not {"device", "models"} <= settings.keys():
        raise ValueError("Config must contain device and models")
    settings = {key: value for key, value in settings.items() if key in allowed}
    normalize_device(settings["device"])
    models = settings["models"]
    if not isinstance(models, dict) or set(models) != {"layout", "detection", "recognition"}:
        raise ValueError("models must contain layout, detection and recognition")
    if any(not isinstance(value, str) or not value.strip() for value in models.values()):
        raise ValueError("Model names must be non-empty strings")
    for section in (
        "paths",
        "ocr",
        "pdf",
        "environment",
        "recognition_download",
        "postprocessing",
        "text_correction",
    ):
        if section in settings and not isinstance(settings[section], dict):
            raise ValueError(f"{section} must be a mapping")
    ocr = settings.get("ocr", {})
    if ocr.keys() - {"dpi", "options"}:
        raise ValueError("ocr only supports dpi and options")
    if "options" in ocr and not isinstance(ocr["options"], dict):
        raise ValueError("ocr.options must be a mapping")
    reserved = {
        "device",
        "layout_detection_model_name",
        "text_detection_model_name",
        "text_recognition_model_name",
    }
    if reserved & ocr.get("options", {}).keys():
        raise ValueError("Set device and model names in device/models, not ocr.options")
    if "paths" in settings:
        if settings["paths"].keys() - {"input", "output", "recognition_model"}:
            raise ValueError("paths only supports input, output and recognition_model")
        if any(not isinstance(v, str) or not v.strip() for v in settings["paths"].values()):
            raise ValueError("Configured paths must be non-empty strings")
    return settings


def _merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for name, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _merge(result[name], value)
        else:
            result[name] = deepcopy(value)
    return result


_DEFAULTS = _read_settings(DEFAULT_CONFIG)
VI_RECOGNITION = _DEFAULTS["recognition_download"]["repository"]


def _environment(settings: dict, root: Path) -> dict[str, str]:
    result = {}
    for name, value in settings["environment"].items():
        if name in {"PADDLE_PDX_CACHE_HOME", "HF_HOME", "PADDLE_HOME", "XDG_CACHE_HOME"}:
            value = str((root / Path(value).expanduser()).resolve())
        result[name] = str(value)
    return result


def _ocr_options(settings: dict, root: Path) -> dict:
    models = settings["models"]
    recognition = models["recognition"]
    download = settings["recognition_download"]
    if "/" in recognition and recognition != download["repository"]:
        raise ValueError(
            "Use a Paddle model name or the configured recognition_download.repository"
        )
    options = deepcopy(settings["ocr"]["options"])
    if options.get("use_table_recognition") is False:
        raise ValueError(
            "ocr.options.use_table_recognition must be true: table structure belongs to "
            "PP-StructureV3, not the text recognition model"
        )
    for name, value in list(options.items()):
        if name.endswith("_model_dir") and value:
            options[name] = str((root / Path(value).expanduser()).resolve())
    options.update(
        device=normalize_device(settings["device"]),
        layout_detection_model_name=models["layout"],
        text_detection_model_name=models["detection"],
        text_recognition_model_name=(
            download["model_name"] if recognition == download["repository"] else recognition
        ),
    )
    if recognition == download["repository"]:
        options["text_recognition_model_dir"] = str(
            (root / settings["paths"]["recognition_model"]).resolve()
        )
    return options


@dataclass(frozen=True)
class PDFAnalyzerConfig:
    minimum_text_chars: int = field(default_factory=lambda: setting("pdf.minimum_text_chars"))
    minimum_text_blocks: int = field(default_factory=lambda: setting("pdf.minimum_text_blocks"))
    minimum_printable_ratio: float = field(default_factory=lambda: setting("pdf.minimum_printable_ratio"))
    max_invalid_char_ratio: float = field(default_factory=lambda: setting("pdf.max_invalid_char_ratio"))
    minimum_text_quality: float = field(default_factory=lambda: setting("pdf.minimum_text_quality"))
    minimum_text_density: float = field(default_factory=lambda: setting("pdf.minimum_text_density"))
    scan_image_coverage_threshold: float = field(default_factory=lambda: setting("pdf.scan_image_coverage_threshold"))
    hybrid_image_coverage_threshold: float = field(default_factory=lambda: setting("pdf.hybrid_image_coverage_threshold"))


@dataclass(frozen=True)
class ParserConfig:
    pdf: PDFAnalyzerConfig = field(default_factory=PDFAnalyzerConfig)
    ocr_dpi: int = field(default_factory=lambda: setting("ocr.dpi"))
    ocr_options: dict[str, Any] = field(
        default_factory=lambda: _ocr_options(_read_settings(config_path()), config_path().parent)
    )
    recognition_model: str = field(default_factory=lambda: setting("models.recognition"))
    max_excel_region_cells: int = field(default_factory=lambda: setting("max_excel_region_cells"))
    input_dir: str = field(default_factory=lambda: str(configured_path("paths.input")))
    output_dir: str = field(default_factory=lambda: str(configured_path("paths.output")))
    environment: dict[str, str] = field(
        default_factory=lambda: _environment(_read_settings(config_path()), config_path().parent)
    )
    postprocessing: dict[str, Any] = field(
        default_factory=lambda: setting("postprocessing")
    )
    text_correction: dict[str, Any] = field(
        default_factory=lambda: setting("text_correction")
    )

    def __post_init__(self) -> None:
        if type(self.ocr_dpi) is not int or self.ocr_dpi <= 0:
            raise ValueError("ocr_dpi must be a positive integer")
        if self.ocr_options.get("use_table_recognition") is False:
            raise ValueError(
                "PP-StructureV3 table recognition cannot be disabled in the document pipeline"
            )
        if not isinstance(self.recognition_model, str) or not self.recognition_model.strip():
            raise ValueError("recognition_model must be a non-empty configured model name")
        if type(self.max_excel_region_cells) is not int or self.max_excel_region_cells <= 0:
            raise ValueError("max_excel_region_cells must be a positive integer")
        options = self.postprocessing
        if options.keys() - _DEFAULTS["postprocessing"].keys():
            raise ValueError("Unknown postprocessing option")
        for key in (
            "enabled",
            "reconstruct_lines",
            "normalize_punctuation",
            "flag_years",
            "flag_spelling",
            "retry_numbering",
        ):
            if type(options[key]) is not bool:
                raise ValueError(f"postprocessing.{key} must be a boolean")
        if options.get("reading_order") not in {"auto", "single_column", "preserve"}:
            raise ValueError("postprocessing.reading_order must be auto, single_column or preserve")
        for key in (
            "line_assignment_ratio",
            "line_match_ratio",
            "line_row_tolerance",
            "duplicate_overlap",
            "column_min_width_ratio",
            "column_gap_ratio",
            "column_overlap_ratio",
            "page_number_margin_ratio",
            "page_number_center_tolerance",
            "retry_min_confidence",
        ):
            if type(options[key]) not in (int, float) or not 0 < options[key] <= 1:
                raise ValueError(f"postprocessing.{key} must be in (0, 1]")
        if not isinstance(options["spelling_vocabulary"], list) or any(
            not isinstance(word, str) for word in options["spelling_vocabulary"]
        ):
            raise ValueError("postprocessing.spelling_vocabulary must be a list of words")
        scales = options["retry_scales"]
        if (
            not isinstance(scales, list)
            or not scales
            or any(type(scale) not in (int, float) or not 1 <= scale <= 4 for scale in scales)
        ):
            raise ValueError("postprocessing.retry_scales must contain scales between 1 and 4")
        if type(options["retry_min_agreement"]) is not int or not 1 <= options[
            "retry_min_agreement"
        ] <= len(set(scales)):
            raise ValueError(
                "postprocessing.retry_min_agreement must fit the distinct retry_scales"
            )
        if type(options["retry_crop_padding"]) is not int or options["retry_crop_padding"] < 0:
            raise ValueError("postprocessing.retry_crop_padding must be a non-negative integer")
        self._validate_text_correction()

    def _validate_text_correction(self) -> None:
        options = self.text_correction
        defaults = _DEFAULTS["text_correction"]
        if not isinstance(options, dict) or options.keys() != defaults.keys():
            raise ValueError("text_correction must contain exactly the configured fields")
        for key in ("enabled", "only_ocr"):
            if type(options[key]) is not bool:
                raise ValueError(f"text_correction.{key} must be a boolean")
        valid_backends = {"protonx", "protonx_legal", "bmd1905", "bravend"}
        if options["backend"] not in valid_backends:
            raise ValueError(
                "text_correction.backend must be one of: " + ", ".join(sorted(valid_backends))
            )
        models = options["models"]
        if not isinstance(models, dict) or set(models) != valid_backends:
            raise ValueError(
                "text_correction.models must contain protonx, protonx_legal, bmd1905 and bravend"
            )
        for backend, model in models.items():
            expected_fields = (
                {"model_name", "revision", "trust_remote_code"}
                if backend == "bravend"
                else {"model_name"}
            )
            if not isinstance(model, dict) or set(model) != expected_fields:
                raise ValueError(
                    f"text_correction.models.{backend} must contain exactly "
                    + ", ".join(sorted(expected_fields))
                )
            if not isinstance(model["model_name"], str) or not model["model_name"].strip():
                raise ValueError(
                    f"text_correction.models.{backend}.model_name must be a non-empty string"
                )
        bravend = models["bravend"]
        if not isinstance(bravend["revision"], str) or not bravend["revision"].strip():
            raise ValueError("text_correction.models.bravend.revision must be a non-empty string")
        if bravend["trust_remote_code"] is not True:
            raise ValueError("text_correction.models.bravend.trust_remote_code must be true")
        if not re.fullmatch(r"(?:auto|cpu|(?:gpu|cuda)(?::\d+)?)", options["device"]):
            raise ValueError("text_correction.device must be auto, cpu, gpu[:N] or cuda[:N]")
        allowed_types = {
            "heading",
            "paragraph",
            "list",
            "table",
            "image",
            "formula",
            "header",
            "footer",
            "unknown",
        }
        element_types = options["eligible_element_types"]
        if (
            not isinstance(element_types, list)
            or not element_types
            or any(value not in allowed_types for value in element_types)
        ):
            raise ValueError(
                "text_correction.eligible_element_types contains an invalid element type"
            )

        detector = options["detector"]
        if not isinstance(detector, dict) or detector.keys() != defaults["detector"].keys():
            raise ValueError("text_correction.detector contains invalid fields")
        for key in ("enabled", "use_ocr_confidence", "use_vocabulary", "use_spelling_flags"):
            if type(detector[key]) is not bool:
                raise ValueError(f"text_correction.detector.{key} must be a boolean")
        threshold = detector["min_ocr_confidence"]
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("text_correction.detector.min_ocr_confidence must be in [0, 1]")

        generation = options["generation"]
        if not isinstance(generation, dict) or generation.keys() != defaults["generation"].keys():
            raise ValueError("text_correction.generation contains invalid fields")
        for key in ("num_beams", "max_input_tokens", "max_new_tokens"):
            if type(generation[key]) is not int or generation[key] <= 0:
                raise ValueError(f"text_correction.generation.{key} must be a positive integer")
        if (
            type(generation["length_penalty"]) not in (int, float)
            or generation["length_penalty"] <= 0
        ):
            raise ValueError("text_correction.generation.length_penalty must be positive")
        if type(generation["early_stopping"]) is not bool:
            raise ValueError("text_correction.generation.early_stopping must be a boolean")

        safety = options["safety"]
        if not isinstance(safety, dict) or safety.keys() != defaults["safety"].keys():
            raise ValueError("text_correction.safety contains invalid fields")
        for key in (
            "enabled",
            "preserve_numbers",
            "preserve_dates",
            "preserve_identifiers",
            "preserve_urls",
            "preserve_emails",
            "preserve_structure",
        ):
            if type(safety[key]) is not bool:
                raise ValueError(f"text_correction.safety.{key} must be a boolean")
        ratio = safety["max_auto_edit_ratio"]
        if type(ratio) not in (int, float) or not 0 <= ratio <= 1:
            raise ValueError("text_correction.safety.max_auto_edit_ratio must be in [0, 1]")

    def apply_environment(self) -> None:
        for name, value in self.environment.items():
            os.environ.setdefault(name, value)


@dataclass(frozen=True)
class ProjectConfig:
    settings: dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ProjectConfig":
        path = Path(path).expanduser().resolve() if path is not None else config_path()
        root = PROJECT_ROOT if path == DEFAULT_CONFIG.resolve() else path.parent
        return cls(_merge(_DEFAULTS, _read_settings(path)), root)

    def path(self, name: str) -> Path:
        return (self.root / Path(self.settings["paths"][name]).expanduser()).resolve()

    @property
    def recognition_download(self) -> dict | None:
        model = self.settings["recognition_download"]
        return (
            deepcopy(model)
            if self.settings["models"]["recognition"] == model["repository"]
            else None
        )

    def parser_config(
        self,
        *,
        device: str | None = None,
        layout_model: str | None = None,
        detection_model: str | None = None,
        recognition_model: str | None = None,
        corrector_backend: str | None = None,
    ) -> ParserConfig:
        settings = deepcopy(self.settings)
        if device is not None:
            settings["device"] = device
        for name, value in {
            "layout": layout_model,
            "detection": detection_model,
            "recognition": recognition_model,
        }.items():
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Model names must be non-empty strings")
                settings["models"][name] = value
        if corrector_backend is not None:
            settings["text_correction"]["backend"] = corrector_backend
        return ParserConfig(
            pdf=PDFAnalyzerConfig(**settings["pdf"]),
            ocr_dpi=settings["ocr"]["dpi"],
            ocr_options=_ocr_options(settings, self.root),
            recognition_model=settings["models"]["recognition"],
            max_excel_region_cells=settings["max_excel_region_cells"],
            input_dir=str(self.path("input")),
            output_dir=str(self.path("output")),
            environment=_environment(settings, self.root),
            postprocessing=deepcopy(settings["postprocessing"]),
            text_correction=deepcopy(settings["text_correction"]),
        )
