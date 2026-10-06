#!/usr/bin/env python3
"""Parse every supported fixture without editing source files."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document_parser.config import ParserConfig
from document_parser.pipeline import DocumentPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "documents")
    parser.add_argument("--output", type=Path, default=ROOT / "parsed_test_document")
    parser.add_argument("--ocr-options", type=Path, help="JSON object with PPStructureV3 constructor options")
    parser.add_argument("--ocr-dpi", type=int, default=150)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Keep downloaded models within this project; callers can override each cache.
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache" / "paddlex"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("PADDLE_HOME", str(ROOT / ".cache" / "paddle"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    options = dict(ParserConfig().ocr_options)
    if args.ocr_options:
        overrides = json.loads(args.ocr_options.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            parser.error("--ocr-options must contain a JSON object")
        for key, value in overrides.items():
            if key.endswith("_model_dir") and value:
                model_path = Path(value).expanduser()
                if not model_path.is_absolute():
                    model_path = args.ocr_options.resolve().parent / model_path
                overrides[key] = str(model_path.resolve())
            elif key == "paddlex_config" and isinstance(value, str):
                config_path = Path(value).expanduser()
                if not config_path.is_absolute():
                    config_path = args.ocr_options.resolve().parent / config_path
                overrides[key] = str(config_path.resolve())
        options.update(overrides)
    if args.ocr_dpi <= 0:
        parser.error("--ocr-dpi must be positive")
    summary = DocumentPipeline(ParserConfig(ocr_dpi=args.ocr_dpi, ocr_options=options)).parse_all(args.input, args.output)
    logging.info("Completed: %s; output: %s", summary["counts"], args.output)
    if not summary["files"]:
        logging.error("No supported documents found")
        return 1
    return 1 if summary["counts"]["failed"] or summary["counts"]["partial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
