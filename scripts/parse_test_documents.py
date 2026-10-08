#!/usr/bin/env python3
"""Parse every supported fixture without editing source files."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document_parser.config import DEFAULT_CONFIG, ProjectConfig
from document_parser.parsers.image.paddle_image_parser import require_requested_gpu
from document_parser.pipeline import DocumentPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Shared project config YAML"
    )
    parser.add_argument("--layout-model", help="Override the layout model")
    parser.add_argument("--detection-model", help="Override the text detection model")
    parser.add_argument("--recognition-model", help="Override the text recognition model")
    parser.add_argument("--input", type=Path, help="Override the configured input directory")
    parser.add_argument("--output", type=Path, help="Override the configured output directory")
    parser.add_argument("--device", help="cpu or gpu (optionally gpu:N)")
    parser.add_argument(
        "--corrector",
        choices=("protonx", "protonx_legal", "bmd1905", "bravend"),
        help="Override only text_correction.backend",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        project = ProjectConfig.load(args.config)
        config = project.parser_config(
            device=args.device,
            layout_model=args.layout_model,
            detection_model=args.detection_model,
            recognition_model=args.recognition_model,
            corrector_backend=args.corrector,
        )
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    config.apply_environment()
    device = str(config.ocr_options.get("device") or "cpu")
    if device.startswith("gpu"):
        import paddle

        require_requested_gpu(device, paddle)
        paddle.set_device(device)
        logging.info(
            "Initialized Paddle %s on %s (%s)",
            paddle.__version__,
            paddle.device.get_device(),
            paddle.device.cuda.get_device_name(),
        )
    input_dir = args.input if args.input is not None else config.input_dir
    output_dir = args.output if args.output is not None else config.output_dir
    summary = DocumentPipeline(config).parse_all(input_dir, output_dir)
    logging.info("Completed: %s; output: %s", summary["counts"], output_dir)
    if not summary["files"]:
        logging.error("No supported documents found")
        return 1
    return 1 if summary["counts"]["failed"] or summary["counts"]["partial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
