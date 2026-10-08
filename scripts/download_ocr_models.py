#!/usr/bin/env python3
"""Download the Vietnamese recognition model selected in the shared config."""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document_parser.config import DEFAULT_CONFIG, ProjectConfig
from document_parser.utils.file_utils import file_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Shared project config YAML"
    )
    parser.add_argument(
        "--output", type=Path, help="Override the configured recognition model directory"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        project = ProjectConfig.load(args.config)
        project.parser_config().apply_environment()
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    model = project.recognition_download
    if model is None:
        logging.info(
            "The selected Paddle recognition model will be downloaded automatically during OCR"
        )
        return
    output_dir = args.output if args.output is not None else project.path("recognition_model")
    from huggingface_hub import snapshot_download

    directory = Path(
        snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            allow_patterns=model["files"],
            local_dir=str(output_dir.resolve()),
            max_workers=model["max_workers"],
        )
    )
    checksums = {name: file_digest(directory / name) for name in model["files"]}
    manifest = {
        "repository": model["repository"],
        "revision": model["revision"],
        "source_url": f"https://huggingface.co/{model['repository']}/tree/{model['revision']}",
        "sha256": checksums,
    }
    (directory / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Model ready: %s", directory)


if __name__ == "__main__":
    main()
