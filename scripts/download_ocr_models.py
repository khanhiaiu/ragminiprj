#!/usr/bin/env python3
"""Provision the pinned Vietnamese recognition model used by the scan profile."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "tieubaoca/pp-ocrv6-medium-rec-vietnamese"
REVISION = "edc19ca6890bedb7a68c055d21190b44fc1aaa00"
FILES = ["inference.json", "inference.pdiparams", "inference.yml", "ppocr_keys.txt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/models/ppocrv6_vi")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(repo_id=REPOSITORY, revision=REVISION, allow_patterns=FILES,
                                      local_dir=str(args.output.resolve()), max_workers=4))
    checksums = {}
    for name in FILES:
        digest = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums[name] = digest.hexdigest()
    manifest = {"repository": REPOSITORY, "revision": REVISION,
                "source_url": f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}", "sha256": checksums}
    (directory / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Model ready: %s", directory)


if __name__ == "__main__":
    main()
