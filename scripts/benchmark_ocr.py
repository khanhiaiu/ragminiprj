#!/usr/bin/env python3
"""Measure the selected OCR models on PDF pages; sample VRAM on GPU only.

Run each model selection in a separate process so allocations stay independent.
The first page includes model initialization; later pages measure a warm engine.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pymupdf

from document_parser.config import DEFAULT_CONFIG, ProjectConfig
from document_parser.parsers.image.paddle_image_parser import PaddleEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Shared project config YAML"
    )
    parser.add_argument("--device", help="cpu or gpu (optionally gpu:N)")
    parser.add_argument("--layout-model", help="Override the layout model")
    parser.add_argument("--detection-model", help="Override the text detection model")
    parser.add_argument("--recognition-model", help="Override the text recognition model")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--pages", type=int, nargs="+", default=[1, 15, 31])
    parser.add_argument("--output", type=Path, required=True, help="Benchmark JSON path")
    args = parser.parse_args()
    try:
        project = ProjectConfig.load(args.config)
        config = project.parser_config(
            device=args.device,
            layout_model=args.layout_model,
            detection_model=args.detection_model,
            recognition_model=args.recognition_model,
        )
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    config.apply_environment()
    input_path = args.input if args.input is not None else project.path("input") / "hp.pdf"
    pages = args.pages
    options = config.ocr_options
    engine = PaddleEngine(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renders = args.output.parent / "inputs"
    renders.mkdir(exist_ok=True)
    stop = threading.Event()
    memory_samples: list[tuple[int, int]] = []

    def monitor() -> None:
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        f"--id={gpu_index}",
                    ],
                    text=True,
                    timeout=3,
                )
                values = output.strip().split(",")
                memory_samples.append((int(values[0]), int(values[1])))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    using_gpu = str(options.get("device", "cpu")).startswith("gpu")
    gpu_index = str(options["device"]).partition(":")[2].split(",")[0] or "0"
    if using_gpu:
        thread.start()
    report = {
        "device": options["device"],
        "models": {
            "layout": options["layout_detection_model_name"],
            "detection": options["text_detection_model_name"],
            "recognition": options["text_recognition_model_name"],
        },
        "config": str(args.config.resolve()),
        "options": options,
        "pages": [],
        "status": "ok",
    }
    try:
        with pymupdf.open(input_path) as pdf:
            for number in pages:
                if number < 1 or number > len(pdf):
                    raise ValueError(f"Page {number} outside PDF with {len(pdf)} pages")
                path = renders / f"p{number:04d}_render.png"
                pdf[number - 1].get_pixmap(dpi=config.ocr_dpi, alpha=False).save(path)
                start = time.monotonic()
                elements = engine.predict(path)
                elapsed = time.monotonic() - start
                report["pages"].append({"page": number, "seconds": elapsed, "elements": elements})
                print(f"Page {number}: {len(elements)} elements, {elapsed:.2f}s", flush=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        print(f"Benchmark failed: {exc}", file=sys.stderr)
    finally:
        stop.set()
        if using_gpu:
            thread.join(timeout=5)
        report.update(
            peak_total_vram_mib_sampled=max((s[0] for s in memory_samples), default=None),
            peak_gpu_util_percent_sampled=max((s[1] for s in memory_samples), default=None),
            memory_samples=len(memory_samples),
        )
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if using_gpu:
        print(f"Peak sampled GPU VRAM: {report['peak_total_vram_mib_sampled']} MiB", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
