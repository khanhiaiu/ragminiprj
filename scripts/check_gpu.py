#!/usr/bin/env python3
"""Verify CUDA kernels on the selected GPU before loading OCR models."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document_parser.config import DEFAULT_CONFIG, ProjectConfig, normalize_device
from document_parser.parsers.image.paddle_image_parser import require_requested_gpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Shared project config YAML"
    )
    parser.add_argument("--device", help="gpu or gpu:N (default: gpu)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        project = ProjectConfig.load(args.config)
        selected = args.device or project.settings["device"]
        device = normalize_device("gpu" if selected == "cpu" and args.device is None else selected)
        if not device.startswith("gpu") or "," in device:
            parser.error("--device must select one GPU")
        project.parser_config(device=device).apply_environment()
        import paddle

        require_requested_gpu(device, paddle)
        paddle.set_device(device)
        # Exercise Paddle CUDA kernels together with cuBLAS/cuDNN.
        with paddle.no_grad():
            matrix = paddle.ones([32, 32], dtype="float32")
            matmul_mean = float(paddle.matmul(matrix, matrix).mean())
            assert math.isclose(matmul_mean, 32.0, rel_tol=1e-5)
            image = paddle.ones([1, 3, 32, 32], dtype="float32")
            kernel = paddle.ones([4, 3, 3, 3], dtype="float32")
            output = paddle.nn.functional.conv2d(image, kernel)
            conv_mean = float(output.mean())
            assert math.isclose(conv_mean, 27.0, rel_tol=1e-5)
        report = {
            "status": "ok",
            "paddle_version": paddle.__version__,
            "cuda_version": paddle.version.cuda(),
            "device": paddle.device.get_device(),
            "device_name": paddle.device.cuda.get_device_name(),
            "compute_capability": paddle.device.cuda.get_device_capability(),
            "matmul_mean": matmul_mean,
            "conv_mean": conv_mean,
            "matmul_and_conv_verified": True,
        }
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        print(serialized)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"GPU verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
