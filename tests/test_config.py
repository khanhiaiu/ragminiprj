import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
import yaml
from PIL import Image

from document_parser.config import DEFAULT_CONFIG, VI_RECOGNITION, ParserConfig, ProjectConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def custom_config(tmp_path):
    settings = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    settings["device"] = "cpu"
    settings["models"].update(
        layout="PP-DocLayout-S", detection="PP-OCRv6_tiny_det", recognition=VI_RECOGNITION
    )
    settings["paths"].update(input="documents", output="parsed_test_document")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    source = tmp_path / "documents"
    source.mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Readable native text. " * 4)
        pdf.save(source / "hp.pdf")
    return path


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_defaults_match_shared_config():
    assert asdict(ParserConfig()) == asdict(ProjectConfig.load().parser_config())
    assert ParserConfig().recognition_model == VI_RECOGNITION


def test_text_correction_models_have_independent_defaults():
    correction = ParserConfig().text_correction
    assert correction["backend"] == "protonx"
    assert correction["models"] == {
        "protonx": {"model_name": "protonx-models/nano-protonx-legal-tc"},
        "protonx_legal": {"model_name": "protonx-models/protonx-legal-tc"},
        "bmd1905": {"model_name": "bmd1905/vietnamese-correction-v2"},
        "bravend": {
            "model_name": "bravend/bartpho-syllable-vi-spellcheck",
            "revision": "31f323bd148f9f0e23722bf3226a752d124c1023",
            "trust_remote_code": True,
        },
    }


@pytest.mark.parametrize("backend", ["protonx", "protonx_legal", "bmd1905", "bravend"])
def test_corrector_backend_override_changes_only_selection(backend):
    project = ProjectConfig.load()
    before = project.parser_config().text_correction
    selected = project.parser_config(corrector_backend=backend).text_correction
    assert selected["backend"] == backend
    assert selected["models"] == before["models"]
    assert project.parser_config().text_correction == before


@pytest.mark.parametrize(
    "device,expected", [("cpu", "cpu"), ("gpu", "gpu:0"), ("gpu:1", "gpu:1"), ("gpu:2", "gpu:2")]
)
def test_device_selection_keeps_configured_models(custom_config, device, expected):
    project = ProjectConfig.load(custom_config)
    options = project.parser_config(device=device).ocr_options
    cpu = project.parser_config(device="cpu").ocr_options
    assert options["device"] == expected
    for key in [
        "layout_detection_model_name",
        "text_detection_model_name",
        "text_recognition_model_name",
        "text_recognition_model_dir",
    ]:
        assert options[key] == cpu[key]
    assert options["layout_detection_model_name"] == "PP-DocLayout-S"
    assert options["text_detection_model_name"] == "PP-OCRv6_tiny_det"
    assert "paddlex_config" not in options


def test_official_model_selection_does_not_reuse_vietnamese_weights():
    config = ProjectConfig.load().parser_config(recognition_model="PP-OCRv6_small_rec")
    assert config.recognition_model == "PP-OCRv6_small_rec"
    assert config.ocr_options["text_recognition_model_name"] == "PP-OCRv6_small_rec"
    assert "text_recognition_model_dir" not in config.ocr_options


def test_model_overrides_do_not_mutate_shared_settings():
    project = ProjectConfig.load()
    before = asdict(project.parser_config())
    first = project.parser_config(device="gpu", layout_model="PP-DocLayout-S")
    first.ocr_options["text_detection_model_name"] = "changed"
    first.environment["HF_HOME"] = "/changed"
    assert asdict(project.parser_config()) == before


@pytest.mark.parametrize("device", ["gpu:-1", "gpu:0,1", "cuda", "", None])
def test_invalid_devices_are_rejected(custom_config, device):
    settings = yaml.safe_load(custom_config.read_text())
    settings["device"] = device
    custom_config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="device must be"):
        ProjectConfig.load(custom_config)


def test_config_rejects_unknown_ocr_sections(custom_config):
    settings = yaml.safe_load(custom_config.read_text())
    settings["ocr"] = {"profiles": {}}
    custom_config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="ocr only supports"):
        ProjectConfig.load(custom_config)


def test_runtime_settings_flow_from_yaml_to_parser(custom_config):
    settings = yaml.safe_load(custom_config.read_text())
    settings["ocr"]["dpi"] = 240
    settings["ocr"]["options"].update(
        text_recognition_batch_size=3, text_det_limit_side_len=2048, precision="fp16"
    )
    settings["pdf"]["minimum_text_chars"] = 80
    settings["postprocessing"].update(reading_order="preserve", retry_crop_padding=5)
    settings["text_correction"].update(device="cpu", only_ocr=False)
    settings["text_correction"]["generation"]["num_beams"] = 10
    settings["max_excel_region_cells"] = 2000
    settings["paths"].update(input="incoming", output="results")
    settings["environment"]["PADDLE_PDX_CACHE_HOME"] = "model_cache"
    custom_config.write_text(yaml.safe_dump(settings))
    config = ProjectConfig.load(custom_config).parser_config()
    assert config.ocr_dpi == 240
    assert config.ocr_options["text_recognition_batch_size"] == 3
    assert config.ocr_options["text_det_limit_side_len"] == 2048
    assert config.ocr_options["precision"] == "fp16"
    assert config.pdf.minimum_text_chars == 80
    assert config.postprocessing["reading_order"] == "preserve"
    assert config.postprocessing["retry_crop_padding"] == 5
    assert config.text_correction["device"] == "cpu"
    assert not config.text_correction["only_ocr"]
    assert config.text_correction["generation"]["num_beams"] == 10
    assert config.max_excel_region_cells == 2000
    assert config.input_dir == str(custom_config.parent / "incoming")
    assert config.output_dir == str(custom_config.parent / "results")
    assert config.environment["PADDLE_PDX_CACHE_HOME"] == str(custom_config.parent / "model_cache")


@pytest.mark.parametrize(
    "path,value,error",
    [
        (("backend",), "unknown", "backend"),
        (("device",), "cuda:-1", "device"),
        (("models", "bravend", "trust_remote_code"), False, "trust_remote_code"),
        (("generation", "max_input_tokens"), 0, "max_input_tokens"),
        (("safety", "max_auto_edit_ratio"), 1.1, "max_auto_edit_ratio"),
    ],
)
def test_invalid_text_correction_config_is_rejected(custom_config, path, value, error):
    settings = yaml.safe_load(custom_config.read_text())
    target = settings["text_correction"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    custom_config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match=error):
        ProjectConfig.load(custom_config).parser_config()


def test_custom_config_resolves_paths_from_its_directory(custom_config):
    project = ProjectConfig.load(custom_config)
    config = project.parser_config()
    assert config.input_dir == str(custom_config.parent / "documents")
    assert config.output_dir == str(custom_config.parent / "parsed_test_document")
    assert config.ocr_options["text_recognition_model_dir"] == str(
        custom_config.parent / ".cache/models/ppocrv6_vi"
    )
    assert config.environment["HF_HOME"] == str(custom_config.parent / ".cache/huggingface")


def test_environment_preserves_caller_overrides_and_applies_gpu_defaults(monkeypatch):
    config = ProjectConfig.load().parser_config(device="gpu")
    for key in [
        *config.environment,
        "FLAGS_allocator_strategy",
        "PADDLE_PDX_DISABLE_DEVICE_FALLBACK",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HOME", "/caller/cache")
    config.apply_environment()
    assert os.environ["HF_HOME"] == "/caller/cache"
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(ROOT / ".cache/paddlex")
    assert os.environ["FLAGS_allocator_strategy"] == "auto_growth"
    assert os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] == "True"


def test_parse_cli_uses_simple_config_outside_repository(custom_config, tmp_path, monkeypatch):
    work = tmp_path / "different_working_directory"
    work.mkdir()
    script = load_script("parse_test_documents")
    captured = {}

    class Pipeline:
        def __init__(self, config):
            captured["config"] = config

        def parse_all(self, input_dir, output_dir):
            captured["input_dir"] = Path(input_dir)
            captured["output_dir"] = Path(output_dir)
            return {
                "files": [{"status": "ok"}],
                "counts": {"ok": 1, "partial": 0, "failed": 0},
            }

    monkeypatch.setattr(script, "DocumentPipeline", Pipeline)
    monkeypatch.chdir(work)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_test_documents.py",
            "--config",
            str(custom_config),
            "--device",
            "gpu",
            "--corrector",
            "bmd1905",
        ],
    )
    assert script.main() == 0

    config = captured["config"]
    options = config.ocr_options
    assert options["device"] == "gpu:0"
    assert options["layout_detection_model_name"] == "PP-DocLayout-S"
    assert config.text_correction["backend"] == "bmd1905"
    assert captured["input_dir"] == tmp_path / "documents"
    assert captured["output_dir"] == tmp_path / "parsed_test_document"


def test_parse_cli_rejects_invalid_device_before_writing_output(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/parse_test_documents.py"), "--device", "cuda"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "device must be" in result.stderr
    assert not (tmp_path / "parsed_test_document").exists()


def test_benchmark_uses_simple_config_and_requested_pages(custom_config, monkeypatch):
    script = load_script("benchmark_ocr")
    rendered = []

    class Engine:
        def __init__(self, config):
            assert config.ocr_options["text_detection_model_name"] == "PP-OCRv6_tiny_det"

        def predict(self, path):
            with Image.open(path) as image:
                rendered.append(image.size)
            return [{"text": "Recognized text"}]

    report = custom_config.parent / "benchmark/report.json"
    monkeypatch.setattr(script, "PaddleEngine", Engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_ocr.py",
            "--config",
            str(custom_config),
            "--pages",
            "1",
            "--output",
            str(report),
        ],
    )
    assert script.main() == 0
    with pymupdf.open(custom_config.parent / "documents/hp.pdf") as pdf:
        pixmap = pdf[0].get_pixmap(dpi=ParserConfig().ocr_dpi)
        assert rendered == [(pixmap.width, pixmap.height)]
    data = json.loads(report.read_text())
    assert [page["page"] for page in data["pages"]] == [1]
    assert data["device"] == "cpu"
    assert data["models"]["layout"] == "PP-DocLayout-S"


def test_model_downloader_uses_selected_model_and_pinned_revision(custom_config, monkeypatch):
    import huggingface_hub

    script = load_script("download_ocr_models")
    model = ProjectConfig.load(custom_config).recognition_download

    def download(**kwargs):
        assert kwargs["repo_id"] == model["repository"]
        assert kwargs["revision"] == model["revision"]
        directory = Path(kwargs["local_dir"])
        assert directory == custom_config.parent / ".cache/models/ppocrv6_vi"
        directory.mkdir(parents=True)
        for name in kwargs["allow_patterns"]:
            (directory / name).write_bytes(name.encode())
        return str(directory)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(sys, "argv", ["download_ocr_models.py", "--config", str(custom_config)])
    script.main()
    manifest = json.loads(
        (custom_config.parent / ".cache/models/ppocrv6_vi/model_manifest.json").read_text()
    )
    assert manifest["sha256"] == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in model["files"]
    }


def test_official_recognition_does_not_download_custom_weights(custom_config, monkeypatch):
    import huggingface_hub

    settings = yaml.safe_load(custom_config.read_text())
    settings["models"]["recognition"] = "PP-OCRv6_small_rec"
    custom_config.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("Unexpected Vietnamese model download"),
    )
    monkeypatch.setattr(sys, "argv", ["download_ocr_models.py", "--config", str(custom_config)])
    load_script("download_ocr_models").main()
    assert not (custom_config.parent / ".cache/models").exists()


def test_gpu_check_uses_selected_gpu_index(custom_config, monkeypatch, capsys):
    settings = yaml.safe_load(custom_config.read_text())
    settings["device"] = "gpu:2"
    custom_config.write_text(yaml.safe_dump(settings))
    selected = []

    def tensor(value):
        return SimpleNamespace(mean=lambda: value)

    paddle = SimpleNamespace(
        __version__="test",
        is_compiled_with_cuda=lambda: True,
        set_device=selected.append,
        no_grad=nullcontext,
        ones=lambda *args, **kwargs: tensor(1),
        matmul=lambda *args: tensor(32),
        nn=SimpleNamespace(functional=SimpleNamespace(conv2d=lambda *args: tensor(27))),
        version=SimpleNamespace(cuda=lambda: "test"),
        device=SimpleNamespace(
            get_device=lambda: selected[-1],
            cuda=SimpleNamespace(
                device_count=lambda: 3,
                get_device_name=lambda: "Generic CUDA GPU",
                get_device_capability=lambda: [8, 6],
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setattr(sys, "argv", ["check_gpu.py", "--config", str(custom_config)])
    assert load_script("check_gpu").main() == 0
    assert selected == ["gpu:2"]
    assert json.loads(capsys.readouterr().out)["device"] == "gpu:2"
