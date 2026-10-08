"""Shared lazy Hugging Face Seq2Seq implementation for correction adapters."""

import logging
from typing import Any

from .base import CorrectionCandidate
from .segmentation import CorrectionSegment, CorrectionSegmenter

logger = logging.getLogger(__name__)


class TransformersSeq2SeqCorrector:
    """Keep runtime imports and model weights lazy until first inference."""

    def __init__(
        self,
        *,
        backend: str,
        model_name: str,
        device: str,
        generation: dict[str, Any],
        revision: str | None = None,
        trust_remote_code: bool = False,
        sentence_level: bool = False,
    ):
        self.backend = backend
        self.model_name = model_name
        self.generation = generation
        self.device_setting = device
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.sentence_level = sentence_level
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device: str | None = None
        self._initialization_error: str | None = None

    @property
    def initialized(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._initialization_error:
            raise RuntimeError(self._initialization_error)
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self._resolve_device(torch)
            load_options = {
                "revision": self.revision,
                "trust_remote_code": self.trust_remote_code,
            }
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_options)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **load_options)
            model.to(device)
            model.eval()
        except Exception as exc:
            self._initialization_error = (
                f"Cannot initialize {self.backend} text correction model "
                f"{self.model_name!r} on {self.device_setting!r}: {exc}"
            )
            raise RuntimeError(self._initialization_error) from exc
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        logger.info(
            "Initialized %s text correction model %s on %s",
            self.backend,
            self.model_name,
            device,
        )

    def _resolve_device(self, torch) -> str:
        requested = self.device_setting
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested == "cpu":
            return "cpu"
        device = requested.replace("gpu", "cuda", 1)
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {requested!r} was requested but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
        return f"cuda:{index}"

    def _token_length(self, text: str) -> int:
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        ids = encoded["input_ids"]
        return len(ids[0] if ids and isinstance(ids[0], list) else ids)

    def _segments(self, text: str) -> list[CorrectionSegment]:
        return CorrectionSegmenter(self._token_length, self.generation["max_input_tokens"]).split(
            text, sentence_level=self.sentence_level
        )

    def _prepare_input(self, text: str) -> str:
        return text

    def _correct_segment(self, text: str) -> str:
        leading = text[: len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()) :]
        core = text.strip()
        if not core:
            return text
        inputs = self._tokenizer(self._prepare_input(core), return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"]
        if input_ids.shape[-1] > self.generation["max_input_tokens"]:
            raise ValueError("Correction segment exceeds max_input_tokens")
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                num_beams=self.generation["num_beams"],
                max_new_tokens=self.generation["max_new_tokens"],
                length_penalty=self.generation["length_penalty"],
                early_stopping=self.generation["early_stopping"],
            )
        corrected = self._tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return leading + corrected + trailing

    def correct(self, text: str) -> CorrectionCandidate:
        self._load()
        corrected = "".join(self._correct_segment(segment.text) for segment in self._segments(text))
        return CorrectionCandidate(
            original=text,
            corrected=corrected,
            backend=self.backend,
            model_name=self.model_name,
            changed=corrected != text,
        )
