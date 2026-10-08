"""ProtonX legal correction adapter."""

from typing import Any

from .transformers import TransformersSeq2SeqCorrector


class ProtonXCorrector(TransformersSeq2SeqCorrector):
    backend = "protonx"

    def __init__(self, config: dict[str, Any]):
        super().__init__(
            backend=self.backend,
            model_name=config["models"][self.backend]["model_name"],
            device=config["device"],
            generation=config["generation"],
        )
