"""BMD1905 Vietnamese correction adapter."""

from typing import Any

from .transformers import TransformersSeq2SeqCorrector


class BMD1905Corrector(TransformersSeq2SeqCorrector):
    backend = "bmd1905"

    def __init__(self, config: dict[str, Any]):
        super().__init__(
            backend=self.backend,
            model_name=config["models"][self.backend]["model_name"],
            device=config["device"],
            generation=config["generation"],
        )
