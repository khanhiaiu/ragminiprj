"""BARTpho syllable spell-check adapter from bravend."""

import re
import unicodedata
from typing import Any

from .transformers import TransformersSeq2SeqCorrector

_TONE = {
    "òa": "oà",
    "óa": "oá",
    "ỏa": "oả",
    "õa": "oã",
    "ọa": "oạ",
    "òe": "oè",
    "óe": "oé",
    "ỏe": "oẻ",
    "õe": "oẽ",
    "ọe": "oẹ",
    "ùy": "uỳ",
    "úy": "uý",
    "ủy": "uỷ",
    "ũy": "uỹ",
    "ụy": "uỵ",
}
_TONE.update({key.capitalize(): value.capitalize() for key, value in list(_TONE.items())})
_TONE_RE = re.compile("|".join(map(re.escape, _TONE)))
_PUNCTUATION = str.maketrans(
    {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "..."}
)


class BravendCorrector(TransformersSeq2SeqCorrector):
    backend = "bravend"

    def __init__(self, config: dict[str, Any]):
        model = config["models"][self.backend]
        super().__init__(
            backend=self.backend,
            model_name=model["model_name"],
            device=config["device"],
            generation=config["generation"],
            revision=model["revision"],
            trust_remote_code=model["trust_remote_code"],
            sentence_level=True,
        )

    def _prepare_input(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text.strip()).translate(_PUNCTUATION)
        return _TONE_RE.sub(lambda match: _TONE[match.group()], normalized)
