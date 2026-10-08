"""High-precision signals deciding whether OCR text merits model inference."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorrectionDecision:
    should_correct: bool
    reasons: list[str]
    suspicious_tokens: list[str] = field(default_factory=list)
    replacement_options: dict[str, list[str]] = field(default_factory=dict)


_KNOWN_REPLACEMENTS = {
    "hòna": ["hòa"],
    "vhiệt": ["việt"],
    "lhàm": ["làm"],
    "lưực": ["lực"],
    "lích": ["ích"],
    "nêền": ["nền"],
    "phuục": ["phục"],
    "chiịu": ["chịu"],
    "duùng": ["dùng"],
    "giìn": ["gìn"],
    "ngưười": ["người"],
}
_KNOWN_OCR_ERRORS = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _KNOWN_REPLACEMENTS)) + r")\b", re.IGNORECASE
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _base_letter(character: str) -> str:
    decomposed = unicodedata.normalize("NFD", character.casefold())
    return decomposed[0] if decomposed else character.casefold()


class CorrectionDetector:
    """Favor precision over recall so clean legal text bypasses the model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def decide(self, text: str, metadata: dict[str, Any]) -> CorrectionDecision:
        if not text.strip():
            return CorrectionDecision(False, [])
        if not self.config.get("enabled", True):
            return CorrectionDecision(True, ["detector_disabled"])

        reasons = []
        suspicious_tokens = []
        replacement_options: dict[str, list[str]] = {}
        confidence = metadata.get("ocr_confidence")
        if (
            self.config.get("use_ocr_confidence", True)
            and isinstance(confidence, (int, float))
            and confidence < self.config["min_ocr_confidence"]
        ):
            reasons.append("low_ocr_confidence")

        if self.config.get("use_spelling_flags", True):
            flagged = [
                issue.get("observed")
                for issue in metadata.get("ocr_review", [])
                if issue.get("status") == "needs_review"
                and issue.get("reason") in {"repeated_letter", "spelling", "spelling_anomaly"}
                and isinstance(issue.get("observed"), str)
            ]
            if flagged:
                reasons.append("ocr_review_spelling_flag")
                suspicious_tokens.extend(flagged)
                for issue in metadata.get("ocr_review", []):
                    observed = issue.get("observed")
                    suggestions = issue.get("suggestions", [])
                    if observed in flagged and suggestions:
                        replacement_options.setdefault(observed.casefold(), []).extend(suggestions)

        if self.config.get("use_vocabulary", True):
            known = [match.group(0) for match in _KNOWN_OCR_ERRORS.finditer(text)]
            if known:
                reasons.append("known_vietnamese_ocr_pattern")
                suspicious_tokens.extend(known)
                for token in known:
                    replacement_options[token.casefold()] = _KNOWN_REPLACEMENTS[token.casefold()]
            repeated = [
                token for token in _WORD.findall(text) if self._has_repeated_base_character(token)
            ]
            if repeated:
                reasons.append("repeated_base_character")
                suspicious_tokens.extend(repeated)
                for token in repeated:
                    candidates = self._repeated_base_candidates(token)
                    replacement_options.setdefault(token.casefold(), []).extend(candidates)

        punctuation = bool(re.search(r"(?:\s+[,:;!?])|(?:[,:;!?]{2,})", text))
        if punctuation:
            reasons.append("punctuation_anomaly")

        # Low confidence is supporting evidence, not enough by itself to expose clean
        # legal text to a generative model.
        should_correct = bool(suspicious_tokens or punctuation)
        return CorrectionDecision(
            should_correct,
            list(dict.fromkeys(reasons)),
            list(dict.fromkeys(suspicious_tokens)),
            {token: list(dict.fromkeys(value)) for token, value in replacement_options.items()},
        )

    @staticmethod
    def _has_repeated_base_character(token: str) -> bool:
        # Require at least one accented character or differing code point. This avoids
        # flagging ordinary Vietnamese/foreign double letters such as "oo" too broadly.
        return any(
            left != right and _base_letter(left) == _base_letter(right)
            for left, right in zip(token, token[1:])
        )

    @staticmethod
    def _repeated_base_candidates(token: str) -> list[str]:
        candidates = []
        for index, (left, right) in enumerate(zip(token, token[1:])):
            if left != right and _base_letter(left) == _base_letter(right):
                candidates.extend(
                    (token[:index] + token[index + 1 :], token[: index + 1] + token[index + 2 :])
                )
        return candidates
