"""Safety checks that keep correction candidates from rewriting legal facts."""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .base import CorrectionCandidate


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    status: str
    reason: str
    edit_ratio: float
    protected_tokens_preserved: bool
    structure_preserved: bool


_URL = re.compile(r"(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_IP_ADDRESS = re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)*(?:[-+][A-Za-z0-9.-]+)?\b", re.IGNORECASE)
_DATE = re.compile(
    r"\b(?:ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"\b(?:\d+|[A-ZĐ]{2,})(?:[/.-](?:\d+|[A-ZĐ]{2,})){1,}(?:-[A-ZĐ]{2,})*\b")
_CODE = re.compile(r"\b(?=[A-Za-zĐđ0-9_-]*\d)(?=[A-Za-zĐđ0-9_-]*[A-Za-zĐđ])[A-Za-zĐđ0-9_-]{3,}\b")
_SOURCE_IDENTIFIER = re.compile(r"\b[A-Za-zĐđ][A-Za-zĐđ0-9]*(?:[_-][A-Za-zĐđ0-9]+)+\b")
_KNOWN_TECHNICAL_NAME = re.compile(r"\b(?:Qdrant)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?!\w)")
_PERCENT = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*\s*%")
_CURRENCY = re.compile(
    r"(?:\b(?:VND|VNĐ|USD|EUR)\s*[+-]?\d[\d.,]*|[+-]?\d[\d.,]*\s*(?:đồng|VNĐ|VND|USD|EUR|₫|\$|€))\b",
    re.IGNORECASE,
)
_LEGAL_MARKER = re.compile(
    r"\b(?:CHƯƠNG|PHẦN|MỤC|Điều|Khoản|Điểm)\s+"
    r"(?:\d+[A-Za-z]?|[IVXLCDM]+|[a-zđ])(?:[.)])?",
    re.IGNORECASE,
)
_LINE_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[a-zđ][.)]|[-•])")
_EDIT_TOKEN = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


class CorrectionSafetyGate:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _protected(self, text: str) -> dict[str, list[str]]:
        protected = {}
        if self.config.get("preserve_urls", True):
            protected["urls"] = [token.rstrip(".,;:!?") for token in _URL.findall(text)]
        if self.config.get("preserve_emails", True):
            protected["emails"] = _EMAIL.findall(text)
        if self.config.get("preserve_dates", True):
            protected["dates"] = _DATE.findall(text)
        if self.config.get("preserve_identifiers", True):
            protected["identifiers"] = (
                _IDENTIFIER.findall(text)
                + _CODE.findall(text)
                + _SOURCE_IDENTIFIER.findall(text)
                + _KNOWN_TECHNICAL_NAME.findall(text)
            )
            protected["ip_addresses"] = _IP_ADDRESS.findall(text)
            protected["versions"] = _VERSION.findall(text)
        if self.config.get("preserve_numbers", True):
            protected["numbers"] = _NUMBER.findall(text)
            protected["percentages"] = _PERCENT.findall(text)
            protected["currency"] = _CURRENCY.findall(text)
        return protected

    @staticmethod
    def _structure(text: str) -> tuple[list[str], list[str], int, int]:
        legal = [re.sub(r"\s+", " ", token).casefold() for token in _LEGAL_MARKER.findall(text)]
        line = [re.sub(r"\s+", "", token).casefold() for token in _LINE_MARKER.findall(text)]
        nonempty_lines = sum(bool(value.strip()) for value in text.splitlines())
        sentence_boundaries = len(re.findall(r"[.!?]+(?=\s|$)", text))
        return legal, line, nonempty_lines, sentence_boundaries

    @staticmethod
    def _edits_are_local(
        original: str,
        corrected: str,
        allowed_tokens: list[str],
        replacement_options: dict[str, list[str]],
        allow_punctuation: bool,
    ) -> bool:
        allowed = {token.casefold() for token in allowed_tokens}
        replacements = {
            token.casefold(): {value.casefold() for value in values}
            for token, values in replacement_options.items()
        }
        original_tokens = _EDIT_TOKEN.findall(original)
        corrected_tokens = _EDIT_TOKEN.findall(corrected)
        matcher = SequenceMatcher(None, original_tokens, corrected_tokens, autojunk=False)
        for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes():
            if operation == "equal":
                continue
            removed = original_tokens[left_start:left_end]
            inserted = corrected_tokens[right_start:right_end]
            removed_words = [token for token in removed if token.isalpha()]
            inserted_words = [token for token in inserted if token.isalpha()]
            if any(token.casefold() not in allowed for token in removed_words):
                return False
            if operation == "insert" and any(token.isalpha() for token in inserted):
                return False
            if operation == "replace":
                if len(removed_words) != len(inserted_words):
                    return False
                for source, target in zip(removed_words, inserted_words):
                    options = replacements.get(source.casefold())
                    if options and target.casefold() not in options:
                        return False
            punctuation_changed = any(not token.isalnum() for token in [*removed, *inserted])
            if punctuation_changed and not allow_punctuation:
                return False
        return True

    def evaluate(
        self,
        candidate: CorrectionCandidate,
        *,
        allowed_edit_tokens: list[str] | None = None,
        replacement_options: dict[str, list[str]] | None = None,
        allow_punctuation_edits: bool = False,
    ) -> SafetyDecision:
        original = unicodedata.normalize("NFC", candidate.original)
        corrected = unicodedata.normalize("NFC", candidate.corrected)
        ratio = _levenshtein(original, corrected) / max(len(original), 1)
        protected = self._protected(original) == self._protected(corrected)
        structure = not self.config.get("preserve_structure", True) or (
            bool(corrected.strip()) and self._structure(original) == self._structure(corrected)
        )

        if not candidate.changed:
            return SafetyDecision(
                False, "not_needed", "candidate_unchanged", ratio, protected, structure
            )
        if not self.config.get("enabled", True):
            return SafetyDecision(True, "accepted", "safety_disabled", ratio, protected, structure)
        if not protected:
            return SafetyDecision(
                False, "rejected", "protected_tokens_changed", ratio, protected, structure
            )
        if not structure:
            return SafetyDecision(
                False, "rejected", "structure_changed", ratio, protected, structure
            )
        if ratio > self.config["max_auto_edit_ratio"]:
            return SafetyDecision(
                False, "needs_review", "edit_ratio_exceeded", ratio, protected, structure
            )
        if allowed_edit_tokens is not None and not self._edits_are_local(
            original,
            corrected,
            allowed_edit_tokens,
            replacement_options or {},
            allow_punctuation_edits,
        ):
            return SafetyDecision(
                False, "rejected", "unsupported_edit_scope", ratio, protected, structure
            )
        return SafetyDecision(True, "accepted", "safe_candidate", ratio, protected, structure)
