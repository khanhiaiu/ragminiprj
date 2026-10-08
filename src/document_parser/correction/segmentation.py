"""Tokenizer-aware segmentation shared by all correction backends."""

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectionSegment:
    text: str


class CorrectionSegmenter:
    """Split without truncation while preferring natural/legal boundaries."""

    _LEGAL_LABEL = re.compile(
        r"^\s*(?:Điều\s+\d+|Khoản\s+\d+|Điểm\s+[a-zđ]|CHƯƠNG\s+[IVXLCDM]+)\.\s*$",
        re.IGNORECASE,
    )

    def __init__(self, token_length: Callable[[str], int], max_input_tokens: int):
        self.token_length = token_length
        self.max_input_tokens = max_input_tokens

    def split(self, text: str, *, sentence_level: bool = False) -> list[CorrectionSegment]:
        if not sentence_level and self.token_length(text) <= self.max_input_tokens:
            return [CorrectionSegment(text)]
        units = self._natural_units(text)
        if sentence_level:
            segments = []
            for unit in units:
                if self.token_length(unit) <= self.max_input_tokens:
                    segments.append(unit)
                else:
                    segments.extend(self._pack_words(unit))
            return [CorrectionSegment(segment) for segment in segments]
        return [CorrectionSegment(segment) for segment in self._pack(units)]

    def _natural_units(self, text: str) -> list[str]:
        units = []
        start = 0
        for match in re.finditer(r"(?<=[.!?;:])(?:[ \t]+|\n+)|\n+", text):
            units.append(text[start : match.end()])
            start = match.end()
        if start < len(text):
            units.append(text[start:])
        if not units:
            return [text]

        merged = []
        index = 0
        while index < len(units):
            if index + 1 < len(units) and self._LEGAL_LABEL.match(units[index]):
                merged.append(units[index] + units[index + 1])
                index += 2
            else:
                merged.append(units[index])
                index += 1
        return merged

    @staticmethod
    def _word_units(text: str) -> list[str]:
        matches = list(re.finditer(r"\S+\s*", text))
        if not matches:
            return [text]
        units = []
        if matches[0].start():
            units.append(text[: matches[0].start()])
        units.extend(match.group(0) for match in matches)
        return units

    def _pack_words(self, text: str) -> list[str]:
        words = self._word_units(text)
        if len(words) == 1 and self.token_length(words[0]) > self.max_input_tokens:
            raise ValueError("A correction token is longer than max_input_tokens")
        packed = []
        current = ""
        for word in words:
            if self.token_length(word) > self.max_input_tokens:
                raise ValueError("A correction token is longer than max_input_tokens")
            if current and self.token_length(current + word) > self.max_input_tokens:
                packed.append(current)
                current = word
            else:
                current += word
        if current:
            packed.append(current)
        return packed

    def _pack(self, units: list[str]) -> list[str]:
        packed = []
        current = ""
        for unit in units:
            if self.token_length(unit) > self.max_input_tokens:
                if current:
                    packed.append(current)
                    current = ""
                packed.extend(self._pack_words(unit))
                continue
            proposed = current + unit
            if current and self.token_length(proposed) > self.max_input_tokens:
                packed.append(current)
                current = unit
            else:
                current = proposed
        if current:
            packed.append(current)
        return packed
