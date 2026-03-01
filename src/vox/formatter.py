"""Rule-based post-processing for transcribed text.

Each transform is a pure function (str -> str). The Formatter runs them
in sequence. Keeps the speaker's words intact — only cleans up
speech-to-text artifacts.
"""

from __future__ import annotations

import re
from typing import Callable

Transform = Callable[[str], str]


def strip_fillers(text: str) -> str:
    """Remove common filler words (um, uh, er, ah, hmm, you know)."""
    text = re.sub(r"\b(?:um|uh|er|ah|hmm)\b,?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou know,?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", text).strip()


def fix_capitalization(text: str) -> str:
    """Capitalize first character and after sentence-ending punctuation."""
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = re.sub(
        r"([.!?])\s+(\w)",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )
    return text


def ensure_trailing_punctuation(text: str) -> str:
    """Add a period if the text doesn't end with terminal punctuation."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def collapse_whitespace(text: str) -> str:
    """Normalize runs of whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# Ordered pipeline — each transform feeds into the next.
DEFAULT_TRANSFORMS: list[Transform] = [
    strip_fillers,
    fix_capitalization,
    ensure_trailing_punctuation,
    collapse_whitespace,
]


class Formatter:
    """Runs a pipeline of text transforms on transcribed output."""

    def __init__(
        self,
        enabled: bool = True,
        transforms: list[Transform] | None = None,
    ) -> None:
        self.enabled = enabled
        self._transforms = transforms or list(DEFAULT_TRANSFORMS)

    def format(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for transform in self._transforms:
            text = transform(text)
        return text
