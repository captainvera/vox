"""Tests for the rule-based text formatter pipeline.

Covers all four transforms (strip_fillers, fix_capitalization,
ensure_trailing_punctuation, collapse_whitespace) and the Formatter
class that runs them in sequence.
"""

from __future__ import annotations

import pytest

from vox.formatter import (
    Formatter,
    collapse_whitespace,
    ensure_trailing_punctuation,
    fix_capitalization,
    strip_fillers,
)


# -- strip_fillers --


class TestStripFillers:
    def test_removes_um(self):
        assert strip_fillers("um I think so") == "I think so"

    def test_removes_uh(self):
        assert strip_fillers("I uh need help") == "I need help"

    def test_removes_er(self):
        assert strip_fillers("er what was that") == "what was that"

    def test_removes_ah(self):
        assert strip_fillers("ah yes") == "yes"

    def test_removes_hmm(self):
        assert strip_fillers("hmm let me think") == "let me think"

    def test_removes_you_know(self):
        assert strip_fillers("it was you know pretty good") == "it was pretty good"

    def test_case_insensitive(self):
        assert strip_fillers("Um I think UM so") == "I think so"

    def test_removes_filler_with_comma(self):
        assert strip_fillers("um, I think so") == "I think so"

    def test_multiple_fillers(self):
        assert strip_fillers("um uh er I think") == "I think"

    def test_only_fillers(self):
        assert strip_fillers("um uh er") == ""

    def test_empty_string(self):
        assert strip_fillers("") == ""

    def test_no_fillers(self):
        assert strip_fillers("hello world") == "hello world"

    def test_collapses_double_spaces(self):
        assert strip_fillers("I  um  think") == "I think"


# -- fix_capitalization --


class TestFixCapitalization:
    def test_capitalizes_first_char(self):
        assert fix_capitalization("hello world") == "Hello world"

    def test_capitalizes_after_period(self):
        assert fix_capitalization("done. next thing") == "Done. Next thing"

    def test_capitalizes_after_exclamation(self):
        assert fix_capitalization("wow! that's great") == "Wow! That's great"

    def test_capitalizes_after_question_mark(self):
        assert fix_capitalization("really? yes") == "Really? Yes"

    def test_already_capitalized(self):
        assert fix_capitalization("Hello World") == "Hello World"

    def test_empty_string(self):
        assert fix_capitalization("") == ""

    def test_single_char(self):
        assert fix_capitalization("a") == "A"

    def test_multiple_sentences(self):
        result = fix_capitalization("one. two. three")
        assert result == "One. Two. Three"


# -- ensure_trailing_punctuation --


class TestEnsureTrailingPunctuation:
    def test_adds_period(self):
        assert ensure_trailing_punctuation("hello world") == "hello world."

    def test_keeps_existing_period(self):
        assert ensure_trailing_punctuation("hello world.") == "hello world."

    def test_keeps_exclamation(self):
        assert ensure_trailing_punctuation("wow!") == "wow!"

    def test_keeps_question_mark(self):
        assert ensure_trailing_punctuation("really?") == "really?"

    def test_strips_trailing_whitespace(self):
        assert ensure_trailing_punctuation("hello world  ") == "hello world."

    def test_empty_string(self):
        assert ensure_trailing_punctuation("") == ""


# -- collapse_whitespace --


class TestCollapseWhitespace:
    def test_collapses_spaces(self):
        assert collapse_whitespace("hello   world") == "hello world"

    def test_collapses_newlines(self):
        assert collapse_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_strips_outer_whitespace(self):
        assert collapse_whitespace("  hello  ") == "hello"

    def test_preserves_double_newline(self):
        assert collapse_whitespace("a\n\nb") == "a\n\nb"

    def test_empty_string(self):
        assert collapse_whitespace("") == ""

    def test_mixed(self):
        assert collapse_whitespace("  a   b\n\n\nc  ") == "a b\n\nc"


# -- Formatter pipeline --


class TestFormatter:
    def test_disabled_returns_input(self):
        f = Formatter(enabled=False)
        assert f.format("um hello") == "um hello"

    def test_empty_returns_empty(self):
        f = Formatter(enabled=True)
        assert f.format("") == ""

    def test_full_pipeline(self):
        f = Formatter(enabled=True)
        result = f.format("um  I think  you know  it works")
        assert result == "I think it works."

    def test_custom_transforms(self):
        f = Formatter(enabled=True, transforms=[str.upper])
        assert f.format("hello") == "HELLO"

    def test_pipeline_order(self):
        """Transforms run in order: fillers -> caps -> punct -> whitespace."""
        f = Formatter(enabled=True)
        result = f.format("um hello world")
        # strip_fillers -> "hello world"
        # fix_capitalization -> "Hello world"
        # ensure_trailing_punctuation -> "Hello world."
        # collapse_whitespace -> "Hello world."
        assert result == "Hello world."
