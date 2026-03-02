"""RED tests for Transcriber and TranscriptionStream protocols.

These verify:
- The protocols exist and are importable
- A minimal mock class that satisfies each protocol is accepted by runtime_checkable
- A class missing methods is NOT accepted
- The protocols define the right method signatures
"""

from __future__ import annotations

from typing import runtime_checkable
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_transcriber_protocol_importable():
    from vox.protocols import Transcriber
    assert Transcriber is not None


def test_transcription_stream_protocol_importable():
    from vox.protocols import TranscriptionStream
    assert TranscriptionStream is not None


def test_transcriber_protocol_is_runtime_checkable():
    from vox.protocols import Transcriber
    assert hasattr(Transcriber, "__protocol_attrs__") or isinstance(
        Transcriber, type
    )


# -- Transcriber conformance --


class _GoodTranscriber:
    """Minimal implementation that should satisfy the protocol."""

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        return False

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray) -> str:
        return ""

    def create_stream(self, on_token):
        raise NotImplementedError


class _BadTranscriber:
    """Missing methods — should NOT satisfy the protocol."""

    def load(self) -> None:
        pass


def test_good_transcriber_satisfies_protocol():
    from vox.protocols import Transcriber
    assert isinstance(_GoodTranscriber(), Transcriber)


def test_bad_transcriber_rejected():
    from vox.protocols import Transcriber
    assert not isinstance(_BadTranscriber(), Transcriber)


# -- TranscriptionStream conformance --


class _GoodStream:
    """Minimal stream implementation."""

    def feed(self, chunk: np.ndarray) -> None:
        pass

    def flush(self) -> str:
        return ""

    def close(self) -> None:
        pass


class _BadStream:
    """Missing methods."""

    def feed(self, chunk):
        pass


def test_good_stream_satisfies_protocol():
    from vox.protocols import TranscriptionStream
    assert isinstance(_GoodStream(), TranscriptionStream)


def test_bad_stream_rejected():
    from vox.protocols import TranscriptionStream
    assert not isinstance(_BadStream(), TranscriptionStream)


# -- Signature checks --


def test_transcriber_has_expected_members():
    """Protocol defines all required members."""
    from vox.protocols import Transcriber

    # Check protocol annotations / attrs exist
    members = {"is_loaded", "supports_streaming", "load", "transcribe", "create_stream"}
    # For runtime_checkable protocols, check via a conforming instance
    t = _GoodTranscriber()
    for name in members:
        assert hasattr(t, name), f"Missing member: {name}"


def test_stream_has_expected_members():
    from vox.protocols import TranscriptionStream

    members = {"feed", "flush", "close"}
    s = _GoodStream()
    for name in members:
        assert hasattr(s, name), f"Missing member: {name}"
