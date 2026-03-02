"""Protocols for transcription backends.

Defines the contract that all model backends must satisfy.
Models that don't support streaming return supports_streaming=False
and raise NotImplementedError from create_stream().
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TranscriptionStream(Protocol):
    """Incremental transcription session.

    Created per recording via Transcriber.create_stream().
    Holds the encoder/decoder state for one streaming session.
    """

    def feed(self, chunk: np.ndarray) -> None:
        """Push an audio chunk into the streaming pipeline."""
        ...

    def flush(self) -> str:
        """Drain remaining tokens and return final accumulated text."""
        ...

    def close(self) -> None:
        """Release model caches and session state."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Backend-agnostic transcription interface.

    Every model backend implements this. The app only depends on
    this protocol �� never on a concrete transcriber class.
    """

    @property
    def is_loaded(self) -> bool: ...

    @property
    def supports_streaming(self) -> bool: ...

    def load(self) -> None:
        """Load model weights. Safe to call multiple times."""
        ...

    def transcribe(self, audio: np.ndarray) -> str:
        """Batch-transcribe a complete audio array to text."""
        ...

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a streaming transcription session.

        Raises NotImplementedError if supports_streaming is False.
        """
        ...
