"""Mic capture — start/stop recording, return numpy audio array."""

from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)

import numpy as np
import sounddevice as sd


class Recorder:
    """Records mono audio from the default mic at a given sample rate."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._on_chunk = on_chunk
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def on_chunk(self) -> Callable[[np.ndarray], None] | None:
        return self._on_chunk

    @on_chunk.setter
    def on_chunk(self, callback: Callable[[np.ndarray], None] | None) -> None:
        self._on_chunk = callback

    def start(self) -> None:
        """Open the mic and begin buffering audio."""
        self._chunks = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1-D float32 array."""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if self._chunks:
                audio = np.concatenate(self._chunks)
                self._chunks = []
                return audio
            return np.zeros(0, dtype=np.float32)

    # -- private --

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        chunk = indata[:, 0].copy()
        if self._on_chunk is not None:
            # Streaming mode: forward chunk, skip buffering (caller
            # discards stop() return value, so _chunks would be a leak).
            self._on_chunk(chunk)
        else:
            with self._lock:
                self._chunks.append(chunk)
