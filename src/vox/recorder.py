"""Mic capture — start/stop recording, return numpy audio array."""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd


class Recorder:
    """Records mono audio from the default mic at a given sample rate."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

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
        with self._lock:
            self._chunks.append(indata[:, 0].copy())
