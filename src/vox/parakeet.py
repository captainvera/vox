"""Parakeet MLX STT — alternative lightweight backend.

Uses parakeet-mlx (600M params) as an alternative to Voxtral (4B).
Implements the Transcriber and TranscriptionStream protocols.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from typing import Callable

import numpy as np
import soundfile as sf

from .protocols import TranscriptionStream

log = logging.getLogger(__name__)


class ParakeetTranscriber:
    """Wraps parakeet-mlx with a cached model for repeated transcriptions."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._sample_rate: int = 16_000

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supports_streaming(self) -> bool:
        return True

    def load(self) -> None:
        """Pre-load model weights. Safe to call multiple times."""
        if self._model is not None:
            return
        from parakeet_mlx import from_pretrained

        log.info("Loading Parakeet model: %s", self.model_name)
        self._model = from_pretrained(self.model_name)
        self._sample_rate = self._model.preprocessor_config.sample_rate
        log.info("Parakeet model loaded (sample_rate=%d)", self._sample_rate)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe a numpy audio array to text.

        Writes a temp WAV, calls model.transcribe(path), returns result.text.
        """
        self.load()

        duration_s = len(audio) / sample_rate
        log.info("Audio %.1fs -> Parakeet batch transcribe", duration_s)

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, sample_rate)
            result = self._model.transcribe(f.name)

        text = result.text.strip()
        log.info("  result: %r", text[:120])
        return text

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a streaming transcription session."""
        self.load()
        log.info("Creating ParakeetStream")
        ctx_mgr = self._model.transcribe_stream(
            context_size=(256, 256),
        )
        ctx = ctx_mgr.__enter__()
        return ParakeetStream(
            ctx=ctx,
            sample_rate=self._sample_rate,
            on_token=on_token,
            _ctx_mgr=ctx_mgr,
        )


class ParakeetStream:
    """Incremental Parakeet transcription session.

    Created per recording via ParakeetTranscriber.create_stream().
    Wraps parakeet-mlx's StreamingParakeet context with a background
    thread that drains audio, calls add_audio(), and diffs result text
    to emit token deltas.
    """

    def __init__(
        self,
        ctx,
        sample_rate: int,
        on_token: Callable[[str], None],
        _ctx_mgr=None,
    ) -> None:
        self._ctx = ctx
        self._sample_rate = sample_rate
        self._on_token = on_token
        self._ctx_mgr = _ctx_mgr

        # Thread-safe audio buffer
        self._lock = threading.Lock()
        self._audio_buf = np.zeros(0, dtype=np.float32)

        # Text tracking for diffing
        self._prev_text = ""
        self._accumulated: list[str] = []

        # Stats
        self._start_time = time.monotonic()
        self._feed_calls = 0
        self._total_audio_samples = 0

        # Lifecycle
        self._running = True
        self._closed = False
        self._done_event = threading.Event()
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._thread.start()
        log.info("ParakeetStream started")

    # -- Public API (TranscriptionStream protocol) --

    def feed(self, chunk: np.ndarray) -> None:
        """Thread-safe. Append audio chunk to internal buffer."""
        with self._lock:
            self._audio_buf = np.append(self._audio_buf, chunk)
        self._feed_calls += 1
        self._total_audio_samples += len(chunk)

    def flush(self) -> str:
        """Stop processing, return accumulated text."""
        elapsed = time.monotonic() - self._start_time
        log.info(
            "ParakeetStream flush after %.1fs (feeds=%d, audio=%.1fs)",
            elapsed,
            self._feed_calls,
            self._total_audio_samples / self._sample_rate,
        )

        self._running = False
        if not self._done_event.wait(timeout=10):
            log.warning("ParakeetStream processing thread did not stop within 10s")

        # Feed any remaining audio
        remaining = self._drain_audio()
        if len(remaining) > 0 and self._ctx is not None:
            try:
                import mlx.core as mx

                self._ctx.add_audio(mx.array(remaining))
                new_text = self._ctx.result.text
                self._diff_and_emit(new_text)
            except Exception:
                log.exception("Final audio flush failed")

        result = "".join(self._accumulated)
        log.info("ParakeetStream flush complete: %d chars", len(result))
        return result

    def close(self) -> None:
        """Release streaming context and stop background thread."""
        self._running = False
        self._closed = True

        if self._ctx_mgr is not None:
            try:
                self._ctx_mgr.__exit__(None, None, None)
            except Exception:
                log.exception("Error exiting streaming context")
            self._ctx_mgr = None

        self._ctx = None
        log.info("ParakeetStream closed")

    # -- Internal helpers --

    def _emit_token(self, text: str) -> None:
        """Accumulate text and fire callback."""
        self._accumulated.append(text)
        try:
            self._on_token(text)
        except Exception:
            log.exception("on_token callback failed for %r", text)

    def _diff_and_emit(self, new_text: str) -> None:
        """Compare new text against previous, emit the delta via on_token."""
        if new_text == self._prev_text:
            return
        if new_text.startswith(self._prev_text):
            delta = new_text[len(self._prev_text):]
            if delta:
                self._emit_token(delta)
        elif new_text:
            # Text was corrected (draft tokens changed) — emit full new text
            # This shouldn't happen often with finalized tokens, but handle it
            self._emit_token(new_text)
        self._prev_text = new_text

    def _drain_audio(self) -> np.ndarray:
        """Drain audio buffer (called from processing thread)."""
        with self._lock:
            new_audio = self._audio_buf
            self._audio_buf = np.zeros(0, dtype=np.float32)
        return new_audio

    # -- Background processing thread --

    def _process_loop(self) -> None:
        """Drain audio, feed to parakeet ctx, diff text, emit tokens."""
        import mlx.core as mx

        log.info("ParakeetStream processing thread started")
        loop_count = 0
        try:
            while self._running:
                loop_count += 1
                new_audio = self._drain_audio()

                if len(new_audio) == 0:
                    time.sleep(0.05)
                    continue

                self._ctx.add_audio(mx.array(new_audio))
                new_text = self._ctx.result.text
                self._diff_and_emit(new_text)

                time.sleep(0.02)
        except Exception:
            log.exception(
                "ParakeetStream processing thread crashed after %d loops",
                loop_count,
            )
        finally:
            elapsed = time.monotonic() - self._start_time
            log.info(
                "ParakeetStream processing thread exited after %.1fs (loops=%d)",
                elapsed,
                loop_count,
            )
            self._done_event.set()
