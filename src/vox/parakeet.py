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
        # Periodic batch transcription: re-run model.generate() on the
        # growing audio buffer every ~2s during recording, diff text,
        # emit stable deltas.  See parakeet-realtime-issues.md.
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
        """Create a periodic-batch streaming transcription session.

        Passes the model directly to ParakeetStream — no transcribe_stream
        context manager.  See parakeet-realtime-issues.md for rationale.
        """
        self.load()
        log.info("Creating ParakeetStream (periodic batch)")
        return ParakeetStream(
            model=self._model,
            sample_rate=self._sample_rate,
            on_token=on_token,
        )


class ParakeetStream:
    """Periodic-batch Parakeet transcription session.

    Created per recording via ParakeetTranscriber.create_stream().
    Receives the model directly (no transcribe_stream context).
    A background thread periodically batch-transcribes the full
    accumulated audio buffer and emits text deltas via on_token.
    """

    # Minimum total audio before the first batch transcription.
    _MIN_FIRST_SAMPLES = 16000  # 1.0 s @ 16 kHz

    # Minimum new audio before subsequent transcriptions.
    _MIN_NEW_SAMPLES = 8000  # 0.5 s @ 16 kHz

    # Seconds between batch transcriptions.
    _INTERVAL = 1.5

    # RMS energy threshold for speech detection.  Audio chunks below
    # this are considered silence and won't trigger re-transcription
    # (but are still added to the buffer so flush has full audio).
    _SILENCE_RMS = 0.01

    def __init__(
        self,
        model,
        sample_rate: int,
        on_token: Callable[[str], None],
    ) -> None:
        self._model = model
        self._sample_rate = sample_rate
        self._on_token = on_token

        # Thread-safe audio buffer — O(1) append per feed() call.
        self._lock = threading.Lock()
        self._audio_chunks: list[np.ndarray] = []
        self._audio_chunks_len = 0

        # Growing audio buffer (never trimmed) — concatenated from
        # _audio_chunks on each transcription cycle.
        self._all_audio = np.zeros(0, dtype=np.float32)

        # Confirmation buffer — text is only emitted once it's stable
        # across 2 consecutive batch transcriptions.  This prevents
        # garbage from prefix instability (model revising punctuation,
        # spelling, word choices as more audio is added).
        self._last_batch_text = ""   # previous batch result
        self._confirmed_text = ""    # longest common prefix across batches
        self._emitted_text = ""      # what's been typed at cursor
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
        log.info("ParakeetStream started (periodic batch)")

    # -- Public API (TranscriptionStream protocol) --

    def feed(self, chunk: np.ndarray) -> None:
        """Thread-safe. O(1) append to internal buffer."""
        n = len(chunk)
        with self._lock:
            self._audio_chunks.append(chunk)
            self._audio_chunks_len += n
        self._feed_calls += 1
        self._total_audio_samples += n

    def flush(self) -> str:
        """Stop processing, do final batch transcription, return text."""
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

        # Drain any remaining audio into the growing buffer.
        remaining = self._drain_audio()
        if len(remaining) > 0:
            self._all_audio = (
                np.concatenate([self._all_audio, remaining])
                if len(self._all_audio) > 0
                else remaining
            )

        # Final batch transcription of full audio.
        # This is authoritative — emit everything not yet typed,
        # bypassing confirmation (no future batch to confirm against).
        if len(self._all_audio) > 0 and self._model is not None:
            try:
                text = self._transcribe_buffer(self._all_audio)
                delta = self._compute_delta(self._emitted_text, text)
                if delta:
                    self._emit_token(delta)
                    self._emitted_text = text
            except Exception:
                log.exception("Final batch transcription failed")

        result = "".join(self._accumulated)
        log.info("ParakeetStream flush complete: %d chars", len(result))
        return result

    def close(self) -> None:
        """Release model reference and stop background thread."""
        self._running = False
        self._closed = True
        self._model = None
        log.info("ParakeetStream closed")

    # -- Internal helpers --

    def _emit_token(self, text: str) -> None:
        """Accumulate text and fire callback."""
        self._accumulated.append(text)
        try:
            self._on_token(text)
        except Exception:
            log.exception("on_token callback failed for %r", text)

    @staticmethod
    def _common_prefix(a: str, b: str) -> str:
        """Return the longest common prefix of two strings."""
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return a[:i]

    def _update_confirmed(self, new_batch_text: str) -> None:
        """Update confirmation buffer with a new batch result.

        Computes the common prefix between this batch and the previous
        one.  If it's longer than the current confirmed text, extend
        confirmed text to that prefix.  Always updates _last_batch_text.

        First call (no previous batch) sets _last_batch_text only —
        nothing can be confirmed from a single observation.
        """
        if self._last_batch_text == "":
            # First batch — nothing to compare against.
            self._last_batch_text = new_batch_text
            return

        prefix = self._common_prefix(self._last_batch_text, new_batch_text)
        if len(prefix) > len(self._confirmed_text):
            self._confirmed_text = prefix
        self._last_batch_text = new_batch_text

    def _emit_confirmed(self) -> None:
        """Emit confirmed text trimmed to the last word boundary.

        Avoids emitting partial words (e.g. "para" when the model
        hasn't settled on "parate" vs "parakeet" yet).  The held-back
        partial word will be emitted once it stabilises in a later
        batch, or by flush() which bypasses word-boundary trimming.
        """
        if not self._confirmed_text:
            return
        # Trim to last word boundary (space) so we never emit mid-word.
        emit_up_to = self._confirmed_text
        last_space = emit_up_to.rfind(" ")
        if last_space > len(self._emitted_text):
            emit_up_to = emit_up_to[: last_space + 1]
        delta = self._compute_delta(self._emitted_text, emit_up_to)
        if delta:
            self._emit_token(delta)
            self._emitted_text = emit_up_to

    @staticmethod
    def _compute_delta(old_text: str, new_text: str) -> str:
        """Return the new suffix when text extends, or empty string.

        Strict prefix matching only.  If new_text starts with old_text,
        return the suffix.  Otherwise return "" — the model revised
        earlier text and we can't un-type what's already at cursor.

        This prevents garbage re-emission when batch transcription
        changes punctuation, word choices, or casing as more audio
        is added.
        """
        if new_text.startswith(old_text):
            return new_text[len(old_text):]
        return ""

    @staticmethod
    def _is_speech(audio: np.ndarray) -> bool:
        """Return True if audio chunk has energy above silence threshold."""
        if len(audio) == 0:
            return False
        rms = float(np.sqrt(np.mean(audio**2)))
        return rms > ParakeetStream._SILENCE_RMS

    def _transcribe_buffer(self, audio: np.ndarray) -> str:
        """Batch-transcribe audio via get_logmel + model.generate.

        No temp files — computes mel directly from numpy array.
        """
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(
            mx.array(audio, dtype=mx.float32),
            self._model.preprocessor_config,
        )
        results = self._model.generate(mel)
        return results[0].text.strip()

    def _drain_audio(self) -> np.ndarray:
        """Drain audio buffer (called from processing thread)."""
        with self._lock:
            chunks = self._audio_chunks
            self._audio_chunks = []
            self._audio_chunks_len = 0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    # -- Background processing thread --

    def _process_loop(self) -> None:
        """Periodically batch-transcribe accumulated audio, emit deltas."""
        sr = self._sample_rate
        transcribe_count = 0
        last_transcribe_time = 0.0
        first_done = False

        try:
            while self._running:
                # Peek at buffered audio length without draining.
                with self._lock:
                    new_len = self._audio_chunks_len
                total_len = len(self._all_audio) + new_len

                # Gate: need minimum audio before first transcription.
                if not first_done and total_len < self._MIN_FIRST_SAMPLES:
                    time.sleep(0.05)
                    continue

                # Gate: need new audio to bother transcribing.
                if new_len == 0:
                    time.sleep(0.05)
                    continue

                # Gate: respect interval between transcriptions.
                now = time.monotonic()
                if first_done and (now - last_transcribe_time) < self._INTERVAL:
                    time.sleep(0.05)
                    continue

                # Drain new audio into growing buffer.
                new_audio = self._drain_audio()
                if len(new_audio) == 0:
                    continue
                has_speech = self._is_speech(new_audio)
                self._all_audio = (
                    np.concatenate([self._all_audio, new_audio])
                    if len(self._all_audio) > 0
                    else new_audio
                )

                # Skip transcription if new audio is silence.
                # Buffer still grows (flush needs full audio) but we
                # don't re-transcribe — prevents regurgitation and
                # normalization drift during silence.
                if not has_speech and first_done:
                    log.debug("silence detected, skipping transcription")
                    continue

                # Batch transcribe full buffer.
                transcribe_count += 1
                t0 = time.monotonic()
                text = self._transcribe_buffer(self._all_audio)
                dt = time.monotonic() - t0

                log.info(
                    "batch #%d: %.1fs audio, %.3fs compute, %d chars %r",
                    transcribe_count,
                    len(self._all_audio) / sr,
                    dt,
                    len(text),
                    text[:80],
                )

                # Update confirmation buffer and emit stable text.
                self._update_confirmed(text)
                self._emit_confirmed()

                first_done = True
                last_transcribe_time = time.monotonic()

        except Exception:
            log.exception(
                "ParakeetStream processing thread crashed "
                "(transcriptions=%d)",
                transcribe_count,
            )
        finally:
            elapsed = time.monotonic() - self._start_time
            log.info(
                "ParakeetStream processing thread exited after %.1fs "
                "(transcriptions=%d)",
                elapsed, transcribe_count,
            )
            self._done_event.set()
