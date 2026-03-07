"""Moonshine STT — alternative backend using moonshine-voice.

Uses Moonshine Voice (OnnxRuntime-based, C++ core) as an alternative
to Voxtral/Parakeet. Supports batch and true incremental streaming
with encoder caching, built-in VAD, and phrase segmentation.
Implements the Transcriber and TranscriptionStream protocols.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from .protocols import TranscriptionStream

log = logging.getLogger(__name__)


class MoonshineTranscriber:
    """Wraps moonshine-voice with a cached model for repeated transcriptions."""

    def __init__(
        self,
        language: str = "en",
        model_arch: str = "medium-streaming",
    ) -> None:
        self.language = language
        self.model_arch_name = model_arch
        self._transcriber = None
        self._model_path: str | None = None
        self._model_arch = None  # moonshine_voice.ModelArch enum

    @property
    def is_loaded(self) -> bool:
        return self._transcriber is not None

    @property
    def supports_streaming(self) -> bool:
        return "streaming" in self.model_arch_name

    def load(self) -> None:
        """Download model (if needed) and load. Safe to call multiple times."""
        if self._transcriber is not None:
            return

        from moonshine_voice import (
            Transcriber,
            get_model_for_language,
            string_to_model_arch,
        )

        arch = string_to_model_arch(self.model_arch_name)

        log.info(
            "Loading Moonshine model: language=%s, arch=%s",
            self.language,
            self.model_arch_name,
        )
        self._model_path, self._model_arch = get_model_for_language(
            wanted_language=self.language,
            wanted_model_arch=arch,
        )
        log.info("Model path: %s", self._model_path)

        self._transcriber = Transcriber(
            model_path=self._model_path,
            model_arch=self._model_arch,
        )
        log.info("Moonshine model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Batch-transcribe a numpy audio array to text."""
        self.load()

        duration_s = len(audio) / sample_rate
        log.info("Audio %.1fs -> Moonshine batch transcribe", duration_s)

        transcript = self._transcriber.transcribe_without_streaming(
            audio.tolist(),
            sample_rate=sample_rate,
        )

        text = " ".join(line.text for line in transcript.lines).strip()
        log.info("  result: %r", text[:120])
        return text

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a streaming transcription session."""
        self.load()
        log.info("Creating MoonshineStream")
        return MoonshineStream(
            transcriber=self._transcriber,
            on_token=on_token,
            sample_rate=16_000,
        )


class MoonshineStream:
    """Moonshine streaming transcription session.

    Bridges Moonshine's event-based streaming API to Vox's
    TranscriptionStream protocol (feed/flush/close).

    A background thread drains audio from feed() and pushes it to
    Moonshine's stream.  Moonshine handles VAD, encoder caching,
    and phrase segmentation internally.

    Emit-on-complete strategy: text is only emitted via on_token when
    a line is completed (LineCompleted event).  Intermediate
    LineTextChanged events are tracked but not emitted, because
    Moonshine's streaming model routinely revises earlier text via
    speculative decoding — emitting intermediate hypotheses would
    produce wrong text that can't be un-typed.
    """

    def __init__(
        self,
        transcriber,  # moonshine_voice.Transcriber
        on_token: Callable[[str], None],
        sample_rate: int = 16_000,
        update_interval: float = 0.3,
    ) -> None:
        from moonshine_voice import (
            LineCompleted,
            LineTextChanged,
            TranscriptEventListener,
        )

        self._ms_transcriber = transcriber
        self._on_token = on_token
        self._sample_rate = sample_rate

        # Audio buffer (thread-safe: written by feed, read by processing thread)
        self._lock = threading.Lock()
        self._audio_chunks: list[np.ndarray] = []

        # Text tracking for delta computation (per-line)
        self._current_line_id: int | None = None
        self._current_line_text: str = ""
        self._has_emitted_text: bool = False
        self._accumulated: list[str] = []

        # Stats
        self._start_time = time.monotonic()
        self._feed_calls = 0
        self._total_audio_samples = 0
        self._tokens_emitted = 0
        self._lines_completed = 0

        # Create Moonshine stream with event listener.
        # Capture `self` via closure (inner_self avoids shadowing).
        outer = self

        class _Listener(TranscriptEventListener):
            def on_line_text_changed(self, event):
                outer._handle_text_changed(event)

            def on_line_completed(self, event):
                outer._handle_line_completed(event)

        self._listener = _Listener()
        self._stream = transcriber.create_stream(
            update_interval=update_interval,
        )
        self._stream.add_listener(self._listener)

        # Lifecycle
        self._running = True
        self._done_event = threading.Event()
        self._stream.start()
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True,
        )
        self._thread.start()
        log.info("MoonshineStream started (update_interval=%.2fs)", update_interval)

    # -- Public API (TranscriptionStream protocol) --

    def feed(self, chunk: np.ndarray) -> None:
        """Thread-safe. Append audio chunk to internal buffer."""
        with self._lock:
            self._audio_chunks.append(chunk)
        self._feed_calls += 1
        self._total_audio_samples += len(chunk)

    def flush(self) -> str:
        """Stop processing, flush remaining audio, return accumulated text."""
        elapsed = time.monotonic() - self._start_time
        log.info(
            "MoonshineStream flush after %.1fs "
            "(feeds=%d, audio=%.1fs, tokens=%d, lines=%d)",
            elapsed,
            self._feed_calls,
            self._total_audio_samples / self._sample_rate,
            self._tokens_emitted,
            self._lines_completed,
        )

        # Stop the processing loop and wait for it to exit.
        self._running = False
        if not self._done_event.wait(timeout=10):
            log.warning("MoonshineStream processing thread did not stop within 10s")

        # Drain remaining audio and feed to Moonshine.
        remaining = self._drain_audio()
        if len(remaining) > 0:
            self._stream.add_audio(remaining, self._sample_rate)

        # Stop stream — triggers final transcription + events.
        self._stream.stop()

        result = "".join(self._accumulated)
        log.info(
            "MoonshineStream flush complete: %d chars, %r",
            len(result),
            result[:120],
        )
        return result

    def close(self) -> None:
        """Release stream resources (not the shared transcriber)."""
        self._running = False
        try:
            self._stream.remove_all_listeners()
            self._stream.close()
        except Exception:
            log.exception("Error closing Moonshine stream")
        log.info("MoonshineStream closed")

    # -- Event handlers (called from processing thread via add_audio) --

    def _handle_text_changed(self, event) -> None:
        """Track current line state — no emission until LineCompleted.

        Moonshine's streaming model revises text via speculative decoding
        (e.g. "Ever heard?" → "Ever failed.").  Emitting intermediate
        hypotheses would produce wrong text that can't be retracted.
        """
        line = event.line
        self._current_line_id = line.line_id
        self._current_line_text = line.text

    def _handle_line_completed(self, event) -> None:
        """Moonshine line completed — emit full final text."""
        line = event.line
        final_text = line.text
        self._lines_completed += 1

        if final_text:
            if self._has_emitted_text:
                self._emit_token(" ")
            self._emit_token(final_text)

        self._current_line_id = None
        self._current_line_text = ""

    # -- Internal helpers --

    def _emit_token(self, text: str) -> None:
        """Accumulate text and fire callback."""
        if not text:
            return
        self._accumulated.append(text)
        self._tokens_emitted += 1
        if text.strip():
            self._has_emitted_text = True
        try:
            self._on_token(text)
        except Exception:
            log.exception("on_token callback failed for %r", text)

    @staticmethod
    def _compute_delta(old_text: str, new_text: str) -> str:
        """Return the new suffix when text extends, or empty string.

        Strict prefix match only.  Not used in the emit-on-complete
        strategy (kept for potential future use in incremental mode).
        """
        if new_text.startswith(old_text):
            return new_text[len(old_text):]
        return ""

    def _drain_audio(self) -> np.ndarray:
        """Drain audio buffer (called from processing thread)."""
        with self._lock:
            chunks = self._audio_chunks
            self._audio_chunks = []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    # -- Background processing thread --

    def _process_loop(self) -> None:
        """Drain audio buffer and feed to Moonshine stream.

        Moonshine's add_audio() triggers transcription updates at the
        configured interval.  Events fire synchronously within add_audio(),
        so on_token callbacks happen on this thread.

        Chunks are fed individually (not concatenated) because Moonshine's
        add_audio() checks whether to run update_transcription() AT MOST
        ONCE per call.  Feeding one large array would skip intermediate
        updates and produce text in large jumps instead of incrementally.
        """
        log.info("Processing thread started")
        loop_count = 0
        try:
            while self._running:
                loop_count += 1
                with self._lock:
                    chunks = self._audio_chunks
                    self._audio_chunks = []
                if chunks:
                    for chunk in chunks:
                        self._stream.add_audio(chunk, self._sample_rate)
                else:
                    time.sleep(0.02)
        except Exception:
            log.exception(
                "MoonshineStream processing thread crashed after %d loops",
                loop_count,
            )
        finally:
            elapsed = time.monotonic() - self._start_time
            log.info(
                "MoonshineStream processing thread exited after %.1fs "
                "(loops=%d, tokens=%d, lines=%d)",
                elapsed,
                loop_count,
                self._tokens_emitted,
                self._lines_completed,
            )
            self._done_event.set()
