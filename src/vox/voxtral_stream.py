"""Streaming transcription session for Voxtral.

Extracts the closure state from voxmlx/stream.py into a class that
implements the TranscriptionStream protocol. Audio chunks are fed
via feed() from any thread; a background processing thread runs the
incremental encoder/decoder pipeline and emits tokens via on_token.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import mlx.core as mx
import numpy as np
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx.audio import SAMPLES_PER_TOKEN, log_mel_spectrogram_step
from voxmlx.cache import RotatingKVCache

log = logging.getLogger(__name__)

N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17


class VoxtralStream:
    """Incremental Voxtral transcription session.

    Created per recording via VoxtralTranscriber.create_stream().
    Mirrors the logic in voxmlx/stream.py but as a reusable class.
    """

    def __init__(
        self,
        model,
        sp,
        text_embeds: mx.array,
        t_cond: mx.array,
        prefix_len: int,
        eos_token_id: int,
        on_token: Callable[[str], None],
        temperature: float = 0.0,
        encoder_window: int | None = None,
    ) -> None:
        # Shared model references (not owned, from VoxtralTranscriber)
        self._model = model
        self._sp = sp
        self._text_embeds = text_embeds
        self._t_cond = t_cond
        self._prefix_len = prefix_len
        self._eos_token_id = eos_token_id
        self._on_token = on_token
        self._temperature = temperature

        # Encoder window: the encoder's trained sliding_window size.
        # When the encoder KV cache reaches this many frames, reset
        # encoder state to prevent RoPE positional drift.  The decoder
        # (KV cache, last token) is preserved for seamless output.
        # None = no periodic resets (legacy behavior).
        self._encoder_window = encoder_window

        self._n_layers = len(model.language_model.layers)

        # Audio buffer (thread-safe: written by feed(), read by _process_loop)
        # Uses a list of chunks (O(1) append) instead of np.append (O(n) copy).
        self._lock = threading.Lock()
        self._audio_chunks: list[np.ndarray] = []

        # Accumulated text for flush()
        self._accumulated: list[str] = []

        # Per-session decoder state
        self._cache = None
        self._y = None

        # Per-session incremental encoder state
        self._audio_tail = None
        self._conv1_tail = None
        self._conv2_tail = None
        self._encoder_cache = None
        self._ds_buf = None

        # Buffers and counters
        self._pending_audio = np.zeros(0, dtype=np.float32)
        self._audio_embeds = None
        self._n_audio_samples_fed = 0
        self._n_total_decoded = 0
        self._first_cycle = True
        self._prefilled = False

        # Silence detection
        self._consecutive_empty = 0
        self._silence_threshold = 25  # ~2s at 12.5 tokens/sec
        self._silence_paused = False
        self._speech_rms_threshold = 0.003  # RMS above this = speech

        # Stats
        self._start_time = time.monotonic()
        self._tokens_emitted = 0
        self._eos_count = 0
        self._encode_calls = 0
        self._encoder_resets = 0
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
        log.info(
            "Stream started (prefix_len=%d, eos_id=%d, encoder_window=%s)",
            prefix_len, eos_token_id, encoder_window,
        )

    # -- Public API (TranscriptionStream protocol) --

    def feed(self, chunk: np.ndarray) -> None:
        """Thread-safe. Append audio chunk to internal buffer."""
        with self._lock:
            self._audio_chunks.append(chunk)
        self._feed_calls += 1
        self._total_audio_samples += len(chunk)

    def flush(self) -> str:
        """Stop processing, flush remaining audio with right padding,
        return accumulated text."""
        elapsed = time.monotonic() - self._start_time
        log.info(
            "Stream flush requested after %.1fs "
            "(tokens=%d, eos=%d, encodes=%d, enc_resets=%d, feeds=%d, audio=%.1fs)",
            elapsed,
            self._tokens_emitted,
            self._eos_count,
            self._encode_calls,
            self._encoder_resets,
            self._feed_calls,
            self._total_audio_samples / 16_000,
        )

        self._running = False
        if not self._done_event.wait(timeout=10):
            log.warning(
                "Stream processing thread did not stop within 10s "
                "— returning partial text (skipping final flush to "
                "avoid concurrent model access)"
            )
            result = "".join(self._accumulated)
            log.info("Stream flush (partial): %d chars, %r", len(result), result[:120])
            return result

        # Final flush: feed remaining audio + right padding
        remaining = self._drain_audio()
        self._pending_audio = np.append(self._pending_audio, remaining)

        if self._cache is not None and self._y is not None:
            log.info(
                "Final flush: pending=%.3fs, adding right pad (%d tokens)",
                len(self._pending_audio) / 16_000,
                N_RIGHT_PAD_TOKENS,
            )
            right_pad = np.zeros(
                N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
            )
            flush_chunk = np.concatenate([self._pending_audio, right_pad])
            self._encode_chunk(flush_chunk)

            if self._audio_embeds is not None:
                self._decode_available(self._audio_embeds.shape[0])
        else:
            log.info(
                "Final flush skipped (cache=%s, y=%s)",
                self._cache is not None,
                self._y is not None,
            )

        # Flush last pending token
        if self._y is not None:
            token_id = self._y.item()
            if token_id != self._eos_token_id:
                text = self._sp.decode(
                    [token_id],
                    special_token_policy=SpecialTokenPolicy.IGNORE,
                )
                self._emit_token(text)

        result = "".join(self._accumulated)
        log.info("Stream flush complete: %d chars, %r", len(result), result[:120])
        return result

    def close(self) -> None:
        """Release caches and stop background thread."""
        self._running = False
        self._closed = True
        self._cache = None
        self._encoder_cache = None
        self._y = None
        self._audio_embeds = None
        log.info("Stream closed")

    # -- Internal helpers --

    def _emit_token(self, text: str) -> None:
        """Accumulate text and fire callback.

        Filters empty/whitespace-only tokens (produced during silence).
        Tracks consecutive empty tokens for silence detection.
        """
        if not text or not text.strip():
            self._consecutive_empty += 1
            if (
                not self._silence_paused
                and self._consecutive_empty >= self._silence_threshold
            ):
                self._silence_paused = True
                log.info(
                    "Silence detected (%d consecutive empty tokens) "
                    "— pausing decode, will reset on next speech",
                    self._consecutive_empty,
                )
            return

        self._consecutive_empty = 0

        self._accumulated.append(text)
        self._tokens_emitted += 1
        try:
            self._on_token(text)
        except Exception:
            log.exception("on_token callback failed for %r", text)

    def _drain_audio(self) -> np.ndarray:
        """Drain audio buffer (called from processing thread)."""
        with self._lock:
            chunks = self._audio_chunks
            self._audio_chunks = []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def _sample(self, logits: mx.array) -> mx.array:
        if self._temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(
            logits[0, -1:] / self._temperature
        ).squeeze()

    def _encode_chunk(self, chunk: np.ndarray) -> None:
        """Run incremental mel + encoder on a chunk of audio."""
        self._encode_calls += 1
        mel, self._audio_tail = log_mel_spectrogram_step(
            chunk, self._audio_tail
        )
        new_embeds, self._conv1_tail, self._conv2_tail, self._encoder_cache, self._ds_buf = (
            self._model.encode_step(
                mel,
                self._conv1_tail,
                self._conv2_tail,
                self._encoder_cache,
                self._ds_buf,
            )
        )
        if new_embeds is not None:
            mx.eval(new_embeds)
            n_new = new_embeds.shape[0]
            if self._audio_embeds is not None:
                self._audio_embeds = mx.concatenate(
                    [self._audio_embeds, new_embeds]
                )
            else:
                self._audio_embeds = new_embeds
            log.debug(
                "Encoded %d samples -> %d new embeds (total undecoded: %d)",
                len(chunk),
                n_new,
                self._audio_embeds.shape[0],
            )

    def _decode_available(self, max_steps: int | None = None) -> None:
        """Decode available positions, emitting tokens via callback."""
        if self._audio_embeds is None:
            return

        safe_total = (
            N_LEFT_PAD_TOKENS
            + self._n_audio_samples_fed // SAMPLES_PER_TOKEN
        )
        n_decodable = min(
            self._audio_embeds.shape[0],
            safe_total - self._n_total_decoded,
        )
        if max_steps is not None:
            n_decodable = min(n_decodable, max_steps)

        if n_decodable <= 0:
            return

        if not self._prefilled:
            if self._n_total_decoded + self._audio_embeds.shape[0] < self._prefix_len:
                log.debug(
                    "Waiting for prefill: have %d embeds, need %d",
                    self._n_total_decoded + self._audio_embeds.shape[0],
                    self._prefix_len,
                )
                return

            log.info(
                "Prefilling decoder (prefix_len=%d, audio_embeds=%d, audio_fed=%.1fs)",
                self._prefix_len,
                self._audio_embeds.shape[0],
                self._n_audio_samples_fed / 16_000,
            )

            self._cache = [
                RotatingKVCache(8192) for _ in range(self._n_layers)
            ]

            prefix_embeds = (
                self._text_embeds + self._audio_embeds[: self._prefix_len]
            )
            prefix_embeds = prefix_embeds[None, :, :]

            logits = self._model.decode(
                prefix_embeds, self._t_cond, "causal", self._cache
            )
            mx.eval(
                logits,
                *[x for c in self._cache for x in (c.keys, c.values)],
            )

            self._y = self._sample(logits)
            mx.async_eval(self._y)

            self._audio_embeds = self._audio_embeds[self._prefix_len :]
            self._n_total_decoded = self._prefix_len
            self._prefilled = True
            log.info("Prefill complete, remaining embeds: %d", self._audio_embeds.shape[0])

            # Recompute decodable after consuming prefix
            n_decodable = min(
                self._audio_embeds.shape[0],
                safe_total - self._n_total_decoded,
            )

        if n_decodable <= 0:
            return

        # Decode loop (mirrors decode_steps in stream.py)
        n_consumed = 0
        hit_eos = False
        for i in range(n_decodable):
            if not self._running:
                log.info("Decode interrupted (_running=False) at step %d/%d", i, n_decodable)
                break

            token_embed = self._model.language_model.embed(
                self._y.reshape(1, 1)
            )[0, 0]
            step_embed = (self._audio_embeds[i] + token_embed)[None, None, :]
            logits = self._model.decode(
                step_embed, self._t_cond, mask=None, cache=self._cache
            )
            next_y = self._sample(logits)
            mx.async_eval(next_y)

            token_id = self._y.item()
            if token_id == self._eos_token_id:
                log.info(
                    "EOS at position %d (total_decoded=%d, audio_fed=%.1fs, tokens_emitted=%d)",
                    i,
                    self._n_total_decoded + i,
                    self._n_audio_samples_fed / 16_000,
                    self._tokens_emitted,
                )
                self._cache = None
                self._y = None
                n_consumed = i
                hit_eos = True
                self._eos_count += 1
                break

            text = self._sp.decode(
                [token_id],
                special_token_policy=SpecialTokenPolicy.IGNORE,
            )
            self._emit_token(text)

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            self._y = next_y
            n_consumed = i + 1

        self._n_total_decoded += n_consumed

        # Trim consumed embeddings
        if self._audio_embeds.shape[0] > n_consumed:
            self._audio_embeds = self._audio_embeds[n_consumed:]
        else:
            self._audio_embeds = None

        if hit_eos:
            log.info("EOS reset — starting new segment")
            self._reset_state()

    def _reset_state(self) -> None:
        """Reset all encoder/decoder state (EOS or silence boundary)."""
        self._cache = None
        self._y = None
        self._audio_tail = None
        self._conv1_tail = None
        self._conv2_tail = None
        self._encoder_cache = None
        self._ds_buf = None
        self._pending_audio = np.zeros(0, dtype=np.float32)
        self._audio_embeds = None
        self._n_audio_samples_fed = 0
        self._n_total_decoded = 0
        self._first_cycle = True
        self._prefilled = False

    def _reset_encoder_state(self) -> None:
        """Reset encoder state while preserving the decoder.

        The voxmlx encoder was trained with a sliding_window of 750
        positions but the streaming path creates its KV cache with
        max_size=100,000 and never applies a sliding-window mask.
        After ~encoder_window frames the RoPE positional encodings
        leave the training distribution and the encoder produces
        degraded embeddings that decode to empty tokens.

        This method resets all encoder-side state (conv tails, encoder
        KV cache, downsampling buffer, mel overlap) and the associated
        bookkeeping counters.  The next processing-loop iteration will
        start a fresh encoder segment with left-pad silence, while the
        decoder continues with its warm KV cache for seamless text
        output.  Undecoded embeddings from the old encoder are
        discarded (at most a few 80ms frames at the boundary).
        """
        self._encoder_resets += 1
        encoder_offset = (
            self._encoder_cache[0].offset
            if self._encoder_cache
            else 0
        )
        log.info(
            "Encoder reset #%d (encoder_offset=%d, window=%s, "
            "audio_fed=%.1fs, tokens_emitted=%d)",
            self._encoder_resets,
            encoder_offset,
            self._encoder_window,
            self._n_audio_samples_fed / 16_000,
            self._tokens_emitted,
        )

        # Clear encoder pipeline state.
        self._audio_tail = None
        self._conv1_tail = None
        self._conv2_tail = None
        self._encoder_cache = None
        self._ds_buf = None

        # Discard stale embeds from the old encoder segment.
        self._audio_embeds = None

        # Reset counters so safe_total / n_decodable math stays
        # correct for the new segment.
        self._n_audio_samples_fed = 0
        self._n_total_decoded = 0

        # Next encode will add left-pad silence for the fresh segment.
        self._first_cycle = True

        # Decoder state intentionally preserved:
        #   _cache, _y, _prefilled, _accumulated

    # -- Background processing thread --

    def _process_loop(self) -> None:
        """Main processing loop — mirrors stream.py's while True loop."""
        log.info("Processing thread started")
        loop_count = 0
        try:
            while self._running:
                loop_count += 1
                new_audio = self._drain_audio()

                # -- Silence pause: drain audio, check RMS, skip encode/decode --
                if self._silence_paused:
                    # Reset state once on entering silence so the model
                    # starts fresh when speech resumes (like a new segment).
                    if self._cache is not None:
                        log.info("Resetting encoder/decoder state for silence pause")
                        self._reset_state()
                        mx.clear_cache()

                    if len(new_audio) > 0:
                        rms = float(np.sqrt(np.mean(new_audio ** 2)))
                        log.debug("Silence probe RMS=%.5f (threshold=%.4f)", rms, self._speech_rms_threshold)
                        if rms > self._speech_rms_threshold:
                            self._silence_paused = False
                            self._consecutive_empty = 0
                            # Keep this audio — it's speech.  _first_cycle
                            # is True from _reset_state, so next iteration
                            # does left-pad + prefill.  Lookback buffer
                            # already has the onset in _pending_audio.
                            self._pending_audio = np.append(
                                self._pending_audio, new_audio
                            )
                            log.info(
                                "Speech detected (RMS=%.4f) — resuming "
                                "(will re-prefill as new segment, "
                                "lookback=%.3fs)",
                                rms,
                                len(self._pending_audio) / 16_000,
                            )
                            continue

                        # Keep a lookback buffer (~0.5s) to capture
                        # speech onset.  The beginning of a word often
                        # has low RMS (partial consonant), so discarding
                        # all silence audio loses the first syllable.
                        self._pending_audio = np.append(
                            self._pending_audio, new_audio
                        )
                        max_lookback = int(0.5 * 16_000)
                        if len(self._pending_audio) > max_lookback:
                            self._pending_audio = self._pending_audio[
                                -max_lookback:
                            ]

                    time.sleep(0.1)
                    continue

                # -- Normal path: buffer audio, encode, decode --
                if len(new_audio) > 0:
                    self._pending_audio = np.append(
                        self._pending_audio, new_audio
                    )

                if (
                    self._first_cycle
                    and len(self._pending_audio) < SAMPLES_PER_TOKEN
                ):
                    time.sleep(0.02)
                    continue

                # Encode new audio
                if (
                    self._first_cycle
                    and len(self._pending_audio) >= SAMPLES_PER_TOKEN
                ):
                    # First cycle: add left pad
                    left_pad = np.zeros(
                        N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN,
                        dtype=np.float32,
                    )
                    n_feed = (
                        len(self._pending_audio) // SAMPLES_PER_TOKEN
                    ) * SAMPLES_PER_TOKEN
                    chunk = np.concatenate(
                        [left_pad, self._pending_audio[:n_feed]]
                    )
                    self._pending_audio = self._pending_audio[n_feed:]
                    self._n_audio_samples_fed += n_feed
                    log.info(
                        "First encode: %.3fs audio + %d left-pad tokens",
                        n_feed / 16_000,
                        N_LEFT_PAD_TOKENS,
                    )
                    self._encode_chunk(chunk)
                    self._first_cycle = False

                elif (
                    not self._first_cycle
                    and len(self._pending_audio) >= SAMPLES_PER_TOKEN
                ):
                    n_feed = (
                        len(self._pending_audio) // SAMPLES_PER_TOKEN
                    ) * SAMPLES_PER_TOKEN
                    chunk = self._pending_audio[:n_feed]
                    self._pending_audio = self._pending_audio[n_feed:]
                    self._n_audio_samples_fed += n_feed
                    self._encode_chunk(chunk)

                # Check if encoder cache has exceeded its trained
                # sliding window.  If so, reset encoder state to
                # prevent RoPE positional drift.  Pending audio is
                # kept and will be re-encoded in the next iteration.
                if (
                    self._encoder_window is not None
                    and self._encoder_cache is not None
                    and self._encoder_cache[0].offset
                    >= self._encoder_window
                ):
                    self._reset_encoder_state()
                    continue

                if self._audio_embeds is None:
                    time.sleep(0.02)
                    continue

                self._decode_available()
                mx.clear_cache()
                time.sleep(0.02)
        except Exception:
            log.exception(
                "Processing thread crashed after %d loops, %.1fs, %d tokens emitted",
                loop_count,
                time.monotonic() - self._start_time,
                self._tokens_emitted,
            )
        finally:
            elapsed = time.monotonic() - self._start_time
            log.info(
                "Processing thread exited after %.1fs "
                "(loops=%d, tokens=%d, eos=%d, encodes=%d, enc_resets=%d, running=%s)",
                elapsed,
                loop_count,
                self._tokens_emitted,
                self._eos_count,
                self._encode_calls,
                self._encoder_resets,
                self._running,
            )
            self._done_event.set()
