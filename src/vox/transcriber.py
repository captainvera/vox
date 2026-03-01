"""Voxtral STT — load model once, transcribe numpy audio arrays.

Splits audio at silence boundaries so each chunk gets a clean
encode+decode pass. This prevents the model from emitting EOS
mid-recording when it detects a pause in speech.
"""

from __future__ import annotations

import logging
import tempfile

import numpy as np
import soundfile as sf
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx import _build_prompt_tokens, load_model
from voxmlx.generate import generate

log = logging.getLogger(__name__)

# -- silence detection defaults --
MIN_SILENCE_MS = 800  # pause must be at least this long to split
SILENCE_THRESHOLD = 0.015  # RMS energy below this = silence
ENERGY_WINDOW_MS = 30  # window size for energy computation
MIN_CHUNK_MS = 1000  # discard chunks shorter than this
MAX_CHUNK_S = 60  # force-split chunks longer than this


def _split_on_silence(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    min_silence_ms: int = MIN_SILENCE_MS,
    silence_threshold: float = SILENCE_THRESHOLD,
    min_chunk_ms: int = MIN_CHUNK_MS,
    max_chunk_s: int = MAX_CHUNK_S,
) -> list[np.ndarray]:
    """Split audio into chunks at silence boundaries.

    Returns a list of numpy arrays, each a contiguous speech segment.
    Falls back to the full audio if no silences are found.
    """
    window_samples = int(sample_rate * ENERGY_WINDOW_MS / 1000)
    hop_samples = window_samples // 2

    if len(audio) < window_samples:
        return [audio]

    # Compute RMS energy per window.
    n_windows = (len(audio) - window_samples) // hop_samples + 1
    energies = np.array(
        [
            np.sqrt(
                np.mean(
                    audio[i * hop_samples : i * hop_samples + window_samples] ** 2
                )
            )
            for i in range(n_windows)
        ]
    )

    is_silent = energies < silence_threshold
    min_silent_windows = max(1, int(min_silence_ms / ENERGY_WINDOW_MS))

    # Find contiguous silence runs long enough to split at.
    split_samples: list[int] = []
    run_start: int | None = None
    for i, silent in enumerate(is_silent):
        if silent:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_silent_windows:
                mid = (run_start + i) // 2
                split_samples.append(mid * hop_samples)
            run_start = None

    # Build chunks from split points.
    boundaries = [0] + split_samples + [len(audio)]
    min_chunk_samples = int(sample_rate * min_chunk_ms / 1000)
    max_chunk_samples = int(sample_rate * max_chunk_s)

    chunks: list[np.ndarray] = []
    for i in range(len(boundaries) - 1):
        chunk = audio[boundaries[i] : boundaries[i + 1]]
        if len(chunk) < min_chunk_samples:
            continue
        # Skip chunks that are mostly silence (no real speech).
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < silence_threshold:
            continue
        # Force-split oversized chunks.
        while len(chunk) > max_chunk_samples:
            chunks.append(chunk[:max_chunk_samples])
            chunk = chunk[max_chunk_samples:]
        if len(chunk) >= min_chunk_samples:
            chunks.append(chunk)

    return chunks if chunks else [audio]


class VoxtralTranscriber:
    """Wraps voxmlx with a cached model for repeated transcriptions."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._sp = None
        self._config = None
        self._prompt_tokens: list[int] | None = None
        self._n_delay_tokens: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Pre-load model weights and tokenizer. Safe to call multiple times."""
        if self._model is not None:
            return
        self._model, self._sp, self._config = load_model(self.model_path)
        self._prompt_tokens, self._n_delay_tokens = _build_prompt_tokens(self._sp)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe a numpy audio array to text.

        Sends the full audio as a single chunk — no splitting, no filtering.
        """
        self.load()

        duration_s = len(audio) / sample_rate
        log.info("Audio %.1fs -> single chunk (no splitting)", duration_s)

        text = self._transcribe_chunk(audio, sample_rate)
        log.info("  result: %r", text[:120])
        return text.strip()

    def _transcribe_chunk(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a single audio chunk via voxmlx."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, sample_rate)
            output_tokens = generate(
                self._model,
                f.name,
                self._prompt_tokens,
                n_delay_tokens=self._n_delay_tokens,
                temperature=0.0,
                eos_token_id=self._sp.eos_id,
            )

        return self._sp.decode(
            output_tokens,
            special_token_policy=SpecialTokenPolicy.IGNORE,
        )
