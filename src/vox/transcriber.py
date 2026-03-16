"""Voxtral STT — load model once, transcribe numpy audio arrays."""

from __future__ import annotations

import logging
import tempfile
from typing import Callable

import mlx.core as mx
import numpy as np
import soundfile as sf
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx import _build_prompt_tokens, load_model
from voxmlx.generate import generate

from .protocols import TranscriptionStream
from .voxtral_stream import VoxtralStream

log = logging.getLogger(__name__)


class VoxtralTranscriber:
    """Wraps voxmlx with a cached model for repeated transcriptions."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._sp = None
        self._config = None
        self._prompt_tokens: list[int] | None = None
        self._n_delay_tokens: int | None = None
        # Precomputed for streaming (set during load)
        self._text_embeds = None
        self._t_cond = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supports_streaming(self) -> bool:
        return True

    def load(self) -> None:
        """Pre-load model weights and tokenizer. Safe to call multiple times."""
        if self._model is not None:
            return
        self._model, self._sp, self._config = load_model(self.model_path)
        self._prompt_tokens, self._n_delay_tokens = _build_prompt_tokens(self._sp)

        # Precompute embeddings needed for streaming
        self._t_cond = self._model.time_embedding(
            mx.array([self._n_delay_tokens], dtype=mx.float32)
        )
        prompt_ids = mx.array([self._prompt_tokens])
        self._text_embeds = self._model.language_model.embed(prompt_ids)[0]
        mx.eval(self._t_cond, self._text_embeds)

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

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a streaming transcription session."""
        self.load()
        encoder_window = getattr(
            self._model.encoder, "sliding_window", None
        )
        log.info(
            "Creating VoxtralStream (prefix_len=%d, encoder_window=%s)",
            len(self._prompt_tokens),
            encoder_window,
        )
        return VoxtralStream(
            model=self._model,
            sp=self._sp,
            text_embeds=self._text_embeds,
            t_cond=self._t_cond,
            prefix_len=len(self._prompt_tokens),
            eos_token_id=self._sp.eos_id,
            on_token=on_token,
            encoder_window=encoder_window,
        )

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
