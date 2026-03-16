"""Integration test: streaming transcription survives beyond 60 seconds.

Generates a 90-second synthetic WAV file via macOS `say` + `afconvert`,
feeds it to VoxtralStream in real-time-ish chunks, and asserts that
non-empty tokens are still produced in the final 30 seconds.

Requires:
  - macOS (for `say` and `afconvert`)
  - Voxtral model weights at ~/models/Voxtral-Mini-4B-Realtime-6bit

Skipped automatically when the model is unavailable or not on macOS.
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Skip entire module when not on macOS or model is absent.
_MODEL_PATH = Path.home() / "models" / "Voxtral-Mini-4B-Realtime-6bit"
_REQUIRES_MODEL = pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason=f"Model not found at {_MODEL_PATH}",
)
_REQUIRES_MACOS = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="macOS required for `say` / `afconvert`",
)

# How long the synthetic audio should be (seconds).
_DURATION = 90
# We assert non-empty tokens appear in the final N seconds.
_TAIL_WINDOW = 30

SAMPLES_PER_TOKEN = 1280  # from voxmlx.audio
SAMPLE_RATE = 16_000


def _generate_speech_wav(path: str, duration: int = _DURATION) -> None:
    """Generate a WAV file of continuous speech using macOS TTS.

    Uses `say` → AIFF → `afconvert` → 16kHz mono WAV.
    The sentence is repeated enough times to fill *duration* seconds.
    """
    sentence = "The quick brown fox jumps over the lazy dog. "
    # `say` speaks ~150 wpm ≈ 2.5 words/sec. 9 words × reps.
    reps = max(1, (duration * 3) // 9)  # generous overshoot
    text = sentence * reps

    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as aiff:
        aiff_path = aiff.name

    try:
        subprocess.run(
            ["say", "-o", aiff_path, text],
            check=True,
            timeout=duration + 30,
        )
        subprocess.run(
            [
                "afconvert",
                "-f", "WAVE",
                "-d", "LEI16@16000",
                "-c", "1",
                aiff_path,
                path,
            ],
            check=True,
            timeout=30,
        )
    finally:
        os.unlink(aiff_path)

    # Trim or verify duration.
    audio, sr = sf.read(path, dtype="float32")
    actual_dur = len(audio) / sr
    if actual_dur < duration:
        pytest.skip(
            f"Generated audio too short ({actual_dur:.1f}s < {duration}s)"
        )


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory) -> str:
    """Module-scoped fixture: 90-second WAV file of synthetic speech."""
    wav_dir = tmp_path_factory.mktemp("audio")
    wav_path = str(wav_dir / "speech_90s.wav")
    _generate_speech_wav(wav_path)
    return wav_path


@_REQUIRES_MACOS
@_REQUIRES_MODEL
def test_streaming_produces_tokens_after_60s(speech_wav):
    """Feed 90s of speech and verify tokens are emitted in the final 30s.

    This is the regression test for the encoder window bug: without
    periodic encoder resets, the model stops producing meaningful
    tokens after ~60 seconds because the encoder's RoPE positional
    encodings exceed its trained sliding_window (750 frames).

    We track the **audio position** (how many seconds of audio have
    been fed to the stream) at the time each token is emitted, not
    wall-clock time.  This makes the assertion independent of feed
    pacing.
    """
    from vox.voxtral_stream import VoxtralStream

    import mlx.core as mx
    from voxmlx import _build_prompt_tokens, load_model

    model, sp, config = load_model(str(_MODEL_PATH))
    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
    t_cond = model.time_embedding(
        mx.array([n_delay_tokens], dtype=mx.float32)
    )
    text_embeds = model.language_model.embed(mx.array([prompt_tokens]))[0]
    mx.eval(t_cond, text_embeds)

    encoder_window = getattr(model.encoder, "sliding_window", 750)

    # Collect (audio_position_seconds, text) pairs for every non-empty token.
    # audio_pos tracks how much audio has been fed at the time of emission.
    token_log: list[tuple[float, str]] = []
    audio_pos = {"samples_fed": 0}  # mutable so the callback can read it

    def on_token(text: str) -> None:
        if text.strip():
            pos_sec = audio_pos["samples_fed"] / SAMPLE_RATE
            token_log.append((pos_sec, text))

    stream = VoxtralStream(
        model=model,
        sp=sp,
        text_embeds=text_embeds,
        t_cond=t_cond,
        prefix_len=len(prompt_tokens),
        eos_token_id=sp.eos_id,
        on_token=on_token,
        encoder_window=encoder_window,
    )

    # Read the WAV and feed it in SAMPLES_PER_TOKEN-sized chunks,
    # paced at roughly real time so the processing loop can keep up.
    audio, sr = sf.read(speech_wav, dtype="float32")
    assert sr == SAMPLE_RATE

    chunk_samples = SAMPLES_PER_TOKEN
    chunk_duration = chunk_samples / SAMPLE_RATE  # 0.08s

    for offset in range(0, len(audio), chunk_samples):
        chunk = audio[offset : offset + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        stream.feed(chunk)
        audio_pos["samples_fed"] = min(offset + chunk_samples, len(audio))
        # Pace feeding at ~real time (slightly faster is fine).
        time.sleep(chunk_duration * 0.5)

    text = stream.flush()
    stream.close()

    # -- Assertions --
    total_text = "".join(t for _, t in token_log)
    total_dur = len(audio) / SAMPLE_RATE

    # Basic sanity: the stream produced some output.
    assert len(total_text) > 0, "Stream produced no text at all"

    # Core assertion: tokens were emitted when audio position was
    # past the first 60 seconds.  This proves the encoder reset
    # kept the pipeline alive well beyond the encoder window limit.
    tail_start = total_dur - _TAIL_WINDOW
    tail_tokens = [t for pos, t in token_log if pos >= tail_start]
    tail_text = "".join(tail_tokens)

    assert len(tail_text) > 0, (
        f"No tokens emitted after audio position {tail_start:.1f}s "
        f"(total_dur={total_dur:.1f}s, total_tokens={len(token_log)}, "
        f"encoder_window={encoder_window}, "
        f"resets={stream._encoder_resets}, "
        f"last_token_pos={token_log[-1][0]:.1f}s)"
        if token_log
        else f"No tokens emitted at all"
    )

    # Informational: print summary for manual inspection.
    last_pos = token_log[-1][0] if token_log else 0.0
    print(
        f"\n--- Streaming integration test ---\n"
        f"Audio duration:     {total_dur:.1f}s\n"
        f"Total tokens:       {len(token_log)}\n"
        f"Total text chars:   {len(total_text)}\n"
        f"Encoder resets:     {stream._encoder_resets}\n"
        f"Encoder window:     {encoder_window}\n"
        f"Last token at:      {last_pos:.1f}s audio position\n"
        f"Tail tokens (>{tail_start:.0f}s): {len(tail_tokens)}\n"
        f"Tail text:          {tail_text[:120]!r}\n"
        f"Full text:          {total_text[:200]!r}\n"
    )
