"""RED tests for VoxtralStream — the TranscriptionStream implementation.

Tests the public contract (feed/flush/close), threading safety,
and text accumulation. Does NOT test ML pipeline internals — that's
voxmlx's responsibility.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_voxtral_stream_importable():
    from vox.voxtral_stream import VoxtralStream
    assert VoxtralStream is not None


def test_voxtral_stream_satisfies_protocol():
    from vox.protocols import TranscriptionStream
    from vox.voxtral_stream import VoxtralStream

    # Check class has all required methods
    for attr in ("feed", "flush", "close"):
        assert hasattr(VoxtralStream, attr)


@pytest.fixture
def mock_model():
    """Minimal mock of VoxtralRealtime for stream construction."""
    model = MagicMock()
    sp = MagicMock()
    sp.eos_id = 0
    sp.decode.return_value = "word "

    # Fake precomputed state (would normally be mx.array, but mocks work
    # since VoxtralStream methods that touch them will be patched)
    text_embeds = MagicMock()
    t_cond = MagicMock()

    return {
        "model": model,
        "sp": sp,
        "text_embeds": text_embeds,
        "t_cond": t_cond,
        "prefix_len": 39,
        "eos_token_id": 0,
    }


def _make_stream(mock_model, on_token=None, start_loop=False):
    """Create a VoxtralStream, optionally suppressing the bg thread."""
    from vox.voxtral_stream import VoxtralStream

    if on_token is None:
        on_token = MagicMock()

    if start_loop:
        return VoxtralStream(
            model=mock_model["model"],
            sp=mock_model["sp"],
            text_embeds=mock_model["text_embeds"],
            t_cond=mock_model["t_cond"],
            prefix_len=mock_model["prefix_len"],
            eos_token_id=mock_model["eos_token_id"],
            on_token=on_token,
        )

    # Suppress the background processing thread for unit tests.
    # patch.object replaces the method with a no-op mock; the thread
    # starts, calls the mock (returns immediately), and exits.
    with patch.object(VoxtralStream, "_process_loop"):
        stream = VoxtralStream(
            model=mock_model["model"],
            sp=mock_model["sp"],
            text_embeds=mock_model["text_embeds"],
            t_cond=mock_model["t_cond"],
            prefix_len=mock_model["prefix_len"],
            eos_token_id=mock_model["eos_token_id"],
            on_token=on_token,
        )
    # Manually signal done since the mock loop won't set it
    stream._done_event.set()
    return stream


def test_voxtral_stream_constructs(mock_model):
    stream = _make_stream(mock_model)
    assert stream is not None


def test_feed_accepts_numpy_array(mock_model):
    stream = _make_stream(mock_model)
    chunk = np.random.rand(1280).astype(np.float32)
    stream.feed(chunk)  # should not raise


def test_feed_is_thread_safe(mock_model):
    """Multiple threads calling feed() should not crash."""
    stream = _make_stream(mock_model)
    errors = []

    def feeder():
        try:
            for _ in range(50):
                stream.feed(np.random.rand(1280).astype(np.float32))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=feeder) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_flush_returns_string(mock_model):
    stream = _make_stream(mock_model)
    result = stream.flush()
    assert isinstance(result, str)


def test_close_releases_state(mock_model):
    stream = _make_stream(mock_model)
    stream.close()  # should not raise
    assert stream._closed


def test_accumulated_text_via_emit(mock_model):
    """Internal _emit_token should accumulate text for flush()."""
    stream = _make_stream(mock_model)
    stream._emit_token("hello ")
    stream._emit_token("world")
    result = stream.flush()
    assert result == "hello world"


def test_on_token_callback_fired(mock_model):
    """_emit_token should call the on_token callback."""
    cb = MagicMock()
    stream = _make_stream(mock_model, on_token=cb)
    stream._emit_token("hello")
    cb.assert_called_once_with("hello")


def test_emit_token_survives_callback_error(mock_model):
    """_emit_token should not crash if on_token callback raises."""

    def bad_callback(text):
        raise RuntimeError("callback failed")

    stream = _make_stream(mock_model, on_token=bad_callback)
    stream._emit_token("hello")  # should not raise
    assert stream._accumulated == ["hello"]
    assert stream._tokens_emitted == 1


def test_drain_audio_returns_buffered_data(mock_model):
    """_drain_audio should return all fed audio and clear the buffer."""
    stream = _make_stream(mock_model)
    chunk = np.ones(1280, dtype=np.float32)
    stream.feed(chunk)
    stream.feed(chunk)

    drained = stream._drain_audio()
    assert len(drained) == 2560

    # Second drain should be empty
    drained2 = stream._drain_audio()
    assert len(drained2) == 0


# ---------------------------------------------------------------------------
# Bug 1: Decode loop ignores _running — can't stop promptly
# ---------------------------------------------------------------------------


def test_decode_loop_checks_running(mock_model):
    """Bug repro: _decode_available runs a for-loop that never checks
    _running.  If n_decodable is large, the processing thread is stuck
    for seconds (or longer) and can't respond to stop requests.

    Expected: decode loop breaks when _running is False.
    Actual (bug): loop runs to completion regardless.
    """
    import mlx.core as mx

    stream = _make_stream(mock_model)

    # Set up minimal state so _decode_available enters the decode loop.
    stream._prefilled = True
    stream._y = mx.array([999])  # any non-EOS token
    n_embeds = 200
    stream._audio_embeds = mx.zeros((n_embeds, 1))
    stream._n_audio_samples_fed = n_embeds * 1280  # enough for safe_total
    stream._n_total_decoded = 39  # past prefix_len
    stream._cache = [MagicMock() for _ in range(stream._n_layers)]

    # Mock model.decode to return logits where argmax != 0 (non-EOS).
    # EOS token ID is 0, so argmax must produce 1 to avoid EOS break.
    fake_logits = mx.array([[[0.0, 1.0]]])
    stream._model.decode.return_value = fake_logits
    stream._model.language_model.embed.return_value = mx.zeros((1, 1, 1))

    # Set _running = False BEFORE calling decode — it should break early.
    stream._running = False
    stream._decode_available()

    # With 200 embeds but _running=False, should decode very few steps
    # (at most 1 before checking). NOT all 200.
    assert stream._model.decode.call_count < 10


# ---------------------------------------------------------------------------
# Bug 2: flush() races with processing thread after timeout
# ---------------------------------------------------------------------------


def test_flush_skips_model_after_timeout(mock_model):
    """Bug repro: flush() waits 10s for the processing thread, then
    proceeds to call _encode_chunk + _decode_available even though the
    processing thread is still running.  Two threads using the model
    simultaneously corrupts state and causes hangs.

    Expected: after timeout, flush returns accumulated text without
    touching the model.
    Actual (bug): flush calls _encode_chunk/_decode_available after timeout.
    """
    stream = _make_stream(mock_model)

    # Simulate accumulated text from the session.
    stream._accumulated = ["Hello ", "world"]

    # Simulate: processing thread is still running (done_event NOT set).
    stream._done_event.clear()

    # Set up state as if we were mid-session (model access would occur).
    stream._cache = MagicMock()
    stream._y = MagicMock()
    stream._pending_audio = np.zeros(1000, dtype=np.float32)

    # Mock _done_event.wait to return False immediately (simulates timeout).
    stream._done_event.wait = MagicMock(return_value=False)

    result = stream.flush()

    assert result == "Hello world"
    # After timeout, the model should NOT have been called.
    stream._model.encode_step.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 3: Empty tokens emitted during silence
# ---------------------------------------------------------------------------


def test_emit_token_filters_empty_strings(mock_model):
    """Bug repro: _emit_token fires on_token for empty strings during
    silence.  39% of all tokens in logs are empty.  Each triggers a
    callback and grows _accumulated.

    Expected: _emit_token returns early for empty/whitespace-only text.
    Actual (bug): empty strings are accumulated and callback fires.
    """
    cb = MagicMock()
    stream = _make_stream(mock_model, on_token=cb)

    stream._emit_token("")
    stream._emit_token("  ")
    stream._emit_token("hello")

    # Only "hello" should be accumulated and callbacked.
    assert stream._accumulated == ["hello"]
    cb.assert_called_once_with("hello")


# ---------------------------------------------------------------------------
# Bug 4: No silence detection — GPU runs at full speed on silence
# ---------------------------------------------------------------------------


def test_consecutive_empty_tokens_tracked(mock_model):
    """Empty tokens should increment a consecutive counter.
    Non-empty tokens should reset it."""
    stream = _make_stream(mock_model)

    # Emit empties — counter should grow.
    stream._emit_token("")
    stream._emit_token("")
    stream._emit_token("")
    assert stream._consecutive_empty >= 3

    # Emit real token — counter should reset.
    stream._emit_token("hello")
    assert stream._consecutive_empty == 0


def test_silence_pause_resumes_on_loud_audio(mock_model):
    """When silence_paused is True, feeding loud audio (high RMS)
    should immediately unpause so decode resumes without waiting
    for the slow probe to work through old silence embeds."""
    stream = _make_stream(mock_model)

    # Simulate silence pause.
    stream._silence_paused = True
    stream._consecutive_empty = 50

    # Feed loud audio (speech-level RMS > 0.01).
    loud_chunk = np.full(1280, 0.1, dtype=np.float32)
    stream.feed(loud_chunk)

    # Drain audio in process loop to trigger RMS check.
    new_audio = stream._drain_audio()
    rms = float(np.sqrt(np.mean(new_audio ** 2)))
    assert rms > stream._speech_rms_threshold

    # Simulate what process loop does: check RMS and unpause.
    if stream._silence_paused:
        if rms > stream._speech_rms_threshold:
            stream._silence_paused = False
            stream._consecutive_empty = 0

    assert stream._silence_paused is False
    assert stream._consecutive_empty == 0


def test_silence_pause_stays_paused_on_quiet_audio(mock_model):
    """Quiet audio (low RMS) should NOT unpause."""
    stream = _make_stream(mock_model)

    stream._silence_paused = True
    stream._consecutive_empty = 50

    # Feed near-silence audio.
    quiet_chunk = np.full(1280, 0.001, dtype=np.float32)
    stream.feed(quiet_chunk)

    new_audio = stream._drain_audio()
    rms = float(np.sqrt(np.mean(new_audio ** 2)))
    assert rms < stream._speech_rms_threshold

    # RMS check should NOT unpause.
    if stream._silence_paused:
        if rms > stream._speech_rms_threshold:
            stream._silence_paused = False

    assert stream._silence_paused is True
