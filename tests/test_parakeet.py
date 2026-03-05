"""RED tests for ParakeetTranscriber + ParakeetStream.

These verify:
- ParakeetTranscriber satisfies the Transcriber protocol
- ParakeetStream satisfies the TranscriptionStream protocol
- load() calls from_pretrained, is idempotent
- transcribe() returns string, writes temp WAV
- supports_streaming is True (periodic batch mode)
- create_stream() returns a ParakeetStream (passes model directly)
- Stream feed/flush/close contract
- Stream prefix-diff fires on_token with correct deltas
- Thread safety of feed()
- Factory dispatch returns correct types
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock parakeet_mlx at module level before any vox.parakeet import
# ---------------------------------------------------------------------------

_mock_parakeet_mlx_mod = MagicMock()
if "parakeet_mlx" not in sys.modules:
    sys.modules["parakeet_mlx"] = _mock_parakeet_mlx_mod


# ---------------------------------------------------------------------------
# Fixtures �� mock parakeet_mlx so nothing real loads
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_parakeet_mlx():
    """Mock parakeet_mlx internals so ParakeetTranscriber can be tested."""
    mock_model = MagicMock()
    mock_model.preprocessor_config.sample_rate = 16000

    # Batch via file: model.transcribe(path) returns AlignedResult with .text
    mock_result = MagicMock()
    mock_result.text = "Hello world."
    mock_model.transcribe.return_value = mock_result

    # Batch via mel: model.generate(mel) returns [AlignedResult]
    mock_gen_result = MagicMock()
    mock_gen_result.text = "Hello world."
    mock_model.generate.return_value = [mock_gen_result]

    _mock_parakeet_mlx_mod.from_pretrained = MagicMock(return_value=mock_model)

    yield {
        "from_pretrained": _mock_parakeet_mlx_mod.from_pretrained,
        "model": mock_model,
        "result": mock_result,
        "gen_result": mock_gen_result,
    }


def _make_transcriber():
    """Import and construct a ParakeetTranscriber."""
    from vox.parakeet import ParakeetTranscriber

    return ParakeetTranscriber(model_name="mlx-community/parakeet-tdt-0.6b-v3")


def _make_stream(mock_parakeet_mlx, on_token=None):
    """Create a ParakeetStream with bg thread suppressed."""
    from vox.parakeet import ParakeetStream

    if on_token is None:
        on_token = MagicMock()

    model = mock_parakeet_mlx["model"]

    with patch.object(ParakeetStream, "_process_loop"):
        stream = ParakeetStream(
            model=model,
            sample_rate=16000,
            on_token=on_token,
        )
    stream._done_event.set()
    return stream


# ---------------------------------------------------------------------------
# ParakeetTranscriber — protocol conformance
# ---------------------------------------------------------------------------


def test_parakeet_transcriber_importable():
    from vox.parakeet import ParakeetTranscriber

    assert ParakeetTranscriber is not None


def test_parakeet_satisfies_transcriber_protocol(mock_parakeet_mlx):
    from vox.protocols import Transcriber

    t = _make_transcriber()
    assert isinstance(t, Transcriber)


def test_parakeet_has_supports_streaming(mock_parakeet_mlx):
    t = _make_transcriber()
    assert isinstance(t.supports_streaming, bool)


def test_parakeet_supports_streaming_is_true(mock_parakeet_mlx):
    """Streaming via periodic batch transcription. See issues doc."""
    t = _make_transcriber()
    assert t.supports_streaming is True


def test_parakeet_has_create_stream(mock_parakeet_mlx):
    t = _make_transcriber()
    assert callable(t.create_stream)


# ---------------------------------------------------------------------------
# ParakeetTranscriber — load lifecycle
# ---------------------------------------------------------------------------


def test_parakeet_is_loaded_false_initially(mock_parakeet_mlx):
    t = _make_transcriber()
    assert t.is_loaded is False


def test_parakeet_is_loaded_true_after_load(mock_parakeet_mlx):
    t = _make_transcriber()
    t.load()
    assert t.is_loaded is True


def test_parakeet_load_calls_from_pretrained(mock_parakeet_mlx):
    t = _make_transcriber()
    t.load()
    mock_parakeet_mlx["from_pretrained"].assert_called_once_with(
        "mlx-community/parakeet-tdt-0.6b-v3"
    )


def test_parakeet_load_idempotent(mock_parakeet_mlx):
    t = _make_transcriber()
    t.load()
    t.load()
    assert mock_parakeet_mlx["from_pretrained"].call_count == 1


# ---------------------------------------------------------------------------
# ParakeetTranscriber — batch transcribe
# ---------------------------------------------------------------------------


def test_parakeet_transcribe_returns_string(mock_parakeet_mlx):
    t = _make_transcriber()
    t.load()

    audio = np.random.rand(16_000).astype(np.float32)
    result = t.transcribe(audio)
    assert isinstance(result, str)
    assert result == "Hello world."


def test_parakeet_transcribe_calls_model_with_temp_file(mock_parakeet_mlx):
    """transcribe() should write a temp WAV and pass the path to model."""
    t = _make_transcriber()
    t.load()

    audio = np.random.rand(16_000).astype(np.float32)
    t.transcribe(audio)

    mock_parakeet_mlx["model"].transcribe.assert_called_once()
    call_args = mock_parakeet_mlx["model"].transcribe.call_args
    path_arg = call_args[0][0]
    assert str(path_arg).endswith(".wav")


def test_parakeet_transcribe_auto_loads(mock_parakeet_mlx):
    """transcribe() should auto-load model if not yet loaded."""
    t = _make_transcriber()
    audio = np.random.rand(16_000).astype(np.float32)
    t.transcribe(audio)
    assert t.is_loaded is True


# ---------------------------------------------------------------------------
# ParakeetTranscriber — create_stream
# ---------------------------------------------------------------------------


def test_parakeet_create_stream_returns_stream(mock_parakeet_mlx):
    from vox.parakeet import ParakeetStream

    t = _make_transcriber()
    t.load()

    cb = MagicMock()
    with patch.object(ParakeetStream, "_process_loop"):
        stream = t.create_stream(on_token=cb)

    assert isinstance(stream, ParakeetStream)
    stream._done_event.set()
    stream.close()


def test_parakeet_create_stream_does_not_use_transcribe_stream(mock_parakeet_mlx):
    """create_stream passes model directly — no transcribe_stream context."""
    from vox.parakeet import ParakeetStream

    t = _make_transcriber()
    t.load()

    cb = MagicMock()
    with patch.object(ParakeetStream, "_process_loop"):
        t.create_stream(on_token=cb)

    # transcribe_stream should never be called
    mock_parakeet_mlx["model"].transcribe_stream.assert_not_called()


def test_parakeet_create_stream_passes_model(mock_parakeet_mlx):
    """ParakeetStream should receive the model object directly."""
    from vox.parakeet import ParakeetStream

    t = _make_transcriber()
    t.load()

    cb = MagicMock()
    with patch.object(ParakeetStream, "_process_loop"):
        stream = t.create_stream(on_token=cb)

    assert stream._model is mock_parakeet_mlx["model"]
    stream._done_event.set()
    stream.close()


# ---------------------------------------------------------------------------
# ParakeetStream — protocol conformance
# ---------------------------------------------------------------------------


def test_parakeet_stream_importable():
    from vox.parakeet import ParakeetStream

    assert ParakeetStream is not None


def test_parakeet_stream_satisfies_protocol():
    from vox.parakeet import ParakeetStream
    from vox.protocols import TranscriptionStream

    for attr in ("feed", "flush", "close"):
        assert hasattr(ParakeetStream, attr)


# ---------------------------------------------------------------------------
# ParakeetStream — feed / flush / close
# ---------------------------------------------------------------------------


def test_stream_constructs(mock_parakeet_mlx):
    stream = _make_stream(mock_parakeet_mlx)
    assert stream is not None


def test_stream_feed_accepts_numpy(mock_parakeet_mlx):
    stream = _make_stream(mock_parakeet_mlx)
    chunk = np.random.rand(1280).astype(np.float32)
    stream.feed(chunk)  # should not raise


def test_stream_feed_is_thread_safe(mock_parakeet_mlx):
    """Multiple threads calling feed() should not crash."""
    stream = _make_stream(mock_parakeet_mlx)
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


def test_stream_flush_returns_string(mock_parakeet_mlx):
    stream = _make_stream(mock_parakeet_mlx)
    result = stream.flush()
    assert isinstance(result, str)


def test_stream_close_releases_state(mock_parakeet_mlx):
    stream = _make_stream(mock_parakeet_mlx)
    stream.close()
    assert stream._closed
    assert stream._model is None


def test_stream_drain_audio(mock_parakeet_mlx):
    """_drain_audio should return all fed audio and clear the buffer."""
    stream = _make_stream(mock_parakeet_mlx)
    chunk = np.ones(1280, dtype=np.float32)
    stream.feed(chunk)
    stream.feed(chunk)

    drained = stream._drain_audio()
    assert len(drained) == 2560

    drained2 = stream._drain_audio()
    assert len(drained2) == 0


# ---------------------------------------------------------------------------
# ParakeetStream — text diffing and on_token callback
# ---------------------------------------------------------------------------


def test_stream_emit_token_accumulates(mock_parakeet_mlx):
    stream = _make_stream(mock_parakeet_mlx)
    stream._emit_token("hello ")
    stream._emit_token("world")
    result = stream.flush()
    assert result == "hello world"


def test_stream_on_token_callback_fired(mock_parakeet_mlx):
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)
    stream._emit_token("hello")
    cb.assert_called_once_with("hello")


def test_stream_emit_token_survives_callback_error(mock_parakeet_mlx):
    def bad_callback(text):
        raise RuntimeError("callback failed")

    stream = _make_stream(mock_parakeet_mlx, on_token=bad_callback)
    stream._emit_token("hello")  # should not raise
    assert stream._accumulated == ["hello"]


def test_stream_compute_delta_appended_text(mock_parakeet_mlx):
    """_compute_delta returns only the new suffix when text extends."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._compute_delta("Hello ", "Hello world") == "world"


def test_stream_compute_delta_no_change(mock_parakeet_mlx):
    """_compute_delta returns empty string when text is unchanged."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._compute_delta("Hello", "Hello") == ""


def test_stream_compute_delta_from_empty(mock_parakeet_mlx):
    """_compute_delta returns full text when previous was empty."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._compute_delta("", "Hello world") == "Hello world"


def test_stream_compute_delta_revision_skipped(mock_parakeet_mlx):
    """_compute_delta returns empty when model revises (strict prefix only)."""
    from vox.parakeet import ParakeetStream

    # Same length, no common prefix — can't un-type.
    assert ParakeetStream._compute_delta("abc", "xyz") == ""

    # Longer but different prefix — NOT emitted (would re-type garbage).
    assert ParakeetStream._compute_delta("abc", "xyzw") == ""

    # Punctuation revision (common in batch mode):
    # "Hello." → "Hello world" — period removed, text extended.
    assert ParakeetStream._compute_delta("Hello.", "Hello world") == ""

    # Partial prefix match then divergence:
    assert ParakeetStream._compute_delta(
        "Hi, this is uh Parakeet.",
        "Hi, this is uh Parakeet live test",
    ) == ""


def test_stream_compute_delta_shorter_text(mock_parakeet_mlx):
    """_compute_delta returns empty string when new text is shorter (regression)."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._compute_delta("Hello world", "Hello") == ""


# ---------------------------------------------------------------------------
# ParakeetStream — flush with batch transcription
# ---------------------------------------------------------------------------


def test_stream_flush_transcribes_remaining_audio(mock_parakeet_mlx):
    """flush() should batch-transcribe any audio fed before flushing."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    # Feed audio directly into the growing buffer (bypass _process_loop).
    audio = np.random.rand(16000).astype(np.float32)
    stream.feed(audio)

    # Mock _transcribe_buffer to return known text.
    with patch.object(stream, "_transcribe_buffer", return_value="Hello world") as mock_tb:
        result = stream.flush()
        mock_tb.assert_called_once()

    assert result == "Hello world"


def test_stream_flush_emits_delta_via_callback(mock_parakeet_mlx):
    """flush() should emit the text delta via on_token callback."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    audio = np.random.rand(16000).astype(np.float32)
    stream.feed(audio)

    with patch.object(stream, "_transcribe_buffer", return_value="Hello world"):
        stream.flush()

    cb.assert_called_once_with("Hello world")


def test_stream_flush_emits_only_new_delta(mock_parakeet_mlx):
    """flush() should only emit text not already emitted by the process loop."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    # Simulate process loop having already emitted partial text.
    stream._emitted_text = "Hello "
    stream._accumulated = ["Hello "]

    audio = np.random.rand(16000).astype(np.float32)
    stream.feed(audio)

    with patch.object(stream, "_transcribe_buffer", return_value="Hello world"):
        result = stream.flush()

    # Should emit only the delta.
    cb.assert_called_once_with("world")
    assert result == "Hello world"


def test_stream_flush_no_audio_returns_empty(mock_parakeet_mlx):
    """flush() with no audio fed should return empty string."""
    stream = _make_stream(mock_parakeet_mlx)
    result = stream.flush()
    assert result == ""


def test_stream_flush_appends_to_growing_buffer(mock_parakeet_mlx):
    """flush() should drain remaining chunks into _all_audio before transcribing."""
    stream = _make_stream(mock_parakeet_mlx)

    # Simulate process loop having already consumed some audio.
    stream._all_audio = np.ones(8000, dtype=np.float32)

    # Feed more audio that hasn't been drained yet.
    stream.feed(np.ones(8000, dtype=np.float32))

    with patch.object(stream, "_transcribe_buffer", return_value="text") as mock_tb:
        stream.flush()

    # _transcribe_buffer should receive the full combined buffer.
    call_audio = mock_tb.call_args[0][0]
    assert len(call_audio) == 16000


# ---------------------------------------------------------------------------
# ParakeetStream — process loop (tested via controlled invocation)
# ---------------------------------------------------------------------------


def test_stream_confirm_text_common_prefix(mock_parakeet_mlx):
    """_confirm_text returns the common prefix of old and new batch results."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._common_prefix(
        "Hello world.", "Hello world!"
    ) == "Hello world"

    assert ParakeetStream._common_prefix(
        "Hey test for parate real time.",
        "Hey test for parakeet real time.",
    ) == "Hey test for para"


def test_stream_confirm_text_identical(mock_parakeet_mlx):
    """_common_prefix of identical strings is the full string."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._common_prefix("Hello", "Hello") == "Hello"


def test_stream_confirm_text_empty(mock_parakeet_mlx):
    """_common_prefix with empty string returns empty."""
    from vox.parakeet import ParakeetStream

    assert ParakeetStream._common_prefix("", "Hello") == ""
    assert ParakeetStream._common_prefix("Hello", "") == ""


def test_stream_confirmation_emits_stable_text(mock_parakeet_mlx):
    """Text should only be emitted once confirmed across 2 consecutive batches."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    # Simulate batch sequence: model revises "parate" → "parakeet".
    # Batch 1: first result, nothing confirmed yet (no previous batch).
    stream._update_confirmed("Hey test for parate real time.")
    assert stream._confirmed_text == ""  # nothing confirmed after 1 batch
    cb.assert_not_called()

    # Batch 2: model revises. Common prefix = "Hey test for para".
    stream._update_confirmed("Hey test for parakeet real time.")
    assert stream._confirmed_text == "Hey test for para"

    # Emit the confirmed text.
    stream._emit_confirmed()
    cb.assert_called_once_with("Hey test for para")
    assert stream._emitted_text == "Hey test for para"

    # Batch 3: model stabilizes. Common prefix extends.
    cb.reset_mock()
    stream._update_confirmed("Hey test for parakeet real time. It works.")
    assert stream._confirmed_text == "Hey test for parakeet real time."

    stream._emit_confirmed()
    cb.assert_called_once_with("keet real time.")
    assert stream._emitted_text == "Hey test for parakeet real time."


def test_stream_confirmation_no_emit_without_two_batches(mock_parakeet_mlx):
    """Nothing should be emitted until text is confirmed (2+ batches)."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    stream._update_confirmed("Hello world")
    stream._emit_confirmed()
    cb.assert_not_called()


def test_stream_confirmation_stable_prefix_extends(mock_parakeet_mlx):
    """When batches extend text without revision, confirmed text grows."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    stream._update_confirmed("Hello ")
    stream._update_confirmed("Hello world")
    # Common prefix = "Hello " (6 chars). Confirmed = "Hello ".
    assert stream._confirmed_text == "Hello "

    stream._emit_confirmed()
    cb.assert_called_once_with("Hello ")

    cb.reset_mock()
    stream._update_confirmed("Hello world. More text.")
    # Common prefix of "Hello world" and "Hello world. More text." = "Hello world"
    assert stream._confirmed_text == "Hello world"

    stream._emit_confirmed()
    cb.assert_called_once_with("world")


# ---------------------------------------------------------------------------
# ParakeetStream — silence detection
# ---------------------------------------------------------------------------


def test_stream_is_speech_detects_speech(mock_parakeet_mlx):
    """_is_speech returns True for audio with energy above threshold."""
    from vox.parakeet import ParakeetStream

    # Sine wave — clearly speech-level energy.
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    speech = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert ParakeetStream._is_speech(speech) is True


def test_stream_is_speech_detects_silence(mock_parakeet_mlx):
    """_is_speech returns False for near-silent audio."""
    from vox.parakeet import ParakeetStream

    silence = np.zeros(16000, dtype=np.float32)
    assert ParakeetStream._is_speech(silence) is False

    # Very quiet noise — below threshold.
    quiet = (np.random.rand(16000).astype(np.float32) - 0.5) * 0.001
    assert ParakeetStream._is_speech(quiet) is False


def test_stream_silence_still_added_to_buffer(mock_parakeet_mlx):
    """Silent audio should be added to _all_audio (flush needs it)
    but should NOT trigger a re-transcription."""
    stream = _make_stream(mock_parakeet_mlx)

    # Simulate: process loop drains silence and appends to buffer.
    silence = np.zeros(8000, dtype=np.float32)
    stream._all_audio = np.ones(16000, dtype=np.float32)  # existing speech

    # After drain, silence is appended to buffer.
    stream._all_audio = np.concatenate([stream._all_audio, silence])
    assert len(stream._all_audio) == 24000  # buffer grew


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_returns_parakeet_for_parakeet_backend(mock_parakeet_mlx):
    from vox.config import Config
    from vox.parakeet import ParakeetTranscriber

    config = Config(backend="parakeet")

    from vox.__main__ import _make_transcriber

    t = _make_transcriber(config)
    assert isinstance(t, ParakeetTranscriber)


def test_factory_returns_voxtral_for_voxtral_backend():
    from vox.config import Config

    config = Config(backend="voxtral")

    from vox.__main__ import _make_transcriber

    t = _make_transcriber(config)

    from vox.transcriber import VoxtralTranscriber

    assert isinstance(t, VoxtralTranscriber)
