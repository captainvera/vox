"""RED tests for ParakeetTranscriber + ParakeetStream.

These verify:
- ParakeetTranscriber satisfies the Transcriber protocol
- ParakeetStream satisfies the TranscriptionStream protocol
- load() calls from_pretrained, is idempotent
- transcribe() returns string, writes temp WAV
- supports_streaming is True
- create_stream() returns a ParakeetStream
- Stream feed/flush/close contract
- Stream text diffing fires on_token with correct deltas
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

    # Batch: model.transcribe(path) returns AlignedResult with .text
    mock_result = MagicMock()
    mock_result.text = "Hello world."
    mock_model.transcribe.return_value = mock_result

    # Streaming: model.transcribe_stream() returns a context manager
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ctx.result.text = ""
    mock_ctx.finalized_tokens = []
    mock_ctx.draft_tokens = []
    mock_model.transcribe_stream.return_value = mock_ctx

    _mock_parakeet_mlx_mod.from_pretrained = MagicMock(return_value=mock_model)

    yield {
        "from_pretrained": _mock_parakeet_mlx_mod.from_pretrained,
        "model": mock_model,
        "result": mock_result,
        "ctx": mock_ctx,
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

    ctx = mock_parakeet_mlx["ctx"]

    with patch.object(ParakeetStream, "_process_loop"):
        stream = ParakeetStream(
            ctx=ctx,
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

    assert stream is not None
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


def test_stream_diff_text_emits_delta(mock_parakeet_mlx):
    """_diff_and_emit should detect new text and fire on_token."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    # Simulate ctx.result.text progressing
    stream._prev_text = ""
    stream._diff_and_emit("Hello ")
    cb.assert_called_with("Hello ")

    stream._diff_and_emit("Hello world")
    cb.assert_called_with("world")


def test_stream_diff_text_no_emit_when_unchanged(mock_parakeet_mlx):
    """_diff_and_emit should not fire on_token if text hasn't changed."""
    cb = MagicMock()
    stream = _make_stream(mock_parakeet_mlx, on_token=cb)

    stream._prev_text = "Hello"
    stream._diff_and_emit("Hello")
    cb.assert_not_called()


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
