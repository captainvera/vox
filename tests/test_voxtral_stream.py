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
