"""RED tests for Recorder on_chunk callback.

These verify:
- Recorder accepts an on_chunk callback parameter
- Callback is invoked with each audio chunk during recording
- When on_chunk is None, existing behavior is preserved (chunks buffered)
- When on_chunk is set, chunks are NOT buffered (streaming mode skips it)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest


def test_recorder_accepts_on_chunk_param(mock_sd):
    from vox.recorder import Recorder
    # Should not raise
    r = Recorder(sample_rate=16_000, on_chunk=lambda chunk: None)
    assert r is not None


def test_recorder_on_chunk_default_is_none(mock_sd):
    from vox.recorder import Recorder
    r = Recorder(sample_rate=16_000)
    assert r.on_chunk is None


def test_recorder_on_chunk_called_with_audio(mock_sd):
    from vox.recorder import Recorder

    chunks_received = []
    r = Recorder(sample_rate=16_000, on_chunk=chunks_received.append)

    # Simulate start (captures the callback sd.InputStream was called with)
    r.start()
    sd_callback = mock_sd.InputStream.call_args[1]["callback"]

    # Simulate audio arriving
    fake_audio = np.random.rand(1280, 1).astype(np.float32)
    sd_callback(fake_audio, 1280, None, MagicMock())

    assert len(chunks_received) == 1
    np.testing.assert_array_equal(chunks_received[0], fake_audio[:, 0])


def test_recorder_skips_buffering_with_on_chunk(mock_sd):
    """With on_chunk set, stop() should return empty (no buffering)."""
    from vox.recorder import Recorder

    r = Recorder(sample_rate=16_000, on_chunk=lambda c: None)
    r.start()
    sd_callback = mock_sd.InputStream.call_args[1]["callback"]

    # Feed two chunks — should NOT be buffered
    chunk1 = np.ones((1280, 1), dtype=np.float32)
    chunk2 = np.ones((1280, 1), dtype=np.float32) * 0.5
    sd_callback(chunk1, 1280, None, MagicMock())
    sd_callback(chunk2, 1280, None, MagicMock())

    audio = r.stop()
    assert len(audio) == 0


def test_recorder_no_on_chunk_preserves_behavior(mock_sd):
    """Without on_chunk, existing batch behavior is unchanged."""
    from vox.recorder import Recorder

    r = Recorder(sample_rate=16_000)
    r.start()
    sd_callback = mock_sd.InputStream.call_args[1]["callback"]

    fake_audio = np.random.rand(1280, 1).astype(np.float32)
    sd_callback(fake_audio, 1280, None, MagicMock())

    audio = r.stop()
    assert len(audio) == 1280


def test_recorder_on_chunk_multiple_calls(mock_sd):
    from vox.recorder import Recorder

    cb = MagicMock()
    r = Recorder(sample_rate=16_000, on_chunk=cb)
    r.start()
    sd_callback = mock_sd.InputStream.call_args[1]["callback"]

    for _ in range(5):
        sd_callback(np.zeros((1280, 1), dtype=np.float32), 1280, None, MagicMock())

    assert cb.call_count == 5
