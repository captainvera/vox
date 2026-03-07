"""RED tests for MoonshineTranscriber + MoonshineStream.

These verify:
- MoonshineTranscriber satisfies the Transcriber protocol
- MoonshineStream satisfies the TranscriptionStream protocol
- load() calls moonshine_voice APIs, is idempotent
- transcribe() returns string
- supports_streaming depends on model arch name
- create_stream() returns a MoonshineStream
- Stream feed/flush/close contract
- Stream delta computation fires on_token with correct deltas
- Thread safety of feed()
- Factory dispatch returns correct type

Bug reproductions:
- Bug 1: _process_loop concatenates chunks -> fewer Moonshine updates
- Bug 2: Missing spaces between completed lines
"""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock moonshine_voice at module level before any vox.moonshine import
# ---------------------------------------------------------------------------


class _FakeTranscriptEventListener:
    """Stand-in ABC for moonshine_voice.TranscriptEventListener."""

    def on_line_started(self, event):
        pass

    def on_line_updated(self, event):
        pass

    def on_line_text_changed(self, event):
        pass

    def on_line_completed(self, event):
        pass

    def on_error(self, event):
        pass


_mock_mv = MagicMock()
_mock_mv.TranscriptEventListener = _FakeTranscriptEventListener
_mock_mv.LineTextChanged = type("LineTextChanged", (), {})
_mock_mv.LineCompleted = type("LineCompleted", (), {})

if "moonshine_voice" not in sys.modules:
    sys.modules["moonshine_voice"] = _mock_mv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_moonshine():
    """Mock moonshine_voice internals so nothing real loads."""
    ms_stream = MagicMock()
    ms_transcriber = MagicMock()
    ms_transcriber.create_stream.return_value = ms_stream

    # Module-level functions used by MoonshineTranscriber.load()
    _mock_mv.string_to_model_arch = MagicMock(return_value=5)
    _mock_mv.get_model_for_language = MagicMock(
        return_value=("/fake/model/path", 5),
    )
    _mock_mv.Transcriber = MagicMock(return_value=ms_transcriber)

    # Batch transcribe mock
    mock_line = MagicMock()
    mock_line.text = "Hello world."
    mock_transcript = MagicMock()
    mock_transcript.lines = [mock_line]
    ms_transcriber.transcribe_without_streaming.return_value = mock_transcript

    yield {
        "ms_transcriber": ms_transcriber,
        "ms_stream": ms_stream,
    }


def _make_transcriber(model_arch="medium-streaming"):
    from vox.moonshine import MoonshineTranscriber

    return MoonshineTranscriber(model_arch=model_arch)


def _make_stream(mock_moonshine, on_token=None):
    """Create a MoonshineStream with the background thread suppressed."""
    from vox.moonshine import MoonshineStream

    if on_token is None:
        on_token = MagicMock()

    with patch.object(MoonshineStream, "_process_loop"):
        stream = MoonshineStream(
            transcriber=mock_moonshine["ms_transcriber"],
            on_token=on_token,
            sample_rate=16_000,
        )
    # Thread ran the mock (no-op) and exited; signal done manually.
    stream._done_event.set()
    return stream


def _make_line_event(line_id, text):
    """Create a mock event with event.line.line_id and event.line.text."""
    line = MagicMock()
    line.line_id = line_id
    line.text = text
    event = MagicMock()
    event.line = line
    return event


# ---------------------------------------------------------------------------
# MoonshineTranscriber — protocol conformance
# ---------------------------------------------------------------------------


def test_moonshine_transcriber_importable():
    from vox.moonshine import MoonshineTranscriber

    assert MoonshineTranscriber is not None


def test_moonshine_satisfies_transcriber_protocol():
    from vox.protocols import Transcriber

    t = _make_transcriber()
    assert isinstance(t, Transcriber)


def test_moonshine_has_supports_streaming():
    t = _make_transcriber()
    assert isinstance(t.supports_streaming, bool)


def test_moonshine_supports_streaming_true_for_streaming_arch():
    t = _make_transcriber(model_arch="medium-streaming")
    assert t.supports_streaming is True


def test_moonshine_supports_streaming_false_for_base_arch():
    t = _make_transcriber(model_arch="base")
    assert t.supports_streaming is False


def test_moonshine_has_create_stream():
    t = _make_transcriber()
    assert callable(t.create_stream)


# ---------------------------------------------------------------------------
# MoonshineTranscriber — load lifecycle
# ---------------------------------------------------------------------------


def test_moonshine_is_loaded_false_initially(mock_moonshine):
    t = _make_transcriber()
    assert t.is_loaded is False


def test_moonshine_is_loaded_true_after_load(mock_moonshine):
    t = _make_transcriber()
    t.load()
    assert t.is_loaded is True


def test_moonshine_load_idempotent(mock_moonshine):
    t = _make_transcriber()
    t.load()
    t.load()
    assert _mock_mv.Transcriber.call_count == 1


# ---------------------------------------------------------------------------
# MoonshineTranscriber — batch transcribe
# ---------------------------------------------------------------------------


def test_moonshine_transcribe_returns_string(mock_moonshine):
    t = _make_transcriber()
    t.load()
    audio = np.random.rand(16_000).astype(np.float32)
    result = t.transcribe(audio)
    assert isinstance(result, str)
    assert result == "Hello world."


def test_moonshine_transcribe_auto_loads(mock_moonshine):
    t = _make_transcriber()
    audio = np.random.rand(16_000).astype(np.float32)
    t.transcribe(audio)
    assert t.is_loaded is True


# ---------------------------------------------------------------------------
# MoonshineStream — protocol conformance
# ---------------------------------------------------------------------------


def test_moonshine_stream_importable():
    from vox.moonshine import MoonshineStream

    assert MoonshineStream is not None


def test_moonshine_stream_satisfies_protocol():
    from vox.moonshine import MoonshineStream
    from vox.protocols import TranscriptionStream

    for attr in ("feed", "flush", "close"):
        assert hasattr(MoonshineStream, attr)


# ---------------------------------------------------------------------------
# MoonshineStream — feed / flush / close
# ---------------------------------------------------------------------------


def test_stream_constructs(mock_moonshine):
    stream = _make_stream(mock_moonshine)
    assert stream is not None


def test_stream_feed_accepts_numpy(mock_moonshine):
    stream = _make_stream(mock_moonshine)
    chunk = np.random.rand(1024).astype(np.float32)
    stream.feed(chunk)


def test_stream_feed_is_thread_safe(mock_moonshine):
    """Multiple threads calling feed() should not crash."""
    stream = _make_stream(mock_moonshine)
    errors = []

    def feeder():
        try:
            for _ in range(50):
                stream.feed(np.random.rand(1024).astype(np.float32))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=feeder) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_stream_flush_returns_string(mock_moonshine):
    stream = _make_stream(mock_moonshine)
    result = stream.flush()
    assert isinstance(result, str)


def test_stream_close(mock_moonshine):
    stream = _make_stream(mock_moonshine)
    stream.close()


def test_stream_drain_audio(mock_moonshine):
    stream = _make_stream(mock_moonshine)
    chunk = np.ones(1024, dtype=np.float32)
    stream.feed(chunk)
    stream.feed(chunk)

    drained = stream._drain_audio()
    assert len(drained) == 2048

    drained2 = stream._drain_audio()
    assert len(drained2) == 0


# ---------------------------------------------------------------------------
# MoonshineStream — text delta + on_token callback
# ---------------------------------------------------------------------------


def test_stream_emit_token_accumulates(mock_moonshine):
    stream = _make_stream(mock_moonshine)
    stream._emit_token("hello ")
    stream._emit_token("world")
    result = "".join(stream._accumulated)
    assert result == "hello world"


def test_stream_on_token_callback_fired(mock_moonshine):
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)
    stream._emit_token("hello")
    cb.assert_called_once_with("hello")


def test_stream_emit_token_survives_callback_error(mock_moonshine):
    def bad_callback(text):
        raise RuntimeError("oops")

    stream = _make_stream(mock_moonshine, on_token=bad_callback)
    stream._emit_token("hello")
    assert stream._accumulated == ["hello"]
    assert stream._tokens_emitted == 1


def test_stream_compute_delta_appended_text():
    from vox.moonshine import MoonshineStream

    assert MoonshineStream._compute_delta("Hello ", "Hello world") == "world"


def test_stream_compute_delta_no_change():
    from vox.moonshine import MoonshineStream

    assert MoonshineStream._compute_delta("Hello", "Hello") == ""


def test_stream_compute_delta_from_empty():
    from vox.moonshine import MoonshineStream

    assert MoonshineStream._compute_delta("", "Hello world") == "Hello world"


def test_stream_compute_delta_revision_skipped():
    from vox.moonshine import MoonshineStream

    assert MoonshineStream._compute_delta("abc", "xyz") == ""
    assert MoonshineStream._compute_delta("abc", "xyzw") == ""
    assert MoonshineStream._compute_delta("Hello.", "Hello world") == ""


def test_stream_compute_delta_shorter_text():
    from vox.moonshine import MoonshineStream

    assert MoonshineStream._compute_delta("Hello world", "Hello") == ""


# ---------------------------------------------------------------------------
# MoonshineStream ��� event handler: single line incremental text
# ---------------------------------------------------------------------------


def test_handle_text_changed_does_not_emit(mock_moonshine):
    """TextChanged tracks state but does NOT emit — emission is deferred
    to LineCompleted to avoid emitting speculative hypotheses."""
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    stream._handle_text_changed(_make_line_event(1, "Hello"))
    cb.assert_not_called()

    stream._handle_text_changed(_make_line_event(1, "Hello world"))
    cb.assert_not_called()
    assert stream._accumulated == []


def test_handle_line_completed_emits_full_text(mock_moonshine):
    """LineCompleted emits the full final text for the line."""
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    stream._handle_text_changed(_make_line_event(1, "Hello"))
    stream._handle_line_completed(_make_line_event(1, "Hello world"))
    cb.assert_called_once_with("Hello world")


def test_handle_line_completed_no_leading_space_on_first_line(mock_moonshine):
    """First completed line should NOT have a leading space."""
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    stream._handle_line_completed(_make_line_event(1, "Hello"))

    result = "".join(stream._accumulated)
    assert result == "Hello"


# ---------------------------------------------------------------------------
# Bug 1: _process_loop concatenates chunks -> fewer Moonshine updates
# ---------------------------------------------------------------------------


def test_process_loop_feeds_chunks_individually(mock_moonshine):
    """Bug repro: _process_loop concatenates all buffered chunks and calls
    add_audio() once.  Moonshine's add_audio() triggers update_transcription()
    AT MOST ONCE per call, so feeding one large array skips intermediate
    transcription updates — text appears in large jumps instead of
    incrementally.

    Expected: each chunk results in a separate add_audio() call.
    Actual (bug): all chunks merged into 1 add_audio() call.
    """
    stream = _make_stream(mock_moonshine)
    ms_stream = mock_moonshine["ms_stream"]

    # Buffer 5 chunks before the loop starts (no race — loop is suppressed).
    for _ in range(5):
        stream.feed(np.ones(1024, dtype=np.float32))

    # Run the REAL _process_loop on a thread (patch was undone after _make_stream).
    stream._done_event.clear()
    stream._running = True
    t = threading.Thread(target=stream._process_loop, daemon=True)
    t.start()
    time.sleep(0.15)
    stream._running = False
    stream._done_event.wait(timeout=2)

    # Each of the 5 chunks should produce its own add_audio() call.
    assert ms_stream.add_audio.call_count >= 5


# ---------------------------------------------------------------------------
# Bug 2: Missing space between completed lines
# ---------------------------------------------------------------------------


def test_space_between_completed_lines(mock_moonshine):
    """Two completed lines should be separated by a space."""
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    # Line 1: text appears, then completed.
    stream._handle_text_changed(_make_line_event(1, "Hello"))
    stream._handle_line_completed(_make_line_event(1, "Hello"))

    # Line 2: text appears, then completed.
    stream._handle_text_changed(_make_line_event(2, "World"))
    stream._handle_line_completed(_make_line_event(2, "World"))

    result = "".join(stream._accumulated)
    assert result == "Hello World"


def test_spaces_between_multiple_completed_lines(mock_moonshine):
    """Three completed lines should produce space-separated text."""
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    for i, text in enumerate(["Alpha", "Beta", "Gamma"], start=1):
        stream._handle_text_changed(_make_line_event(i, text))
        stream._handle_line_completed(_make_line_event(i, text))

    result = "".join(stream._accumulated)
    assert result == "Alpha Beta Gamma"


# ---------------------------------------------------------------------------
# Bug 3: Text revisions silently dropped by prefix-match delta
# ---------------------------------------------------------------------------


def test_text_revision_preserved_in_output(mock_moonshine):
    """Bug repro: Moonshine's streaming model revises earlier text via
    speculative decoding.  _compute_delta uses strict prefix matching,
    so when "Ever heard?" is revised to "Ever failed.", the revision is
    silently dropped — output retains the wrong initial hypothesis.

    Expected: accumulated text contains "Ever failed." (the revision).
    Actual (bug): accumulated text contains "Ever heard?" (the initial).
    """
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    # Simulate Moonshine's event sequence for a revised line:
    # 1. TextChanged with initial hypothesis
    # 2. TextChanged with corrected text
    # 3. LineCompleted with final text
    stream._handle_text_changed(_make_line_event(1, "Ever heard?"))
    stream._handle_text_changed(_make_line_event(1, "Ever failed."))
    stream._handle_line_completed(_make_line_event(1, "Ever failed."))

    result = "".join(stream._accumulated)
    assert "Ever failed." in result
    assert "Ever heard?" not in result


def test_beckett_revision_sequence(mock_moonshine):
    """Bug repro: full Beckett quote from reference test (beckett.wav).
    3 of 6 lines are revised.  All revisions must appear in final output.

    Reference impl output:
      Ever tried? Ever failed. No matter. Try again. Fail again. Fail better.

    Our code (bug) produces:
      Ever tried? Ever heard? No matter. Try it. Fail again. 5.
    """
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    # Line 1: no revision
    stream._handle_text_changed(_make_line_event(1, ""))
    stream._handle_text_changed(_make_line_event(1, "Ever tried?"))
    stream._handle_line_completed(_make_line_event(1, "Ever tried?"))

    # Line 2: "Ever heard?" revised to "Ever failed."
    stream._handle_text_changed(_make_line_event(2, "Ever heard?"))
    stream._handle_text_changed(_make_line_event(2, "Ever failed."))
    stream._handle_line_completed(_make_line_event(2, "Ever failed."))

    # Line 3: no revision
    stream._handle_text_changed(_make_line_event(3, "No matter."))
    stream._handle_line_completed(_make_line_event(3, "No matter."))

    # Line 4: "Try it." revised to "Try again."
    stream._handle_text_changed(_make_line_event(4, "Try it."))
    stream._handle_text_changed(_make_line_event(4, "Try again."))
    stream._handle_line_completed(_make_line_event(4, "Try again."))

    # Line 5: late text (empty → "Fail again."), no revision
    stream._handle_text_changed(_make_line_event(5, ""))
    stream._handle_text_changed(_make_line_event(5, "Fail again."))
    stream._handle_line_completed(_make_line_event(5, "Fail again."))

    # Line 6: "5." revised to "Fail better."
    stream._handle_text_changed(_make_line_event(6, "5."))
    stream._handle_text_changed(_make_line_event(6, "Fail better."))
    stream._handle_line_completed(_make_line_event(6, "Fail better."))

    result = "".join(stream._accumulated)
    assert result == "Ever tried? Ever failed. No matter. Try again. Fail again. Fail better."


def test_on_token_only_fires_on_completion(mock_moonshine):
    """on_token must NOT fire for intermediate LineTextChanged events.
    Only LineCompleted should trigger on_token, ensuring the emitted
    text is the model's final answer (not a speculative hypothesis).
    """
    cb = MagicMock()
    stream = _make_stream(mock_moonshine, on_token=cb)

    # Two TextChanged events — neither should fire on_token.
    stream._handle_text_changed(_make_line_event(1, "Ever heard?"))
    stream._handle_text_changed(_make_line_event(1, "Ever failed."))
    cb.assert_not_called()

    # Only completion fires on_token.
    stream._handle_line_completed(_make_line_event(1, "Ever failed."))
    cb.assert_called_once_with("Ever failed.")


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_returns_moonshine_for_moonshine_backend(mock_moonshine):
    from vox.__main__ import _make_transcriber
    from vox.config import Config
    from vox.moonshine import MoonshineTranscriber

    config = Config(backend="moonshine", dev_mode=True)
    t = _make_transcriber(config)
    assert isinstance(t, MoonshineTranscriber)
