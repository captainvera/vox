"""RED tests for app pipeline branching (transcript vs realtime mode).

These verify:
- App constructor accepts a Transcriber (protocol type, not just VoxtralTranscriber)
- App type-hints transcriber as the protocol
- Transcript mode uses the batch flow (existing behavior)
- Realtime mode starts streaming when model supports it
- Realtime mode falls back to transcript when model doesn't support streaming
- Mode toggle in menu works
- The STREAMING state exists
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# -- Helpers: fake transcribers that satisfy the protocol --


class FakeBatchTranscriber:
    """Satisfies Transcriber protocol, batch-only."""

    def __init__(self):
        self._loaded = False
        self.transcribe_calls = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supports_streaming(self) -> bool:
        return False

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribe_calls.append(audio)
        return "batch result"

    def create_stream(self, on_token):
        raise NotImplementedError


class FakeStream:
    """Fake TranscriptionStream."""

    def __init__(self, on_token):
        self.on_token = on_token
        self.fed_chunks = []
        self.flushed = False
        self.closed = False

    def feed(self, chunk: np.ndarray) -> None:
        self.fed_chunks.append(chunk)
        self.on_token("word ")

    def flush(self) -> str:
        self.flushed = True
        return "streamed result"

    def close(self) -> None:
        self.closed = True


class FakeStreamingTranscriber:
    """Satisfies Transcriber protocol, supports streaming."""

    def __init__(self):
        self._loaded = False
        self.transcribe_calls = []
        self.last_stream = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def supports_streaming(self) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribe_calls.append(audio)
        return "batch result"

    def create_stream(self, on_token):
        self.last_stream = FakeStream(on_token)
        return self.last_stream


@pytest.fixture
def config():
    from vox.config import Config

    return Config(mode="transcript")


@pytest.fixture
def recorder(mock_sd):
    from vox.recorder import Recorder

    return Recorder(sample_rate=16_000)


@pytest.fixture
def formatter():
    from vox.formatter import Formatter

    return Formatter(enabled=False)


def _make_app(config, recorder, formatter, transcriber, **config_overrides):
    """Build VoxApp with mocked macOS internals."""
    import vox.app as app_mod

    for k, v in config_overrides.items():
        setattr(config, k, v)

    with (
        patch.object(app_mod.VoxApp, "_write_pid"),
        patch("vox.app.signal"),
        patch("vox.app.threading.Thread"),
        patch("vox.app.AppHelper"),
        patch("vox.app._make_toggle_view", return_value=MagicMock()),
    ):
        return app_mod.VoxApp(
            config=config,
            recorder=recorder,
            transcriber=transcriber,
            formatter=formatter,
        )


# -- Constructor: accepts protocol --


def test_app_accepts_batch_transcriber(config, recorder, formatter, mock_sd):
    app = _make_app(config, recorder, formatter, FakeBatchTranscriber())
    assert app is not None


def test_app_accepts_streaming_transcriber(config, recorder, formatter, mock_sd):
    app = _make_app(config, recorder, formatter, FakeStreamingTranscriber())
    assert app is not None


# -- Type hint uses protocol --


def test_app_type_hint_is_protocol():
    """Constructor should type-hint transcriber as Transcriber protocol."""
    import ast
    import inspect
    import textwrap

    from vox.app import VoxApp

    # With `from __future__ import annotations`, all annotations are strings.
    # Use get_type_hints or just inspect the source for the annotation.
    source = inspect.getsource(VoxApp.__init__)
    assert "Transcriber" in source
    assert "VoxtralTranscriber" not in source


# -- Transcript mode (existing batch flow) --


def test_transcript_mode_calls_batch_transcribe(
    config, recorder, formatter, mock_sd
):
    transcriber = FakeBatchTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="transcript")

    fake_audio = np.random.rand(16_000).astype(np.float32)

    with (
        patch("vox.app.subprocess"),
        patch("vox.app.rumps"),
        patch("vox.app.AppHelper"),
    ):
        app._transcribe_worker(fake_audio)

    assert len(transcriber.transcribe_calls) == 1
    np.testing.assert_array_equal(transcriber.transcribe_calls[0], fake_audio)


# -- Realtime mode --


def test_streaming_state_exists():
    from vox.app import STREAMING

    assert STREAMING == "streaming"


def test_realtime_mode_starts_streaming_on_record(
    config, recorder, formatter, mock_sd
):
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"
    app._hotkey_active = True

    with patch("vox.app.AppHelper"):
        app._start_recording()

    assert app._stream is not None
    assert transcriber.last_stream is not None


def test_stop_streaming_dispatches_to_background(
    config, recorder, formatter, mock_sd
):
    """_stop_streaming should spawn a thread instead of blocking the caller."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"

    with patch("vox.app.AppHelper"):
        app._start_recording()

    app._recorder.stop = MagicMock(return_value=np.zeros(0, dtype=np.float32))
    with (
        patch("vox.app.AppHelper"),
        patch("vox.app.threading.Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        app._stop_streaming()

    # Thread was created targeting _finalize_streaming.
    mock_thread_cls.assert_called_once()
    kwargs = mock_thread_cls.call_args[1]
    assert kwargs["target"] == app._finalize_streaming
    mock_thread.start.assert_called_once()


def test_stop_streaming_sets_transcribing_state(
    config, recorder, formatter, mock_sd
):
    """_stop_streaming should transition to TRANSCRIBING immediately."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"

    with patch("vox.app.AppHelper"):
        app._start_recording()

    app._recorder.stop = MagicMock(return_value=np.zeros(0, dtype=np.float32))
    with (
        patch("vox.app.AppHelper"),
        patch("vox.app.threading.Thread") as mock_thread_cls,
    ):
        mock_thread_cls.return_value = MagicMock()
        app._stop_streaming()

    assert app._state == "transcribing"


def test_finalize_streaming_flushes_and_cleans_up(
    config, recorder, formatter, mock_sd
):
    """_finalize_streaming should flush, close, stop keystroke worker, pbcopy."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"

    with patch("vox.app.AppHelper"):
        app._start_recording()

    stream = transcriber.last_stream

    with (
        patch("vox.app.subprocess"),
        patch("vox.app.rumps"),
        patch("vox.app.AppHelper"),
    ):
        app._finalize_streaming(stream)

    assert stream.flushed
    assert stream.closed


def test_realtime_fallback_when_no_streaming(
    config, recorder, formatter, mock_sd
):
    """Batch-only model in realtime mode: no stream created, normal record."""
    transcriber = FakeBatchTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"
    app._hotkey_active = True

    with patch("vox.app.AppHelper"):
        app._start_recording()

    assert not hasattr(app, "_stream") or app._stream is None


# -- Mode toggle --


def test_mode_toggle_menu_item_exists(config, recorder, formatter, mock_sd):
    app = _make_app(config, recorder, formatter, FakeStreamingTranscriber())
    assert hasattr(app, "_mode_toggle")


def test_mode_toggle_switches_config(config, recorder, formatter, mock_sd):
    app = _make_app(
        config, recorder, formatter, FakeStreamingTranscriber(), mode="transcript"
    )
    with patch.object(app._config, "save"):
        app._on_toggle_mode(True)
    assert app._config.mode == "realtime"


# -- Recorder <-> Stream bridge --


def test_start_streaming_wires_recorder_on_chunk(
    config, recorder, formatter, mock_sd
):
    """_start_streaming should set recorder.on_chunk = stream.feed."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"

    with patch("vox.app.AppHelper"):
        app._start_recording()

    stream = transcriber.last_stream
    # Bound methods create new wrappers; use == not is
    assert app._recorder.on_chunk == stream.feed


def test_stop_streaming_clears_recorder_on_chunk(
    config, recorder, formatter, mock_sd
):
    """_stop_streaming should clear recorder.on_chunk on the main thread."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"

    with patch("vox.app.AppHelper"):
        app._start_recording()

    app._recorder.stop = MagicMock(return_value=np.zeros(0, dtype=np.float32))
    with (
        patch("vox.app.AppHelper"),
        patch("vox.app.threading.Thread") as mock_thread_cls,
    ):
        mock_thread_cls.return_value = MagicMock()
        app._stop_streaming()

    assert app._recorder.on_chunk is None


# -- Keystroke output --


def test_on_stream_token_queues_text(config, recorder, formatter, mock_sd):
    """_on_stream_token should queue non-empty text for async keystroke."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "streaming"

    app._on_stream_token("hello")

    assert not app._keystroke_queue.empty()
    assert app._keystroke_queue.get_nowait() == "hello"


def test_on_stream_token_skips_empty(config, recorder, formatter, mock_sd):
    """_on_stream_token should skip empty tokens (silence)."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "streaming"

    app._on_stream_token("")

    assert app._keystroke_queue.empty()


def test_keystroke_worker_processes_queue(config, recorder, formatter, mock_sd):
    """_keystroke_worker should consume queue and call _keystroke."""
    app = _make_app(
        config, recorder, formatter, FakeStreamingTranscriber(), mode="realtime"
    )

    with patch.object(app, "_keystroke") as mock_ks:
        app._keystroke_queue.put("hello")
        app._keystroke_queue.put("world")
        app._keystroke_queue.put(None)  # sentinel
        app._keystroke_worker()

    assert mock_ks.call_count == 2
    mock_ks.assert_any_call("hello")
    mock_ks.assert_any_call("world")


def test_keystroke_calls_osascript(config, recorder, formatter, mock_sd):
    """_keystroke should call osascript to type text."""
    app = _make_app(
        config, recorder, formatter, FakeStreamingTranscriber(), mode="realtime"
    )

    with patch("vox.app.subprocess.run") as mock_run:
        app._keystroke("hello")

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "osascript"
    assert "keystroke" in args[2]
    assert "hello" in args[2]


def test_keystroke_escapes_quotes(config, recorder, formatter, mock_sd):
    """Quotes in text must be escaped for osascript."""
    app = _make_app(
        config, recorder, formatter, FakeStreamingTranscriber(), mode="realtime"
    )

    with patch("vox.app.subprocess.run") as mock_run:
        app._keystroke('say "hi"')

    args = mock_run.call_args[0][0]
    # The osascript string should have escaped quotes
    assert '\\"' in args[2]


# -- Subprocess timeouts --


def test_keystroke_has_timeout(config, recorder, formatter, mock_sd):
    """_keystroke osascript call must have a timeout to prevent hangs."""
    app = _make_app(
        config, recorder, formatter, FakeStreamingTranscriber(), mode="realtime"
    )

    with patch("vox.app.subprocess.run") as mock_run:
        app._keystroke("hello")

    kwargs = mock_run.call_args[1]
    assert "timeout" in kwargs
    assert kwargs["timeout"] > 0


def test_paste_at_cursor_has_timeout(config, recorder, formatter, mock_sd):
    """_paste_at_cursor osascript call must have a timeout."""
    app = _make_app(
        config, recorder, formatter, FakeStreamingTranscriber(), mode="realtime"
    )

    with (
        patch("vox.app.subprocess.run") as mock_run,
        patch("vox.app.time.sleep"),
    ):
        app._paste_at_cursor()

    kwargs = mock_run.call_args[1]
    assert "timeout" in kwargs
    assert kwargs["timeout"] > 0


def test_finalize_streaming_pbcopy_has_timeout(
    config, recorder, formatter, mock_sd
):
    """pbcopy in _finalize_streaming must have a timeout."""
    transcriber = FakeStreamingTranscriber()
    transcriber.load()
    app = _make_app(config, recorder, formatter, transcriber, mode="realtime")
    app._state = "idle"

    with patch("vox.app.AppHelper"):
        app._start_recording()

    stream = transcriber.last_stream

    with (
        patch("vox.app.subprocess.run") as mock_run,
        patch("vox.app.rumps"),
        patch("vox.app.AppHelper"),
    ):
        app._finalize_streaming(stream)

    # Find the pbcopy call.
    pbcopy_calls = [
        c for c in mock_run.call_args_list if c[0][0][0] == "pbcopy"
    ]
    assert len(pbcopy_calls) == 1
    assert pbcopy_calls[0][1]["timeout"] > 0
