"""RED tests for VoxtralTranscriber protocol conformance.

These verify:
- VoxtralTranscriber satisfies the Transcriber protocol
- supports_streaming property exists and returns a bool
- create_stream() exists and is callable
- Existing batch transcribe() behavior is preserved
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_voxmlx():
    """Mock voxmlx internals so VoxtralTranscriber can be instantiated."""
    with (
        patch("vox.transcriber.load_model") as mock_load,
        patch("vox.transcriber._build_prompt_tokens") as mock_prompt,
        patch("vox.transcriber.generate") as mock_gen,
        patch("vox.transcriber.mx") as mock_mx,
    ):
        mock_model = MagicMock()
        mock_sp = MagicMock()
        mock_sp.eos_id = 0
        mock_sp.decode.return_value = "hello world"
        mock_config = MagicMock()
        mock_load.return_value = (mock_model, mock_sp, mock_config)
        mock_prompt.return_value = ([1, 2, 3], 5)
        mock_gen.return_value = [10, 20, 30]

        # load() now precomputes streaming embeddings
        mock_model.time_embedding.return_value = MagicMock()
        mock_model.language_model.embed.return_value = MagicMock(
            __getitem__=lambda self, idx: MagicMock()
        )

        yield {
            "load_model": mock_load,
            "build_prompt": mock_prompt,
            "generate": mock_gen,
            "model": mock_model,
            "sp": mock_sp,
            "mx": mock_mx,
        }


# -- Protocol conformance --


def test_voxtral_satisfies_transcriber_protocol(mock_voxmlx):
    from vox.protocols import Transcriber
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    assert isinstance(t, Transcriber)


def test_voxtral_has_supports_streaming(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    assert isinstance(t.supports_streaming, bool)


def test_voxtral_has_create_stream(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    assert callable(t.create_stream)


# -- Batch behavior preserved --


def test_voxtral_transcribe_returns_string(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    t.load()

    audio = np.random.rand(16_000).astype(np.float32)
    result = t.transcribe(audio)
    assert isinstance(result, str)
    assert result == "hello world"


def test_voxtral_load_idempotent(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    t.load()
    t.load()  # second call should not re-load
    assert mock_voxmlx["load_model"].call_count == 1


def test_voxtral_is_loaded_false_initially(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    assert t.is_loaded is False


def test_voxtral_is_loaded_true_after_load(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    t.load()
    assert t.is_loaded is True


# -- Streaming support --


def test_voxtral_supports_streaming_is_true(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    assert t.supports_streaming is True


def test_voxtral_create_stream_returns_stream(mock_voxmlx):
    from vox.transcriber import VoxtralTranscriber

    t = VoxtralTranscriber(model_path="/fake/path")
    t.load()

    cb = MagicMock()
    # Mock VoxtralStream so we don't spin up real ML threads
    with patch("vox.transcriber.VoxtralStream") as MockStream:
        MockStream.return_value = MagicMock()
        stream = t.create_stream(on_token=cb)

    assert stream is not None
    MockStream.assert_called_once()
    # Verify on_token was passed through
    call_kwargs = MockStream.call_args[1]
    assert call_kwargs["on_token"] is cb
