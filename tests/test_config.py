"""RED tests for Config fields: mode and backend.

These verify:
- Config has `mode`, `backend`, `parakeet_model` fields
- Default values are correct
- Fields persist through save/load round-trip
- Invalid values raise ValueError
- Old config files without new fields load with defaults
"""

from __future__ import annotations

import json

import pytest


def test_config_has_mode_field():
    from vox.config import Config
    c = Config()
    assert hasattr(c, "mode")


def test_config_mode_default_is_transcript():
    from vox.config import Config
    c = Config()
    assert c.mode == "transcript"


def test_config_mode_can_be_set_to_realtime():
    from vox.config import Config
    c = Config(mode="realtime")
    assert c.mode == "realtime"


def test_config_mode_round_trips(tmp_config):
    from vox.config import Config

    c = Config(mode="realtime")
    c.save()

    loaded = Config.load()
    assert loaded.mode == "realtime"


def test_config_mode_default_when_file_missing(tmp_config):
    from vox.config import Config
    c = Config.load()
    assert c.mode == "transcript"


def test_config_mode_survives_unknown_keys(tmp_config):
    """Old config files without mode should load with the default."""
    from vox.config import CONFIG_FILE

    # Write a config without the mode field (simulates pre-mode config)
    CONFIG_FILE.write_text(json.dumps({"type_at_cursor": True}))

    from vox.config import Config
    c = Config.load()
    assert c.mode == "transcript"
    assert c.type_at_cursor is True


# -- Backend field --


def test_config_has_backend_field():
    from vox.config import Config
    c = Config()
    assert hasattr(c, "backend")


def test_config_backend_default_is_voxtral():
    from vox.config import Config
    c = Config()
    assert c.backend == "voxtral"


def test_config_backend_can_be_set_to_parakeet():
    from vox.config import Config
    c = Config(backend="parakeet", dev_mode=True)
    assert c.backend == "parakeet"


def test_config_non_dev_forces_voxtral():
    from vox.config import Config
    c = Config(backend="parakeet", dev_mode=False)
    assert c.backend == "voxtral"


def test_config_backend_invalid_raises():
    from vox.config import Config
    with pytest.raises(ValueError, match="Invalid backend"):
        Config(backend="invalid")


def test_config_has_parakeet_model_field():
    from vox.config import Config
    c = Config()
    assert hasattr(c, "parakeet_model")


def test_config_parakeet_model_default():
    from vox.config import Config
    c = Config()
    assert c.parakeet_model == "mlx-community/parakeet-tdt-0.6b-v3"


def test_config_backend_round_trips(tmp_config):
    from vox.config import Config

    c = Config(backend="parakeet", dev_mode=True)
    c.save()

    loaded = Config.load()
    assert loaded.backend == "parakeet"


def test_config_backend_default_when_file_missing(tmp_config):
    from vox.config import Config
    c = Config.load()
    assert c.backend == "voxtral"


def test_config_old_file_without_backend_loads_default(tmp_config):
    """Pre-backend config files should load with backend='voxtral'."""
    from vox.config import CONFIG_FILE

    CONFIG_FILE.write_text(json.dumps({"mode": "realtime"}))

    from vox.config import Config
    c = Config.load()
    assert c.backend == "voxtral"
    assert c.mode == "realtime"
