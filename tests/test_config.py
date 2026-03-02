"""RED tests for Config.mode field.

These verify:
- Config has a `mode` field
- Default value is "transcript"
- Mode persists through save/load round-trip
- Unknown mode values are loaded as-is (no validation at load time)
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
