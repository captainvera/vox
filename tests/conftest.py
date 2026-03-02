"""Shared fixtures and mocks for vox tests.

Mocks the heavy macOS / ML dependencies so tests run without hardware.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


# -- Stub rumps so VoxApp can inherit from a real class. --
# rumps.App must be a real class (not MagicMock) because VoxApp
# inherits from it, and Python's MRO breaks with a MagicMock base.


class _FakeRumpsApp:
    """Minimal stub of rumps.App for test-time inheritance."""

    def __init__(self, name, title=None, quit_button=None, **kwargs):
        self.name = name
        self.title = title
        self.icon = None
        self.template = None
        self.menu = []

    def run(self):
        pass


class _FakeMenuItem:
    """Stub of rumps.MenuItem."""

    def __init__(self, title="", callback=None, **kwargs):
        self.title = title
        self._menuitem = MagicMock()

    def set_callback(self, cb):
        pass


_rumps_mock = MagicMock()
_rumps_mock.App = _FakeRumpsApp
_rumps_mock.MenuItem = _FakeMenuItem
_rumps_mock.notification = MagicMock()
_rumps_mock.quit_application = MagicMock()


# -- Mock macOS-only modules before any vox imports touch them. --
_mock_modules = {
    "rumps": _rumps_mock,
    "AppKit": MagicMock(),
    "PyObjCTools": MagicMock(),
    "PyObjCTools.AppHelper": MagicMock(),
    "Foundation": MagicMock(),
}

for mod_name, mock in _mock_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect Config file I/O to a temp directory."""
    import vox.config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    return tmp_path


@pytest.fixture
def mock_sd(monkeypatch):
    """Mock sounddevice so Recorder never opens a real mic."""
    mock = MagicMock()
    monkeypatch.setattr("vox.recorder.sd", mock)
    return mock
