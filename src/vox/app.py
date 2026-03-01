"""Menubar app — state machine, hotkey, clipboard integration."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import rumps
from AppKit import NSEvent
from PyObjCTools import AppHelper

log = logging.getLogger(__name__)

# PID file �� lets ``vox stop`` / ``vox reload`` find the running process.
_PID_FILE = Path.home() / ".local" / "share" / "vox" / "vox.pid"

from .config import Config
from .formatter import Formatter
from .recorder import Recorder
from .transcriber import VoxtralTranscriber

# State constants.
LOADING = "loading"
IDLE = "idle"
RECORDING = "recording"
TRANSCRIBING = "transcribing"

# Menubar icons (template images — adapt to light/dark automatically).
_ICON_DIR = Path(__file__).parent / "icons"
_ICON_LOGO = str(_ICON_DIR / "logo.svg")
_ICON_MIC = str(_ICON_DIR / "mic.svg")
_ICON_WAVE = str(_ICON_DIR / "wave.svg")

# macOS key constants for the global hotkey (Option + Space).
_KEYCODE_SPACE = 49
_FLAG_OPTION = 1 << 19   # NSEventModifierFlagOption
_FLAG_CMD = 1 << 20      # NSEventModifierFlagCommand
_FLAG_CTRL = 1 << 18     # NSEventModifierFlagControl
_FLAG_SHIFT = 1 << 17    # NSEventModifierFlagShift
_MOD_MASK = _FLAG_OPTION | _FLAG_CMD | _FLAG_CTRL | _FLAG_SHIFT
_MASK_KEYDOWN = 1 << 10  # NSEventMaskKeyDown


class VoxApp(rumps.App):
    """Always-on menubar STT app.

    States: loading -> idle <-> recording -> transcribing -> idle
    """

    def __init__(
        self,
        config: Config,
        recorder: Recorder,
        transcriber: VoxtralTranscriber,
        formatter: Formatter,
    ) -> None:
        super().__init__("vox", title="V", quit_button="Quit")

        self._config = config
        self._recorder = recorder
        self._transcriber = transcriber
        self._formatter = formatter
        self._state = LOADING
        self._hotkey_active = False

        # -- menu items --
        self._status_item = rumps.MenuItem("Loading model...")
        self._status_item.set_callback(None)

        self._pp_toggle = rumps.MenuItem(
            "Post-processing",
            callback=self._on_toggle_pp,
        )
        self._pp_toggle.state = self._config.post_processing

        self._type_toggle = rumps.MenuItem(
            "Type at cursor",
            callback=self._on_toggle_type,
        )
        self._type_toggle.state = self._config.type_at_cursor

        self.menu = [
            self._status_item,
            None,
            self._pp_toggle,
            self._type_toggle,
        ]

        # -- PID file + clean shutdown on SIGTERM --
        self._write_pid()
        signal.signal(signal.SIGTERM, lambda *_: rumps.quit_application())

        # -- load model in background --
        threading.Thread(target=self._load_model, daemon=True).start()

    # -- lifecycle --

    def _write_pid(self) -> None:
        """Write PID file so ``vox stop/reload`` can find us."""
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()))
        atexit.register(lambda: _PID_FILE.unlink(missing_ok=True))
        log.info("PID %d written to %s", os.getpid(), _PID_FILE)

    def _load_model(self) -> None:
        """Load Voxtral weights (runs in background thread at startup)."""
        try:
            self._transcriber.load()
            self._set_state(IDLE)
        except Exception as exc:
            self._status_item.title = f"Error: {exc}"

    def _start_hotkey(self) -> None:
        """Register a global hotkey via Cocoa NSEvent monitor.

        Runs on the main thread (NSApplication event loop) — no threading
        issues unlike pynput which calls TSM from a background thread and
        crashes when running as a .app bundle.
        """

        def handler(event):
            mods = event.modifierFlags() & _MOD_MASK
            if event.keyCode() == _KEYCODE_SPACE and mods == _FLAG_OPTION:
                self._on_hotkey()

        self._hotkey_monitor = (
            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                _MASK_KEYDOWN, handler,
            )
        )
        log.info("Global hotkey registered (Option+Space)")

    # -- hotkey handler --

    def _on_hotkey(self) -> None:
        if self._state == IDLE:
            self._start_recording()
        elif self._state == RECORDING:
            self._stop_and_transcribe()
        # Ignore presses during loading / transcribing.

    # -- state machine --

    def _set_state(self, state: str) -> None:
        """Update state machine — dispatches UI changes to the main thread."""
        self._state = state

        def _apply_ui():
            _STATES = {
                #              (icon,        template, status text)
                LOADING:       (_ICON_LOGO, True,     "Loading model..."),
                IDLE:          (_ICON_LOGO, True,     "Ready"),
                RECORDING:     (_ICON_MIC,  True,     "Recording..."),
                TRANSCRIBING:  (_ICON_WAVE, True,     "Transcribing..."),
            }
            icon, template, status = _STATES[state]

            self.icon = icon
            self.template = template
            # Clear title AFTER icon is set — rumps' fallbackOnName()
            # re-sets title to app name when both title and image are
            # empty.  With the image already set, it won't fall back.
            self.title = None

            self._status_item.title = status

            # Start hotkey only once model is loaded.
            if state == IDLE and not self._hotkey_active:
                self._start_hotkey()
                self._hotkey_active = True

        # UI updates must happen on the main thread (NSApplication).
        AppHelper.callAfter(_apply_ui)

    def _start_recording(self) -> None:
        self._set_state(RECORDING)
        self._recorder.start()

    def _stop_and_transcribe(self) -> None:
        audio = self._recorder.stop()
        if len(audio) == 0:
            self._set_state(IDLE)
            return

        self._set_state(TRANSCRIBING)
        threading.Thread(
            target=self._transcribe_worker,
            args=(audio,),
            daemon=True,
        ).start()

    # -- transcription worker (background thread) --

    def _transcribe_worker(self, audio) -> None:
        try:
            text = self._transcriber.transcribe(audio)

            if self._config.post_processing:
                text = self._formatter.format(text)

            # Always copy to clipboard.
            subprocess.run(
                ["pbcopy"],
                input=text.encode("utf-8"),
                check=True,
            )

            if self._config.type_at_cursor:
                self._paste_at_cursor()
                subtitle = "Typed at cursor"
            else:
                subtitle = "Copied to clipboard"

            preview = text[:80] + ("\u2026" if len(text) > 80 else "")
            rumps.notification(
                title="vox",
                subtitle=subtitle,
                message=preview,
            )
        except Exception as exc:
            rumps.notification("vox", "Error", str(exc))
        finally:
            self._set_state(IDLE)

    # -- output --

    @staticmethod
    def _paste_at_cursor() -> None:
        """Simulate Cmd+V via macOS System Events (more reliable than pynput)."""
        time.sleep(0.15)  # let clipboard settle
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True,
        )

    # -- menu callbacks --

    def _on_toggle_pp(self, sender: rumps.MenuItem) -> None:
        sender.state = not sender.state
        self._config.post_processing = bool(sender.state)
        self._formatter.enabled = self._config.post_processing
        self._config.save()

    def _on_toggle_type(self, sender: rumps.MenuItem) -> None:
        sender.state = not sender.state
        self._config.type_at_cursor = bool(sender.state)
        self._config.save()
