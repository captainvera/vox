"""Menubar app — state machine, hotkey, clipboard integration."""

from __future__ import annotations

import atexit
import logging
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import rumps
from AppKit import NSColor, NSEvent, NSFont, NSImage, NSImageView, NSTextField, NSView
from PyObjCTools import AppHelper

log = logging.getLogger(__name__)

# PID file — lets ``vox stop`` / ``vox reload`` find the running process.
_PID_FILE = Path.home() / ".local" / "share" / "vox" / "vox.pid"

from .config import Config
from .formatter import Formatter
from .protocols import Transcriber, TranscriptionStream
from .recorder import Recorder

# State constants.
LOADING = "loading"
IDLE = "idle"
RECORDING = "recording"
TRANSCRIBING = "transcribing"
STREAMING = "streaming"

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


# -- Toggle menu items (icon indicator pattern) ----------------------------
#
# Menu items show a circular icon indicator: blue circle + white SF Symbol
# when on, dim icon when off — matching the macOS Wi-Fi/Bluetooth
# selection pattern.  Custom views keep the menu open on click.


class _ClickableView(NSView):
    """NSView that fires a Python callback on click."""

    def mouseUp_(self, event):
        if hasattr(self, "_py_callback"):
            self._py_callback()


class _ToggleIndicator:
    """Manages the circle + icon visual state of a toggle item."""

    def __init__(self, circle, image_view, state):
        self._circle = circle
        self._iv = image_view
        self._state = state
        self._apply()

    def toggle(self):
        self._state = not self._state
        self._apply()
        return self._state

    def _apply(self):
        if self._state:
            bg = NSColor.systemBlueColor().CGColor()
            tint = NSColor.whiteColor()
        else:
            bg = NSColor.clearColor().CGColor()
            tint = NSColor.secondaryLabelColor()
        self._circle.layer().setBackgroundColor_(bg)
        self._iv.setContentTintColor_(tint)


def _make_toggle_view(menu_item, label, symbol, state, callback):
    """Set an icon-toggle + label as a menu item's custom view.

    Active: blue circle with white SF Symbol.
    Inactive: no circle, dim SF Symbol.
    Clicking anywhere on the row toggles state (menu stays open).

    Returns _ToggleIndicator — caller must retain to prevent GC.
    """
    width, height = 250, 30
    pad = 14
    circle_d = 24
    inset = 4

    container = _ClickableView.alloc().initWithFrame_(((0, 0), (width, height)))

    # Circle background.
    cy = (height - circle_d) / 2
    circle = NSView.alloc().initWithFrame_(((pad, cy), (circle_d, circle_d)))
    circle.setWantsLayer_(True)
    circle.layer().setCornerRadius_(circle_d / 2)
    circle.layer().setMasksToBounds_(True)

    # SF Symbol icon inside circle.
    image = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None)
    icon_size = circle_d - inset * 2
    iv = NSImageView.alloc().initWithFrame_(
        ((inset, inset), (icon_size, icon_size))
    )
    iv.setImage_(image)
    iv.setImageScaling_(3)  # NSImageScaleProportionallyUpOrDown
    circle.addSubview_(iv)

    # Label.
    tf = NSTextField.labelWithString_(label)
    tf.setFont_(NSFont.menuFontOfSize_(0))
    tf.sizeToFit()
    tf.setFrameOrigin_((
        pad + circle_d + 8,
        (height - tf.frame().size.height) / 2,
    ))

    container.addSubview_(circle)
    container.addSubview_(tf)

    # State management + click handler.
    indicator = _ToggleIndicator(circle, iv, state)

    def on_click():
        new_state = indicator.toggle()
        callback(new_state)

    container._py_callback = on_click

    menu_item._menuitem.setView_(container)
    return indicator


class VoxApp(rumps.App):
    """Always-on menubar STT app.

    States: loading -> idle <-> recording -> transcribing -> idle
            idle <-> streaming (realtime mode)
    """

    def __init__(
        self,
        config: Config,
        recorder: Recorder,
        transcriber: Transcriber,
        formatter: Formatter,
    ) -> None:
        super().__init__("vox", title="V", quit_button="Quit")

        self._config = config
        self._recorder = recorder
        self._transcriber = transcriber
        self._formatter = formatter
        self._state = LOADING
        self._hotkey_active = False
        self._stream: TranscriptionStream | None = None
        self._keystroke_queue: queue.Queue[str | None] = queue.Queue()
        self._keystroke_thread: threading.Thread | None = None

        # -- menu items --
        self._status_item = rumps.MenuItem("Loading model...")
        self._status_item.set_callback(None)

        self._pp_toggle = rumps.MenuItem("Post-processing")
        self._pp_indicator = _make_toggle_view(
            self._pp_toggle, "Post-processing", "wand.and.stars",
            self._config.post_processing, self._on_toggle_pp,
        )

        self._type_toggle = rumps.MenuItem("Type at cursor")
        self._type_indicator = _make_toggle_view(
            self._type_toggle, "Type at cursor", "keyboard",
            self._config.type_at_cursor, self._on_toggle_type,
        )

        self._mode_toggle = rumps.MenuItem("Realtime mode")
        self._mode_indicator = _make_toggle_view(
            self._mode_toggle, "Realtime mode", "waveform",
            self._config.mode == "realtime", self._on_toggle_mode,
        )

        self.menu = [
            self._status_item,
            None,
            self._pp_toggle,
            self._type_toggle,
            self._mode_toggle,
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
        """Load model weights (runs in background thread at startup)."""
        try:
            self._transcriber.load()
            self._set_state(IDLE)
        except Exception as exc:
            log.exception("Model failed to load")
            AppHelper.callAfter(
                setattr, self._status_item, "title", f"Error: {exc}",
            )

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
        elif self._state == STREAMING:
            self._stop_streaming()
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
                STREAMING:     (_ICON_WAVE, True,     "Streaming..."),
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

    @property
    def _use_realtime(self) -> bool:
        """True if config is realtime AND the model supports streaming."""
        return (
            self._config.mode == "realtime"
            and self._transcriber.supports_streaming
        )

    def _start_recording(self) -> None:
        if self._use_realtime:
            self._start_streaming()
        else:
            self._set_state(RECORDING)
            self._recorder.start()

    def _start_streaming(self) -> None:
        """Start recording + streaming transcription simultaneously."""
        log.info("Starting realtime stream")
        self._stream = self._transcriber.create_stream(
            on_token=self._on_stream_token,
        )
        # Bridge recorder audio callback → stream.feed
        self._recorder._on_chunk = self._stream.feed
        # Start async keystroke worker
        self._keystroke_thread = threading.Thread(
            target=self._keystroke_worker, daemon=True,
        )
        self._keystroke_thread.start()
        self._set_state(STREAMING)
        self._recorder.start()
        log.info("Recorder started, streaming active")

    def _on_stream_token(self, text: str) -> None:
        """Called by the streaming transcriber for each decoded token.

        Runs on the stream's background processing thread.
        Queues non-empty text for async keystroke output.
        """
        if not text:
            return
        log.info("token: %r", text)
        self._keystroke_queue.put(text)

    @staticmethod
    def _keystroke(text: str) -> None:
        """Type text at cursor via macOS System Events keystroke."""
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "System Events" to keystroke "{escaped}"',
            ],
            check=True,
        )

    def _keystroke_worker(self) -> None:
        """Background thread: consume keystroke queue, type via osascript."""
        while True:
            text = self._keystroke_queue.get()
            if text is None:
                break
            try:
                self._keystroke(text)
            except Exception:
                log.exception("Keystroke failed for %r", text)

    def _stop_keystroke_worker(self) -> None:
        """Signal keystroke worker to exit and wait for it to drain."""
        self._keystroke_queue.put(None)
        if self._keystroke_thread and self._keystroke_thread.is_alive():
            self._keystroke_thread.join(timeout=5)
        self._keystroke_thread = None

    def _stop_streaming(self) -> None:
        """Stop recording, flush remaining tokens, output final text."""
        log.info("Stopping realtime stream")
        self._recorder.stop()
        self._recorder._on_chunk = None
        stream = self._stream
        self._stream = None

        if stream is None:
            log.warning("_stop_streaming called but no active stream")
            self._stop_keystroke_worker()
            self._set_state(IDLE)
            return

        try:
            text = stream.flush()
            stream.close()
            self._stop_keystroke_worker()

            if self._config.post_processing:
                text = self._formatter.format(text)

            # Copy final text to clipboard.
            subprocess.run(
                ["pbcopy"],
                input=text.encode("utf-8"),
                check=True,
            )

            if self._config.type_at_cursor:
                subtitle = "Typed at cursor (streamed)"
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

    def _on_toggle_pp(self, new_state: bool) -> None:
        self._config.post_processing = new_state
        self._formatter.enabled = new_state
        self._config.save()

    def _on_toggle_type(self, new_state: bool) -> None:
        self._config.type_at_cursor = new_state
        self._config.save()

    def _on_toggle_mode(self, new_state: bool) -> None:
        self._config.mode = "realtime" if new_state else "transcript"
        self._config.save()
