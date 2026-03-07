# Vox: A Complete Code Walkthrough

*2026-03-05T17:09:11Z by Showboat 0.6.1*
<!-- showboat-id: 32af3007-1be4-48ed-bc3d-cc8aa7e4961d -->

## What is Vox?

Vox is a menubar speech-to-text app for macOS. Press a hotkey (Option+Space), speak, press again — your words land on the clipboard or get typed at the cursor. Everything runs locally on Apple Silicon via MLX, no cloud APIs.

The codebase is ~2,200 lines of Python across 10 modules, plus a compiled C launcher that gives the app a native macOS identity. Two transcription backends are supported: Voxtral (4B params, streaming capable) and Parakeet (600M params, fast batch).

Here is the project layout:

```bash
find src/vox -name "*.py" | sort | while read f; do wc -l < "$f" | tr -d " " | xargs -I{} echo "{} $f"; done
```

```output
5 src/vox/__init__.py
239 src/vox/__main__.py
789 src/vox/app.py
54 src/vox/config.py
337 src/vox/daemon.py
76 src/vox/formatter.py
343 src/vox/parakeet.py
65 src/vox/protocols.py
87 src/vox/recorder.py
106 src/vox/transcriber.py
462 src/vox/voxtral_stream.py
```

The packaging is a standard Python project built with hatchling, installed as an editable uv tool. The `vox` console script entry point routes to `vox.__main__:main`:

```bash
cat pyproject.toml
```

```output
[project]
name = "vox"
version = "0.1.0"
description = "Menubar STT for macOS — record, transcribe, clipboard"
requires-python = ">=3.12"
dependencies = [
    "rumps>=0.4.0",
    "voxmlx>=0.0.2",
    "sounddevice>=0.5.0",
    "soundfile>=0.13.0",
    "parakeet-mlx>=0.3.0",
]

[project.scripts]
vox = "vox.__main__:main"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vox"]
```

Key dependencies: **rumps** for the menubar UI, **voxmlx** (private API from Awni Hannun at Apple) for Voxtral inference on MLX, **parakeet-mlx** for the lightweight alternative backend, **sounddevice** for mic capture, and **soundfile** for WAV encoding. The version comes from package metadata at runtime:

```bash
cat src/vox/__init__.py
```

```output
"""vox — menubar STT for macOS."""

from importlib.metadata import version

__version__ = version("vox")
```

## The Entry Point: `__main__.py`

Everything starts at `main()`. When you type `vox` in a terminal, Python calls `vox.__main__:main` (registered as a console script in pyproject.toml). This function is a simple command dispatcher — it reads `sys.argv[1]` and routes to the right handler:

```bash
sed -n "202,235p" src/vox/__main__.py
```

```output
def main() -> None:
    from . import daemon

    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd is None:
        _run_foreground()
    elif cmd == "start":
        daemon.start()
    elif cmd == "stop":
        daemon.stop()
    elif cmd == "restart":
        daemon.restart()
    elif cmd == "reload":
        daemon.reload()
    elif cmd == "status":
        daemon.status()
    elif cmd == "logs":
        try:
            subprocess.run(
                ["tail", "-f", str(daemon.LOG_FILE)],
            )
        except KeyboardInterrupt:
            pass
    elif cmd == "setup":
        _setup()
    elif cmd == "update":
        _update()
    elif cmd == "uninstall":
        daemon.uninstall()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: vox [start|stop|restart|reload|status|logs|setup|update|uninstall]")
        sys.exit(1)
```

With no subcommand, `vox` runs the menubar app in the current terminal — useful for debugging since you see logs in real time. The production path is `vox start`, which delegates to `daemon.start()` to build the .app bundle and launch it.

The most interesting function here is `_run_foreground()`. It detects whether it is running inside the compiled .app bundle (via `NSBundle.mainBundle().bundleIdentifier()`), and if so, redirects stdout/stderr to a log file — because the C launcher has no terminal attached:

```bash
sed -n "28,50p" src/vox/__main__.py
```

```output
def _is_app_bundle() -> bool:
    """Return True if running inside the Vox.app bundle (compiled launcher)."""
    try:
        from Foundation import NSBundle

        return NSBundle.mainBundle().bundleIdentifier() == "com.vox.app"
    except Exception:
        return False


def _redirect_to_log() -> None:
    """Redirect stdout/stderr to the log file when inside the .app bundle.

    The compiled C launcher's Py_RunMain() inherits stdout/stderr that go
    nowhere when launched via ``open -n``.  This captures all output —
    logging, print(), unhandled exceptions — into the log file.
    """
    from .daemon import DATA_DIR, LOG_FILE

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG_FILE, "a")  # noqa: SIM115 — kept open for process lifetime
    sys.stdout = log_fh
    sys.stderr = log_fh
```

`_run_foreground()` then assembles the four core components (Config, Recorder, Transcriber, Formatter) and hands them to VoxApp. The transcriber backend is selected dynamically via `_make_transcriber()` — lazy imports ensure only the chosen backend's dependencies load:

```bash
sed -n "53,102p" src/vox/__main__.py
```

```output
def _make_transcriber(config):
    """Create the transcriber backend based on config.backend.

    Lazy imports so only the selected backend's dependencies are loaded.
    """
    if config.backend == "parakeet":
        from .parakeet import ParakeetTranscriber

        return ParakeetTranscriber(model_name=config.parakeet_model)
    from .transcriber import VoxtralTranscriber

    return VoxtralTranscriber(model_path=config.model_path)


def _run_foreground() -> None:
    """Run the menubar app in the foreground (original behaviour)."""
    if _is_app_bundle():
        _redirect_to_log()

    from .app import VoxApp
    from .config import Config
    from .formatter import Formatter
    from .recorder import Recorder

    if _is_app_bundle():
        # Rotating log: 5 MB max, keep 3 backups.
        from .daemon import LOG_FILE

        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3,
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S",
        ))
        logging.basicConfig(level=logging.INFO, handlers=[handler])
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    config = Config.load()
    app = VoxApp(
        config=config,
        recorder=Recorder(sample_rate=config.sample_rate),
        transcriber=_make_transcriber(config),
        formatter=Formatter(enabled=config.post_processing),
    )
    app.run()
```

Notice the two logging paths: inside the .app bundle, logs go to a 5 MB rotating file at `~/.local/share/vox/vox.log`; in foreground mode, they go straight to the terminal.

The module also contains `_setup()` (interactive model download wizard), `_update()` (self-update from GitHub tags via uv), and the `_is_app_bundle()` / `_redirect_to_log()` pair we already saw. These are utilities, not the main flow.

## Configuration: `config.py`

Before diving into the app, we need to understand the config. It is a simple dataclass persisted as JSON at `~/.config/vox/config.json`. The entire file is 54 lines:

```bash
cat src/vox/config.py
```

```output
"""Persistent configuration for vox."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "vox"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_MODEL_PATH = str(Path.home() / "models" / "Voxtral-Mini-4B-Realtime-6bit")


VALID_MODES = ("transcript", "realtime")
VALID_BACKENDS = ("voxtral", "parakeet")

DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


@dataclass
class Config:
    model_path: str = DEFAULT_MODEL_PATH
    post_processing: bool = True
    type_at_cursor: bool = False
    sample_rate: int = 16_000
    mode: str = "transcript"  # "transcript" | "realtime"
    backend: str = "voxtral"  # "voxtral" | "parakeet"
    parakeet_model: str = DEFAULT_PARAKEET_MODEL

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode {self.mode!r}, must be one of {VALID_MODES}"
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend {self.backend!r}, must be one of {VALID_BACKENDS}"
            )

    @classmethod
    def load(cls) -> Config:
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
                return cls(**known)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2) + "\n")
```

Key design choices:

- **Forward-compatible loading**: `load()` only picks keys that match `__dataclass_fields__`, so adding new config fields never breaks existing config files. Unknown keys are silently ignored.
- **Validation at construction**: `__post_init__` rejects invalid `mode` or `backend` values immediately.
- **Two modes**: `"transcript"` (record-then-transcribe) and `"realtime"` (transcribe while recording).
- **Two backends**: `"voxtral"` (Voxtral 4B, supports streaming) and `"parakeet"` (Parakeet 600M, periodic batch streaming).
- **type_at_cursor**: When enabled, vox pastes text at the cursor via Cmd+V (osascript). When disabled, it only copies to clipboard.

The menu toggles in `app.py` call `config.save()` after every change, so preferences persist across restarts.

## The .app Bundle and C Launcher: `daemon.py`

This is the most interesting systems-level piece of vox. macOS will not display an NSStatusBar (menubar icon) for a script-based .app bundle — the process identity becomes `Python.app` and the menubar item silently fails. The solution: compile a native C binary that embeds the Python interpreter and runs `python -m vox` from within.

Here is the embedded C source:

```bash
sed -n "48,71p" src/vox/daemon.py
```

```output
_LAUNCHER_C = """\
#define PY_SSIZE_T_CLEAN
#include <Python.h>

int main(int argc, char *argv[]) {
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    /* Point Python at the vox venv so it finds rumps, voxmlx, etc. */
    PyConfig_SetBytesString(&config, &config.executable,
        PYTHON_EXECUTABLE);

    /* argv = ["vox"] */
    wchar_t *wargv[] = { L"vox" };
    PyConfig_SetWideStringList(&config, &config.argv, 1, wargv);

    /* Run: python -m vox */
    config.run_module = Py_DecodeLocale("vox", NULL);

    Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    return Py_RunMain();
}
"""
```

The key line is `PyConfig_SetBytesString(&config, &config.executable, PYTHON_EXECUTABLE)`. `PYTHON_EXECUTABLE` is injected as a `-D` flag at compile time, pointing to the venv's Python binary. This tells the embedded interpreter where to find site-packages (rumps, voxmlx, etc.), even though the executing binary is `Vox.app/Contents/MacOS/Vox`.

The argv is hardcoded to `["vox"]` and `config.run_module` is set to `"vox"` — equivalent to `python -m vox`, which hits `__main__.py`.

Compilation is handled by `_compile_launcher()`, which uses `cc` (Xcode CLI tools) and links against the Python framework. It skips recompilation when nothing changed, tracked by a sha256 hash of the C source + `sys.executable`:

```bash
sed -n "107,166p" src/vox/daemon.py
```

```output
def _launcher_hash(python_exe: str) -> str:
    """Content hash used to skip recompilation when nothing changed."""
    payload = (_LAUNCHER_C + python_exe).encode()
    return hashlib.sha256(payload).hexdigest()


def _compile_launcher(macos_dir: Path) -> Path:
    """Compile the C launcher into ``macos_dir/Vox``.

    Skips compilation if the binary already exists and the source + config
    haven't changed (checked via a hash file).
    """
    if not shutil.which("cc"):
        raise RuntimeError(
            "C compiler not found. Install Xcode CLI tools: xcode-select --install"
        )

    binary = macos_dir / "Vox"
    hash_file = macos_dir / ".launcher_hash"
    python_exe = sys.executable
    current_hash = _launcher_hash(python_exe)

    # Skip if up to date.
    if (
        binary.exists()
        and hash_file.exists()
        and hash_file.read_text().strip() == current_hash
    ):
        return binary

    include, lib_path, rpath = _python_compile_flags()

    # Write C source to a temp file for compilation.
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(_LAUNCHER_C)
        c_src = f.name

    try:
        result = subprocess.run(
            [
                "cc",
                "-o", str(binary),
                f"-I{include}",
                lib_path,
                f"-DPYTHON_EXECUTABLE=\"{python_exe}\"",
                f"-Wl,-rpath,{rpath}",
                c_src,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Launcher compilation failed:\n{result.stderr}"
            )
    finally:
        os.unlink(c_src)

    hash_file.write_text(current_hash)
    return binary
```

The `_python_compile_flags()` function handles the difference between macOS framework builds (Homebrew) and non-framework builds — framework builds need to link against the dylib directly since Homebrew's framework lacks the standard top-level symlinks.

The .app bundle itself is a standard macOS bundle structure:

```bash
if [ -d ~/Applications/Vox.app ]; then find ~/Applications/Vox.app -maxdepth 4 -not -name ".DS_Store" | sed "s|$HOME/Applications/||" | sort; else echo "Vox.app/"; echo "Vox.app/Contents/"; echo "Vox.app/Contents/Info.plist"; echo "Vox.app/Contents/MacOS/"; echo "Vox.app/Contents/MacOS/Vox          # compiled C binary"; echo "Vox.app/Contents/MacOS/.launcher_hash"; echo "Vox.app/Contents/Resources/"; echo "Vox.app/Contents/Resources/Vox.icns"; fi
```

```output
Vox.app
Vox.app/Contents
Vox.app/Contents/Info.plist
Vox.app/Contents/MacOS
Vox.app/Contents/MacOS/.launcher_hash
Vox.app/Contents/MacOS/Vox
Vox.app/Contents/Resources
Vox.app/Contents/Resources/Vox.icns
```

`_create_app_bundle()` generates this structure, writes `Info.plist` with `LSUIElement: true` (no Dock icon), copies the .icns, compiles the launcher, and calls `lsregister` to force Launch Services to re-read the plist.

The lifecycle functions are straightforward — `start()` kills orphans, builds the bundle, and launches with `open -n`. `stop()` uses `pkill` patterns matching both the C launcher and legacy Python script paths. Here is the `start()` function:

```bash
sed -n "276,294p" src/vox/daemon.py
```

```output
def start() -> None:
    """Create the .app bundle and launch it."""
    _cleanup_legacy_launchd()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _read_pid():
        print("vox is already running. Use 'vox reload' to restart.")
        return

    # Kill any orphaned instances before launching.
    _kill_all_vox()

    _create_app_bundle()
    subprocess.run(["open", "-n", str(APP_PATH)], check=True)
    print(f"vox started. Logs: {LOG_FILE}")
    print(
        "First time? Grant Accessibility to Vox in "
        "System Settings > Privacy & Security > Accessibility."
    )
```

The `open -n` flag tells macOS to open a new instance even if one is already registered. The PID file at `~/.local/share/vox/vox.pid` is written by `VoxApp.__init__()` once the app actually starts (not by the daemon module). This matters because there is a timing gap between `open -n` returning and the Python process writing its PID.

## Protocols: `protocols.py`

Before looking at any concrete backend, we need to see the contract they all implement. This is a clean Protocol-based design — the app never imports a concrete transcriber class directly, only these two interfaces:

```bash
cat src/vox/protocols.py
```

```output
"""Protocols for transcription backends.

Defines the contract that all model backends must satisfy.
Models that don't support streaming return supports_streaming=False
and raise NotImplementedError from create_stream().
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TranscriptionStream(Protocol):
    """Incremental transcription session.

    Created per recording via Transcriber.create_stream().
    Holds the encoder/decoder state for one streaming session.
    """

    def feed(self, chunk: np.ndarray) -> None:
        """Push an audio chunk into the streaming pipeline."""
        ...

    def flush(self) -> str:
        """Drain remaining tokens and return final accumulated text."""
        ...

    def close(self) -> None:
        """Release model caches and session state."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Backend-agnostic transcription interface.

    Every model backend implements this. The app only depends on
    this protocol �� never on a concrete transcriber class.
    """

    @property
    def is_loaded(self) -> bool: ...

    @property
    def supports_streaming(self) -> bool: ...

    def load(self) -> None:
        """Load model weights. Safe to call multiple times."""
        ...

    def transcribe(self, audio: np.ndarray) -> str:
        """Batch-transcribe a complete audio array to text."""
        ...

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a streaming transcription session.

        Raises NotImplementedError if supports_streaming is False.
        """
        ...
```

Two protocols, two levels:

1. **`Transcriber`** — the backend itself. Has `load()`, `transcribe()` (batch), `create_stream()` (streaming), and two properties: `is_loaded` and `supports_streaming`. The app checks `supports_streaming` before attempting realtime mode — if the backend does not support it, it falls back to batch transcript mode.

2. **`TranscriptionStream`** — a per-recording session. Three methods: `feed(chunk)` pushes audio in (called from the sounddevice callback thread), `flush()` drains remaining audio and returns the full text, `close()` releases caches.

Both are `@runtime_checkable`, meaning `isinstance(obj, Transcriber)` works at runtime. This is used in tests to verify that concrete backends satisfy the protocol.

## Audio Recording: `recorder.py`

The Recorder wraps `sounddevice.InputStream` for mic capture. It is only 87 lines and has a clever dual-mode design:

```bash
cat src/vox/recorder.py
```

```output
"""Mic capture — start/stop recording, return numpy audio array."""

from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)

import numpy as np
import sounddevice as sd


class Recorder:
    """Records mono audio from the default mic at a given sample rate."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._on_chunk = on_chunk
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def on_chunk(self) -> Callable[[np.ndarray], None] | None:
        return self._on_chunk

    @on_chunk.setter
    def on_chunk(self, callback: Callable[[np.ndarray], None] | None) -> None:
        self._on_chunk = callback

    def start(self) -> None:
        """Open the mic and begin buffering audio."""
        self._chunks = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a 1-D float32 array."""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if self._chunks:
                audio = np.concatenate(self._chunks)
                self._chunks = []
                return audio
            return np.zeros(0, dtype=np.float32)

    # -- private --

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        chunk = indata[:, 0].copy()
        if self._on_chunk is not None:
            # Streaming mode: forward chunk, skip buffering (caller
            # discards stop() return value, so _chunks would be a leak).
            self._on_chunk(chunk)
        else:
            with self._lock:
                self._chunks.append(chunk)
```

The critical design is in `_on_audio()` (the sounddevice callback, invoked from a C audio thread):

- **Batch mode** (`on_chunk is None`): Audio chunks are appended to `_chunks` under a lock. When recording stops, `stop()` concatenates them into one numpy array and returns it for batch transcription.
- **Streaming mode** (`on_chunk` is set): Chunks are forwarded directly to the callback (which will be `stream.feed()`). Nothing is buffered in `_chunks` — this avoids a memory leak since `stop()` return value is discarded in streaming mode.

The `on_chunk` property is a settable callback. The app sets it to `stream.feed` when starting a streaming session, and clears it to `None` before stopping — this ordering matters to prevent deadlocks (the sounddevice callback must not block in `feed()` during shutdown while the main thread waits in `sd.InputStream.stop()`).

## The Menubar App: `app.py`

This is the largest file (789 lines) and the heart of vox. It contains:
1. The VoxApp state machine
2. Global hotkey registration via Cocoa NSEvent
3. Custom macOS-native menu UI (toggle indicators, disclosure sections, model picker)
4. The two transcription paths (batch and streaming)
5. Output handling (clipboard + type-at-cursor)

### State Machine and Hotkey

The state machine has five states. The hotkey (Option+Space) is a toggle — press once to start recording, press again to stop and transcribe:

```bash
sed -n "35,55p" src/vox/app.py
```

```output
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
```

The hotkey is registered using Cocoa's `NSEvent.addGlobalMonitorForEventsMatchingMask:handler:` instead of pynput. This is a hard-won design decision — pynput's keyboard listener calls macOS TSM (Text Services Manager) APIs from a background thread, which triggers `dispatch_assert_queue` assertion failures when running as a .app bundle. The NSEvent approach runs on the main NSApplication event loop:

```bash
sed -n "430,459p" src/vox/app.py
```

```output
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
```

The handler filters for exactly Option+Space — the modifier mask (`_MOD_MASK`) includes Cmd, Ctrl, and Shift bits to ensure those are NOT pressed simultaneously. The `_on_hotkey()` dispatcher is the state machine transition table: IDLE starts recording, RECORDING stops and transcribes, STREAMING stops streaming. Presses during LOADING or TRANSCRIBING are silently ignored.

### State Transitions and UI

`_set_state()` is the single point through which all state transitions flow. It updates the menubar icon and status text, dispatched to the main thread via `AppHelper.callAfter()`:

```bash
sed -n "463,493p" src/vox/app.py
```

```output
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
```

A subtle detail: `self.title = None` is set AFTER `self.icon` to avoid a rumps quirk — when both title and icon are empty, rumps' `fallbackOnName()` resets the title to the app name. Setting the icon first prevents this.

The hotkey is only registered once, on the first transition to IDLE (after model loading). This prevents the user from triggering recording before the model is ready.

### The Batch Transcription Path

When the hotkey is pressed in IDLE and the mode is `transcript`, `_start_recording()` starts the recorder in batch mode (no `on_chunk` callback). A second hotkey press calls `_stop_and_transcribe()`, which grabs the audio and dispatches a background thread:

```bash
sed -n "638,683p" src/vox/app.py
```

```output
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
                timeout=3,
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
```

The batch path is linear: stop recorder -> get audio array -> call `transcriber.transcribe()` -> optionally format -> `pbcopy` to clipboard -> optionally `_paste_at_cursor()` -> send notification -> return to IDLE.

`_paste_at_cursor()` uses osascript to simulate Cmd+V — it waits 150ms for the clipboard to settle, then tells System Events to keystroke "v" with command down. This requires Accessibility permission granted to Vox.app:

```bash
sed -n "687,699p" src/vox/app.py
```

```output
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
            timeout=5,
        )
```

### The Streaming Transcription Path

When the mode is `realtime` and the backend supports streaming, the hotkey triggers `_start_streaming()` instead. This sets up a pipeline: Recorder -> `stream.feed()` -> background encoder/decoder -> `on_token` callback -> keystroke queue -> osascript:

```bash
sed -n "510,562p" src/vox/app.py
```

```output
    def _start_streaming(self) -> None:
        """Start recording + streaming transcription simultaneously."""
        log.info("Starting realtime stream")
        self._stream = self._transcriber.create_stream(
            on_token=self._on_stream_token,
        )
        # Bridge recorder audio callback → stream.feed
        self._recorder.on_chunk = self._stream.feed
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
            timeout=5,
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

```

The streaming pipeline has three threads working together:

1. **Sounddevice callback thread**: Calls `recorder._on_audio()` -> forwards chunk to `stream.feed()`.
2. **Stream processing thread** (inside VoxtralStream/ParakeetStream): Drains audio, runs encoder/decoder, calls `_on_stream_token()` per decoded token.
3. **Keystroke worker thread**: Reads from `_keystroke_queue`, types each token at the cursor via osascript.

The queue-based keystroke worker decouples inference speed from osascript latency. Tokens are typed as fast as osascript can handle them, without blocking the decoder.

Stopping the stream is carefully ordered to avoid deadlocks:

```bash
sed -n "570,636p" src/vox/app.py
```

```output
    def _stop_streaming(self) -> None:
        """Stop recording, dispatch flush/output to a background thread.

        Only the fast, non-blocking parts run on the main thread:
        clear the audio callback, stop the recorder, grab the stream
        reference.  The expensive work (flush with MLX inference,
        keystroke worker join, pbcopy, notification) runs in
        ``_finalize_streaming`` on a daemon thread so the menubar
        stays responsive.
        """
        log.info("Stopping realtime stream")
        # Clear callback BEFORE stopping — prevents the sounddevice
        # callback from blocking in feed() during shutdown, which would
        # deadlock the main thread waiting in sd.InputStream.stop().
        self._recorder.on_chunk = None
        self._recorder.stop()
        stream = self._stream
        self._stream = None

        if stream is None:
            log.warning("_stop_streaming called but no active stream")
            self._stop_keystroke_worker()
            self._set_state(IDLE)
            return

        # Transition to TRANSCRIBING so the hotkey handler ignores
        # presses while the background thread flushes remaining audio.
        self._set_state(TRANSCRIBING)
        threading.Thread(
            target=self._finalize_streaming,
            args=(stream,),
            daemon=True,
        ).start()

    def _finalize_streaming(self, stream: TranscriptionStream) -> None:
        """Flush stream, output text, clean up (runs on background thread)."""
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
                timeout=3,
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
```

The shutdown sequence in `_stop_streaming()`:
1. Clear `recorder.on_chunk` to `None` **before** stopping the recorder — this prevents a deadlock where the sounddevice callback would block in `stream.feed()` while the main thread waits in `sd.InputStream.stop()`.
2. Stop the recorder (returns immediately since chunks were forwarded, not buffered).
3. Grab the stream reference and clear it on self.
4. Transition to TRANSCRIBING (so the hotkey ignores further presses).
5. Dispatch `_finalize_streaming()` to a background thread.

`_finalize_streaming()` calls `stream.flush()` (which feeds right-padding silence and decodes remaining tokens), copies the final text to clipboard, and posts a notification. The formatter is applied to the full accumulated text, not to individual tokens.

### Custom Menu UI

The menu uses Cocoa-native custom views to create macOS-style toggle indicators (blue circle + white SF Symbol). This is not standard rumps — it drops down to `NSView`, `NSImageView`, and `NSTextField` directly:

```bash
sed -n "105,128p" src/vox/app.py
```

```output
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

```

The toggle items (post-processing, type-at-cursor, realtime mode) each get a `_ClickableView` (an NSView subclass that fires a Python callback on `mouseUp:`). Using a custom view keeps the menu open on click — standard menu items dismiss the menu. The toggle visual matches the macOS Wi-Fi/Bluetooth selection pattern.

The backend picker uses a disclosure pattern — a collapsible "Other Models" section that shows/hides alternative backends. The disclosure state resets every time the menu opens (via an `NSMenu` delegate):

```bash
sed -n "89,103p" src/vox/app.py
```

```output
class _MenuDelegate(NSObject):
    """NSMenu delegate — fires a Python callback on menuWillOpen:."""

    def menuWillOpen_(self, menu):
        if hasattr(self, "_py_on_open"):
            self._py_on_open()


class _ClickableView(NSView):
    """NSView that fires a Python callback on click."""

    def mouseUp_(self, event):
        if hasattr(self, "_py_callback"):
            self._py_callback()

```

The VoxApp constructor wires everything together — it creates all menu items, sets up the custom views, writes the PID file, registers SIGTERM to quit cleanly, and kicks off model loading in a background thread:

```bash
sed -n "303,407p" src/vox/app.py
```

```output
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

        # -- backend picker (collapsible model selector) --
        self._backend_expanded = False
        self._backend_active_label = _BACKEND_LABELS[self._config.backend]

        self._backend_section = rumps.MenuItem("Model")
        _make_section_header(self._backend_section, "Model")

        self._backend_active = rumps.MenuItem("Active model")
        _make_toggle_view(
            self._backend_active, self._backend_active_label, "brain",
            True, lambda _state: None,  # active row is display-only
        )

        self._backend_disclosure = rumps.MenuItem("Other Models")
        self._disclosure_chevron = _make_disclosure_view(
            self._backend_disclosure, "Other Models",
            expanded=False,
            callback=lambda: self._on_toggle_disclosure(),
        )

        # One menu item per alternative backend (all except the active one).
        self._backend_alternatives: list[rumps.MenuItem] = []
        for key in VALID_BACKENDS:
            if key == self._config.backend:
                continue
            item = rumps.MenuItem(_BACKEND_LABELS[key])
            # Capture `key` in default arg to avoid late-binding closure bug.
            _make_model_view(
                item, _BACKEND_LABELS[key], "brain",
                callback=lambda k=key: self._on_select_backend(k),
            )
            self._backend_alternatives.append(item)

        self.menu = [
            self._status_item,
            None,
            self._pp_toggle,
            self._type_toggle,
            self._mode_toggle,
            None,
            self._backend_section,
            self._backend_active,
            None,
            self._backend_disclosure,
            *self._backend_alternatives,
        ]

        # Hide alternatives initially (collapsed).
        for item in self._backend_alternatives:
            item._menuitem.setHidden_(True)

        # Collapse disclosure every time the menu opens.
        self._menu_delegate = _MenuDelegate.alloc().init()
        self._menu_delegate._py_on_open = self._on_menu_open
        # The NSMenu backing self.menu is the first item's parent menu.
        ns_menu = self._status_item._menuitem.menu()
        ns_menu.setDelegate_(self._menu_delegate)

        # -- PID file + clean shutdown on SIGTERM --
        self._write_pid()
        signal.signal(signal.SIGTERM, lambda *_: rumps.quit_application())

        # -- load model in background --
        threading.Thread(target=self._load_model, daemon=True).start()
```

The initial state is LOADING. The model loads on a daemon thread. When it completes, `_load_model` calls `_set_state(IDLE)`, which registers the hotkey. If model loading fails, the error is shown in the status menu item (wrapped to 30-char lines via `_format_error`).

Selecting a different backend from the disclosure menu triggers `_on_select_backend()`, which saves the config, rebuilds the menu alternatives, creates a new transcriber via `_make_transcriber()`, and reloads the model in the background — transitioning back through LOADING -> IDLE.

## Voxtral Batch Backend: `transcriber.py`

The `VoxtralTranscriber` wraps the private `voxmlx` library. It loads the model once and caches it for repeated transcriptions. The model is Voxtral-Mini-4B-Realtime (6-bit quantized, ~2.6 GB on disk):

```bash
cat src/vox/transcriber.py
```

```output
"""Voxtral STT — load model once, transcribe numpy audio arrays."""

from __future__ import annotations

import logging
import tempfile
from typing import Callable

import mlx.core as mx
import numpy as np
import soundfile as sf
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx import _build_prompt_tokens, load_model
from voxmlx.generate import generate

from .protocols import TranscriptionStream
from .voxtral_stream import VoxtralStream

log = logging.getLogger(__name__)


class VoxtralTranscriber:
    """Wraps voxmlx with a cached model for repeated transcriptions."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None
        self._sp = None
        self._config = None
        self._prompt_tokens: list[int] | None = None
        self._n_delay_tokens: int | None = None
        # Precomputed for streaming (set during load)
        self._text_embeds = None
        self._t_cond = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supports_streaming(self) -> bool:
        return True

    def load(self) -> None:
        """Pre-load model weights and tokenizer. Safe to call multiple times."""
        if self._model is not None:
            return
        self._model, self._sp, self._config = load_model(self.model_path)
        self._prompt_tokens, self._n_delay_tokens = _build_prompt_tokens(self._sp)

        # Precompute embeddings needed for streaming
        self._t_cond = self._model.time_embedding(
            mx.array([self._n_delay_tokens], dtype=mx.float32)
        )
        prompt_ids = mx.array([self._prompt_tokens])
        self._text_embeds = self._model.language_model.embed(prompt_ids)[0]
        mx.eval(self._t_cond, self._text_embeds)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe a numpy audio array to text.

        Sends the full audio as a single chunk — no splitting, no filtering.
        """
        self.load()

        duration_s = len(audio) / sample_rate
        log.info("Audio %.1fs -> single chunk (no splitting)", duration_s)

        text = self._transcribe_chunk(audio, sample_rate)
        log.info("  result: %r", text[:120])
        return text.strip()

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a streaming transcription session."""
        self.load()
        log.info("Creating VoxtralStream (prefix_len=%d)", len(self._prompt_tokens))
        return VoxtralStream(
            model=self._model,
            sp=self._sp,
            text_embeds=self._text_embeds,
            t_cond=self._t_cond,
            prefix_len=len(self._prompt_tokens),
            eos_token_id=self._sp.eos_id,
            on_token=on_token,
        )

    def _transcribe_chunk(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a single audio chunk via voxmlx."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, sample_rate)
            output_tokens = generate(
                self._model,
                f.name,
                self._prompt_tokens,
                n_delay_tokens=self._n_delay_tokens,
                temperature=0.0,
                eos_token_id=self._sp.eos_id,
            )

        return self._sp.decode(
            output_tokens,
            special_token_policy=SpecialTokenPolicy.IGNORE,
        )
```

Key details:

- **`load()`** does three things: loads model weights + tokenizer via `voxmlx.load_model()`, builds the prompt token sequence via `_build_prompt_tokens()`, and precomputes the time-conditional embedding (`t_cond`) and text prompt embeddings (`text_embeds`). These precomputed tensors are passed to streaming sessions to avoid recomputing them per recording.
- **`transcribe()`** writes the numpy audio to a temp WAV (because `voxmlx.generate()` expects a file path), generates tokens with temperature=0 (greedy), and decodes them via the sentencepiece tokenizer. Special tokens are ignored in the decoded output.
- **`create_stream()`** passes the precomputed embeddings to `VoxtralStream`, which handles the incremental encode/decode pipeline.
- **`_transcribe_chunk()`** is the low-level call to `voxmlx.generate()` — it returns raw token IDs that are decoded by the Mistral sentencepiece tokenizer.

## Voxtral Streaming: `voxtral_stream.py`

This is the most complex module (462 lines) and the technical core of realtime mode. It implements the `TranscriptionStream` protocol for Voxtral, performing incremental audio encoding and autoregressive token decoding in a background thread.

The architecture mirrors `voxmlx/stream.py` but is refactored into a reusable class. The key idea: Voxtral is an audio-conditioned language model. Audio is encoded into embeddings, and these embeddings condition the decoder (which generates text tokens one at a time). By feeding audio incrementally and decoding as embeddings become available, we get realtime output.

### Initialization and State

```bash
sed -n "25,102p" src/vox/voxtral_stream.py
```

```output
N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17


class VoxtralStream:
    """Incremental Voxtral transcription session.

    Created per recording via VoxtralTranscriber.create_stream().
    Mirrors the logic in voxmlx/stream.py but as a reusable class.
    """

    def __init__(
        self,
        model,
        sp,
        text_embeds: mx.array,
        t_cond: mx.array,
        prefix_len: int,
        eos_token_id: int,
        on_token: Callable[[str], None],
        temperature: float = 0.0,
    ) -> None:
        # Shared model references (not owned, from VoxtralTranscriber)
        self._model = model
        self._sp = sp
        self._text_embeds = text_embeds
        self._t_cond = t_cond
        self._prefix_len = prefix_len
        self._eos_token_id = eos_token_id
        self._on_token = on_token
        self._temperature = temperature

        self._n_layers = len(model.language_model.layers)

        # Audio buffer (thread-safe: written by feed(), read by _process_loop)
        # Uses a list of chunks (O(1) append) instead of np.append (O(n) copy).
        self._lock = threading.Lock()
        self._audio_chunks: list[np.ndarray] = []

        # Accumulated text for flush()
        self._accumulated: list[str] = []

        # Per-session decoder state
        self._cache = None
        self._y = None

        # Per-session incremental encoder state
        self._audio_tail = None
        self._conv1_tail = None
        self._conv2_tail = None
        self._encoder_cache = None
        self._ds_buf = None

        # Buffers and counters
        self._pending_audio = np.zeros(0, dtype=np.float32)
        self._audio_embeds = None
        self._n_audio_samples_fed = 0
        self._n_total_decoded = 0
        self._first_cycle = True
        self._prefilled = False

        # Stats
        self._start_time = time.monotonic()
        self._tokens_emitted = 0
        self._eos_count = 0
        self._encode_calls = 0
        self._feed_calls = 0
        self._total_audio_samples = 0

        # Lifecycle
        self._running = True
        self._closed = False
        self._done_event = threading.Event()
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True
        )
        self._thread.start()
        log.info("Stream started (prefix_len=%d, eos_id=%d)", prefix_len, eos_token_id)
```

There is a lot of state here, so let us break it down by role:

- **Audio buffer** (`_audio_chunks`, `_lock`): Thread-safe O(1) append buffer. `feed()` pushes chunks from the sounddevice thread; `_drain_audio()` concatenates and clears from the processing thread.
- **Encoder state** (`_audio_tail`, `_conv1_tail`, `_conv2_tail`, `_encoder_cache`, `_ds_buf`): Incremental encoder carries convolutional tails and transformer KV cache between chunks. These let the encoder process audio in pieces without reprocessing the whole sequence.
- **Decoder state** (`_cache`, `_y`): `_cache` is a list of `RotatingKVCache` (one per transformer layer, 8192-token window). `_y` is the most recent token prediction.
- **Bookkeeping** (`_pending_audio`, `_audio_embeds`, `_n_audio_samples_fed`, `_n_total_decoded`, `_first_cycle`, `_prefilled`): Track how much audio has been encoded vs. decoded, whether the decoder has been prefilled, and whether left-padding has been applied.

### The Processing Loop

The background thread runs a continuous loop that alternates between encoding new audio and decoding available embeddings:

```bash
sed -n "378,462p" src/vox/voxtral_stream.py
```

```output
    def _process_loop(self) -> None:
        """Main processing loop — mirrors stream.py's while True loop."""
        log.info("Processing thread started")
        loop_count = 0
        try:
            while self._running:
                loop_count += 1
                new_audio = self._drain_audio()
                if len(new_audio) > 0:
                    self._pending_audio = np.append(
                        self._pending_audio, new_audio
                    )

                if (
                    self._first_cycle
                    and len(self._pending_audio) < SAMPLES_PER_TOKEN
                ):
                    time.sleep(0.02)
                    continue

                # Encode new audio
                if (
                    self._first_cycle
                    and len(self._pending_audio) >= SAMPLES_PER_TOKEN
                ):
                    # First cycle: add left pad
                    left_pad = np.zeros(
                        N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN,
                        dtype=np.float32,
                    )
                    n_feed = (
                        len(self._pending_audio) // SAMPLES_PER_TOKEN
                    ) * SAMPLES_PER_TOKEN
                    chunk = np.concatenate(
                        [left_pad, self._pending_audio[:n_feed]]
                    )
                    self._pending_audio = self._pending_audio[n_feed:]
                    self._n_audio_samples_fed += n_feed
                    log.info(
                        "First encode: %.3fs audio + %d left-pad tokens",
                        n_feed / 16_000,
                        N_LEFT_PAD_TOKENS,
                    )
                    self._encode_chunk(chunk)
                    self._first_cycle = False

                elif (
                    not self._first_cycle
                    and len(self._pending_audio) >= SAMPLES_PER_TOKEN
                ):
                    n_feed = (
                        len(self._pending_audio) // SAMPLES_PER_TOKEN
                    ) * SAMPLES_PER_TOKEN
                    chunk = self._pending_audio[:n_feed]
                    self._pending_audio = self._pending_audio[n_feed:]
                    self._n_audio_samples_fed += n_feed
                    self._encode_chunk(chunk)

                if self._audio_embeds is None:
                    time.sleep(0.02)
                    continue

                self._decode_available()
                mx.clear_cache()
                time.sleep(0.02)
        except Exception:
            log.exception(
                "Processing thread crashed after %d loops, %.1fs, %d tokens emitted",
                loop_count,
                time.monotonic() - self._start_time,
                self._tokens_emitted,
            )
        finally:
            elapsed = time.monotonic() - self._start_time
            log.info(
                "Processing thread exited after %.1fs "
                "(loops=%d, tokens=%d, eos=%d, encodes=%d, running=%s)",
                elapsed,
                loop_count,
                self._tokens_emitted,
                self._eos_count,
                self._encode_calls,
                self._running,
            )
            self._done_event.set()
```

The loop runs every 20ms (the `time.sleep(0.02)` calls). Each iteration:

1. **Drain** new audio from the thread-safe buffer.
2. **Encode** when enough pending audio exists (at least `SAMPLES_PER_TOKEN` samples). On the first cycle, 32 tokens of silence are prepended as left-padding. Audio is quantized to whole-token boundaries — leftover samples stay in `_pending_audio`.
3. **Decode** available embeddings via `_decode_available()`.
4. **Clear** MLX cache to control memory.

### The Encoder: `_encode_chunk()`

This runs mel-spectrogram extraction and the audio encoder incrementally:

```bash
sed -n "208,237p" src/vox/voxtral_stream.py
```

```output
    def _encode_chunk(self, chunk: np.ndarray) -> None:
        """Run incremental mel + encoder on a chunk of audio."""
        self._encode_calls += 1
        mel, self._audio_tail = log_mel_spectrogram_step(
            chunk, self._audio_tail
        )
        new_embeds, self._conv1_tail, self._conv2_tail, self._encoder_cache, self._ds_buf = (
            self._model.encode_step(
                mel,
                self._conv1_tail,
                self._conv2_tail,
                self._encoder_cache,
                self._ds_buf,
            )
        )
        if new_embeds is not None:
            mx.eval(new_embeds)
            n_new = new_embeds.shape[0]
            if self._audio_embeds is not None:
                self._audio_embeds = mx.concatenate(
                    [self._audio_embeds, new_embeds]
                )
            else:
                self._audio_embeds = new_embeds
            log.debug(
                "Encoded %d samples -> %d new embeds (total undecoded: %d)",
                len(chunk),
                n_new,
                self._audio_embeds.shape[0],
            )
```

`log_mel_spectrogram_step()` is the incremental mel computation from voxmlx — it carries `_audio_tail` between calls to handle the FFT windowing across chunk boundaries. `model.encode_step()` runs the audio encoder with convolutional tail state (`_conv1_tail`, `_conv2_tail`), encoder transformer cache, and a downsampling buffer. New audio embeddings are appended to `_audio_embeds`.

### The Decoder: `_decode_available()`

This is the autoregressive decoder. It has two phases: prefill (first time) and incremental decode:

```bash
sed -n "239,360p" src/vox/voxtral_stream.py
```

```output
    def _decode_available(self, max_steps: int | None = None) -> None:
        """Decode available positions, emitting tokens via callback."""
        if self._audio_embeds is None:
            return

        safe_total = (
            N_LEFT_PAD_TOKENS
            + self._n_audio_samples_fed // SAMPLES_PER_TOKEN
        )
        n_decodable = min(
            self._audio_embeds.shape[0],
            safe_total - self._n_total_decoded,
        )
        if max_steps is not None:
            n_decodable = min(n_decodable, max_steps)

        if n_decodable <= 0:
            return

        if not self._prefilled:
            if self._n_total_decoded + self._audio_embeds.shape[0] < self._prefix_len:
                log.debug(
                    "Waiting for prefill: have %d embeds, need %d",
                    self._n_total_decoded + self._audio_embeds.shape[0],
                    self._prefix_len,
                )
                return

            log.info(
                "Prefilling decoder (prefix_len=%d, audio_embeds=%d, audio_fed=%.1fs)",
                self._prefix_len,
                self._audio_embeds.shape[0],
                self._n_audio_samples_fed / 16_000,
            )

            self._cache = [
                RotatingKVCache(8192) for _ in range(self._n_layers)
            ]

            prefix_embeds = (
                self._text_embeds + self._audio_embeds[: self._prefix_len]
            )
            prefix_embeds = prefix_embeds[None, :, :]

            logits = self._model.decode(
                prefix_embeds, self._t_cond, "causal", self._cache
            )
            mx.eval(
                logits,
                *[x for c in self._cache for x in (c.keys, c.values)],
            )

            self._y = self._sample(logits)
            mx.async_eval(self._y)

            self._audio_embeds = self._audio_embeds[self._prefix_len :]
            self._n_total_decoded = self._prefix_len
            self._prefilled = True
            log.info("Prefill complete, remaining embeds: %d", self._audio_embeds.shape[0])

            # Recompute decodable after consuming prefix
            n_decodable = min(
                self._audio_embeds.shape[0],
                safe_total - self._n_total_decoded,
            )

        if n_decodable <= 0:
            return

        # Decode loop (mirrors decode_steps in stream.py)
        n_consumed = 0
        hit_eos = False
        for i in range(n_decodable):
            token_embed = self._model.language_model.embed(
                self._y.reshape(1, 1)
            )[0, 0]
            step_embed = (self._audio_embeds[i] + token_embed)[None, None, :]
            logits = self._model.decode(
                step_embed, self._t_cond, mask=None, cache=self._cache
            )
            next_y = self._sample(logits)
            mx.async_eval(next_y)

            token_id = self._y.item()
            if token_id == self._eos_token_id:
                log.info(
                    "EOS at position %d (total_decoded=%d, audio_fed=%.1fs, tokens_emitted=%d)",
                    i,
                    self._n_total_decoded + i,
                    self._n_audio_samples_fed / 16_000,
                    self._tokens_emitted,
                )
                self._cache = None
                self._y = None
                n_consumed = i
                hit_eos = True
                self._eos_count += 1
                break

            text = self._sp.decode(
                [token_id],
                special_token_policy=SpecialTokenPolicy.IGNORE,
            )
            self._emit_token(text)

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            self._y = next_y
            n_consumed = i + 1

        self._n_total_decoded += n_consumed

        # Trim consumed embeddings
        if self._audio_embeds.shape[0] > n_consumed:
            self._audio_embeds = self._audio_embeds[n_consumed:]
        else:
            self._audio_embeds = None

        if hit_eos:
            log.info("EOS reset — starting new segment")
            self._reset_state()
```

This is the heart of the streaming engine. Two phases:

**Prefill** (first call after enough audio): The decoder needs a "prompt" to start. This is the sum of `_text_embeds` (the text prompt tokens embedded) and the first `_prefix_len` audio embeddings. The combined embeddings are fed to the decoder in one shot with causal masking, producing the first token prediction. A `RotatingKVCache` (8192-token window) is created per transformer layer to enable incremental decoding.

**Incremental decode** (subsequent calls): For each decodable position, the previous token is embedded and **added** to the current audio embedding (element-wise sum — this is how Voxtral fuses text and audio). This combined embedding is fed to the decoder with the KV cache, producing the next token. The key insight: the decoder only needs to see one position at a time because the KV cache holds all prior context.

The `safe_total` calculation prevents decoding ahead of what the encoder has produced — it accounts for the 32 left-pad tokens and the total audio samples encoded so far. This prevents the decoder from "seeing" audio it has not yet been fed.

When EOS is hit, `_reset_state()` clears all encoder/decoder state to start a fresh segment — the model can produce multiple sentences within one recording.

### Flush and Right Padding

```bash
sed -n "113,169p" src/vox/voxtral_stream.py
```

```output
    def flush(self) -> str:
        """Stop processing, flush remaining audio with right padding,
        return accumulated text."""
        elapsed = time.monotonic() - self._start_time
        log.info(
            "Stream flush requested after %.1fs "
            "(tokens=%d, eos=%d, encodes=%d, feeds=%d, audio=%.1fs)",
            elapsed,
            self._tokens_emitted,
            self._eos_count,
            self._encode_calls,
            self._feed_calls,
            self._total_audio_samples / 16_000,
        )

        self._running = False
        if not self._done_event.wait(timeout=10):
            log.warning("Stream processing thread did not stop within 10s")

        # Final flush: feed remaining audio + right padding
        remaining = self._drain_audio()
        self._pending_audio = np.append(self._pending_audio, remaining)

        if self._cache is not None and self._y is not None:
            log.info(
                "Final flush: pending=%.3fs, adding right pad (%d tokens)",
                len(self._pending_audio) / 16_000,
                N_RIGHT_PAD_TOKENS,
            )
            right_pad = np.zeros(
                N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
            )
            flush_chunk = np.concatenate([self._pending_audio, right_pad])
            self._encode_chunk(flush_chunk)

            if self._audio_embeds is not None:
                self._decode_available(self._audio_embeds.shape[0])
        else:
            log.info(
                "Final flush skipped (cache=%s, y=%s)",
                self._cache is not None,
                self._y is not None,
            )

        # Flush last pending token
        if self._y is not None:
            token_id = self._y.item()
            if token_id != self._eos_token_id:
                text = self._sp.decode(
                    [token_id],
                    special_token_policy=SpecialTokenPolicy.IGNORE,
                )
                self._emit_token(text)

        result = "".join(self._accumulated)
        log.info("Stream flush complete: %d chars, %r", len(result), result[:120])
        return result
```

`flush()` stops the processing loop, then does a final encode+decode pass. It appends 17 tokens of silence as right-padding (matching the model's expected context window), encodes everything remaining, and decodes all available positions. The last pending token is also emitted if it is not EOS.

The right-padding is critical — without it, the model would not have enough trailing context to decode the final few tokens of speech. The 17-token value comes from voxmlx's stream implementation.

## Parakeet Backend: `parakeet.py`

The Parakeet backend is the lightweight alternative — 600M parameters vs Voxtral's 4B. It takes a fundamentally different approach to streaming: instead of incremental encode/decode, it does periodic batch transcription of the growing audio buffer, emitting text deltas.

### The Transcriber

```bash
sed -n "23,85p" src/vox/parakeet.py
```

```output
class ParakeetTranscriber:
    """Wraps parakeet-mlx with a cached model for repeated transcriptions."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._sample_rate: int = 16_000

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supports_streaming(self) -> bool:
        # Periodic batch transcription: re-run model.generate() on the
        # growing audio buffer every ~2s during recording, diff text,
        # emit stable deltas.  See parakeet-realtime-issues.md.
        return True

    def load(self) -> None:
        """Pre-load model weights. Safe to call multiple times."""
        if self._model is not None:
            return
        from parakeet_mlx import from_pretrained

        log.info("Loading Parakeet model: %s", self.model_name)
        self._model = from_pretrained(self.model_name)
        self._sample_rate = self._model.preprocessor_config.sample_rate
        log.info("Parakeet model loaded (sample_rate=%d)", self._sample_rate)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe a numpy audio array to text.

        Writes a temp WAV, calls model.transcribe(path), returns result.text.
        """
        self.load()

        duration_s = len(audio) / sample_rate
        log.info("Audio %.1fs -> Parakeet batch transcribe", duration_s)

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, sample_rate)
            result = self._model.transcribe(f.name)

        text = result.text.strip()
        log.info("  result: %r", text[:120])
        return text

    def create_stream(
        self, on_token: Callable[[str], None]
    ) -> TranscriptionStream:
        """Create a periodic-batch streaming transcription session.

        Passes the model directly to ParakeetStream — no transcribe_stream
        context manager.  See parakeet-realtime-issues.md for rationale.
        """
        self.load()
        log.info("Creating ParakeetStream (periodic batch)")
        return ParakeetStream(
            model=self._model,
            sample_rate=self._sample_rate,
            on_token=on_token,
        )
```

`ParakeetTranscriber` mirrors `VoxtralTranscriber`'s interface but is simpler — no precomputed embeddings needed. Batch mode writes a temp WAV and calls `model.transcribe()`. The stream version passes the raw model to `ParakeetStream`.

### The Periodic Batch Stream

`ParakeetStream` is an elegant hack. Since Parakeet does not support true incremental decoding, it fakes streaming by periodically re-transcribing the entire accumulated audio buffer and emitting the text delta:

```bash
sed -n "88,105p" src/vox/parakeet.py
```

```output
class ParakeetStream:
    """Periodic-batch Parakeet transcription session.

    Created per recording via ParakeetTranscriber.create_stream().
    Receives the model directly (no transcribe_stream context).
    A background thread periodically batch-transcribes the full
    accumulated audio buffer and emits text deltas via on_token.
    """

    # Minimum total audio before the first batch transcription.
    _MIN_FIRST_SAMPLES = 16000  # 1.0 s @ 16 kHz

    # Minimum new audio before subsequent transcriptions.
    _MIN_NEW_SAMPLES = 8000  # 0.5 s @ 16 kHz

    # Seconds between batch transcriptions.
    _INTERVAL = 2.0

```

The tuning constants: wait for at least 1 second of audio before the first transcription, then re-transcribe every 2 seconds if there is at least 0.5 seconds of new audio. The delta computation is the interesting part:

```bash
sed -n "212,237p" src/vox/parakeet.py
```

```output
        try:
            self._on_token(text)
        except Exception:
            log.exception("on_token callback failed for %r", text)

    @staticmethod
    def _compute_delta(old_text: str, new_text: str) -> str:
        """Return the new suffix when text extends, or empty string.

        Strict prefix matching only.  If new_text starts with old_text,
        return the suffix.  Otherwise return "" — the model revised
        earlier text and we can't un-type what's already at cursor.

        This prevents garbage re-emission when batch transcription
        changes punctuation, word choices, or casing as more audio
        is added.
        """
        if new_text.startswith(old_text):
            return new_text[len(old_text):]
        return ""

    @staticmethod
    def _is_speech(audio: np.ndarray) -> bool:
        """Return True if audio chunk has energy above silence threshold."""
        if len(audio) == 0:
            return False
```

The delta computation exploits the fact that batch transcription is deterministic for the same audio. Extending the audio buffer only appends to the transcription, so the old text is always a prefix of the new text. The fast path (`new_text.startswith(old_text)`) handles this cleanly. If the model revised earlier text (which can happen with added context), the fallback finds the longest common prefix — but only emits if the new text is longer, preventing garbage re-emission when the model changes punctuation or casing.

The processing loop has the same drain-and-transcribe pattern as VoxtralStream, but instead of encode/decode, it calls `_transcribe_buffer()` which computes mel spectrograms directly from numpy (no temp WAV files) and calls `model.generate()`:

```bash
sed -n "239,252p" src/vox/parakeet.py
```

```output
        return rms > ParakeetStream._SILENCE_RMS

    def _transcribe_buffer(self, audio: np.ndarray) -> str:
        """Batch-transcribe audio via get_logmel + model.generate.

        No temp files — computes mel directly from numpy array.
        """
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        mel = get_logmel(
            mx.array(audio, dtype=mx.float32),
            self._model.preprocessor_config,
        )
```

## Text Formatting: `formatter.py`

The formatter is a pipeline of pure functions that clean up raw STT output. It runs after transcription (in batch mode) or after `stream.flush()` (in streaming mode). The formatter is toggled on/off via the menu and config:

```bash
cat src/vox/formatter.py
```

```output
"""Rule-based post-processing for transcribed text.

Each transform is a pure function (str -> str). The Formatter runs them
in sequence. Keeps the speaker's words intact — only cleans up
speech-to-text artifacts.
"""

from __future__ import annotations

import re
from typing import Callable

Transform = Callable[[str], str]


def strip_fillers(text: str) -> str:
    """Remove common filler words (um, uh, er, ah, hmm, you know)."""
    text = re.sub(r"\b(?:um|uh|er|ah|hmm)\b,?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou know,?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", text).strip()


def fix_capitalization(text: str) -> str:
    """Capitalize first character and after sentence-ending punctuation."""
    if not text:
        return text
    text = text[0].upper() + text[1:]
    text = re.sub(
        r"([.!?])\s+(\w)",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )
    return text


def ensure_trailing_punctuation(text: str) -> str:
    """Add a period if the text doesn't end with terminal punctuation."""
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def collapse_whitespace(text: str) -> str:
    """Normalize runs of whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# Ordered pipeline — each transform feeds into the next.
DEFAULT_TRANSFORMS: list[Transform] = [
    strip_fillers,
    fix_capitalization,
    ensure_trailing_punctuation,
    collapse_whitespace,
]


class Formatter:
    """Runs a pipeline of text transforms on transcribed output."""

    def __init__(
        self,
        enabled: bool = True,
        transforms: list[Transform] | None = None,
    ) -> None:
        self.enabled = enabled
        self._transforms = transforms or list(DEFAULT_TRANSFORMS)

    def format(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        for transform in self._transforms:
            text = transform(text)
        return text
```

Four transforms in order:

1. **`strip_fillers`**: Removes "um", "uh", "er", "ah", "hmm", and "you know" (with optional trailing commas). Collapses double spaces left behind.
2. **`fix_capitalization`**: Capitalizes the first character and the first letter after sentence-ending punctuation.
3. **`ensure_trailing_punctuation`**: Adds a period if the text does not end with `.`, `!`, or `?`.
4. **`collapse_whitespace`**: Normalizes triple+ newlines to double, collapses double+ spaces.

Each function is pure `str -> str`. The `Formatter` class just runs them in sequence, with a short-circuit if disabled or empty. The transform list is customizable (passed at construction) but defaults to the four above.

## Full Lifecycle: Putting It All Together

Here is the complete data flow for each mode, from user action to output:

### Batch Mode (transcript)

```
User presses Option+Space (first time)
  -> _on_hotkey() [state=IDLE]
  -> _start_recording()
  -> recorder.start() [sounddevice opens mic, chunks append to list]
  -> _set_state(RECORDING) [icon: mic.svg]

User presses Option+Space (second time)
  -> _on_hotkey() [state=RECORDING]
  -> _stop_and_transcribe()
  -> audio = recorder.stop() [concatenate chunks -> numpy array]
  -> _set_state(TRANSCRIBING) [icon: wave.svg]
  -> background thread: _transcribe_worker(audio)
    -> transcriber.transcribe(audio) [voxmlx or parakeet]
    -> formatter.format(text) [if enabled]
    -> pbcopy [clipboard]
    -> _paste_at_cursor() [if enabled, osascript Cmd+V]
    -> rumps.notification() [preview]
    -> _set_state(IDLE) [icon: logo.svg]
```

### Realtime Mode (streaming)

```
User presses Option+Space (first time)
  -> _on_hotkey() [state=IDLE]
  -> _start_streaming()
  -> stream = transcriber.create_stream(on_token=_on_stream_token)
  -> recorder.on_chunk = stream.feed [bridge audio -> stream]
  -> keystroke_worker thread starts [consumes keystroke queue]
  -> _set_state(STREAMING) [icon: wave.svg]
  -> recorder.start()

  [Audio flows: mic -> recorder._on_audio -> stream.feed -> processing thread]
  [Tokens flow: _decode_available -> _emit_token -> _on_stream_token -> queue -> keystroke_worker -> osascript]

User presses Option+Space (second time)
  -> _on_hotkey() [state=STREAMING]
  -> _stop_streaming()
  -> recorder.on_chunk = None [prevent deadlock]
  -> recorder.stop()
  -> _set_state(TRANSCRIBING) [icon: wave.svg]
  -> background thread: _finalize_streaming(stream)
    -> stream.flush() [right-pad + final decode]
    -> stream.close() [release caches]
    -> keystroke_worker.join()
    -> formatter.format(text) [if enabled]
    -> pbcopy [clipboard]
    -> rumps.notification() [preview]
    -> _set_state(IDLE) [icon: logo.svg]
```

### Threading Model

At peak activity (realtime mode), five threads are running:

| Thread | Role |
|--------|------|
| Main (NSApplication) | Event loop, hotkey handler, UI updates |
| sounddevice callback | Mic capture, forwards to stream.feed() |
| Stream processing | Encode audio, decode tokens |
| Keystroke worker | Type tokens via osascript |
| Model loading (startup only) | Load weights in background |

### File Map Summary

| File | Lines | Role |
|------|-------|------|
| `__main__.py` | 239 | CLI entry point, subcommand routing |
| `config.py` | 54 | Persistent JSON config dataclass |
| `daemon.py` | 337 | .app bundle generation, C launcher, lifecycle |
| `protocols.py` | 65 | Transcriber + TranscriptionStream protocols |
| `recorder.py` | 87 | Mic capture via sounddevice |
| `app.py` | 789 | Menubar app, state machine, hotkey, UI |
| `transcriber.py` | 106 | Voxtral batch backend |
| `voxtral_stream.py` | 462 | Voxtral incremental streaming |
| `parakeet.py` | 343 | Parakeet batch + periodic-batch streaming |
| `formatter.py` | 76 | Text cleanup pipeline |
