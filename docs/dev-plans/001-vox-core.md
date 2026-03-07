# vox

Menubar STT for macOS. Record voice, transcribe with Voxtral, format, copy to clipboard.

## Goals

- High accuracy offline transcription (Voxtral on Apple Silicon via MLX)
- Always-on menubar app with global hotkey (Option+Space)
- Light post-processing: capitalization, punctuation, filler word removal
- Minimal, tasteful UI — icon-only menubar (four states, no text), no clutter
- Runs as native .app — no terminal window, Accessibility scoped to app
- Realtime streaming transcription — type tokens at cursor as decoded
- Backend-agnostic transcriber protocol — support multiple models

## Non-Goals

- Cloud APIs or network dependencies

## Architecture

```
app.py            Menubar + state machine + hotkey (rumps + NSEvent)
protocols.py      Transcriber + TranscriptionStream protocols (backend-agnostic)
transcriber.py    Voxtral batch STT; implements Transcriber protocol
voxtral_stream.py Voxtral streaming STT session; implements TranscriptionStream
recorder.py       Mic capture -> numpy array (sounddevice), optional on_chunk callback
formatter.py      Rule-based text cleanup pipeline
config.py         Settings persistence (~/.config/vox/config.json)
daemon.py         Vox.app bundle (C source + compilation + hash cache) + lifecycle
__main__.py       CLI routing, .app bundle detection + log redirect, foreground run
tests/            55 pytest tests (protocols, config, recorder, transcriber, stream, app)
```

## Protocols

The app depends on protocols, never on concrete transcriber classes:

```python
class Transcriber(Protocol):
    is_loaded: bool
    supports_streaming: bool
    def load() -> None
    def transcribe(audio: np.ndarray) -> str
    def create_stream(on_token: Callable) -> TranscriptionStream

class TranscriptionStream(Protocol):
    def feed(chunk: np.ndarray) -> None
    def flush() -> str
    def close() -> None
```

A model that doesn't support streaming returns `supports_streaming = False` and the app falls back to transcript mode. No other code changes needed.

## Flow

### Transcript mode (default)

1. `vox start` compiles C launcher (or skips if hash unchanged), generates Vox.app, launches via `open -n`
2. App detects .app context (NSBundle), redirects stdout/stderr to log file
3. Loads Voxtral model in background (~5s)
4. User presses Option+Space -> recording starts
5. User presses Option+Space again -> recording stops
6. Audio -> Voxtral transcribes offline (temp file, no streaming)
7. Formatter cleans text (if enabled)
8. Result copied to clipboard, optionally pasted via osascript Cmd+V

### Realtime mode

1. User toggles "Realtime mode" in menubar dropdown
2. Option+Space -> recording starts + VoxtralStream created
3. Audio chunks flow: mic -> recorder on_chunk -> stream.feed() -> bg thread
4. Background thread: mel spectrogram -> encode_step -> decode -> on_token callback
5. Each token typed at cursor via osascript keystroke
6. Option+Space -> stop recording, flush remaining tokens with right padding
7. Final text copied to clipboard, notification shown

## Key Decisions

- **Voxtral only** via voxmlx, local model at ~/models/Voxtral-Mini-4B-Realtime-6bit
- **Compiled C launcher** — embeds Python via Py_InitializeFromConfig. Required because macOS doesn't show NSStatusBar items for script-based .app bundles (process identity becomes Python.app). Source embedded in daemon.py, compiled at start time, hash-cached to skip redundant builds
- **NSEvent for hotkey** — replaced pynput (which crashes in .app context due to TSM thread assertions). Runs on main thread via NSApplication event loop
- **osascript for paste** — replaced pynput keyboard.Controller (unreliable in terminals)
- **osascript keystroke for realtime** — types decoded tokens at cursor via System Events. ~30-50ms per call; tokens emitted word-by-word
- **AppHelper.callAfter** for UI updates from background threads
- **rumps** for menubar (LSUIElement .app, no Dock icon). Title must be set to `None` after icon — rumps' `fallbackOnName()` re-sets title to app name when both title and image are empty, so icon must be set first
- **NSBundle log redirect** — `__main__.py` detects .app context via `NSBundle.mainBundle().bundleIdentifier()`, redirects stdout/stderr to log file. No C recompile needed if log path changes
- **Protocol-based transcriber abstraction** — `app.py` depends on `Transcriber` protocol, not `VoxtralTranscriber`. Adding a new model backend = implement the protocol, pass to VoxApp. No app changes needed
- Installed via `uv tool install -e .`, source changes via `vox reload`

## Dependencies

- rumps (macOS menubar)
- voxmlx (Voxtral STT on MLX)
- sounddevice (mic recording)
- soundfile (audio I/O for temp files)
- PyObjC/AppKit (NSEvent hotkey, AppHelper — comes with rumps)
- Xcode CLI tools (cc — for compiling the C launcher)
- pytest (dev dependency, for tests)
