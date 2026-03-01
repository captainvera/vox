# vox

Menubar STT for macOS. Record voice, transcribe with Voxtral, format, copy to clipboard.

## Goals

- High accuracy offline transcription (Voxtral on Apple Silicon via MLX)
- Always-on menubar app with global hotkey (Option+Space)
- Light post-processing: capitalization, punctuation, filler word removal
- Minimal, tasteful UI — icon-only menubar (four states, no text), no clutter
- Runs as native .app — no terminal window, Accessibility scoped to app

## Non-Goals

- Live/realtime streaming transcription
- Multiple model backends
- Cloud APIs or network dependencies

## Architecture

```
app.py            Menubar + state machine + hotkey (rumps + NSEvent)
recorder.py       Mic capture -> numpy array (sounddevice)
transcriber.py    Voxtral STT via voxmlx (model cached, offline)
formatter.py      Rule-based text cleanup pipeline
config.py         Settings persistence (~/.config/vox/config.json)
daemon.py         Vox.app bundle (C source + compilation + hash cache) + lifecycle
__main__.py       CLI routing, .app bundle detection + log redirect, foreground run
```

## Flow

1. `vox start` compiles C launcher (or skips if hash unchanged), generates Vox.app, launches via `open -n`
2. App detects .app context (NSBundle), redirects stdout/stderr to log file
3. Loads Voxtral model in background (~5s)
4. User presses Option+Space -> recording starts
5. User presses Option+Space again -> recording stops
6. Audio -> Voxtral transcribes offline (temp file, no streaming)
7. Formatter cleans text (if enabled)
8. Result copied to clipboard, optionally pasted via osascript Cmd+V

## Key Decisions

- **Voxtral only** via voxmlx, local model at ~/models/Voxtral-Mini-4B-Realtime-6bit
- **Compiled C launcher** — embeds Python via Py_InitializeFromConfig. Required because macOS doesn't show NSStatusBar items for script-based .app bundles (process identity becomes Python.app). Source embedded in daemon.py, compiled at start time, hash-cached to skip redundant builds
- **NSEvent for hotkey** — replaced pynput (which crashes in .app context due to TSM thread assertions). Runs on main thread via NSApplication event loop
- **osascript for paste** — replaced pynput keyboard.Controller (unreliable in terminals)
- **AppHelper.callAfter** for UI updates from background threads
- **rumps** for menubar (LSUIElement .app, no Dock icon). Title must be set to `None` after icon — rumps' `fallbackOnName()` re-sets title to app name when both title and image are empty, so icon must be set first
- **NSBundle log redirect** — `__main__.py` detects .app context via `NSBundle.mainBundle().bundleIdentifier()`, redirects stdout/stderr to log file. No C recompile needed if log path changes
- Installed via `uv tool install -e .`, source changes via `vox reload`

## Dependencies

- rumps (macOS menubar)
- voxmlx (Voxtral STT on MLX)
- sounddevice (mic recording)
- soundfile (audio I/O for temp files)
- PyObjC/AppKit (NSEvent hotkey, AppHelper — comes with rumps)
- Xcode CLI tools (cc — for compiling the C launcher)
