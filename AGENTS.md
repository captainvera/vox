# vox

Menubar STT for macOS. Record voice via hotkey, transcribe offline on Apple Silicon (Voxtral, Parakeet, Moonshine), output to clipboard or type at cursor. Runs as a native `.app` bundle with a compiled C launcher — no terminal window needed, Accessibility scoped to the app.

## Quick orientation

### Source map — read these to understand the codebase

**Entry points & wiring:**
- `src/vox/__main__.py` — CLI dispatcher + app bootstrap. `_make_transcriber()` (line 53) is the backend factory. `_run_foreground()` (line 71) wires Config → Recorder → Transcriber → Formatter → VoxApp.
- `src/vox/daemon.py` — .app bundle generation (compiled C launcher + Info.plist), start/stop/restart lifecycle, autostart via LaunchAgent plist. The C source is `_LAUNCHER_C` (line 48).

**Core app:**
- `src/vox/app.py` — rumps menubar app (~790 lines). State machine at `_set_state()` (line 464), hotkey at `_start_hotkey()` (line 431), streaming lifecycle at `_start_streaming()` / `_stop_streaming()` / `_finalize_streaming()` (lines 511-636), batch path at `_stop_and_transcribe()` / `_transcribe_worker()` (lines 639-685). Backend hot-swap at `_reload_backend()` (line 784).
- `src/vox/protocols.py` — `Transcriber` and `TranscriptionStream` runtime-checkable protocols. All backends implement these. Read this first to understand the contract.
- `src/vox/recorder.py` — mic capture via sounddevice. `on_chunk` callback enables streaming mode. Thread-safe via lock.
- `src/vox/config.py` — persistent settings dataclass at `~/.config/vox/config.json`. `VALID_BACKENDS`, `VALID_MODES`. `dev_mode` flag gates non-Voxtral backends (env `VOX_DEV=1` overrides).
- `src/vox/formatter.py` — rule-based text cleanup (strip fillers, fix caps, punctuation). Pure functions, no state.

**Backends (each implements `Transcriber` + `TranscriptionStream`):**
- `src/vox/transcriber.py` + `src/vox/voxtral_stream.py` — Voxtral 4B. Batch via temp WAV + voxmlx generate. Streaming via incremental mel encoding + autoregressive decoding with silence detection (RMS-based), EOS handling, and segment reset. The stream file is ~550 lines — the most complex module.
- `src/vox/parakeet.py` — Parakeet 600M. Batch via `parakeet_mlx.from_pretrained()`. Streaming via periodic re-transcription every 1.5s with a confirmation buffer (text emitted only when stable across 2 consecutive batches) + word-boundary trimming.
- `src/vox/moonshine.py` — Moonshine 245M. Batch via `moonshine_voice`. Streaming uses emit-on-complete strategy — only emits on `LineCompleted` events, ignores intermediate `LineTextChanged` revisions. `_compute_delta()` exists but is dead code (kept for potential incremental mode).

**Other:**
- `src/vox/icons/` — SVG template images (logo.svg idle, mic.svg recording, wave.svg transcribing)
- `tests/` — 215 tests across 8 files. `conftest.py` mocks rumps/AppKit/Foundation for headless testing.
- `tests/test_install.sh` — bash test for `install.sh` with mocked externals.
- `docs/dev-plans/` — numbered dev plans. 001 (core) done, 002 (moonshine) done, 003 (voxtral streaming) mostly done.

## Things you need to know

**Compiled C launcher.** Vox.app uses a compiled native binary (not a Python script) as its executable. This is required because macOS won't show NSStatusBar items for script-based .app bundles — the process identity becomes Python.app and the menubar item silently fails. The C launcher embeds Python via `Py_InitializeFromConfig` + `Py_RunMain`, keeping the process identity as Vox.app. Source is embedded in `daemon.py` as `_LAUNCHER_C`, compiled automatically at `vox start` time using `cc` (Xcode CLI tools required). Recompilation is skipped if the binary exists and a sha256 hash of the source + `sys.executable` hasn't changed.

**No pynput.** Replaced with Cocoa's `NSEvent.addGlobalMonitorForEventsMatchingMask:handler:`. pynput's keyboard listener crashes when running as a .app — it calls macOS TSM APIs from a background thread, which triggers `dispatch_assert_queue` failures. The NSEvent approach runs on the main thread (NSApplication event loop) and has no threading issues.

**Type-at-cursor uses osascript.** Copies to clipboard, then sends `Cmd+V` via macOS System Events. Needs Accessibility permission for Vox.app.

**Accessibility is scoped to Vox.app.** Permission is granted to `~/Applications/Vox.app` (not all of Python). The compiled binary is the process identity.

**UI updates dispatch to main thread.** `_set_state()` uses `AppHelper.callAfter()` to ensure menubar changes happen on the main thread. Background threads (model loading, transcription) cannot touch UI directly when running as a .app.

**Model is pre-downloaded.** Voxtral weights at `~/models/Voxtral-Mini-4B-Realtime-6bit`, cloned via git (Python SSL + Cloudflare WARP cert issue). Never downloads at runtime.

**voxmlx private API.** Source: https://github.com/awni/voxmlx (by Awni Hannun, Apple MLX team). Imports `voxmlx._build_prompt_tokens`, `voxmlx.load_model`, `voxmlx.generate.generate`. Check installed source at `~/.local/share/uv/tools/vox/lib/python3.13/site-packages/voxmlx/` if behavior changes. The streaming path also uses `voxmlx.audio.log_mel_spectrogram_step`, `voxmlx.audio.SAMPLES_PER_TOKEN`, `voxmlx.cache.RotatingKVCache`, and `model.encode_step()` / `model.decode()` directly.

**Menubar icons.** All states use SVG template images (adapt to light/dark automatically). Idle/loading = logo.svg (stylized waveform "V"), recording = mic.svg, transcribing = wave.svg. Icons at `src/vox/icons/`, resolved via `Path(__file__).parent / "icons"`. No text titles — `self.title = None` is set after icon to avoid rumps' `fallbackOnName()` re-setting it to the app name.

**Log redirect in .app context.** When launched from the compiled .app bundle, `__main__.py` detects the bundle via `NSBundle.mainBundle().bundleIdentifier()` and redirects stdout/stderr to `~/.local/share/vox/vox.log` before anything else runs. This captures logging, print(), and unhandled exceptions. When running `vox` in foreground (no subcommand), output goes to the terminal as usual.

## Running vox

```
vox start      # compile launcher, generate Vox.app, launch via open -n
vox stop       # kill all vox processes
vox restart    # stop + start
vox reload     # alias for restart (picks up source changes)
vox status     # show if running + PID
vox logs       # tail -f the log file
vox autostart  # show autostart status
vox autostart on   # enable launch at login (launchd plist)
vox autostart off  # disable launch at login
vox uninstall  # stop + remove Vox.app + remove autostart plist
vox            # run in foreground (for debugging)
```

- App bundle: `~/Applications/Vox.app`
- PID file: `~/.local/share/vox/vox.pid`
- Logs: `~/.local/share/vox/vox.log`
- Config: `~/.config/vox/config.json`
- Source changes take effect after `vox reload` (editable install)

**First-time setup:** After `vox start`, grant Accessibility to **Vox** in System Settings > Privacy & Security > Accessibility. In the file picker, press Cmd+Shift+G and type `~/Applications` to navigate to the hidden home Applications folder. If you recompile the launcher (e.g. after changing Python version), toggle Accessibility off/on for Vox — macOS TCC tracks the specific binary.

## State machine

```
LOADING  ->  IDLE  <->  RECORDING  ->  TRANSCRIBING  ->  IDLE
  (model)      (hotkey)    (hotkey)       (background)
                    \
                     `<->  STREAMING  ->  IDLE
                      (hotkey, realtime mode)
```

Two modes controlled by `config.mode`:
- **transcript** (default): Record all audio, then batch-transcribe. `IDLE -> RECORDING -> TRANSCRIBING -> IDLE`.
- **realtime**: Record and transcribe simultaneously, typing tokens at cursor as decoded. `IDLE -> STREAMING -> IDLE`. Falls back to transcript mode if the model doesn't support streaming (`transcriber.supports_streaming`).

## Threading

- Main thread: rumps event loop + NSEvent hotkey monitor
- Background thread: model loading (at startup)
- Background thread: transcription worker (per recording, transcript mode)
- Background thread: VoxtralStream processing loop (per recording, realtime mode) — runs encode/decode, fires `on_token` callback
- Background thread: MoonshineStream/ParakeetStream processing loop (per recording, realtime mode)
- Background thread: keystroke worker (streaming mode) — drains queue, types via osascript
- Callback thread: sounddevice audio capture (lock-protected buffer)

Thread safety: `threading.Lock` in Recorder/VoxtralStream/MoonshineStream/ParakeetStream, `queue.Queue` for keystroke dispatch, `threading.Event` (`_done_event`) for stream shutdown coordination, `AppHelper.callAfter()` for main-thread UI dispatch.

## Install / run

```
uv tool install -e /Users/mvera/dev/personal/vox --python 3.13
vox start
```

Editable install — source changes take effect on `vox reload`. No reinstall needed. Requires Xcode CLI tools (`cc`) for compiling the launcher.

## What's missing

- Parakeet model download fails silently when `hf_hub_download()` errors (SSL/network) — `parakeet_mlx.from_pretrained()` falls back to treating the HF repo ID as a local path. Upstream bug in parakeet_mlx, not vox.
