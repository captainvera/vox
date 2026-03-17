# Vox

Menubar speech-to-text for macOS. Record voice via global hotkey, transcribe offline on Apple Silicon, output to clipboard or type at cursor.

Runs as a native `.app` bundle with a compiled C launcher — no terminal window, Accessibility scoped to the app.

## Requirements

- macOS on Apple Silicon (M1+)
- Xcode CLI tools (`xcode-select --install`)
- ~7 GB disk for the Voxtral model

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/captainvera/vox/main/install.sh | bash
```

This installs `vox` via [uv](https://docs.astral.sh/uv/), downloads the Voxtral model (~6.8 GB), starts the menubar app, and enables autostart at login.

After install, grant Accessibility to **Vox.app**:

> System Settings > Privacy & Security > Accessibility
>
> Press `Cmd+Shift+G`, type `~/Applications`, select **Vox.app**

## Usage

| Command | Description |
|---|---|
| `vox start` | Launch menubar app |
| `vox stop` | Stop all vox processes |
| `vox restart` | Restart (picks up source changes) |
| `vox status` | Show running state + PID |
| `vox logs` | Tail the log file |
| `vox autostart on\|off` | Enable/disable launch at login |
| `vox uninstall` | Stop + remove app + remove autostart |

**Default hotkey:** `Ctrl+Shift+Space` — press to start recording, press again to stop and transcribe.

### Modes

- **Transcript** (default): Record all audio, then batch-transcribe. Result goes to clipboard (or types at cursor if enabled).
- **Realtime**: Transcribe while recording, typing tokens at cursor as they're decoded. Toggle via the menubar menu.

## Manual install

If you prefer not to use the install script:

```bash
# Install vox
uv tool install "vox @ git+https://github.com/captainvera/vox@v0.5.0" --python 3.13

# Download the Voxtral model (~6.8 GB)
git clone https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-6bit ~/models/Voxtral-Mini-4B-Realtime-6bit

# Start
vox start
```

## Dev backends

Vox supports alternative backends (Parakeet 600M, Moonshine 245M) behind a dev flag. These are not installed by default.

```bash
# Install with alt backends
uv tool install "vox[alt-backends] @ git+https://github.com/captainvera/vox@v0.5.0" --python 3.13

# Enable dev mode
VOX_DEV=1 vox start
```

## License

MIT
