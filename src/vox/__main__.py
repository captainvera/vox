"""Entry point: `python -m vox` or the `vox` console script.

Subcommands:
    vox           Run in foreground (default, useful for debugging)
    vox start     Generate Vox.app + launch via macOS `open`
    vox stop      Stop the running Vox process
    vox restart   Stop + start (regenerates .app bundle)
    vox reload    Alias for restart (picks up source changes)
    vox status    Show whether vox is running
    vox logs      Tail the log file
    vox setup     Download transcription model (interactive)
    vox update    Update vox to the latest release
    vox uninstall Remove Vox.app and clean up
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


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


def _make_transcriber(config):
    """Create the transcriber backend based on config.backend.

    Lazy imports so only the selected backend's dependencies are loaded.
    """
    if config.backend == "parakeet":
        from .parakeet import ParakeetTranscriber

        return ParakeetTranscriber(model_name=config.parakeet_model)
    if config.backend == "moonshine":
        from .moonshine import MoonshineTranscriber

        return MoonshineTranscriber(model_arch=config.moonshine_arch)
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


_REPO = "captainvera/vox"
_VOXTRAL_HF_REPO = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"


def _latest_tag() -> str:
    """Fetch the latest vX.Y.Z tag from GitHub."""
    url = f"https://api.github.com/repos/{_REPO}/tags"
    with urllib.request.urlopen(url, timeout=10) as resp:
        tags = json.loads(resp.read())
    for tag in tags:
        if tag["name"].startswith("v"):
            return tag["name"]
    raise RuntimeError("No version tags found")


def _setup() -> None:
    """Interactive model download."""
    from .config import DEFAULT_MODEL_PATH, Config

    voxtral_dir = Path(DEFAULT_MODEL_PATH)

    print("\nChoose a transcription model:\n")
    print("  1) Parakeet  — 600 MB, downloads on first launch, fast")
    print("  2) Voxtral   — 6.8 GB, best quality, realtime streaming")
    print("  3) Moonshine — 245 MB, low latency, built-in streaming\n")

    choice = input("Choice [1]: ").strip() or "1"

    config = Config.load()

    if choice == "3":
        config.backend = "moonshine"
        print("Moonshine model will download on first launch.")
    elif choice == "2":
        config.backend = "voxtral"

        if voxtral_dir.is_dir():
            print(f"Voxtral model already at {voxtral_dir}")
        else:
            print("Downloading Voxtral model (6.8 GB)...")
            voxtral_dir.parent.mkdir(parents=True, exist_ok=True)

            if shutil.which("git-lfs"):
                subprocess.run(
                    ["git", "clone",
                     f"https://huggingface.co/{_VOXTRAL_HF_REPO}",
                     str(voxtral_dir)],
                    check=True,
                )
            elif shutil.which("uvx"):
                subprocess.run(
                    ["uvx", "--from", "huggingface-hub",
                     "huggingface-cli", "download",
                     _VOXTRAL_HF_REPO,
                     "--local-dir", str(voxtral_dir)],
                    check=True,
                )
            else:
                print("Install git-lfs (brew install git-lfs) or uv, then re-run.")
                sys.exit(1)

            print(f"Downloaded to {voxtral_dir}")
    else:
        config.backend = "parakeet"
        print("Parakeet model will download on first launch.")

    config.save()
    print(f"Default backend: {config.backend}")


def _update() -> None:
    """Update vox to the latest tagged release."""
    from . import __version__

    try:
        tag = _latest_tag()
    except Exception as exc:
        print(f"Failed to check for updates: {exc}")
        sys.exit(1)

    remote_version = tag.lstrip("v")
    if remote_version == __version__:
        print(f"Already up to date ({__version__})")
        return

    print(f"Updating vox {__version__} → {remote_version}...")

    uv = shutil.which("uv")
    if not uv:
        print("uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    subprocess.run(
        [uv, "tool", "install", "--force",
         f"vox @ git+https://github.com/{_REPO}@{tag}",
         "--python", "3.13"],
        check=True,
    )
    print(f"Updated to {remote_version}")


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


if __name__ == "__main__":
    main()
