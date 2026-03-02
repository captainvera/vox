"""Entry point: `python -m vox` or the `vox` console script.

Subcommands:
    vox           Run in foreground (default, useful for debugging)
    vox start     Generate Vox.app + launch via macOS `open`
    vox stop      Stop the running Vox process
    vox restart   Stop + start (regenerates .app bundle)
    vox reload    Alias for restart (picks up source changes)
    vox status    Show whether vox is running
    vox logs      Tail the log file
    vox uninstall Remove Vox.app and clean up
"""

from __future__ import annotations

import logging
import logging.handlers
import subprocess
import sys


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
    elif cmd == "uninstall":
        daemon.uninstall()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: vox [start|stop|restart|reload|status|logs|uninstall]")
        sys.exit(1)


if __name__ == "__main__":
    main()
