"""Vox.app bundle management — create, launch, stop, reload.

Instead of a bare launchd daemon (which runs as python3.13 and needs
Accessibility granted to *all* of Python), we generate a minimal .app
bundle at ~/Applications/Vox.app.  When launched via ``open``, macOS
LaunchServices associates the process with the .app, so Accessibility
can be granted to Vox.app alone.
"""

from __future__ import annotations

import hashlib
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

# -- paths ----------------------------------------------------------------

APP_DIR = Path.home() / "Applications"
APP_PATH = APP_DIR / "Vox.app"
DATA_DIR = Path.home() / ".local" / "share" / "vox"
LOG_FILE = DATA_DIR / "vox.log"
PID_FILE = DATA_DIR / "vox.pid"

# Legacy launchd artefacts (from the previous daemon approach).
_LEGACY_LABEL = "com.vox.agent"
_LEGACY_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{_LEGACY_LABEL}.plist"

# -- compiled C launcher --------------------------------------------------
#
# Native binary that embeds Python via Py_InitializeFromConfig + Py_RunMain.
# Required because macOS won't show NSStatusBar items for script-based .app
# bundles — the process identity becomes Python.app and the menubar item
# silently fails.  The compiled binary keeps the identity as Vox.app.
#
# PYTHON_EXECUTABLE is injected at compile time via -D flag, pointing at the
# venv's python so that voxmlx, rumps, etc. are importable.

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


# -- C launcher compilation -----------------------------------------------

def _python_compile_flags() -> tuple[str, str, str]:
    """Resolve compiler flags for embedding Python.

    Returns (include_dir, lib_path, rpath_dir).

    On macOS framework builds (Homebrew Python), links directly against the
    framework dylib — ``-framework Python`` doesn't work because Homebrew's
    framework lacks the standard top-level symlinks.
    """
    include = sysconfig.get_config_var("INCLUDEPY")
    if not include:
        raise RuntimeError("Cannot determine Python include path from sysconfig")

    libdir = sysconfig.get_config_var("LIBDIR")
    framework = sysconfig.get_config_var("PYTHONFRAMEWORK")

    if framework:
        # Framework build: dylib is at <framework_dir>/<framework_name>
        # e.g. .../Versions/3.13/Python
        framework_dir = os.path.dirname(libdir)  # .../Versions/3.13
        lib_path = os.path.join(framework_dir, framework)
        rpath = framework_dir
    else:
        # Non-framework build: link against the shared library directly.
        ldlib = sysconfig.get_config_var("LDLIBRARY")
        lib_path = os.path.join(libdir, ldlib)
        rpath = libdir

    return include, lib_path, rpath


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


# -- helpers --------------------------------------------------------------

def _find_vox_bin() -> str:
    """Resolve the absolute path to the ``vox`` console-script."""
    path = shutil.which("vox")
    if path:
        return str(Path(path).resolve())
    fallback = Path.home() / ".local" / "bin" / "vox"
    if fallback.exists():
        return str(fallback.resolve())
    raise FileNotFoundError(
        "Cannot find 'vox' on PATH. Is it installed via `uv tool install`?"
    )


def _read_pid() -> int | None:
    """Return the running vox PID, or None if not running."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check if process exists (signal 0 = no-op)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


def _kill_all_vox() -> None:
    """Kill any running vox processes (guards against orphans from open -n)."""
    # Match both the compiled C launcher and the legacy Python script launcher.
    _patterns = ["Vox.app/Contents/MacOS/Vox", "Python.*vox/bin/vox"]
    for pat in _patterns:
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    # Brief wait for processes to exit.
    time.sleep(0.5)
    # Force-kill stragglers.
    for pat in _patterns:
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    PID_FILE.unlink(missing_ok=True)


def _cleanup_legacy_launchd() -> None:
    """Unload and remove the old LaunchAgent plist if present."""
    result = subprocess.run(
        ["launchctl", "list", _LEGACY_LABEL],
        capture_output=True,
    )
    if result.returncode == 0:
        subprocess.run(
            ["launchctl", "unload", str(_LEGACY_PLIST)],
            capture_output=True,
        )
    if _LEGACY_PLIST.exists():
        _LEGACY_PLIST.unlink()


# -- app bundle -----------------------------------------------------------

def _create_app_bundle() -> Path:
    """Create or update ~/Applications/Vox.app."""
    contents = APP_PATH / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    # App icon — copy .icns into Resources/.
    icns_src = Path(__file__).parent / "icons" / "Vox.icns"
    if icns_src.exists():
        shutil.copy(icns_src, resources / "Vox.icns")

    # Info.plist
    info = {
        "CFBundleIdentifier": "com.vox.app",
        "CFBundleName": "Vox",
        "CFBundleDisplayName": "Vox",
        "CFBundleExecutable": "Vox",
        "CFBundleIconFile": "Vox",
        "CFBundleVersion": "0.1.0",
        "LSUIElement": True,  # no Dock icon
        "NSMicrophoneUsageDescription": (
            "Vox needs microphone access to transcribe speech."
        ),
    }
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(info, f, fmt=plistlib.FMT_XML)

    # Compiled C launcher — native binary that embeds Python.
    # Log redirection is handled in __main__.py (detects .app context).
    _compile_launcher(macos)

    # Force Launch Services to re-read Info.plist — without this, macOS
    # caches the old plist and changes (e.g. adding an icon) don't appear.
    _LSREGISTER = (
        "/System/Library/Frameworks/CoreServices.framework"
        "/Versions/A/Frameworks/LaunchServices.framework"
        "/Versions/A/Support/lsregister"
    )
    if os.path.exists(_LSREGISTER):
        subprocess.run([_LSREGISTER, "-f", str(APP_PATH)], capture_output=True)

    return APP_PATH


# -- public API (called from __main__) ------------------------------------

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


def stop() -> None:
    """Stop all running vox processes."""
    pid = _read_pid()
    if not pid:
        print("vox is not running.")
        return

    _kill_all_vox()
    print("vox stopped.")


def restart() -> None:
    """Stop then start (full bundle refresh)."""
    _kill_all_vox()
    start()


def reload() -> None:
    """Alias for restart — kill process, relaunch with fresh code."""
    restart()


def status() -> None:
    """Print whether vox is running."""
    pid = _read_pid()
    if pid:
        print(f"vox is running (PID {pid}).")
        print(f"Logs: {LOG_FILE}")
    else:
        print("vox is not running.")


def uninstall() -> None:
    """Stop vox, remove the .app bundle, and clean up."""
    _cleanup_legacy_launchd()
    _kill_all_vox()
    if APP_PATH.exists():
        shutil.rmtree(APP_PATH)
        print(f"Removed {APP_PATH}")
    else:
        print("Vox.app not found.")
