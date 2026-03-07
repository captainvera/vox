#!/usr/bin/env bash
# Vox installer — menubar STT for macOS on Apple Silicon.
# Usage: curl -fsSL https://raw.githubusercontent.com/captainvera/vox/main/install.sh | bash
set -euo pipefail

REPO="captainvera/vox"
VOXTRAL_HF_REPO="mlx-community/Voxtral-Mini-4B-Realtime-6bit"
VOXTRAL_DIR="$HOME/models/Voxtral-Mini-4B-Realtime-6bit"

# -- Helpers ------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { printf "${BLUE}==>${NC} %s\n" "$1"; }
success() { printf "${GREEN}==>${NC} %s\n" "$1"; }
warn()    { printf "${YELLOW}==>${NC} %s\n" "$1"; }
error()   { printf "${RED}==>${NC} %s\n" "$1" >&2; }

# -- Preflight ----------------------------------------------------------------

info "Checking system requirements..."

if [[ "$(uname -s)" != "Darwin" ]]; then
    error "vox only runs on macOS."
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    error "vox requires Apple Silicon (ARM64). Intel Macs are not supported."
    exit 1
fi

if ! xcode-select -p &>/dev/null; then
    error "Xcode CLI tools not found. Install with: xcode-select --install"
    exit 1
fi

success "macOS $(sw_vers -productVersion) on Apple Silicon"

# -- Install uv ---------------------------------------------------------------

if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        error "Failed to install uv. Add ~/.local/bin to your PATH and re-run."
        exit 1
    fi
    success "uv installed"
else
    success "uv found at $(command -v uv)"
fi

# -- Fetch latest tag ----------------------------------------------------------

info "Fetching latest release..."

TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/tags" \
    | python3 -c "
import json, sys
tags = json.load(sys.stdin)
vtags = [t['name'] for t in tags if t['name'].startswith('v')]
if vtags:
    print(vtags[0])
else:
    sys.exit(1)
" 2>/dev/null) || {
    error "Could not determine latest version. Check https://github.com/${REPO}/tags"
    exit 1
}

success "Latest version: ${TAG}"

# -- Install vox ---------------------------------------------------------------

info "Installing vox ${TAG}..."
uv tool install --force "vox @ git+https://github.com/${REPO}@${TAG}" --python 3.13

# Ensure vox is on PATH
export PATH="$HOME/.local/bin:$PATH"

if ! command -v vox &>/dev/null; then
    error "vox command not found after install."
    error "Add this to your shell profile: export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
fi

success "vox installed: $(command -v vox)"

# -- Download Voxtral model ----------------------------------------------------

if [[ -d "$VOXTRAL_DIR" ]]; then
    success "Voxtral model already exists at ${VOXTRAL_DIR}"
else
    info "Downloading Voxtral model (6.8 GB) — this will take a while..."

    if command -v git-lfs &>/dev/null; then
        git clone "https://huggingface.co/${VOXTRAL_HF_REPO}" "$VOXTRAL_DIR"
    else
        warn "git-lfs not found — installing via Homebrew..."
        if command -v brew &>/dev/null; then
            brew install git-lfs
            git lfs install
            git clone "https://huggingface.co/${VOXTRAL_HF_REPO}" "$VOXTRAL_DIR"
        else
            warn "No Homebrew found. Falling back to huggingface-cli..."
            uvx --from huggingface-hub huggingface-cli download \
                "$VOXTRAL_HF_REPO" \
                --local-dir "$VOXTRAL_DIR"
        fi
    fi

    success "Voxtral model downloaded to ${VOXTRAL_DIR}"
fi

# Set voxtral as default backend
mkdir -p "$HOME/.config/vox"
python3 -c "
import json
from pathlib import Path
cfg = Path.home() / '.config' / 'vox' / 'config.json'
data = json.loads(cfg.read_text()) if cfg.exists() else {}
data['backend'] = 'voxtral'
data['mode'] = 'transcript'
cfg.write_text(json.dumps(data, indent=2) + '\n')
"
success "Default backend: voxtral"

# -- Done ----------------------------------------------------------------------

echo ""
success "vox is ready!"
echo ""
echo "  Start:    vox start"
echo "  Stop:     vox stop"
echo "  Logs:     vox logs"
echo "  Update:   vox update"
echo ""
echo "After first launch, grant Accessibility to Vox.app:"
echo "  System Settings → Privacy & Security → Accessibility"
echo "  Press Cmd+Shift+G → type ~/Applications → select Vox.app"
echo ""
