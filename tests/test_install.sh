#!/usr/bin/env bash
# tests/test_install.sh — validates install.sh with mocked externals.
#
# Mocks uname, sw_vers, xcode-select, curl, uv, git, python3, command
# so the script runs end-to-end in a temp $HOME without side effects.
#
# Usage: bash tests/test_install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_SH="${SCRIPT_DIR}/install.sh"

# -- setup temp HOME and mock bin dir ----------------------------------------

FAKE_HOME=$(mktemp -d)
MOCK_BIN="${FAKE_HOME}/.mock_bin"
mkdir -p "${MOCK_BIN}" "${FAKE_HOME}/.local/bin" "${FAKE_HOME}/models/Voxtral-Mini-4B-Realtime-6bit"

cleanup() { rm -rf "$FAKE_HOME"; }
trap cleanup EXIT

# -- create mock commands ----------------------------------------------------

# uname: pretend macOS ARM64
cat > "${MOCK_BIN}/uname" << 'MOCK'
#!/usr/bin/env bash
case "$1" in
    -s) echo "Darwin" ;;
    -m) echo "arm64" ;;
    *)  echo "Darwin" ;;
esac
MOCK

# sw_vers
cat > "${MOCK_BIN}/sw_vers" << 'MOCK'
#!/usr/bin/env bash
echo "15.3"
MOCK

# xcode-select: pretend installed
cat > "${MOCK_BIN}/xcode-select" << 'MOCK'
#!/usr/bin/env bash
echo "/Library/Developer/CommandLineTools"
exit 0
MOCK

# curl: return a fake tag list for the GitHub API call
cat > "${MOCK_BIN}/curl" << 'MOCK'
#!/usr/bin/env bash
# If fetching tags API, return fake JSON
if [[ "$*" == *"api.github.com"* ]]; then
    echo '[{"name":"v0.3.0"},{"name":"v0.2.0"}]'
else
    # For any other curl call (e.g. uv installer), succeed silently
    exit 0
fi
MOCK

# uv: record calls, pretend success
UV_LOG="${FAKE_HOME}/uv_calls.log"
cat > "${MOCK_BIN}/uv" << MOCK
#!/usr/bin/env bash
echo "\$@" >> "${UV_LOG}"
exit 0
MOCK

# git: pretend git-lfs clone works
cat > "${MOCK_BIN}/git" << 'MOCK'
#!/usr/bin/env bash
exit 0
MOCK

# git-lfs: pretend available
cat > "${MOCK_BIN}/git-lfs" << 'MOCK'
#!/usr/bin/env bash
exit 0
MOCK

# python3: handle the config-writing snippet and tag parsing
cat > "${MOCK_BIN}/python3" << 'ENDMOCK'
#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then
    script="$2"
    if [[ "$script" == *"sys.stdin"* ]]; then
        cat > /dev/null  # consume stdin
        echo "v0.3.0"
    elif [[ "$script" == *"config.json"* ]]; then
        /usr/bin/python3 -c "$script"
    fi
fi
ENDMOCK

# vox: fake binary so "command -v vox" succeeds
cat > "${FAKE_HOME}/.local/bin/vox" << 'MOCK'
#!/usr/bin/env bash
echo "vox 0.3.0"
MOCK

chmod +x "${MOCK_BIN}"/* "${FAKE_HOME}/.local/bin/vox"

# -- run install.sh with mocked PATH ----------------------------------------

echo "=== Running install.sh with mocks ==="
echo ""

# Prepend mock bin so our mocks shadow real commands.
# Override HOME so config writes go to temp dir.
env \
    HOME="$FAKE_HOME" \
    PATH="${MOCK_BIN}:${FAKE_HOME}/.local/bin:${PATH}" \
    bash "$INSTALL_SH" 2>&1

echo ""
echo "=== Validating results ==="

# -- shellcheck --------------------------------------------------------------

if command -v shellcheck &>/dev/null; then
    echo ""
    echo "=== Running shellcheck ==="
    shellcheck "$INSTALL_SH" && printf "  PASS: shellcheck clean\n" || printf "  FAIL: shellcheck found issues\n"
fi

# -- assertions --------------------------------------------------------------

PASS=0
FAIL=0

assert() {
    local desc="$1"
    if eval "$2"; then
        printf "  PASS: %s\n" "$desc"
        PASS=$((PASS + 1))
    else
        printf "  FAIL: %s\n" "$desc"
        FAIL=$((FAIL + 1))
    fi
}

# 1. uv tool install was called with correct args
assert "uv tool install called" \
    "grep -q 'tool install' '${UV_LOG}'"

assert "uv install references v0.3.0 tag" \
    "grep -q 'v0.3.0' '${UV_LOG}'"

assert "uv install uses --force" \
    "grep -q '\-\-force' '${UV_LOG}'"

assert "uv install uses --python 3.13" \
    "grep -q '\-\-python 3.13' '${UV_LOG}'"

# 2. Config file was written with voxtral backend
CONFIG_FILE="${FAKE_HOME}/.config/vox/config.json"
assert "config.json exists" \
    "[[ -f '${CONFIG_FILE}' ]]"

assert "config backend is voxtral" \
    "/usr/bin/python3 -c \"import json; d=json.load(open('${CONFIG_FILE}')); assert d['backend']=='voxtral'\""

assert "config mode is transcript" \
    "/usr/bin/python3 -c \"import json; d=json.load(open('${CONFIG_FILE}')); assert d['mode']=='transcript'\""

# 3. No parakeet references in the script
assert "no parakeet fallback in script" \
    "! grep -q 'parakeet' '${INSTALL_SH}'"

# 4. No interactive read in the script
assert "no interactive read prompt" \
    "! grep -q 'read -rp' '${INSTALL_SH}'"

# -- summary -----------------------------------------------------------------

echo ""
echo "=== ${PASS} passed, ${FAIL} failed ==="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
