#!/usr/bin/env bash
# ── Doc QA Tool — Mac/Linux Launcher ─────────────────────────────
# Run: ./run.sh  (or: bash run.sh)
# First run: creates venv, installs dependencies, copies config.
# Subsequent runs: just starts the server.
# ──────────────────────────────────────────────────────────────────

set -e

cd "$(dirname "$0")"

echo ""
echo " ===================================="
echo "  Doc QA Tool"
echo " ===================================="
echo ""

# ── Check Python ──────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo " [ERROR] Python not found."
    echo " Please install Python 3.11+:"
    echo "   macOS:  brew install python@3.11"
    echo "   Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# Verify Python version >= 3.11
PYVER=$($PYTHON --version 2>&1 | awk '{print $2}')
PYMAJOR=$(echo "$PYVER" | cut -d. -f1)
PYMINOR=$(echo "$PYVER" | cut -d. -f2)

if [ "$PYMAJOR" -lt 3 ] || { [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -lt 11 ]; }; then
    echo " [ERROR] Python 3.11+ required, found $PYVER."
    exit 1
fi
echo " Python $PYVER OK"

# ── Create virtual environment (first run) ────────────────────
if [ ! -f ".venv/bin/activate" ]; then
    echo ""
    echo " Creating virtual environment..."
    $PYTHON -m venv .venv
    echo " Virtual environment created."
fi

# ── Activate venv ─────────────────────────────────────────────
source .venv/bin/activate

# ── Install / upgrade dependencies (first run or update) ──────
if [ ! -f ".venv/.installed" ]; then
    echo ""
    echo " Installing dependencies (this may take a few minutes)..."
    pip install --quiet --upgrade pip
    pip install -e .
    echo "installed" > .venv/.installed
    echo " Dependencies installed."
else
    echo " Dependencies already installed. Delete .venv/.installed to force reinstall."
fi

# ── Copy example config (first run) ───────────────────────────
if [ ! -f "config.yaml" ] && [ -f "config.example.yaml" ]; then
    echo ""
    echo " Creating config.yaml from example..."
    cp config.example.yaml config.yaml
    echo " Config created."
fi

# ── Check Cody access token ──────────────────────────────────
if [ -n "$SRC_ACCESS_TOKEN" ]; then
    echo " Cody token: found via SRC_ACCESS_TOKEN env var"
else
    echo ""
    echo " [WARNING] SRC_ACCESS_TOKEN environment variable is not set."
    echo " Set it to your Sourcegraph access token:"
    echo "   export SRC_ACCESS_TOKEN=sgp_xxxxxxxxxxxx"
    echo " Or add it to your ~/.bashrc / ~/.zshrc for persistence."
    echo ""
fi
if [ -n "$SRC_ENDPOINT" ]; then
    echo " Cody endpoint: $SRC_ENDPOINT"
else
    echo " Cody endpoint: https://sourcegraph.com (default)"
fi

# ── Start server ──────────────────────────────────────────────
echo ""
echo " Starting Doc QA server..."
echo " Open http://127.0.0.1:8000 in your browser."
echo " Press Ctrl+C to stop."
echo ""

python -m doc_qa --log-level WARNING serve
