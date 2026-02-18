#!/usr/bin/env bash
# ── Build a distributable release zip ─────────────────────────
# This script:
#   1. Builds the frontend (npm run build)
#   2. Downloads all ML models (bundle-models)
#   3. Packages everything into doc-qa-tool-vX.Y.Z.zip
#
# The resulting zip is fully self-contained — users just need Python 3.11+.
# No internet required after extraction (except for Cody LLM calls).
# ──────────────────────────────────────────────────────────────

set -e

cd "$(dirname "$0")"

VERSION=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
")
echo "Building release v${VERSION}..."

# ── Step 1: Build frontend ────────────────────────────────────
echo ""
echo "=== Building frontend ==="
cd frontend
npm ci --silent
npm run build
cd ..
echo "Frontend built."

# ── Step 2: Bundle ML models ──────────────────────────────────
echo ""
echo "=== Bundling ML models ==="
python3 -m doc_qa bundle-models
echo "Models bundled."

# ── Step 3: Create zip ────────────────────────────────────────
echo ""
echo "=== Packaging release ==="

RELEASE_NAME="doc-qa-tool-v${VERSION}"
RELEASE_DIR="/tmp/${RELEASE_NAME}"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# Copy source code
rsync -a --exclude='.venv' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='.ruff_cache' \
         --exclude='.pytest_cache' \
         --exclude='*.egg-info' \
         --exclude='frontend/node_modules' \
         --exclude='logs' \
         --exclude='build' \
         --exclude='dist' \
         --exclude='.git' \
         --exclude='tests' \
         --exclude='config.yaml' \
         ./ "$RELEASE_DIR/"

# Ensure data/models is included (overrides .gitignore exclusion from rsync)
if [ -d "data/models" ]; then
    mkdir -p "$RELEASE_DIR/data"
    cp -r data/models "$RELEASE_DIR/data/"
fi

# Include example config
cp config.example.yaml "$RELEASE_DIR/config.example.yaml"

# Create the zip
cd /tmp
rm -f "${RELEASE_NAME}.zip"
zip -r -q "${RELEASE_NAME}.zip" "$RELEASE_NAME"
mv "${RELEASE_NAME}.zip" "$(cd - > /dev/null && pwd)/"
rm -rf "$RELEASE_DIR"

echo ""
echo "=== Release built ==="
echo "  File: ${RELEASE_NAME}.zip"
echo "  Size: $(du -h "${RELEASE_NAME}.zip" 2>/dev/null | cut -f1 || echo 'unknown')"
echo ""
echo "Users extract the zip, set SRC_ACCESS_TOKEN, and run:"
echo "  Windows: run.bat"
echo "  Mac/Linux: ./run.sh"
