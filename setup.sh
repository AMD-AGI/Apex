#!/usr/bin/env bash
# setup.sh — One-shot environment setup for the RL kernel-optimization sandbox.
#
# What this does:
#   1. Creates a Python virtual environment
#   2. Installs Python dependencies (test runner + prompt/grader libs)
#   3. Validates the environment (imports, script permissions)
#   4. Creates the output/ directory
#   5. Optionally runs files/setup_files.sh and tools/setup_tools.sh
#
# Usage:
#   bash setup.sh                   # full setup
#   bash setup.sh --skip-downloads  # skip cloning repos and downloading docs
#   bash setup.sh --skip-tools      # skip Magpie + RAG tool install
#   bash setup.sh --venv=.venv      # custom venv path (default: .venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SKIP_DOWNLOADS=false
SKIP_TOOLS=false

for arg in "$@"; do
    case "$arg" in
        --skip-downloads) SKIP_DOWNLOADS=true ;;
        --skip-tools)     SKIP_TOOLS=true ;;
        --venv=*)         VENV_DIR="${arg#*=}" ;;
        -h|--help)
            echo "Usage: bash setup.sh [--skip-downloads] [--skip-tools] [--venv=PATH]"
            exit 0 ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' required but not found"; exit 1; }; }

echo ""
echo "=== RL Kernel Optimization — Environment Setup ==="
echo ""

# ── pre-flight ────────────────────────────────────────────────────────────────
need python3
need git
need curl

PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required (found 3.$PYTHON_MINOR)"
    exit 1
fi

# ── 1. Python virtual environment ─────────────────────────────────────────────
echo "--- 1. Python virtual environment ($VENV_DIR) ---"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  created $VENV_DIR"
else
    echo "  exists: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python3 -m pip install --quiet --upgrade pip

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "--- 2. Python dependencies ---"

# Core for graders + prompts (always needed)
python3 -m pip install --quiet \
    anthropic \
    pyyaml \
    rich

# Test runner
python3 -m pip install --quiet \
    pytest \
    pytest-asyncio

echo "  installed: anthropic, pyyaml, rich, pytest"

# RAG tool deps (needed for eval + tools tests)
python3 -m pip install --quiet \
    chromadb \
    "sentence-transformers>=2.7" \
    mcp \
    tiktoken

echo "  installed: chromadb, sentence-transformers, mcp, tiktoken"

# ── 3. Create output/ directory ───────────────────────────────────────────────
echo ""
echo "--- 3. Output directory ---"
mkdir -p "$SCRIPT_DIR/output"
echo "  created/verified: output/"

# ── 4. files/setup_files.sh (repos + docs) ───────────────────────────────────
echo ""
if $SKIP_DOWNLOADS; then
    echo "--- 4. Skipping files/setup_files.sh (--skip-downloads) ---"
else
    echo "--- 4. Downloading code repos and documentation ---"
    bash "$SCRIPT_DIR/files/setup_files.sh"
fi

# ── 5. tools/setup_tools.sh (Magpie + RAG) ───────────────────────────────────
echo ""
if $SKIP_TOOLS; then
    echo "--- 5. Skipping tools/setup_tools.sh (--skip-tools) ---"
else
    echo "--- 5. Installing tools (Magpie + RAG tool) ---"
    bash "$SCRIPT_DIR/tools/setup_tools.sh"
fi

# ── 6. Environment validation ─────────────────────────────────────────────────
echo ""
echo "--- 6. Validating environment ---"

validate_import() {
    local module="$1"
    python3 -c "import $module" 2>/dev/null \
        && echo "  ✓ import $module" \
        || echo "  ✗ import $module FAILED"
}

validate_import anthropic
validate_import pytest
validate_import chromadb
validate_import yaml

# Validate graders are importable
python3 -c "
import sys; sys.path.insert(0, 'graders')
from score import total_score, KernelResult, ModelResult
from kernel_grader import find_tasks, grade_all
from model_grader import grade_all as model_grade_all
print('  ✓ graders module')
"

# Validate prompts are importable
python3 -c "
import sys; sys.path.insert(0, 'prompts')
from models import MODELS
from configs import CONFIGS
from kernel_prompt import all_prompts as kp
from model_prompt import all_prompts as mp
kc = len(list(kp()))
mc = len(list(mp()))
print(f'  ✓ prompts: {len(MODELS)} models, {len(CONFIGS)} configs → {kc} kernel tasks, {mc} model tasks')
"

# Check script permissions
for sh in files/setup_files.sh tools/setup_tools.sh; do
    if [ -x "$SCRIPT_DIR/$sh" ]; then
        echo "  ✓ executable: $sh"
    else
        echo "  ✗ NOT executable: $sh"
    fi
done

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "  Activate venv:  source $VENV_DIR/bin/activate"
echo "  Run tests:      pytest tests/ -v"
echo "  Run mini eval:  python3 eval.py"
echo ""
echo "  Kernel prompts: python3 prompts/kernel_prompt.py --list"
echo "  Model prompts:  python3 prompts/model_prompt.py  --list"
echo "  Grade output:   python3 graders/kernel_grader.py"
