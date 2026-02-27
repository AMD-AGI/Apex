#!/usr/bin/env bash
# setup_tools.sh
# Downloads and installs tools/MCPs/skills for the RL kernel-optimization sandbox.
#
# Usage:
#   ./setup_tools.sh --claude      # install everything + register with Claude Code
#   ./setup_tools.sh --cursor      # install everything + register with Cursor
#   ./setup_tools.sh               # install dependencies only (no registration)
#
# Options:
#   --skip-repos       Skip cloning ROCm repos
#   --skip-docs        Skip downloading documentation PDFs
#
# Tools installed:
#   1. Magpie          — GPU kernel correctness & performance evaluation
#   2. RAG tool        — multi-retriever RAG for kernel optimization docs/code/snippets
#   3. Fusion Advisor  — kernel fusion opportunity detection and code generation
#   4. GPU Info        — GPU architecture detection and optimization hints
#   5. Source Finder   — kernel source lookup from trace signatures
#
# Shared data (cloned/downloaded once, used by rag_tool + source_finder):
#   tools/rocm/   — ROCm library repos
#   tools/doc/    — AMD/ROCm documentation PDFs
#   tools/jsons/  — optimization snippet sheets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MCPS_DIR="$SCRIPT_DIR/mcps"

# Load shared helpers (clone_rocm_repos, download_docs)
source "$MCPS_DIR/_shared.sh"

# ── parse arguments ──────────────────────────────────────────────────────────

TARGET=""
TARGET_FLAG=""
SKIP_REPOS=false
SKIP_DOCS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --claude) TARGET="claude"; TARGET_FLAG="--claude"; shift ;;
        --cursor) TARGET="cursor"; TARGET_FLAG="--cursor"; shift ;;
        --skip-repos) SKIP_REPOS=true; shift ;;
        --skip-docs)  SKIP_DOCS=true; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--claude|--cursor] [--skip-repos] [--skip-docs]"; exit 1 ;;
    esac
done

# ── helpers ──────────────────────────────────────────────────────────────────

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found in PATH"; exit 1; }
}

pip_install() {
    python3 -m pip install --quiet "$@"
}

register_mcp() {
    local name="$1"
    local cmd="$2"
    shift 2
    local args=("$@")

    if [[ "$TARGET" == "claude" ]]; then
        claude mcp add --transport stdio "$name" -- $cmd "${args[@]}"
        echo "  [ok] registered: $name (Claude Code)"

    elif [[ "$TARGET" == "cursor" ]]; then
        local cursor_dir="$REPO_ROOT/.cursor"
        local mcp_json="$cursor_dir/mcp.json"
        mkdir -p "$cursor_dir"

        local args_json
        args_json=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" "${args[@]}")

        python3 -c "
import json, os
path = '$mcp_json'
cfg = {}
if os.path.exists(path):
    with open(path) as f:
        cfg = json.load(f)
cfg.setdefault('mcpServers', {})['$name'] = {
    'command': '$cmd',
    'args': $args_json
}
with open(path, 'w') as f:
    json.dump(cfg, f, indent=2)
"
        echo "  [ok] registered: $name (Cursor)"
    fi
}

# ── pre-flight ───────────────────────────────────────────────────────────────

need git
need python3
need pip3

if [[ "$TARGET" == "claude" ]]; then
    need claude
fi

echo ""
echo "================================================================"
echo "  Keystone Tools Setup"
if [[ -n "$TARGET" ]]; then
    echo "  Target: $TARGET"
else
    echo "  Target: deps only (pass --claude or --cursor to register MCPs)"
fi
echo "================================================================"

# ── 1. Clone ROCm repos (shared by rag_tool + source_finder) ────────────────

echo ""
if [[ "$SKIP_REPOS" == "false" ]]; then
    clone_rocm_repos
else
    echo "  [skip] ROCm repos (--skip-repos)"
fi

# ── 2. Download documentation PDFs (used by rag_tool) ───────────────────────

echo ""
if [[ "$SKIP_DOCS" == "false" ]]; then
    download_docs
else
    echo "  [skip] docs (--skip-docs)"
fi

# ── 3. Magpie ────────────────────────────────────────────────────────────────

echo ""
echo "=== 1/5. Magpie — GPU kernel evaluation framework ==="

MAGPIE_DIR="$SCRIPT_DIR/magpie"

if [[ -d "$MAGPIE_DIR/.git" ]]; then
    echo "  [skip] $MAGPIE_DIR already exists"
else
    echo "  cloning AMD-AGI/Magpie..."
    if git clone --depth=1 git@github.com:AMD-AGI/Magpie.git "$MAGPIE_DIR" 2>/dev/null; then
        echo "  [ok] cloned via SSH"
    elif git clone --depth=1 https://github.com/AMD-AGI/Magpie.git "$MAGPIE_DIR" 2>/dev/null; then
        echo "  [ok] cloned via HTTPS"
    else
        # Fallback: copy from a sibling checkout if available
        local_magpie="$(cd "$REPO_ROOT/.." && pwd)/Magpie"
        if [[ -d "$local_magpie/.git" ]]; then
            echo "  [info] SSH/HTTPS failed — copying local clone from $local_magpie"
            cp -a "$local_magpie" "$MAGPIE_DIR"
        else
            echo "  [ERROR] Could not clone Magpie. Provide access or place it at $MAGPIE_DIR"
            exit 1
        fi
    fi
fi

echo "  installing Magpie..."
pip_install -e "$MAGPIE_DIR[mcp]"
register_mcp "magpie" "python3" "-m" "Magpie.mcp"

# ── 4. RAG Tool ──────────────────────────────────────────────────────────────

echo ""
echo "=== 2/5. RAG Tool — kernel optimization knowledge base ==="

RAG_DIR="$MCPS_DIR/rag_tool"
pip_install -e "$RAG_DIR"
echo "  [ok] dependencies installed"
register_mcp "kernel-rag" "python3" "$RAG_DIR/server.py"

# ── 5. Fusion Advisor ────────────────────────────────────────────────────────

echo ""
echo "=== 3/5. Fusion Advisor — kernel fusion detection ==="

FUSION_DIR="$MCPS_DIR/fusion_advisor"
pip_install -e "$FUSION_DIR"
echo "  [ok] dependencies installed"
register_mcp "fusion-advisor" "python3" "$FUSION_DIR/server.py"

# ── 6. GPU Info ──────────────────────────────────────────────────────────────

echo ""
echo "=== 4/5. GPU Info — architecture detection & hints ==="

GPU_DIR="$MCPS_DIR/gpu_info"
pip_install -e "$GPU_DIR"
echo "  [ok] dependencies installed"
register_mcp "gpu-info" "python3" "$GPU_DIR/server.py"

# ── 7. Source Finder ─────────────────────────────────────────────────────────

echo ""
echo "=== 5/5. Source Finder — kernel source lookup ==="

SRC_DIR="$MCPS_DIR/source_finder"
pip_install -e "$SRC_DIR"
echo "  [ok] dependencies installed"
register_mcp "source-finder" "python3" "$SRC_DIR/server.py"

# ── 8. Skills ─────────────────────────────────────────────────────────────────

SKILLS_SRC="$SCRIPT_DIR/skills"
SKILLS_COUNT=$(find "$SKILLS_SRC" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')

echo ""
echo "=== Skills — $SKILLS_COUNT kernel-optimization skills ==="

if [[ -n "$TARGET" ]]; then
    if [[ "$TARGET" == "claude" ]]; then
        SKILLS_DST="$REPO_ROOT/.claude/skills"
    elif [[ "$TARGET" == "cursor" ]]; then
        SKILLS_DST="$REPO_ROOT/.cursor/skills"
    fi

    mkdir -p "$SKILLS_DST"
    rsync -a --delete "$SKILLS_SRC/" "$SKILLS_DST/"
    echo "  [ok] synced $SKILLS_COUNT skills → $SKILLS_DST"
else
    echo "  [skip] pass --claude or --cursor to sync skills to IDE"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "  All tools installed successfully!"
echo "================================================================"
echo ""
echo "MCP servers installed:"
echo "  1. magpie          — GPU kernel evaluation"
echo "  2. kernel-rag      — optimization knowledge search"
echo "  3. fusion-advisor  — kernel fusion opportunities"
echo "  4. gpu-info        — GPU arch detection & hints"
echo "  5. source-finder   — kernel source lookup"
echo ""
echo "Skills installed ($SKILLS_COUNT):"
for skill_dir in "$SKILLS_SRC"/*/; do
    echo "  - $(basename "$skill_dir")"
done
echo ""
echo "Shared data:"
echo "  rocm repos:  $ROCM_DIR"
echo "  docs (PDFs): $DOC_DIR"
echo "  json sheets: $JSONS_DIR"
echo ""

if [[ -z "$TARGET" ]]; then
    echo "To register MCPs and sync skills with your IDE, re-run with:"
    echo "  $0 --claude    # for Claude Code"
    echo "  $0 --cursor    # for Cursor"
    echo ""
fi

echo "Next steps:"
echo "  1. (Optional) Build RAG index: python3 $RAG_DIR/index.py"
