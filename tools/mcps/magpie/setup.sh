#!/usr/bin/env bash
# setup.sh — Clone, install, and register the Magpie MCP server.
#
# Magpie is a GPU kernel correctness & performance evaluation framework.
# Repo: https://github.com/AMD-AGI/Magpie
#
# Usage:
#   ./setup.sh --claude      # register with Claude Code
#   ./setup.sh --cursor      # register with Cursor
#   ./setup.sh               # install deps only (no registration)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAGPIE_DIR="$TOOLS_DIR/magpie"
MCP_NAME="magpie"

TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --claude) TARGET="claude"; shift ;;
        --cursor) TARGET="cursor"; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--claude|--cursor]"; exit 1 ;;
    esac
done

# ── Clone Magpie ─────────────────────────────────────────────────────────────

echo "=== Magpie — GPU kernel evaluation framework ==="

if [[ -d "$MAGPIE_DIR/.git" ]]; then
    echo "  [skip] $MAGPIE_DIR already exists"
else
    echo "  cloning AMD-AGI/Magpie..."
    if git clone --depth=1 git@github.com:AMD-AGI/Magpie.git "$MAGPIE_DIR" 2>/dev/null; then
        echo "  [ok] cloned via SSH"
    elif git clone --depth=1 https://github.com/AMD-AGI/Magpie.git "$MAGPIE_DIR" 2>/dev/null; then
        echo "  [ok] cloned via HTTPS"
    else
        REPO_ROOT="$(cd "$TOOLS_DIR/.." && pwd)"
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

# ── Install ──────────────────────────────────────────────────────────────────

echo "  installing Magpie..."
python3 -m pip install --quiet -e "$MAGPIE_DIR[mcp]"
echo "  [ok] installed"

# ── Register with target IDE ─────────────────────────────────────────────────

if [[ "$TARGET" == "claude" ]]; then
    echo "=== Registering with Claude Code ==="
    command -v claude >/dev/null 2>&1 || { echo "ERROR: 'claude' CLI not found"; exit 1; }
    claude mcp add --transport stdio "$MCP_NAME" -- python3 -m Magpie.mcp
    echo "  [ok] registered: $MCP_NAME"

elif [[ "$TARGET" == "cursor" ]]; then
    echo "=== Registering with Cursor ==="
    CURSOR_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo "$TOOLS_DIR/..")/.cursor"
    MCP_JSON="$CURSOR_DIR/mcp.json"
    mkdir -p "$CURSOR_DIR"

    if [[ -f "$MCP_JSON" ]]; then
        python3 -c "
import json, sys
with open('$MCP_JSON') as f:
    cfg = json.load(f)
cfg.setdefault('mcpServers', {})['$MCP_NAME'] = {
    'command': 'python3',
    'args': ['-m', 'Magpie.mcp']
}
with open('$MCP_JSON', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    else
        python3 -c "
import json
cfg = {'mcpServers': {'$MCP_NAME': {'command': 'python3', 'args': ['-m', 'magpie.mcp']}}}
with open('$MCP_JSON', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    fi
    echo "  [ok] registered in $MCP_JSON"

else
    echo ""
    echo "Magpie installed. To register, re-run with --claude or --cursor."
fi

echo ""
echo "=== Done ==="
