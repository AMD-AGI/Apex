#!/usr/bin/env bash
# setup.sh — Install and register the Fusion Advisor MCP server.
#
# Usage:
#   ./setup.sh --claude      # register with Claude Code
#   ./setup.sh --cursor      # register with Cursor
#   ./setup.sh               # install deps only (no registration)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_NAME="fusion-advisor"
SERVER_SCRIPT="$SCRIPT_DIR/server.py"

TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --claude) TARGET="claude"; shift ;;
        --cursor) TARGET="cursor"; shift ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--claude|--cursor]"; exit 1 ;;
    esac
done

# ── Install dependencies ─────────────────────────────────────────────────────

echo "=== Fusion Advisor MCP — installing dependencies ==="
python3 -m pip install --quiet -e "$SCRIPT_DIR"
echo "  [ok] dependencies installed"

# ── Register with target IDE ─────────────────────────────────────────────────

if [[ "$TARGET" == "claude" ]]; then
    echo "=== Registering with Claude Code ==="
    command -v claude >/dev/null 2>&1 || { echo "ERROR: 'claude' CLI not found"; exit 1; }
    claude mcp add --transport stdio "$MCP_NAME" -- python3 "$SERVER_SCRIPT"
    echo "  [ok] registered: $MCP_NAME"

elif [[ "$TARGET" == "cursor" ]]; then
    echo "=== Registering with Cursor ==="
    CURSOR_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../../..")/.cursor"
    MCP_JSON="$CURSOR_DIR/mcp.json"
    mkdir -p "$CURSOR_DIR"

    if [[ -f "$MCP_JSON" ]]; then
        python3 -c "
import json, sys
with open('$MCP_JSON') as f:
    cfg = json.load(f)
cfg.setdefault('mcpServers', {})['$MCP_NAME'] = {
    'command': 'python3',
    'args': ['$SERVER_SCRIPT']
}
with open('$MCP_JSON', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    else
        python3 -c "
import json
cfg = {'mcpServers': {'$MCP_NAME': {'command': 'python3', 'args': ['$SERVER_SCRIPT']}}}
with open('$MCP_JSON', 'w') as f:
    json.dump(cfg, f, indent=2)
"
    fi
    echo "  [ok] registered in $MCP_JSON"

else
    echo ""
    echo "Dependencies installed. To register, re-run with --claude or --cursor."
fi

echo ""
echo "=== Done ==="
