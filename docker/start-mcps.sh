#!/bin/bash
# start-mcps.sh — Launch all MCP servers as background processes.
# Used as the entrypoint for the MCP sidecar container.

set -e

echo "[mcp-sidecar] Starting MCP servers..."

MCP_DIRS=(
    "/workspace/tools/mcps/fusion_advisor"
    "/workspace/tools/mcps/gpu_info"
    "/workspace/tools/mcps/rag_tool"
    "/workspace/tools/mcps/source_finder"
)

PIDS=()

for mcp_dir in "${MCP_DIRS[@]}"; do
    if [ -f "$mcp_dir/server.py" ]; then
        name=$(basename "$mcp_dir")
        echo "[mcp-sidecar]   Starting $name..."
        python "$mcp_dir/server.py" &
        PIDS+=($!)
    fi
done

# Magpie MCP (special — uses python -m)
if python -c "import Magpie" 2>/dev/null; then
    echo "[mcp-sidecar]   Starting magpie..."
    python -m Magpie.mcp &
    PIDS+=($!)
fi

echo "[mcp-sidecar] All MCP servers started (${#PIDS[@]} processes)"

# Wait for any child to exit
wait -n "${PIDS[@]}" 2>/dev/null || true
echo "[mcp-sidecar] A server exited, shutting down..."
kill "${PIDS[@]}" 2>/dev/null || true
