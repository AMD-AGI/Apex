#!/usr/bin/env bash
# setup.sh — Configure Claude Code CLI for the Apex workload optimization pipeline.
#
# Run once from the Apex project root:
#   cd /home/sirafati/code_combine/Apex && bash setup.sh
#
# What it does:
#   1. Verifies prerequisites (claude CLI, python venv, Magpie)
#   2. Registers all 7 MCP servers with claude (project-scoped)
#   3. Creates the results directory
#   4. Prints the one-liner to start the agent
#
# After running setup.sh, just:
#   cd /home/sirafati/code_combine/Apex && claude
#   Then paste your task (e.g., the GPT OSS 120B workflow prompt below)

set -euo pipefail

APEX_ROOT="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-/home/sirafati/results_total_agent}"
VENV="/home/sirafati/Kernel/.venv"
MAGPIE_ROOT="${MAGPIE_ROOT:-/home/sirafati/code_combine/Magpie}"
AGIKIT_MCP="/home/sirafati/Kernel/AGIKIT-V2/mcp_tools/src"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "═══════════════════════════════════════════════════════"
echo " Apex — Claude Code CLI Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── 1. Prerequisites ─────────────────────────────────────────────────────────

echo "▸ Checking prerequisites..."

if command -v claude &>/dev/null; then
    ok "Claude Code CLI: $(claude --version 2>&1 | head -1)"
else
    fail "Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

if [ -d "$VENV" ]; then
    ok "Python venv: $VENV"
else
    warn "Python venv not found at $VENV — some pipeline steps may fail"
fi

if [ -d "$MAGPIE_ROOT" ]; then
    ok "Magpie: $MAGPIE_ROOT"
else
    warn "Magpie not found at $MAGPIE_ROOT — benchmark steps will fail"
fi

if [ -f "$APEX_ROOT/mcp_config.json" ]; then
    ok "mcp_config.json found"
else
    warn "mcp_config.json not found — MCPs will not be available to workload_optimizer.py"
fi

echo ""

# ── 2. Register MCP servers (project-scoped) ─────────────────────────────────

echo "▸ Registering MCP servers with Claude Code (project scope)..."

cd "$APEX_ROOT"

register_mcp() {
    local name="$1"
    local json="$2"
    # Remove existing, ignore errors
    claude mcp remove -s project "$name" 2>/dev/null || true
    if claude mcp add-json -s project "$name" "$json" 2>/dev/null; then
        ok "$name"
    else
        warn "$name — failed to register (check paths)"
    fi
}

register_mcp "magpie" "{
  \"command\": \"$VENV/bin/python\",
  \"args\": [\"-m\", \"Magpie.mcp\"],
  \"env\": { \"PYTHONPATH\": \"$MAGPIE_ROOT\" }
}"

register_mcp "gpu-info" "{
  \"command\": \"$AGIKIT_MCP/mcp_gpu_info/run_server.sh\",
  \"args\": []
}"

register_mcp "kernel-perf" "{
  \"command\": \"$AGIKIT_MCP/mcp_kernel_perf/run_server.sh\",
  \"args\": []
}"

register_mcp "source-finder" "{
  \"command\": \"$AGIKIT_MCP/mcp_source_finder/run_server.sh\",
  \"args\": []
}"

register_mcp "asm-tools" "{
  \"command\": \"$AGIKIT_MCP/mcp_asm_tools/run_server.sh\",
  \"args\": []
}"

register_mcp "fusion-advisor" "{
  \"command\": \"$AGIKIT_MCP/mcp_fusion_advisor/run_server.sh\",
  \"args\": []
}"

register_mcp "rag-server" "{
  \"command\": \"$AGIKIT_MCP/mcp_rag_server/run_server.sh\",
  \"args\": []
}"

echo ""

# ── 3. Create results directory ──────────────────────────────────────────────

echo "▸ Setting up results directory..."
mkdir -p "$RESULTS_DIR"
ok "Results dir: $RESULTS_DIR"
echo ""

# ── 4. Verify MCP registration ──────────────────────────────────────────────

echo "▸ Verifying MCP registration..."
claude mcp list 2>&1 | grep -E "^[a-z]" || true
echo ""

# ── 5. Summary ───────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════"
echo -e " ${GREEN}Setup complete!${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " MCPs registered:  7 (magpie, gpu-info, kernel-perf, source-finder,"
echo "                       asm-tools, fusion-advisor, rag-server)"
echo " Skills available: 13 (in tools/skills/)"
echo " Project context:  CLAUDE.md (auto-loaded by Claude Code)"
echo " Results dir:      $RESULTS_DIR"
echo ""
echo " ── How to use ──────────────────────────────────────"
echo ""
echo " Option 1: Interactive Claude Code session"
echo "   cd $APEX_ROOT && claude"
echo "   Then paste your task prompt (see CLAUDE.md for examples)"
echo ""
echo " Option 2: One-shot CLI"
cat << EXAMPLE
   cd $APEX_ROOT && claude -p "Focus on GPT OSS 120B on vLLM.
   Benchmark it E2E with Magpie, report top 10 triton kernels,
   optimize them, and save all results to $RESULTS_DIR.
   Use all available MCP tools and read relevant skills from
   tools/skills/ before starting. Run the full pipeline via
   workload_optimizer.py."
EXAMPLE
echo ""
echo " Option 3: Automated pipeline (no interactive agent)"
echo "   source $VENV/bin/activate"
echo "   export MAGPIE_ROOT=$MAGPIE_ROOT"
echo "   python3 workload_optimizer.py run \\"
echo "     -r $RESULTS_DIR \\"
echo "     -b \$MAGPIE_ROOT/examples/benchmark_vllm_gptoss_120b.yaml \\"
echo "     --kernel-types triton --top-k 10 \\"
echo "     --max-iterations 3 --max-turns 25 --leaderboard"
echo ""
echo "═══════════════════════════════════════════════════════"
