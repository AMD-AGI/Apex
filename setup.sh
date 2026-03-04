#!/usr/bin/env bash
# setup.sh — Configure Apex for Claude Code CLI, Codex CLI, or both.
#
# Run once from the Apex project root:
#   cd Apex && bash setup.sh
#
# What it does:
#   1. Lets you choose: Claude Code, Codex, or both
#   2. Verifies prerequisites (CLI tools, python venv, Magpie)
#   3. Optionally clones ROCm repos + downloads docs (for source-finder & RAG)
#   4. Installs MCP Python dependencies
#   5. Registers all 5 MCP servers with selected CLI(s)
#   6. Installs 13 domain skills into selected CLI(s)
#   7. Creates the results directory
#   8. Prints usage summary per CLI

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

APEX_ROOT="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$APEX_ROOT/tools"
MCPS_DIR="$TOOLS_DIR/mcps"
SKILLS_DIR="$TOOLS_DIR/skills"
ROCM_DIR="$TOOLS_DIR/rocm"
DOC_DIR="$TOOLS_DIR/doc"
JSONS_DIR="$TOOLS_DIR/jsons"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"

# ── Colors ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${CYAN}→${NC} $1"; }

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Apex — CLI Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 1. CLI Selection & Path Configuration
# ═════════════════════════════════════════════════════════════════════════════

echo -e "${BOLD}Which CLI(s) do you want to configure?${NC}"
echo ""
echo "  1) Claude Code only"
echo "  2) Codex only"
echo "  3) Both Claude Code and Codex"
echo ""
read -rp "  Select [1/2/3]: " cli_choice
echo ""

INSTALL_CLAUDE=false
INSTALL_CODEX=false

case "$cli_choice" in
    1) INSTALL_CLAUDE=true ;;
    2) INSTALL_CODEX=true ;;
    3) INSTALL_CLAUDE=true; INSTALL_CODEX=true ;;
    *)
        fail "Invalid choice. Run again and select 1, 2, or 3."
        exit 1
        ;;
esac

# ── Configurable paths ───────────────────────────────────────────────────────

echo -e "${BOLD}Configure paths${NC}"
echo ""

# Python venv — use existing or create a local one
default_venv="${VIRTUAL_ENV:-}"
if [[ -n "$default_venv" ]] && [[ -x "$default_venv/bin/python3" ]]; then
    info "Active venv detected: $default_venv"
    read -rp "  Venv path (Enter to use detected, or type a different path): " user_venv
else
    read -rp "  Venv path (Enter to create $APEX_ROOT/.venv, or type a path): " user_venv
fi
VENV="${user_venv:-$default_venv}"

if [[ -n "$VENV" ]] && [[ -x "$VENV/bin/python3" ]]; then
    PYTHON="$VENV/bin/python3"
    ok "Using venv: $VENV"
elif [[ -n "$VENV" ]]; then
    warn "No python3 at $VENV/bin/python3 — will create a local venv instead"
    VENV=""
fi

if [[ -z "$VENV" ]]; then
    VENV="$APEX_ROOT/.venv"
    if [[ -x "$VENV/bin/python3" ]]; then
        ok "Local venv already exists: $VENV"
    else
        info "Creating venv at $VENV ..."
        python3 -m venv "$VENV"
        ok "Created venv: $VENV"
    fi
    PYTHON="$VENV/bin/python3"
fi

# Results directory — user-provided or timestamped
default_results="${RESULTS_DIR:-}"
read -rp "  Results directory path? (leave empty for timestamped): " user_results
if [[ -n "$user_results" ]]; then
    RESULTS_DIR="$user_results"
elif [[ -n "$default_results" ]]; then
    RESULTS_DIR="$default_results"
else
    RESULTS_DIR="$APEX_ROOT/results_$(date +%Y%m%d_%H%M%S)"
fi

# Magpie — derived from the magpie MCP clone location (tools/magpie)
MAGPIE_ROOT="$TOOLS_DIR/magpie"

# Export all paths so child processes and MCP servers can see them
export VENV PYTHON MAGPIE_ROOT RESULTS_DIR
export APEX_ROOT TOOLS_DIR MCPS_DIR SKILLS_DIR
export ROCM_DIR DOC_DIR JSONS_DIR

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 2. Prerequisites
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Checking prerequisites..."

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    if command -v claude &>/dev/null; then
        ok "Claude Code CLI: $(claude --version 2>&1 | head -1)"
    else
        fail "Claude Code CLI not found"
        info "Install: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    if command -v codex &>/dev/null; then
        ok "Codex CLI: $(codex --version 2>&1 | head -1)"
    else
        fail "Codex CLI not found"
        info "Install: npm install -g @openai/codex"
        exit 1
    fi
fi

ok "Python: $($PYTHON --version 2>&1) ($PYTHON)"

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 3. Data Dependencies (ROCm repos + docs)
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Checking data dependencies for source-finder & RAG..."

source "$MCPS_DIR/_shared.sh"

if [[ -d "$ROCM_DIR" ]] && [[ -n "$(ls -A "$ROCM_DIR" 2>/dev/null)" ]]; then
    repo_count=$(find "$ROCM_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
    ok "ROCm repos: $ROCM_DIR ($repo_count repos)"
else
    echo ""
    echo -e "  ${YELLOW}ROCm repos are required for source-finder and RAG indexing.${NC}"
    read -rp "  Clone ROCm repos into tools/rocm/? [y/N]: " clone_choice
    if [[ "$clone_choice" =~ ^[Yy]$ ]]; then
        clone_rocm_repos
    else
        warn "Skipped — source-finder and RAG will have limited functionality"
    fi
fi

if [[ -d "$DOC_DIR" ]] && [[ -n "$(ls -A "$DOC_DIR" 2>/dev/null)" ]]; then
    doc_count=$(find "$DOC_DIR" -maxdepth 1 -name '*.pdf' | wc -l)
    ok "Documentation PDFs: $DOC_DIR ($doc_count files)"
else
    echo ""
    echo -e "  ${YELLOW}AMD/ROCm documentation PDFs are used by the RAG server.${NC}"
    read -rp "  Download documentation PDFs into tools/doc/? [y/N]: " docs_choice
    if [[ "$docs_choice" =~ ^[Yy]$ ]]; then
        download_docs
    else
        warn "Skipped — RAG will have limited documentation coverage"
    fi
fi

if [[ -d "$JSONS_DIR" ]]; then
    ok "Optimization snippets: $JSONS_DIR"
else
    warn "JSON snippets not found at $JSONS_DIR"
fi

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 4. Install MCP Python Dependencies
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Installing MCP Python dependencies..."

# Install pip into the venv if missing
$PYTHON -m ensurepip --upgrade --default-pip 2>/dev/null || true

for mcp_dir in "$MCPS_DIR"/*/; do
    [[ ! -d "$mcp_dir" ]] && continue
    mcp_name="$(basename "$mcp_dir")"
    if [[ -f "$mcp_dir/pyproject.toml" ]]; then
        pip_log="$(mktemp)"
        if $PYTHON -m pip install -e "$mcp_dir" > "$pip_log" 2>&1; then
            ok "$mcp_name"
        else
            warn "$mcp_name — pip install failed (retrying without --quiet)..."
            if $PYTHON -m pip install -e "$mcp_dir" 2>&1 | tail -20; then
                ok "$mcp_name (retry succeeded)"
            else
                fail "$mcp_name — pip install failed"
                echo "    pip output (last 10 lines):"
                tail -10 "$pip_log" | while IFS= read -r line; do echo "      $line"; done
            fi
        fi
        rm -f "$pip_log"
    fi
done

# Clone + install Magpie via its own setup.sh (clones into tools/magpie)
echo ""
echo "▸ Setting up Magpie..."
if [[ -x "$MCPS_DIR/magpie/setup.sh" ]]; then
    bash "$MCPS_DIR/magpie/setup.sh" 2>&1 | while IFS= read -r line; do echo "  $line"; done
    if [[ -d "$MAGPIE_ROOT" ]]; then
        ok "Magpie: $MAGPIE_ROOT"
    else
        warn "Magpie clone may have failed — check $MAGPIE_ROOT"
    fi
else
    warn "Magpie setup script not found at $MCPS_DIR/magpie/setup.sh"
fi

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 5. Register MCP Servers
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Registering MCP servers..."

# register_mcp NAME [KEY=VAL ...] -- COMMAND [ARGS...]
#
# Registers a stdio MCP server with whichever CLI(s) the user selected.
# Environment variables go before the "--" separator; the command goes after.
register_mcp() {
    local name="$1"; shift

    local env_pairs=()
    local cmd_parts=()
    local past_separator=false

    for arg in "$@"; do
        if [[ "$arg" == "--" ]]; then
            past_separator=true
            continue
        fi
        if $past_separator; then
            cmd_parts+=("$arg")
        else
            env_pairs+=("$arg")
        fi
    done

    if [[ "$INSTALL_CLAUDE" == "true" ]]; then
        local claude_env_opts=()
        for ev in "${env_pairs[@]+"${env_pairs[@]}"}"; do
            claude_env_opts+=(-e "$ev")
        done
        claude mcp remove -s project "$name" 2>/dev/null || true
        if claude mcp add -s project --transport stdio "$name" "${claude_env_opts[@]+"${claude_env_opts[@]}"}" -- "${cmd_parts[@]}" 2>/dev/null; then
            ok "$name (claude)"
        else
            warn "$name (claude) — registration failed"
        fi
    fi

    if [[ "$INSTALL_CODEX" == "true" ]]; then
        local codex_opts=()
        for ev in "${env_pairs[@]+"${env_pairs[@]}"}"; do
            codex_opts+=(--env "$ev")
        done
        codex mcp remove "$name" 2>/dev/null || true
        if codex mcp add "$name" "${codex_opts[@]+"${codex_opts[@]}"}" -- "${cmd_parts[@]}" 2>/dev/null; then
            ok "$name (codex)"
        else
            warn "$name (codex) — registration failed"
        fi
    fi
}

cd "$APEX_ROOT"

# ── source-finder (tools/mcps/source_finder) ────────────────────────────────
# Reads ROCm repos from tools/rocm/ (path derived from server.py location)
register_mcp "source-finder" \
    -- "$PYTHON" "$MCPS_DIR/source_finder/server.py"

# ── kernel-rag (tools/mcps/rag_tool) ────────────────────────────────────────
# Env vars override default path resolution so it always finds the data
register_mcp "kernel-rag" \
    "MCP_ROCM_DIR=$ROCM_DIR" \
    "MCP_DOC_DIR=$DOC_DIR" \
    "MCP_JSONS_DIR=$JSONS_DIR" \
    -- "$PYTHON" "$MCPS_DIR/rag_tool/server.py"

# ── gpu-info (tools/mcps/gpu_info) ──────────────────────────────────────────
register_mcp "gpu-info" \
    -- "$PYTHON" "$MCPS_DIR/gpu_info/server.py"

# ── fusion-advisor (tools/mcps/fusion_advisor) ──────────────────────────────
register_mcp "fusion-advisor" \
    -- "$PYTHON" "$MCPS_DIR/fusion_advisor/server.py"

# ── magpie ──────────────────────────────────────────────────────────────────
register_mcp "magpie" \
    "PYTHONPATH=$MAGPIE_ROOT" \
    -- "$PYTHON" "-m" "Magpie.mcp"

# ── Generate .mcp.json (for Claude Code IDE / Cursor) ──────────────────────
cat > "$APEX_ROOT/.mcp.json" <<MCPJSON
{
  "mcpServers": {
    "source-finder": {
      "type": "stdio",
      "command": "$PYTHON",
      "args": ["$MCPS_DIR/source_finder/server.py"],
      "env": {}
    },
    "kernel-rag": {
      "type": "stdio",
      "command": "$PYTHON",
      "args": ["$MCPS_DIR/rag_tool/server.py"],
      "env": {
        "MCP_ROCM_DIR": "$ROCM_DIR",
        "MCP_DOC_DIR": "$DOC_DIR",
        "MCP_JSONS_DIR": "$JSONS_DIR"
      }
    },
    "gpu-info": {
      "type": "stdio",
      "command": "$PYTHON",
      "args": ["$MCPS_DIR/gpu_info/server.py"],
      "env": {}
    },
    "fusion-advisor": {
      "type": "stdio",
      "command": "$PYTHON",
      "args": ["$MCPS_DIR/fusion_advisor/server.py"],
      "env": {}
    },
    "magpie": {
      "type": "stdio",
      "command": "$PYTHON",
      "args": ["-m", "Magpie.mcp"],
      "env": {
        "PYTHONPATH": "$MAGPIE_ROOT"
      }
    }
  }
}
MCPJSON
ok "Generated .mcp.json (auto-resolved paths for IDE)"

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 6. Install Skills
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Installing skills..."

SKILL_NAMES=()
for skill_dir in "$SKILLS_DIR"/*/; do
    [[ -d "$skill_dir" ]] && SKILL_NAMES+=("$(basename "$skill_dir")")
done

# ── Claude Code ─────────────────────────────────────────────────────────────
# Skills are referenced in CLAUDE.md, which Claude Code auto-loads from the
# project root. The agent reads SKILL.md files on demand — no CLI registration
# needed. We just verify the project context file exists.
if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    if [[ -f "$APEX_ROOT/CLAUDE.md" ]]; then
        ok "Claude Code: CLAUDE.md found — ${#SKILL_NAMES[@]} skills auto-discoverable from tools/skills/"
    else
        warn "Claude Code: CLAUDE.md not found — skills won't be auto-discoverable"
        info "Create CLAUDE.md with skill paths (see README)"
    fi
fi

# ── Codex ───────────────────────────────────────────────────────────────────
# Codex discovers skills from $CODEX_HOME/skills/. We symlink each skill
# directory so Codex can find and read them. Codex reads AGENTS.md for
# project-level context (equivalent to CLAUDE.md).
if [[ "$INSTALL_CODEX" == "true" ]]; then
    CODEX_SKILLS="$CODEX_HOME/skills"
    mkdir -p "$CODEX_SKILLS"

    linked=0
    for skill_name in "${SKILL_NAMES[@]}"; do
        target="$CODEX_SKILLS/$skill_name"
        source_dir="$SKILLS_DIR/$skill_name"

        # Remove stale symlink
        if [[ -L "$target" ]]; then
            rm -f "$target"
        fi

        # Don't clobber a real directory the user created
        if [[ -d "$target" ]] && [[ ! -L "$target" ]]; then
            warn "$skill_name — real directory at $target, skipping"
            continue
        fi

        ln -s "$source_dir" "$target" && linked=$((linked + 1))
    done
    ok "Codex: $linked skills symlinked into $CODEX_SKILLS/"

    # Create AGENTS.md from CLAUDE.md if it doesn't already exist
    if [[ -f "$APEX_ROOT/CLAUDE.md" ]] && [[ ! -f "$APEX_ROOT/AGENTS.md" ]]; then
        cp "$APEX_ROOT/CLAUDE.md" "$APEX_ROOT/AGENTS.md"
        ok "Codex: Created AGENTS.md from CLAUDE.md"
    elif [[ -f "$APEX_ROOT/AGENTS.md" ]]; then
        ok "Codex: AGENTS.md already exists"
    else
        warn "Codex: Neither CLAUDE.md nor AGENTS.md found"
    fi
fi

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 7. Create Results Directory
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Setting up results directory..."
mkdir -p "$RESULTS_DIR"
ok "Results dir: $RESULTS_DIR"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 8. Verify & Summary
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Verifying MCP registration..."

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    echo -e "  ${CYAN}Claude Code:${NC}"
    claude mcp list 2>&1 | grep -E "^\s" | head -20 || true
    echo ""
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    echo -e "  ${CYAN}Codex:${NC}"
    codex mcp list 2>&1 | head -20 || true
    echo ""
fi

# Build summary line
installed_clis=""
[[ "$INSTALL_CLAUDE" == "true" ]] && installed_clis="Claude Code"
[[ "$INSTALL_CODEX" == "true" ]] && {
    [[ -n "$installed_clis" ]] && installed_clis+=" + "
    installed_clis+="Codex"
}

echo "═══════════════════════════════════════════════════════"
echo -e " ${GREEN}Setup complete!${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " Configured for: $installed_clis"
echo ""
echo " MCPs registered (5):"
echo "   source-finder   — kernel source search      (tools/mcps/source_finder)"
echo "   kernel-rag      — optimization RAG           (tools/mcps/rag_tool)"
echo "   gpu-info        — MI355X / CDNA4 specs       (tools/mcps/gpu_info)"
echo "   fusion-advisor  — kernel fusion detection    (tools/mcps/fusion_advisor)"
echo "   magpie          — kernel eval & benchmarks   (Magpie framework)"
echo ""
echo " Skills: ${#SKILL_NAMES[@]} (in tools/skills/)"
echo ""
echo " Paths (exported as env vars):"
echo "   VENV             $VENV"
echo "   PYTHON           $PYTHON"
echo "   MAGPIE_ROOT      $MAGPIE_ROOT"
echo "   RESULTS_DIR      $RESULTS_DIR"
echo "   ROCM_DIR         $ROCM_DIR"
echo "   DOC_DIR          $DOC_DIR"
echo "   JSONS_DIR        $JSONS_DIR"
echo ""
echo " ── How to use ──────────────────────────────────────"
echo ""

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    echo " Claude Code (interactive):"
    echo "   cd $APEX_ROOT && claude"
    echo ""
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    echo " Codex (interactive):"
    echo "   cd $APEX_ROOT && codex"
    echo ""
fi

echo " Automated pipeline (no interactive agent):"
echo "   source $VENV/bin/activate"
echo "   export MAGPIE_ROOT=$MAGPIE_ROOT"
echo "   export RESULTS_DIR=$RESULTS_DIR"
echo "   python3 workload_optimizer.py run \\"
echo "     -r \$RESULTS_DIR \\"
echo "     -b \$MAGPIE_ROOT/examples/benchmark_vllm_gptoss_120b.yaml \\"
echo "     --kernel-types triton --top-k 10 \\"
echo "     --max-iterations 3 --max-turns 25 --leaderboard"
echo ""
echo "═══════════════════════════════════════════════════════"
