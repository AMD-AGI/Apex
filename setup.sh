#!/usr/bin/env bash
# setup.sh — One-shot setup for Apex GPU kernel optimization pipeline.
#
# Run once from the Apex project root:
#   cd Apex && bash setup.sh
#
# What it does:
#   1. Parse CLI flags (--skip-downloads, --skip-tools, --venv=PATH, --non-interactive)
#   2. Let you choose: Claude Code, Codex, Cursor Agent, or any combination
#   3. Create/reuse a Python venv and install core dependencies + PyTorch for ROCm
#   4. Optionally clone ROCm repos + download architecture docs (for source-finder & RAG)
#   5. Install MCP Python dependencies + Magpie
#   6. Register all 5 MCP servers with selected CLI(s)
#   7. Install 13 domain skills into selected CLI(s)
#   8. Print usage summary

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

# ── CLI Flags ────────────────────────────────────────────────────────────────

SKIP_DOWNLOADS=false
SKIP_TOOLS=false
NON_INTERACTIVE=false
USER_VENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-downloads) SKIP_DOWNLOADS=true; shift ;;
        --skip-tools)     SKIP_TOOLS=true; shift ;;
        --non-interactive) NON_INTERACTIVE=true; shift ;;
        --venv=*)         USER_VENV="${1#--venv=}"; shift ;;
        --venv)           USER_VENV="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv=PATH         Use or create venv at PATH (default: .venv)"
            echo "  --skip-downloads    Skip cloning ROCm repos and downloading docs"
            echo "  --skip-tools        Skip MCP + Magpie installation"
            echo "  --non-interactive   Accept all defaults (no prompts)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            fail "Unknown option: $1"
            echo "  Run: bash setup.sh --help"
            exit 1
            ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Apex — Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 1. CLI Selection
# ═════════════════════════════════════════════════════════════════════════════

INSTALL_CLAUDE=false
INSTALL_CODEX=false
INSTALL_CURSOR=false

if [[ "$NON_INTERACTIVE" == "true" ]]; then
    # Auto-detect installed CLIs
    command -v claude &>/dev/null && INSTALL_CLAUDE=true
    command -v codex &>/dev/null && INSTALL_CODEX=true
    (command -v cursor-agent &>/dev/null || command -v cursor &>/dev/null) && INSTALL_CURSOR=true
    if [[ "$INSTALL_CLAUDE" == "false" && "$INSTALL_CODEX" == "false" && "$INSTALL_CURSOR" == "false" ]]; then
        warn "No agent CLI detected — will set up environment only"
    fi
else
    echo -e "${BOLD}Which CLI(s) do you want to configure?${NC}"
    echo ""
    echo "  1) Claude Code only"
    echo "  2) Codex only"
    echo "  3) Cursor Agent only"
    echo "  4) All available"
    echo "  5) None (environment setup only)"
    echo ""
    read -rp "  Select [1-5]: " cli_choice
    echo ""

    case "$cli_choice" in
        1) INSTALL_CLAUDE=true ;;
        2) INSTALL_CODEX=true ;;
        3) INSTALL_CURSOR=true ;;
        4) INSTALL_CLAUDE=true; INSTALL_CODEX=true; INSTALL_CURSOR=true ;;
        5) ;;
        *)
            fail "Invalid choice. Run again and select 1-5."
            exit 1
            ;;
    esac
fi

# ═════════════════════════════════════════════════════════════════════════════
# 2. Python venv + Core Dependencies
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Setting up Python environment..."

# Resolve venv path
if [[ -n "$USER_VENV" ]]; then
    VENV="$USER_VENV"
elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python3" ]]; then
    VENV="$VIRTUAL_ENV"
    info "Active venv detected: $VENV"
else
    VENV="$APEX_ROOT/.venv"
fi

# Create if needed
if [[ -x "$VENV/bin/python3" ]]; then
    ok "Venv exists: $VENV"
else
    info "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
    ok "Created venv: $VENV"
fi
PYTHON="$VENV/bin/python3"
PIP="$PYTHON -m pip"

# Upgrade pip
$PIP install --upgrade pip --quiet 2>/dev/null || true

# Install core Python dependencies
echo ""
echo "▸ Installing core Python dependencies..."
$PIP install --quiet \
    numpy "PyYAML>=6.0" "requests>=2.28" \
    "pytest>=7.0" "rich" "tiktoken" \
    "mcp>=1.0.0" "chromadb" "sentence-transformers" \
    "pymupdf>=1.24.0" "pdfplumber>=0.10.0" 2>&1 | tail -5 || true
ok "Core packages installed"

# Install PyTorch for ROCm
echo ""
echo "▸ Installing PyTorch for ROCm..."
if $PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "rocm\|hip"; then
    ok "PyTorch (ROCm) already installed: $($PYTHON -c 'import torch; print(torch.__version__)')"
else
    info "Installing torch + torchvision for ROCm 7.2..."
    $PIP install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2 2>&1 | tail -3 || {
        warn "PyTorch ROCm install failed — you may need to install manually:"
        info "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2"
    }
    if $PYTHON -c "import torch" 2>/dev/null; then
        ok "PyTorch installed: $($PYTHON -c 'import torch; print(torch.__version__)')"
    else
        warn "PyTorch not importable — GPU grading may not work"
    fi
fi

# Install Triton
if $PYTHON -c "import triton" 2>/dev/null; then
    ok "Triton already installed"
else
    info "Installing Triton..."
    $PIP install --quiet "triton>=3.0" 2>&1 | tail -3 || warn "Triton install failed"
fi

# Install Claude agent SDKs
$PIP install --quiet "claude-code-sdk>=0.0.10" 2>&1 | tail -3 || true

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 3. Agent CLI Prerequisites
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Checking agent CLI prerequisites..."

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    if command -v claude &>/dev/null; then
        ok "Claude Code CLI: $(claude --version 2>&1 | head -1)"
    else
        fail "Claude Code CLI not found"
        info "Install: npm install -g @anthropic-ai/claude-code && claude login"
        INSTALL_CLAUDE=false
    fi
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    if command -v codex &>/dev/null; then
        ok "Codex CLI: $(codex --version 2>&1 | head -1)"
    else
        fail "Codex CLI not found"
        info "Install: npm install -g @openai/codex && codex login"
        INSTALL_CODEX=false
    fi
fi

if [[ "$INSTALL_CURSOR" == "true" ]]; then
    if command -v cursor-agent &>/dev/null; then
        ok "Cursor Agent CLI: cursor-agent"
    elif command -v cursor &>/dev/null; then
        ok "Cursor CLI: cursor (will use as fallback)"
    else
        fail "Cursor Agent CLI not found"
        info "Install: npm install -g cursor-agent && cursor-agent login"
        info "Or open the Apex folder in Cursor IDE (MCP auto-configured via .mcp.json)"
        INSTALL_CURSOR=false
    fi
fi

ok "Python: $($PYTHON --version 2>&1) ($PYTHON)"
echo ""

# Magpie / Results
MAGPIE_ROOT="$TOOLS_DIR/magpie"

export VENV PYTHON MAGPIE_ROOT
export APEX_ROOT TOOLS_DIR MCPS_DIR SKILLS_DIR
export ROCM_DIR DOC_DIR JSONS_DIR

# ═════════════════════════════════════════════════════════════════════════════
# 4. Data Dependencies (ROCm repos + docs)
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_DOWNLOADS" == "false" ]]; then
    echo "▸ Checking data dependencies for source-finder & RAG..."

    source "$MCPS_DIR/_shared.sh"

    if [[ -d "$ROCM_DIR" ]] && [[ -n "$(ls -A "$ROCM_DIR" 2>/dev/null)" ]]; then
        repo_count=$(find "$ROCM_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
        ok "ROCm repos: $ROCM_DIR ($repo_count repos)"
    else
        if [[ "$NON_INTERACTIVE" == "true" ]]; then
            info "Cloning ROCm repos..."
            clone_rocm_repos
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
    fi

    if [[ -d "$DOC_DIR" ]] && [[ -n "$(ls -A "$DOC_DIR" 2>/dev/null)" ]]; then
        doc_count=$(find "$DOC_DIR" -maxdepth 1 -name '*.pdf' | wc -l)
        ok "Documentation PDFs: $DOC_DIR ($doc_count files)"
    else
        if [[ "$NON_INTERACTIVE" == "true" ]]; then
            info "Downloading documentation PDFs..."
            download_docs
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
    fi

    if [[ -d "$JSONS_DIR" ]]; then
        ok "Optimization snippets: $JSONS_DIR"
    else
        warn "JSON snippets not found at $JSONS_DIR"
    fi

    echo ""
else
    info "Skipping data downloads (--skip-downloads)"
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════════════
# 5. Install MCP Python Dependencies + Magpie
# ═════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_TOOLS" == "false" ]]; then
    echo "▸ Installing MCP Python dependencies..."

    for mcp_dir in "$MCPS_DIR"/*/; do
        [[ ! -d "$mcp_dir" ]] && continue
        mcp_name="$(basename "$mcp_dir")"
        if [[ -f "$mcp_dir/pyproject.toml" ]]; then
            pip_log="$(mktemp)"
            if $PIP install -e "$mcp_dir" > "$pip_log" 2>&1; then
                ok "$mcp_name"
            else
                warn "$mcp_name — pip install failed (retrying)..."
                if $PIP install -e "$mcp_dir" 2>&1 | tail -20; then
                    ok "$mcp_name (retry succeeded)"
                else
                    fail "$mcp_name — pip install failed"
                    tail -10 "$pip_log" | while IFS= read -r line; do echo "      $line"; done
                fi
            fi
            rm -f "$pip_log"
        fi
    done

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
else
    info "Skipping tool installation (--skip-tools)"
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════════════
# 6. Register MCP Servers
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Registering MCP servers..."

# register_mcp NAME [KEY=VAL ...] -- COMMAND [ARGS...]
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

register_mcp "source-finder" \
    -- "$PYTHON" "$MCPS_DIR/source_finder/server.py"

register_mcp "kernel-rag" \
    "MCP_ROCM_DIR=$ROCM_DIR" \
    "MCP_DOC_DIR=$DOC_DIR" \
    "MCP_JSONS_DIR=$JSONS_DIR" \
    -- "$PYTHON" "$MCPS_DIR/rag_tool/server.py"

register_mcp "gpu-info" \
    -- "$PYTHON" "$MCPS_DIR/gpu_info/server.py"

register_mcp "fusion-advisor" \
    -- "$PYTHON" "$MCPS_DIR/fusion_advisor/server.py"

register_mcp "magpie" \
    "PYTHONPATH=$MAGPIE_ROOT" \
    -- "$PYTHON" "-m" "Magpie.mcp"

# Generate .mcp.json (for Cursor IDE and Claude Code IDE)
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
ok "Generated .mcp.json (Cursor IDE + Claude Code IDE auto-discover MCP servers from this file)"

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 7. Install Skills
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Installing skills..."

SKILL_NAMES=()
for skill_dir in "$SKILLS_DIR"/*/; do
    [[ -d "$skill_dir" ]] && SKILL_NAMES+=("$(basename "$skill_dir")")
done

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    if [[ -f "$APEX_ROOT/CLAUDE.md" ]]; then
        ok "Claude Code: CLAUDE.md found — ${#SKILL_NAMES[@]} skills auto-discoverable from tools/skills/"
    else
        warn "Claude Code: CLAUDE.md not found — skills won't be auto-discoverable"
    fi
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    CODEX_SKILLS="$CODEX_HOME/skills"
    mkdir -p "$CODEX_SKILLS"

    linked=0
    for skill_name in "${SKILL_NAMES[@]}"; do
        target="$CODEX_SKILLS/$skill_name"
        source_dir="$SKILLS_DIR/$skill_name"
        [[ -L "$target" ]] && rm -f "$target"
        if [[ -d "$target" ]] && [[ ! -L "$target" ]]; then
            warn "$skill_name — real directory at $target, skipping"
            continue
        fi
        ln -s "$source_dir" "$target" && linked=$((linked + 1))
    done
    ok "Codex: $linked skills symlinked into $CODEX_SKILLS/"

    if [[ -f "$APEX_ROOT/CLAUDE.md" ]] && [[ ! -f "$APEX_ROOT/AGENTS.md" ]]; then
        cp "$APEX_ROOT/CLAUDE.md" "$APEX_ROOT/AGENTS.md"
        ok "Codex: Created AGENTS.md from CLAUDE.md"
    elif [[ -f "$APEX_ROOT/AGENTS.md" ]]; then
        ok "Codex: AGENTS.md already exists"
    fi
fi

if [[ "$INSTALL_CURSOR" == "true" ]]; then
    if [[ -f "$APEX_ROOT/.mcp.json" ]]; then
        ok "Cursor: .mcp.json found — MCP servers auto-configured when Apex is opened in Cursor"
    fi
    # Cursor discovers skills via .cursor/rules/ or project-level settings
    cursor_rules="$APEX_ROOT/.cursor/rules"
    if [[ -d "$cursor_rules" ]]; then
        ok "Cursor: .cursor/rules/ found — ${#SKILL_NAMES[@]} skills available"
    else
        mkdir -p "$cursor_rules"
        ok "Cursor: Created .cursor/rules/"
    fi
fi

echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 8. Results Directory
# ═════════════════════════════════════════════════════════════════════════════

RESULTS_DIR="${RESULTS_DIR:-$APEX_ROOT/results}"
mkdir -p "$RESULTS_DIR"
ok "Results dir: $RESULTS_DIR"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
# 9. Verify & Summary
# ═════════════════════════════════════════════════════════════════════════════

echo "▸ Verifying setup..."

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    echo -e "  ${CYAN}Claude Code MCP servers:${NC}"
    claude mcp list 2>&1 | grep -E "^\s" | head -20 || true
    echo ""
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    echo -e "  ${CYAN}Codex MCP servers:${NC}"
    codex mcp list 2>&1 | head -20 || true
    echo ""
fi

installed_clis=""
[[ "$INSTALL_CLAUDE" == "true" ]] && installed_clis="Claude Code"
[[ "$INSTALL_CODEX" == "true" ]] && {
    [[ -n "$installed_clis" ]] && installed_clis+=" + "
    installed_clis+="Codex"
}
[[ "$INSTALL_CURSOR" == "true" ]] && {
    [[ -n "$installed_clis" ]] && installed_clis+=" + "
    installed_clis+="Cursor"
}
[[ -z "$installed_clis" ]] && installed_clis="(none — environment only)"

echo "═══════════════════════════════════════════════════════"
echo -e " ${GREEN}Setup complete!${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo " Configured for: $installed_clis"
echo ""
echo " Paths:"
echo "   VENV             $VENV"
echo "   PYTHON           $PYTHON"
echo "   MAGPIE_ROOT      $MAGPIE_ROOT"
echo "   RESULTS_DIR      $RESULTS_DIR"
echo ""
echo " ── Quick Start ─────────────────────────────────────"
echo ""
echo " 1. Activate the environment:"
echo "    source $VENV/bin/activate"
echo "    export MAGPIE_ROOT=$MAGPIE_ROOT"
echo ""

if [[ "$INSTALL_CLAUDE" == "true" ]]; then
    echo " 2a. Interactive (Claude Code):"
    echo "     cd $APEX_ROOT && claude"
    echo ""
fi

if [[ "$INSTALL_CODEX" == "true" ]]; then
    echo " 2b. Interactive (Codex):"
    echo "     cd $APEX_ROOT && codex"
    echo ""
fi

if [[ "$INSTALL_CURSOR" == "true" ]]; then
    echo " 2c. Interactive (Cursor):"
    echo "     Open $APEX_ROOT in Cursor IDE"
    echo "     Or: cd $APEX_ROOT && cursor-agent"
    echo ""
fi

echo " 3. Automated pipeline:"
echo "    python3 workload_optimizer.py run \\"
echo "      -r $RESULTS_DIR \\"
echo "      -b \$MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_120b.yaml \\"
echo "      --kernel-types triton --top-k 10 \\"
echo "      --max-iterations 3 --max-turns 25 --leaderboard"
echo ""
echo " 4. Standalone kernel optimization:"
echo "    python3 workload_optimizer.py optimize-kernel \\"
echo "      -r $RESULTS_DIR \\"
echo "      --kernel path/to/kernel.py \\"
echo "      --kernel-name rms_norm --kernel-type triton \\"
echo "      --agent-backend cursor"
echo ""
echo "═══════════════════════════════════════════════════════"
