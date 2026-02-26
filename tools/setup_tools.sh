#!/usr/bin/env bash
# setup_tools.sh
# Downloads and installs tools/MCPs/skills for the RL kernel-optimization sandbox.
#
# Tools installed:
#   1. Magpie  — GPU kernel correctness & performance evaluation (has its own MCP server)
#   2. RAG tool — dependencies for the kernel-file search tool (tools/rag_tool/)
#   3. Agent MCP registrations (Codex default, Claude optional) — magpie + kernel-rag
#   4. AgentKernelArena — cloned locally (not tracked as a git submodule)
#   5. Skills installed for Codex and Claude

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
FILES_DIR="$REPO_ROOT/files"

# ── helpers ────────────────────────────────────────────────────────────────────

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found in PATH"; exit 1; }
}

pip_install() {
    python3 -m pip install --quiet "$@"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

register_codex_mcp() {
    local rag_dir="$1"
    if ! have codex; then
        echo "  [skip] codex CLI not found (MCP registration)"
        return 0
    fi
    echo "  codex: registering MCP servers..."
    codex mcp remove magpie >/dev/null 2>&1 || true
    codex mcp add magpie -- python3 -m magpie.mcp
    echo "    registered: magpie"

    codex mcp remove kernel-rag >/dev/null 2>&1 || true
    codex mcp add kernel-rag -- python3 "$rag_dir/server.py"
    echo "    registered: kernel-rag"
}

register_claude_mcp() {
    local rag_dir="$1"
    if ! have claude; then
        echo "  [skip] claude CLI not found (MCP registration)"
        return 0
    fi
    echo "  claude: registering MCP servers..."
    claude mcp remove magpie >/dev/null 2>&1 || true
    claude mcp add --transport stdio magpie -- python3 -m magpie.mcp
    echo "    registered: magpie"

    claude mcp remove kernel-rag >/dev/null 2>&1 || true
    claude mcp add --transport stdio kernel-rag -- python3 "$rag_dir/server.py"
    echo "    registered: kernel-rag"
}

install_codex_skills() {
    local source_skills_dir="$1"
    local codex_home="${CODEX_HOME:-$HOME/.codex}"
    local codex_skills_dir="$codex_home/skills"
    mkdir -p "$codex_skills_dir"

    for skill_dir in "$source_skills_dir"/*; do
        [ -d "$skill_dir" ] || continue
        local name
        name="$(basename "$skill_dir")"
        rm -rf "$codex_skills_dir/$name"
        cp -a "$skill_dir" "$codex_skills_dir/$name"
        echo "  codex skill: $name -> $codex_skills_dir/$name"
    done
}

install_claude_skills() {
    local repo_root="$1"
    local source_skills_dir="$2"
    local claude_skills_dir="$repo_root/.claude/skills"
    mkdir -p "$(dirname "$claude_skills_dir")"
    rm -rf "$claude_skills_dir"
    cp -a "$source_skills_dir" "$claude_skills_dir"
    echo "  claude skills installed to: $claude_skills_dir"
}

# ── pre-flight ────────────────────────────────────────────────────────────────

need git
need python3
need pip3

echo ""
echo "=== 1. Magpie — GPU kernel evaluation framework ==="

MAGPIE_DIR="$SCRIPT_DIR/magpie"

if [ -d "$MAGPIE_DIR/.git" ]; then
    echo "  [skip] $MAGPIE_DIR already exists"
else
    echo "  cloning AMD-AGI/Magpie..."
    git clone --depth=1 https://github.com/AMD-AGI/Magpie.git "$MAGPIE_DIR"
fi

echo "  installing Magpie..."
pip_install -e "$MAGPIE_DIR"

# ── 2. RAG tool dependencies ──────────────────────────────────────────────────

echo ""
echo "=== 2. RAG tool — installing dependencies ==="

RAG_DIR="$SCRIPT_DIR/rag_tool"
pip_install -r "$RAG_DIR/requirements.txt"

# ── 3. Register MCP servers (Codex default, Claude optional) ─────────────────

echo ""
echo "=== 3. Registering MCP servers ==="
echo "  default target: codex"
register_codex_mcp "$RAG_DIR"
register_claude_mcp "$RAG_DIR"

# ── 4. AgentKernelArena ───────────────────────────────────────────────────────

echo ""
echo "=== 4. AgentKernelArena ==="

AKA_DIR="$SCRIPT_DIR/AgentKernelArena"
if [ -d "$AKA_DIR/.git" ]; then
    echo "  [skip] $AKA_DIR already exists"
else
    echo "  cloning AMD-AGI/AgentKernelArena..."
    git clone --depth=1 https://github.com/AMD-AGI/AgentKernelArena.git "$AKA_DIR"
fi

echo ""
echo "=== 5. Installing agent skills (Codex default, Claude compatible) ==="
SOURCE_SKILLS_DIR="$SCRIPT_DIR/skills"
install_codex_skills "$SOURCE_SKILLS_DIR"
install_claude_skills "$REPO_ROOT" "$SOURCE_SKILLS_DIR"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Run setup_files.sh to populate the files/ directory with code and docs."
echo "  2. Run: python3 $RAG_DIR/index.py   (to build the RAG search index)"
