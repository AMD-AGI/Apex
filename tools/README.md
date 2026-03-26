# Apex Tools

Tools, MCPs, and skills for the RL kernel-optimization sandbox.

## Quick Start

```bash
# Install everything + register MCPs for Claude Code
./setup_tools.sh --claude

# Install everything + register MCPs for Cursor
./setup_tools.sh --cursor

# Install dependencies only (no IDE registration)
./setup_tools.sh

# Skip heavy downloads if already done
./setup_tools.sh --claude --skip-repos --skip-docs
```

## Tools Installed

| Tool | Description |
|------|-------------|
| **Magpie** | GPU kernel correctness & performance evaluation |
| **RAG Tool** | Multi-retriever RAG for docs, snippets, and library code |
| **Fusion Advisor** | Kernel fusion opportunity detection & code generation |
| **GPU Info** | GPU architecture detection & optimization hints |
| **Source Finder** | Kernel source lookup from profiler trace signatures |

## Individual MCP Setup

Each MCP can also be installed independently:

```bash
# Install just one MCP (pick --claude or --cursor)
./mcps/magpie/setup.sh --cursor
./mcps/fusion_advisor/setup.sh --claude
./mcps/gpu_info/setup.sh --cursor
./mcps/rag_tool/setup.sh --claude           # also clones ROCm repos + downloads docs
./mcps/source_finder/setup.sh --cursor      # also clones ROCm repos

# Skip data downloads for rag_tool/source_finder if already done
./mcps/rag_tool/setup.sh --cursor --skip-repos --skip-docs
./mcps/source_finder/setup.sh --cursor --skip-repos
```

## Directory Layout

```
tools/
  setup_tools.sh          # Install all tools + register MCPs
  jsons/                  # Optimization snippet sheets (checked in)
    hip_sheet.json
    triton_sheet.json
    rocm.json             # ROCm library list (used for cloning)
  rocm/                   # Cloned ROCm repos (gitignored)
  doc/                    # Downloaded documentation PDFs (gitignored)
  magpie/                 # Cloned Magpie repo (gitignored)
  mcps/
    _shared.sh            # Common helpers (clone_rocm_repos, download_docs)
    magpie/               # Magpie MCP setup
      setup.sh
    fusion_advisor/       # Detect & generate fused kernels
      server.py
      setup.sh
      pyproject.toml
    gpu_info/             # GPU arch specs & optimization hints
      server.py
      setup.sh
      pyproject.toml
    rag_tool/             # RAG search over docs, code, snippets
      server.py           # Multi-retriever RAG server
      index.py            # ChromaDB vector index builder
      setup.sh
      pyproject.toml
    source_finder/        # Find kernel source from trace names
      server.py
      setup.sh
      pyproject.toml
```
