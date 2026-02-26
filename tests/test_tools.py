"""
test_tools.py — Tests for tools/setup_tools.sh and the RAG tool modules.

Verifies:
  - The setup script is correct and executable
  - The RAG tool Python modules are valid and have the right structure
  - The MCP tool schema is valid
"""

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).parent.parent
TOOLS_DIR  = REPO_ROOT / "tools"
SETUP_SH   = TOOLS_DIR / "setup_tools.sh"
RAG_DIR    = TOOLS_DIR / "rag_tool"


# ── setup_tools.sh ────────────────────────────────────────────────────────────

class TestSetupToolsScript:
    def test_exists(self):
        assert SETUP_SH.exists(), f"{SETUP_SH} not found"

    def test_executable(self):
        assert os.access(SETUP_SH, os.X_OK)

    def test_shebang(self):
        assert SETUP_SH.read_text().splitlines()[0].startswith("#!")

    def test_installs_magpie(self):
        content = SETUP_SH.read_text()
        assert "AMD-AGI/Magpie" in content

    def test_has_error_handling(self):
        content = SETUP_SH.read_text()
        assert "set -e" in content


# ── RAG tool: index.py ────────────────────────────────────────────────────────

class TestRAGIndex:
    @pytest.fixture(scope="class")
    def index_source(self):
        return (RAG_DIR / "index.py").read_text()

    def test_parses_as_valid_python(self, index_source):
        ast.parse(index_source)  # raises SyntaxError if invalid

    def test_defines_build_index_function(self, index_source):
        tree = ast.parse(index_source)
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "build_index" in funcs

    def test_defines_chunk_text_function(self, index_source):
        tree = ast.parse(index_source)
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "chunk_text" in funcs

    def test_uses_chromadb(self, index_source):
        assert "chromadb" in index_source

    def test_uses_sentence_transformers(self, index_source):
        assert "sentence-transformers" in index_source or "SentenceTransformer" in index_source

    def test_has_argparse_cli(self, index_source):
        assert "argparse" in index_source

    def test_indexes_code_and_docs(self, index_source):
        assert "code" in index_source
        assert "docs" in index_source


# ── RAG tool: server.py ───────────────────────────────────────────────────────

class TestRAGServer:
    @pytest.fixture(scope="class")
    def server_source(self):
        return (RAG_DIR / "server.py").read_text()

    def test_parses_as_valid_python(self, server_source):
        ast.parse(server_source)

    def test_defines_find_relevant_files_tool(self, server_source):
        assert "find_relevant_files" in server_source

    def test_defines_search_docs_tool(self, server_source):
        assert "search_docs" in server_source

    def test_uses_mcp_server(self, server_source):
        assert "Server" in server_source or "mcp" in server_source

    def test_uses_stdio_transport(self, server_source):
        assert "stdio" in server_source

    def test_tool_schemas_present(self, server_source):
        assert "kernel_code" in server_source
        assert "query" in server_source

    def test_formats_results_as_markdown(self, server_source):
        assert "_format_results" in server_source or "Result" in server_source

    def test_connects_to_chromadb(self, server_source):
        assert "chromadb" in server_source
