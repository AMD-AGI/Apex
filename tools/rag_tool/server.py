#!/usr/bin/env python3
"""
server.py — MCP server for the kernel-optimization RAG tool.

Exposes two tools:
  - find_relevant_files(kernel_code, n_results) → list of relevant file excerpts
  - search_docs(query, n_results) → list of relevant documentation excerpts

Start:
    python3 server.py [--db-dir PATH]

Register in Codex or Claude:
  - Codex CLI:  `codex mcp add kernel-rag -- python3 /path/to/tools/rag_tool/server.py`
  - Claude CLI: `claude mcp add --transport stdio kernel-rag -- python3 /path/to/tools/rag_tool/server.py`

Claude Code config example (.claude/mcp.json):
    {
      "mcpServers": {
        "kernel-rag": {
          "command": "python3",
          "args": ["/path/to/tools/rag_tool/server.py"]
        }
      }
    }
"""
import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── setup ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--db-dir", default=None)
args, _ = parser.parse_known_args()

SCRIPT_DIR = Path(__file__).parent
DB_DIR = Path(args.db_dir) if args.db_dir else SCRIPT_DIR / "chroma_db"

if not DB_DIR.exists():
    sys.stderr.write(
        f"[kernel-rag] Database not found at {DB_DIR}. "
        "Run index.py first.\n"
    )
    sys.exit(1)

_client = chromadb.PersistentClient(path=str(DB_DIR))
_embed  = SentenceTransformerEmbeddingFunction(
    model_name="jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True,
)
_collection = _client.get_collection("kernel_files", embedding_function=_embed)

# ── MCP server ────────────────────────────────────────────────────────────────

app = Server("kernel-rag")


def _format_results(results) -> str:
    ids        = results["ids"][0]
    docs       = results["documents"][0]
    metas      = results["metadatas"][0]
    distances  = results["distances"][0]
    lines = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = round(1 - dist, 3)          # cosine similarity
        path  = meta.get("path", "?")
        start = meta.get("start", 0)
        lines.append(f"### Result {i+1}  (score={score})  {path}:{start}")
        lines.append("```")
        lines.append(doc[:800] + ("..." if len(doc) > 800 else ""))
        lines.append("```")
        lines.append("")
    return "\n".join(lines) if lines else "No results found."


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="find_relevant_files",
            description=(
                "Given a HIP/Triton/CUDA kernel (or a description of the operation "
                "being optimized), return the most relevant source files, headers, "
                "and examples from the indexed codebase."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_code": {
                        "type": "string",
                        "description": "The kernel source code or a natural-language description of the operation.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 8).",
                        "default": 8,
                    },
                },
                "required": ["kernel_code"],
            },
        ),
        Tool(
            name="search_docs",
            description=(
                "Search the indexed ROCm, HIP, CK, and Triton documentation "
                "for content relevant to a query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Documentation search query.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 6).",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "find_relevant_files":
        query    = arguments["kernel_code"]
        n        = int(arguments.get("n_results", 8))
        results  = _collection.query(
            query_texts=[query],
            n_results=n,
            where={"path": {"$contains": "code"}},   # prefer source files
        )
        # Fall back to all content if code-only returned nothing
        if not results["ids"][0]:
            results = _collection.query(query_texts=[query], n_results=n)
        return [TextContent(type="text", text=_format_results(results))]

    elif name == "search_docs":
        query   = arguments["query"]
        n       = int(arguments.get("n_results", 6))
        results = _collection.query(
            query_texts=[query],
            n_results=n,
            where={"path": {"$contains": "docs"}},
        )
        if not results["ids"][0]:
            results = _collection.query(query_texts=[query], n_results=n)
        return [TextContent(type="text", text=_format_results(results))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
