#!/usr/bin/env python3
"""
index.py — Build a ChromaDB vector index over the files/ directory.

Usage:
    python3 index.py [--files-dir PATH] [--db-dir PATH]

Indexes .cpp .hip .cu .h .hpp .py .md files from files/code/ and files/docs/.
Run this once after setup_files.sh populates the files/ directory.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.progress import track

console = Console()

INDEXED_EXTENSIONS = {".cpp", ".hip", ".cu", ".h", ".hpp", ".py", ".md", ".rst"}
MAX_CHUNK_CHARS = 1500   # keep chunks small enough for good retrieval
OVERLAP_CHARS   = 200


def chunk_text(text: str, path: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunk = text[start:end]
        chunk_id = hashlib.md5(f"{path}:{start}".encode()).hexdigest()
        chunks.append({
            "id":       chunk_id,
            "text":     chunk,
            "metadata": {
                "path":  path,
                "start": start,
                "end":   end,
            },
        })
        if end == len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


def collect_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in INDEXED_EXTENSIONS:
            # Skip .git and build dirs
            parts = p.parts
            if any(d in parts for d in (".git", "__pycache__", "build", "CMakeFiles")):
                continue
            files.append(p)
    return files


def build_index(files_dir: Path, db_dir: Path):
    console.print(f"[bold]Indexing:[/bold] {files_dir}")
    console.print(f"[bold]Database:[/bold] {db_dir}")

    client = chromadb.PersistentClient(path=str(db_dir))
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
    )
    collection = client.get_or_create_collection(
        name="kernel_files",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    code_dir = files_dir / "code"
    docs_dir = files_dir / "docs"

    all_files: list[Path] = []
    for d in [code_dir, docs_dir]:
        if d.exists():
            all_files.extend(collect_files(d))

    console.print(f"Found [green]{len(all_files)}[/green] files to index.")

    BATCH = 64
    batch_ids, batch_docs, batch_metas = [], [], []

    def flush():
        if batch_ids:
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    for path in track(all_files, description="Indexing files..."):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(files_dir))
        for chunk in chunk_text(text, rel):
            batch_ids.append(chunk["id"])
            batch_docs.append(chunk["text"])
            batch_metas.append(chunk["metadata"])
            if len(batch_ids) >= BATCH:
                flush()
    flush()

    count = collection.count()
    console.print(f"[green]Done.[/green] {count} chunks indexed in {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-dir", default=None,
                        help="Path to the files/ directory (default: ../files relative to this script)")
    parser.add_argument("--db-dir", default=None,
                        help="Path to store the ChromaDB database (default: ./chroma_db)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    files_dir  = Path(args.files_dir) if args.files_dir else script_dir.parent.parent / "files"
    db_dir     = Path(args.db_dir)    if args.db_dir    else script_dir / "chroma_db"

    if not files_dir.exists():
        console.print(f"[red]ERROR:[/red] files dir not found: {files_dir}")
        console.print("Run setup_files.sh first.")
        sys.exit(1)

    build_index(files_dir, db_dir)
