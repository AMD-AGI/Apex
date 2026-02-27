#!/usr/bin/env python3
"""
Kernel Optimization RAG MCP Server
===================================
A comprehensive multi-retriever RAG system for AMD GPU kernel optimization.

Features:
- Multi-source indexing (docs, snippets, code)
- Semantic search with embeddings
- Intelligent routing (HIP, Triton, library-specific)
- Code-aware chunking and retrieval

Sources indexed:
1. HIP/Triton documentation (cleaned markdown)
2. Optimization snippets (hip_sheet.json, triton_sheet.json)
3. ROCm library code (headers, examples, implementations)
"""

import asyncio
import json
import os
import re
import pickle
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

# =============================================================================
# Configuration
# =============================================================================

# Derive paths from environment variables or relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_TOOLS_DIR = SCRIPT_DIR.parent.parent  # mcp_tools/

# Allow environment variable overrides for paths
ROCM_DIR = Path(os.environ.get("MCP_ROCM_DIR", str(MCP_TOOLS_DIR / "rocm")))
DOC_DIR = Path(os.environ.get("MCP_DOC_DIR", str(MCP_TOOLS_DIR / "doc")))
JSONS_DIR = Path(os.environ.get("MCP_JSONS_DIR", str(MCP_TOOLS_DIR / "jsons")))

@dataclass
class Config:
    """RAG server configuration."""
    base_dir: str = str(MCP_TOOLS_DIR)
    docs_dir: str = str(MCP_TOOLS_DIR / "docs")  # For markdown docs if any
    rocm_dir: str = str(ROCM_DIR)
    jsons_dir: str = str(JSONS_DIR)
    pdf_dir: str = str(DOC_DIR)  # PDFs are in doc/ folder
    cache_dir: str = str(MCP_TOOLS_DIR / ".cache")
    cache_file: str = "rag_index_cache.pkl"
    
    # Chunking settings
    max_chunk_size: int = 2000
    chunk_overlap: int = 200
    
    # Search settings
    max_results: int = 10
    
    # Cache settings
    cache_enabled: bool = True
    cache_max_age_hours: int = 24  # Rebuild cache after this many hours
    
    # File patterns to index from repos
    code_extensions: tuple = (".hpp", ".h", ".cpp", ".hip", ".py")
    doc_extensions: tuple = (".md", ".rst", ".txt")
    
    # Directories to skip in repos
    skip_dirs: tuple = (
        ".git", ".github", ".azuredevops", ".jenkins",
        "build", "cmake", "docker", "test", "tests",
        "__pycache__", "node_modules"
    )

config = Config()
print(f"RAG Server configured with MCP tools dir: {MCP_TOOLS_DIR}")
print(f"  ROCm dir: {config.rocm_dir}")
print(f"  PDF dir: {config.pdf_dir}")
print(f"  JSONs dir: {config.jsons_dir}")

# =============================================================================
# ROCm Library Knowledge Base - What each library is useful for
# =============================================================================

LIBRARY_KNOWLEDGE = {
    "composable_kernel": {
        "name": "Composable Kernel (CK)",
        "purpose": "High-performance fused GPU kernels using template metaprogramming",
        "use_for": [
            "GEMM (matrix multiplication) kernels",
            "Fused attention kernels (FlashAttention-style)",
            "Fused MLP/FFN kernels",
            "Custom fused operations",
            "Understanding AMD's MFMA (Matrix Fused Multiply-Add) usage",
            "Tile-based kernel design patterns"
        ],
        "key_concepts": ["MFMA instructions", "Tile mapping", "Pipeline prefetching", "Fused kernels"],
        "priority": 1
    },
    "rocWMMA": {
        "name": "rocWMMA (Wave Matrix Multiply-Accumulate)",
        "purpose": "WMMA/Tensor Core-like operations for AMD GPUs",
        "use_for": [
            "Matrix multiply using hardware matrix units",
            "Mixed-precision GEMM (FP16, BF16, FP8)",
            "Understanding wave-level matrix operations",
            "Replacing CUDA WMMA/Tensor Core code"
        ],
        "key_concepts": ["Wave-level operations", "MFMA", "Matrix fragments", "Mixed precision"],
        "priority": 1
    },
    "aiter": {
        "name": "AITER (AMD Inference Transformer Efficient Runtimes)",
        "purpose": "Optimized inference kernels for transformer models",
        "use_for": [
            "Optimized attention kernels for inference",
            "Fused MoE (Mixture of Experts) kernels",
            "PagedAttention implementations",
            "Quantization kernels (FP8, INT8)",
            "KV-cache management",
            "State-of-the-art LLM serving kernels"
        ],
        "key_concepts": ["FlashAttention", "PagedAttention", "Fused MoE", "Quantization", "KV-cache"],
        "priority": 1
    },
    "triton": {
        "name": "Triton (OpenAI Triton for AMD)",
        "purpose": "Python-based GPU kernel programming language",
        "use_for": [
            "Rapid kernel prototyping",
            "Custom fused kernels in Python",
            "Autotuning kernel configurations",
            "Research-oriented kernel development"
        ],
        "key_concepts": ["@triton.jit", "tl.load/store", "Block programs", "Autotuning"],
        "priority": 1
    },
    "MIOpen": {
        "name": "MIOpen",
        "purpose": "AMD's deep learning primitives library (like cuDNN)",
        "use_for": [
            "Convolution operations",
            "Batch normalization",
            "Pooling operations",
            "RNN/LSTM operations",
            "Optimized DL primitives"
        ],
        "key_concepts": ["Convolution algorithms", "Find-DB", "Fusion", "Workspace management"],
        "priority": 2
    },
    "rocBLAS": {
        "name": "rocBLAS",
        "purpose": "BLAS (Basic Linear Algebra) library for AMD GPUs",
        "use_for": [
            "GEMM (matrix multiplication)",
            "GEMV (matrix-vector)",
            "BLAS Level 1/2/3 operations",
            "Batched GEMM operations"
        ],
        "key_concepts": ["GEMM tuning", "Batched operations", "Data layouts", "Tensile backend"],
        "priority": 2
    },
    "hipBLASLt": {
        "name": "hipBLASLt",
        "purpose": "Lightweight GEMM library with advanced features",
        "use_for": [
            "High-performance GEMM",
            "Fused GEMM + epilogue operations",
            "Mixed-precision GEMM",
            "GEMM with bias/activation fusion"
        ],
        "key_concepts": ["Epilogue fusion", "Algorithm selection", "Workspace", "Heuristics"],
        "priority": 2
    },
    "rocPRIM": {
        "name": "rocPRIM",
        "purpose": "Parallel primitives library (like CUB for CUDA)",
        "use_for": [
            "Parallel reductions",
            "Prefix scans (inclusive/exclusive)",
            "Radix sort",
            "Block/warp-level primitives"
        ],
        "key_concepts": ["Block reduce", "Block scan", "Warp primitives", "Device-wide operations"],
        "priority": 2
    },
    "hipCUB": {
        "name": "hipCUB",
        "purpose": "CUB-compatible parallel primitives for HIP",
        "use_for": [
            "Drop-in CUB replacement",
            "Block/warp reductions",
            "Sorting algorithms",
            "Porting CUDA CUB code"
        ],
        "key_concepts": ["CUB compatibility", "Block operations", "Sorting"],
        "priority": 2
    },
    "rocThrust": {
        "name": "rocThrust",
        "purpose": "Thrust-compatible parallel algorithms library",
        "use_for": [
            "High-level parallel algorithms",
            "STL-like interface for GPU",
            "Sorting, searching, transforming",
            "Porting CUDA Thrust code"
        ],
        "key_concepts": ["Thrust compatibility", "Device vectors", "Parallel algorithms"],
        "priority": 3
    },
    "hipTensor": {
        "name": "hipTensor",
        "purpose": "Tensor contraction library",
        "use_for": [
            "General tensor contractions",
            "Einstein summation operations",
            "Multi-dimensional tensor operations"
        ],
        "key_concepts": ["Tensor modes", "Contraction", "Permutation"],
        "priority": 3
    },
    "sglang": {
        "name": "SGLang",
        "purpose": "High-performance LLM serving framework",
        "use_for": [
            "LLM inference serving",
            "Understanding production inference patterns",
            "RadixAttention implementation",
            "Batch scheduling strategies"
        ],
        "key_concepts": ["RadixAttention", "Continuous batching", "KV-cache", "Speculative decoding"],
        "priority": 1
    },
    "vllm": {
        "name": "vLLM",
        "purpose": "High-throughput LLM serving with PagedAttention",
        "use_for": [
            "PagedAttention implementation",
            "KV-cache management",
            "LLM serving patterns",
            "Memory-efficient inference"
        ],
        "key_concepts": ["PagedAttention", "Block allocation", "Continuous batching"],
        "priority": 1
    },
    "AMDMIGraphX": {
        "name": "MIGraphX",
        "purpose": "AMD's graph-based inference optimizer",
        "use_for": [
            "Model optimization",
            "Graph-level fusion",
            "Inference acceleration",
            "ONNX model deployment"
        ],
        "key_concepts": ["Graph optimization", "Operator fusion", "Quantization"],
        "priority": 2
    },
    "rccl": {
        "name": "RCCL (ROCm Communication Collectives Library)",
        "purpose": "Multi-GPU communication (like NCCL)",
        "use_for": [
            "AllReduce operations",
            "Multi-GPU training",
            "Distributed inference",
            "Collective communication"
        ],
        "key_concepts": ["AllReduce", "AllGather", "Ring/Tree algorithms", "NVLink/Infinity Fabric"],
        "priority": 2
    }
}

# =============================================================================
# GPU Specifications Database - Structured specs for AMD Instinct GPUs
# =============================================================================

GPU_SPECS = {
    "MI300X": {
        "name": "AMD Instinct MI300X",
        "generation": "CDNA3",
        "memory": {
            "hbm_capacity_gb": 192,
            "hbm_bandwidth_tb_s": 5.3,
            "hbm_type": "HBM3",
            "memory_interface_bits": 8192,
            "infinity_cache_mb": 256,
        },
        "compute": {
            "compute_units": 304,
            "stream_processors": 19456,
            "matrix_cores": 1216,
            "peak_clock_mhz": 2100,
            "peak_fp64_tflops": 163.4,
            "peak_fp32_tflops": 163.4,
            "peak_fp16_tflops": 1307.4,
            "peak_bf16_tflops": 1307.4,
            "peak_fp8_tflops": 2614.9,
            "peak_int8_tops": 2614.9,
        },
        "architecture": {
            "arch": "CDNA3",
            "process_node": "5nm FinFET (GPU) + 6nm FinFET (I/O)",
            "xcd_count": 8,
            "wave_size": 64,
            "lds_per_cu_kb": 64,
            "vgprs_per_simd": 512,
            "sgprs_per_simd": 128,
            "max_waves_per_cu": 32,
        },
        "interconnect": {
            "infinity_fabric_links": 7,
            "if_bandwidth_per_link_gb_s": 128,
            "pcie_gen": 5,
            "pcie_lanes": 16,
        },
        "power": {
            "tdp_w": 750,
            "form_factor": "OAM",
        },
    },
    "MI325X": {
        "name": "AMD Instinct MI325X",
        "generation": "CDNA3 Enhanced",
        "memory": {
            "hbm_capacity_gb": 256,
            "hbm_bandwidth_tb_s": 6.0,
            "hbm_type": "HBM3E",
            "memory_interface_bits": 8192,
            "infinity_cache_mb": 256,
        },
        "compute": {
            "compute_units": 304,
            "stream_processors": 19456,
            "matrix_cores": 1216,
            "peak_clock_mhz": 2100,
            "peak_fp64_tflops": 163.4,
            "peak_fp32_tflops": 163.4,
            "peak_fp16_tflops": 1307.4,
            "peak_bf16_tflops": 1307.4,
            "peak_fp8_tflops": 2614.9,
            "peak_int8_tops": 2614.9,
        },
        "architecture": {
            "arch": "CDNA3",
            "process_node": "5nm FinFET (GPU) + 6nm FinFET (I/O)",
            "xcd_count": 8,
            "wave_size": 64,
            "lds_per_cu_kb": 64,
            "vgprs_per_simd": 512,
            "sgprs_per_simd": 128,
            "max_waves_per_cu": 32,
        },
        "interconnect": {
            "infinity_fabric_links": 7,
            "if_bandwidth_per_link_gb_s": 128,
            "pcie_gen": 5,
            "pcie_lanes": 16,
        },
        "power": {
            "tdp_w": 750,
            "form_factor": "OAM",
        },
    },
    "MI350X": {
        "name": "AMD Instinct MI350X",
        "generation": "CDNA4",
        "memory": {
            "hbm_capacity_gb": 288,
            "hbm_bandwidth_tb_s": 8.0,
            "hbm_type": "HBM3E",
            "memory_interface_bits": 8192,
            "infinity_cache_mb": 512,
        },
        "compute": {
            "compute_units": 320,
            "stream_processors": 20480,
            "matrix_cores": 1280,
            "peak_clock_mhz": 2300,
            "peak_fp64_tflops": 200,
            "peak_fp32_tflops": 200,
            "peak_fp16_tflops": 1800,
            "peak_bf16_tflops": 1800,
            "peak_fp8_tflops": 3600,
            "peak_fp4_tflops": 7200,
            "peak_int8_tops": 3600,
        },
        "architecture": {
            "arch": "CDNA4",
            "process_node": "3nm FinFET",
            "xcd_count": 8,
            "wave_size": 64,
            "lds_per_cu_kb": 64,
            "vgprs_per_simd": 512,
            "sgprs_per_simd": 128,
            "max_waves_per_cu": 32,
        },
        "interconnect": {
            "infinity_fabric_links": 8,
            "if_bandwidth_per_link_gb_s": 150,
            "pcie_gen": 6,
            "pcie_lanes": 16,
        },
        "power": {
            "tdp_w": 800,
            "form_factor": "OAM",
        },
    },
}

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Chunk:
    """A chunk of indexed content."""
    id: str
    source: str  # 'docs', 'snippets', 'code'
    category: str  # 'hip', 'triton', 'rocblas', etc.
    title: str
    content: str
    file_path: str
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "category": self.category,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "file_path": self.file_path,
            "metadata": self.metadata
        }

@dataclass  
class SearchResult:
    """A search result with relevance score."""
    chunk: Chunk
    score: float
    match_context: str = ""

# =============================================================================
# BM25 Search Implementation
# =============================================================================

import math
from collections import Counter

# =============================================================================
# Query Expansion / Synonyms
# =============================================================================

QUERY_SYNONYMS = {
    # Memory operations
    "gemm": ["matrix multiply", "matmul", "mm", "linear", "dot product"],
    "matmul": ["gemm", "matrix multiply", "mm", "linear"],
    "coalescing": ["coalesced", "memory access", "contiguous access", "memory pattern"],
    "coalesced": ["coalescing", "contiguous", "memory access pattern"],
    
    # Reduction operations
    "reduction": ["reduce", "sum", "accumulate", "fold", "aggregate"],
    "reduce": ["reduction", "sum", "accumulate"],
    "softmax": ["attention", "normalize", "probability"],
    
    # Tiling and memory
    "tiling": ["blocking", "tile", "block", "cache blocking", "shared memory"],
    "tile": ["tiling", "block", "blocking"],
    "lds": ["shared memory", "local data share", "smem", "__shared__"],
    "shared memory": ["lds", "local data share", "smem"],
    
    # Occupancy and performance
    "occupancy": ["register pressure", "wave slots", "active warps", "utilization"],
    "register": ["vgpr", "sgpr", "register pressure", "spill"],
    "bandwidth": ["memory throughput", "hbm", "memory bound"],
    
    # Matrix operations
    "mfma": ["matrix core", "tensor core", "matrix fma", "wmma", "matrix instruction"],
    "wmma": ["mfma", "matrix core", "tensor core"],
    "tensor core": ["mfma", "wmma", "matrix core"],
    
    # Precision
    "fp8": ["float8", "e4m3", "e5m2", "8-bit float", "quantization"],
    "fp16": ["half", "float16", "half precision"],
    "bf16": ["bfloat16", "brain float"],
    "quantization": ["quantize", "dequantize", "fp8", "int8", "mixed precision"],
    
    # Attention
    "attention": ["flash attention", "scaled dot product", "sdpa", "transformer"],
    "flash attention": ["attention", "flashattn", "tiled attention"],
    
    # Parallelism
    "wave": ["warp", "wavefront", "simd"],
    "warp": ["wave", "wavefront"],
    "thread": ["work item", "lane"],
    "block": ["workgroup", "threadblock", "cta"],
    
    # Libraries
    "rocblas": ["blas", "gemm library"],
    "hipblaslt": ["gemm", "epilogue fusion"],
    "composable kernel": ["ck", "fused kernel", "template kernel"],
    "triton": ["python kernel", "jit kernel"],
}

def expand_query(query: str) -> str:
    """Expand query with domain-specific synonyms for better recall."""
    query_lower = query.lower()
    query_terms = set(re.findall(r'\w+', query_lower))
    expanded_terms = set(query_terms)
    
    for term in query_terms:
        # Check for exact matches
        if term in QUERY_SYNONYMS:
            expanded_terms.update(QUERY_SYNONYMS[term])
        
        # Check for partial matches (term is part of a key)
        for key, synonyms in QUERY_SYNONYMS.items():
            if term in key or key in term:
                expanded_terms.update(synonyms)
                expanded_terms.add(key)
    
    return " ".join(expanded_terms)

def fuzzy_match_pattern(pattern: str, available_patterns: list[str], cutoff: float = 0.5) -> list[tuple[str, float]]:
    """Find patterns that fuzzy match the input pattern.
    
    Returns list of (pattern_name, similarity_score) tuples sorted by score.
    """
    pattern_lower = pattern.lower().replace('-', '_').replace(' ', '_')
    
    matches = []
    for avail in available_patterns:
        avail_lower = avail.lower()
        
        # Calculate similarity using SequenceMatcher
        ratio = SequenceMatcher(None, pattern_lower, avail_lower).ratio()
        
        # Boost if pattern is a substring
        if pattern_lower in avail_lower or avail_lower in pattern_lower:
            ratio = min(1.0, ratio + 0.3)
        
        # Boost if words overlap
        pattern_words = set(pattern_lower.split('_'))
        avail_words = set(avail_lower.split('_'))
        word_overlap = len(pattern_words & avail_words) / max(len(pattern_words), 1)
        ratio = min(1.0, ratio + word_overlap * 0.2)
        
        if ratio >= cutoff:
            matches.append((avail, ratio))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

class BM25:
    """Lightweight BM25 implementation for ranking search results."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[str, int] = {}  # Document frequency per term
        self.doc_lens: list[int] = []  # Length of each document
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0
        self.doc_tokens: list[list[str]] = []  # Tokenized documents
        
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    def index(self, documents: list[str]):
        """Build BM25 index from documents."""
        self.corpus_size = len(documents)
        self.doc_tokens = []
        self.doc_lens = []
        self.doc_freqs = Counter()
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            
            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1
        
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
    
    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a document given a query."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        # Count term frequencies in document
        doc_tf = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue
            
            tf = doc_tf[term]
            idf = self._idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search and return top-k document indices with scores."""
        scores = []
        for i in range(self.corpus_size):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# =============================================================================
# Index Storage
# =============================================================================

class RAGIndex:
    """In-memory index for RAG retrieval with BM25 ranking."""
    
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.by_source: dict[str, list[Chunk]] = defaultdict(list)
        self.by_category: dict[str, list[Chunk]] = defaultdict(list)
        self.snippets: dict[str, dict] = {}  # Direct access to optimization snippets
        self.heuristics: dict[str, list[dict]] = {}  # Decision heuristics
        self.initialized = False
        self.bm25: Optional[BM25] = None  # BM25 index
        self.bm25_by_source: dict[str, BM25] = {}  # Per-source BM25 indices
        self.bm25_by_category: dict[str, BM25] = {}  # Per-category BM25 indices
        self._lock = threading.Lock()  # Thread safety for parallel indexing
        
    def add_chunk(self, chunk: Chunk):
        """Thread-safe chunk addition."""
        with self._lock:
            self.chunks.append(chunk)
            self.by_source[chunk.source].append(chunk)
            self.by_category[chunk.category].append(chunk)
    
    def build_bm25_indices(self):
        """Build BM25 indices after all chunks are added."""
        if not self.chunks:
            return
        
        # Build main BM25 index
        documents = [f"{c.title} {c.content}" for c in self.chunks]
        self.bm25 = BM25()
        self.bm25.index(documents)
        
        # Build per-source indices
        for source, chunks in self.by_source.items():
            if chunks:
                docs = [f"{c.title} {c.content}" for c in chunks]
                bm25 = BM25()
                bm25.index(docs)
                self.bm25_by_source[source] = bm25
        
        # Build per-category indices
        for category, chunks in self.by_category.items():
            if chunks:
                docs = [f"{c.title} {c.content}" for c in chunks]
                bm25 = BM25()
                bm25.index(docs)
                self.bm25_by_category[category] = bm25
        
    def search(self, query: str, source: Optional[str] = None, 
               category: Optional[str] = None, max_results: int = 10,
               use_expansion: bool = True) -> list[SearchResult]:
        """BM25-based search with filtering and query expansion."""
        
        # Expand query with synonyms for better recall
        if use_expansion:
            expanded_query = expand_query(query)
        else:
            expanded_query = query
        
        # Select chunks and BM25 index to use
        if source and category:
            candidates = [c for c in self.by_source.get(source, []) 
                         if c.category == category]
            # Use source BM25 then filter by category
            bm25_index = self.bm25_by_source.get(source)
            source_chunks = self.by_source.get(source, [])
        elif source:
            candidates = self.by_source.get(source, [])
            bm25_index = self.bm25_by_source.get(source)
            source_chunks = candidates
        elif category:
            candidates = self.by_category.get(category, [])
            bm25_index = self.bm25_by_category.get(category)
            source_chunks = candidates
        else:
            candidates = self.chunks
            bm25_index = self.bm25
            source_chunks = self.chunks
        
        if not candidates or bm25_index is None:
            return []
        
        # Use BM25 for ranking with expanded query
        bm25_results = bm25_index.search(expanded_query, top_k=max_results * 2)  # Get more to filter
        
        results = []
        query_lower = query.lower()  # Use original query for context extraction
        
        for doc_idx, bm25_score in bm25_results:
            if doc_idx >= len(source_chunks):
                continue
                
            chunk = source_chunks[doc_idx]
            
            # Filter by category if both source and category specified
            if source and category and chunk.category != category:
                continue
            
            # Add title boost
            title_lower = chunk.title.lower()
            title_boost = 0.0
            query_terms = set(re.findall(r'\w+', query_lower))
            for term in query_terms:
                if term in title_lower:
                    title_boost += 2.0
            
            final_score = bm25_score + title_boost
            
            # Extract context around first match
            context = ""
            content_lower = chunk.content.lower()
            for term in query_terms:
                idx = content_lower.find(term)
                if idx >= 0:
                    start = max(0, idx - 100)
                    end = min(len(chunk.content), idx + 200)
                    context = "..." + chunk.content[start:end] + "..."
                    break
            
            results.append(SearchResult(chunk, final_score, context))
            
            if len(results) >= max_results:
                break
        
        # Sort by score descending (in case title boost changed order)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

# Global index instance
index = RAGIndex()

# =============================================================================
# Index Caching Functions
# =============================================================================

def get_cache_path() -> Path:
    """Get the path to the cache file."""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / config.cache_file

def get_source_fingerprint() -> str:
    """Generate a fingerprint of source files to detect changes."""
    fingerprint_parts = []
    
    # Check modification times of key source files
    source_paths = [
        Path(config.jsons_dir) / "hip_sheet.json",
        Path(config.jsons_dir) / "triton_sheet.json",
        Path(config.jsons_dir) / "hip_doc.json",
        Path(config.jsons_dir) / "triton_doc.json",
        Path(config.jsons_dir) / "rocm.json",
    ]
    
    for path in source_paths:
        if path.exists():
            mtime = path.stat().st_mtime
            fingerprint_parts.append(f"{path.name}:{mtime}")
    
    # Check PDF directory modification time
    pdf_dir = Path(config.pdf_dir)
    if pdf_dir.exists():
        pdf_count = len(list(pdf_dir.glob("*.pdf")))
        fingerprint_parts.append(f"pdfs:{pdf_count}")
    
    # Check rocm directory modification time (just count directories)
    rocm_dir = Path(config.rocm_dir)
    if rocm_dir.exists():
        repo_count = len([d for d in rocm_dir.iterdir() if d.is_dir()])
        fingerprint_parts.append(f"repos:{repo_count}")
    
    return hashlib.md5("|".join(fingerprint_parts).encode()).hexdigest()

def save_index_cache():
    """Save the index to disk for faster startup."""
    if not config.cache_enabled:
        return
    
    try:
        cache_path = get_cache_path()
        cache_data = {
            "version": 2,  # Increment when cache format changes
            "timestamp": time.time(),
            "fingerprint": get_source_fingerprint(),
            "chunks": index.chunks,
            "by_source": dict(index.by_source),
            "by_category": dict(index.by_category),
            "snippets": index.snippets,
            "heuristics": index.heuristics,
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"  Index cache saved to {cache_path}")
    except Exception as e:
        print(f"  Warning: Failed to save index cache: {e}")

def load_index_cache() -> bool:
    """Load the index from cache if valid. Returns True if successful."""
    if not config.cache_enabled:
        return False
    
    cache_path = get_cache_path()
    
    if not cache_path.exists():
        print("  No cache file found, will build fresh index")
        return False
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check cache version
        if cache_data.get("version") != 2:
            print("  Cache version mismatch, will rebuild")
            return False
        
        # Check cache age
        cache_age_hours = (time.time() - cache_data.get("timestamp", 0)) / 3600
        if cache_age_hours > config.cache_max_age_hours:
            print(f"  Cache expired ({cache_age_hours:.1f}h old), will rebuild")
            return False
        
        # Check if source files changed
        current_fingerprint = get_source_fingerprint()
        if cache_data.get("fingerprint") != current_fingerprint:
            print("  Source files changed, will rebuild")
            return False
        
        # Restore index from cache
        index.chunks = cache_data["chunks"]
        index.by_source = defaultdict(list, cache_data["by_source"])
        index.by_category = defaultdict(list, cache_data["by_category"])
        index.snippets = cache_data["snippets"]
        index.heuristics = cache_data["heuristics"]
        
        print(f"  Loaded {len(index.chunks)} chunks from cache")
        return True
        
    except Exception as e:
        print(f"  Warning: Failed to load cache: {e}")
        return False

# =============================================================================
# Indexing Functions  
# =============================================================================

def generate_chunk_id(content: str, path: str) -> str:
    """Generate a unique chunk ID."""
    h = hashlib.md5((content[:100] + path).encode()).hexdigest()[:12]
    return h

def clean_markdown_content(content: str) -> str:
    """Clean markdown content by removing navigation/UI elements."""
    lines = content.split('\n')
    cleaned = []
    skip_patterns = [
        r'^:::', r'^\[Skip to', r'^Back to top',
        r'^Navigation Menu', r'^\s*\[.*\]\{\.', r'^Toggle navigation',
        r'^\s*-\s*\[.*\]\(.*\)\{\.reference',
    ]
    
    in_nav_block = False
    for line in lines:
        # Skip navigation blocks
        if '::: {.wy-menu' in line or '::: {#navbarSupportedContent' in line:
            in_nav_block = True
            continue
        if in_nav_block and line.strip() == ':::':
            in_nav_block = False
            continue
        if in_nav_block:
            continue
            
        # Skip UI elements
        skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line):
                skip = True
                break
        if not skip:
            cleaned.append(line)
            
    return '\n'.join(cleaned)

def chunk_text(text: str, max_size: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= max_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        
        # Try to break at paragraph or sentence
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + max_size // 2:
                end = para_break
            else:
                # Look for sentence break
                sent_break = text.rfind('. ', start, end)
                if sent_break > start + max_size // 2:
                    end = sent_break + 1
                    
        chunks.append(text[start:end].strip())
        start = end - overlap
        
    return chunks

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF or pdfplumber."""
    try:
        # Try PyMuPDF first (faster and better quality)
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        pass
    except Exception as e:
        print(f"PyMuPDF error for {pdf_path}: {e}")
    
    try:
        # Fallback to pdfplumber
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_parts)
    except ImportError:
        print("Warning: Neither pymupdf nor pdfplumber installed. Cannot extract PDF text.")
        return ""
    except Exception as e:
        print(f"pdfplumber error for {pdf_path}: {e}")
        return ""

def categorize_pdf(filename: str, content: str) -> tuple[str, str]:
    """Categorize a PDF based on filename and content."""
    filename_lower = filename.lower()
    content_lower = content[:5000].lower() if content else ""
    
    # GPU datasheets
    if any(x in filename_lower for x in ['mi300', 'mi325', 'mi350', 'instinct']):
        if 'isa' in filename_lower or 'instruction' in filename_lower:
            return 'isa', 'AMD Instinct ISA Reference'
        elif 'cdna' in filename_lower:
            return 'architecture', 'CDNA Architecture Whitepaper'
        else:
            return 'datasheet', 'AMD Instinct GPU Datasheet'
    
    # ISA documents
    if 'isa' in filename_lower or 'instruction-set' in filename_lower:
        return 'isa', 'Instruction Set Architecture Reference'
    
    # Architecture whitepapers
    if 'cdna' in filename_lower or 'architecture' in filename_lower or 'whitepaper' in filename_lower:
        return 'architecture', 'Architecture Whitepaper'
    
    # ROCm documentation
    if any(x in filename_lower for x in ['hip', 'rocm', 'rocblas', 'miopen', 'wmma', 'rocprim']):
        return 'rocm_docs', 'ROCm Library Documentation'
    
    # Triton
    if 'triton' in filename_lower or 'triton' in content_lower:
        return 'triton', 'Triton GPU Programming'
    
    # Attention/FlashAttention
    if 'attention' in filename_lower or 'flashattn' in filename_lower or 'flash attention' in content_lower:
        return 'attention', 'Attention Mechanism / FlashAttention'
    
    # FP8/Quantization
    if any(x in filename_lower for x in ['fp8', 'microscaling', 'quantiz', 'ofp8']):
        return 'quantization', 'Quantization / Mixed Precision Formats'
    
    # Optimization papers
    if any(x in filename_lower for x in ['optim', 'kernel', 'cuda']) or 'optimization' in content_lower:
        return 'optimization', 'Kernel Optimization'
    
    # Research papers (arxiv format)
    if re.match(r'^\d{4}\.\d+', filename_lower):
        return 'research', 'Research Paper'
    
    return 'general', 'General Reference'

def index_pdf_documents():
    """Index PDF documents for RAG retrieval."""
    pdf_dir = Path(config.pdf_dir)
    
    if not pdf_dir.exists():
        print(f"  PDF directory not found: {pdf_dir}")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"  Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        try:
            # Extract text from PDF
            content = extract_pdf_text(pdf_file)
            
            if len(content) < 100:
                print(f"    Skipping {pdf_file.name} (too short or extraction failed)")
                continue
            
            # Categorize the PDF
            category, doc_type = categorize_pdf(pdf_file.name, content)
            
            # Create a clean title from filename
            title = pdf_file.stem.replace('%28', '(').replace('%29', ')').replace('%20', ' ')
            title = title.replace('-', ' ').replace('_', ' ').title()
            
            # Chunk the content
            text_chunks = chunk_text(content, config.max_chunk_size, config.chunk_overlap)
            
            for i, chunk_content in enumerate(text_chunks):
                chunk = Chunk(
                    id=generate_chunk_id(chunk_content, str(pdf_file)),
                    source='pdf',
                    category=category,
                    title=f"{title} (Part {i+1}/{len(text_chunks)})" if len(text_chunks) > 1 else title,
                    content=chunk_content,
                    file_path=str(pdf_file),
                    metadata={
                        'doc_type': doc_type,
                        'filename': pdf_file.name,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks)
                    }
                )
                index.add_chunk(chunk)
            
            print(f"    Indexed: {pdf_file.name} -> {len(text_chunks)} chunks ({category})")
            
        except Exception as e:
            print(f"    Error indexing {pdf_file.name}: {e}")

def index_library_knowledge():
    """Index the library knowledge base so RAG knows what each library is for."""
    for lib_id, lib_info in LIBRARY_KNOWLEDGE.items():
        content = f"""# {lib_info['name']}

**Purpose:** {lib_info['purpose']}

**Use this library for:**
{chr(10).join('- ' + use for use in lib_info['use_for'])}

**Key Concepts:**
{chr(10).join('- ' + concept for concept in lib_info['key_concepts'])}

**Priority Level:** {lib_info['priority']} (1=essential for kernel optimization, 2=commonly used, 3=specialized)
"""
        
        chunk = Chunk(
            id=f"library_knowledge_{lib_id}",
            source='knowledge',
            category='library_guide',
            title=f"Library Guide: {lib_info['name']}",
            content=content,
            file_path="built-in-knowledge",
            metadata={
                'library_id': lib_id,
                'priority': lib_info['priority'],
                'use_cases': lib_info['use_for']
            }
        )
        index.add_chunk(chunk)

def index_documentation():
    """Index cleaned documentation files."""
    docs_dir = Path(config.docs_dir)
    
    for category_dir in ['hip', 'triton']:
        cat_path = docs_dir / category_dir
        if not cat_path.exists():
            continue
            
        for md_file in cat_path.glob("*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')
                cleaned = clean_markdown_content(content)
                
                if len(cleaned) < 100:  # Skip empty/tiny files
                    continue
                    
                # Read metadata if exists
                meta_file = md_file.with_suffix('.meta.json')
                metadata = {}
                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata = json.load(f)
                        
                title = metadata.get('name', md_file.stem.replace('_', ' ').title())
                
                # Chunk the content
                text_chunks = chunk_text(cleaned, config.max_chunk_size, config.chunk_overlap)
                
                for i, chunk_text_content in enumerate(text_chunks):
                    chunk = Chunk(
                        id=generate_chunk_id(chunk_text_content, str(md_file)),
                        source='docs',
                        category=category_dir,
                        title=f"{title} (Part {i+1})" if len(text_chunks) > 1 else title,
                        content=chunk_text_content,
                        file_path=str(md_file),
                        metadata=metadata
                    )
                    index.add_chunk(chunk)
                    
            except Exception as e:
                print(f"Error indexing {md_file}: {e}")

def index_optimization_snippets():
    """Index optimization snippets from sheet JSONs."""
    jsons_dir = Path(config.jsons_dir)
    
    for sheet_file, category in [
        ('hip_sheet.json', 'hip'),
        ('triton_sheet.json', 'triton')
    ]:
        sheet_path = jsons_dir / sheet_file
        if not sheet_path.exists():
            continue
            
        try:
            with open(sheet_path) as f:
                data = json.load(f)
                
            # Get the main content key
            key = list(data.keys())[0]
            sheet = data[key]
            
            # Index snippets
            for snippet in sheet.get('snippets', []):
                snippet_id = snippet.get('id', '')
                title = snippet.get('title', '')
                clue = snippet.get('clue', '')
                code = snippet.get('snippet', '')
                watch_for = snippet.get('watch_for', [])
                rewrite_moves = snippet.get('rewrite_moves', [])
                
                content = f"""# {title}

**Category:** {snippet.get('category', '')}

**Clue:** {clue}

**Watch for:**
{chr(10).join('- ' + w for w in watch_for)}

**Code Example:**
```
{code}
```

**Optimization Moves:**
{chr(10).join('- ' + m for m in rewrite_moves)}
"""
                
                chunk = Chunk(
                    id=f"snippet_{category}_{snippet_id}",
                    source='snippets',
                    category=category,
                    title=title,
                    content=content,
                    file_path=str(sheet_path),
                    metadata={
                        'snippet_id': snippet_id,
                        'category_type': snippet.get('category', ''),
                        'watch_for': watch_for,
                        'rewrite_moves': rewrite_moves
                    }
                )
                index.add_chunk(chunk)
                index.snippets[f"{category}_{snippet_id}"] = snippet
                
            # Index heuristics
            heuristics = sheet.get('llm_decision_heuristics', [])
            index.heuristics[category] = heuristics
            
            for heuristic in heuristics:
                h_id = heuristic.get('id', '')
                signals = heuristic.get('if_signals', [])
                actions = heuristic.get('then_try', [])
                
                content = f"""# Decision Heuristic: {h_id}

**If you observe:**
{chr(10).join('- ' + s for s in signals)}

**Then try:**
{chr(10).join('- ' + a for a in actions)}
"""
                
                chunk = Chunk(
                    id=f"heuristic_{category}_{h_id}",
                    source='snippets',
                    category=category,
                    title=f"Optimization Playbook: {h_id.replace('_', ' ').title()}",
                    content=content,
                    file_path=str(sheet_path),
                    metadata={'heuristic_id': h_id, 'type': 'decision_heuristic'}
                )
                index.add_chunk(chunk)
                
        except Exception as e:
            print(f"Error indexing {sheet_file}: {e}")

def index_documentation_jsons():
    """Index documentation reference JSONs (hip_doc, triton_doc, rocm)."""
    jsons_dir = Path(config.jsons_dir)
    
    # Index HIP documentation index
    hip_doc_path = jsons_dir / 'hip_doc.json'
    if hip_doc_path.exists():
        try:
            with open(hip_doc_path) as f:
                data = json.load(f)
            
            hip_docs = data.get('hip_docs_index', {})
            
            # Index documentation sets
            for doc_set in hip_docs.get('documentation_sets', []):
                doc_id = doc_set.get('id', '')
                name = doc_set.get('name', '')
                url = doc_set.get('url', '')
                topics = doc_set.get('topics', [])
                rag_value = doc_set.get('rag_value', '')
                
                content = f"""# {name}

**URL:** {url}

**Topics covered:**
{chr(10).join('- ' + t for t in topics)}

**RAG value:** {rag_value}
"""
                chunk = Chunk(
                    id=f"hip_doc_{doc_id}",
                    source='documentation',
                    category='hip',
                    title=name,
                    content=content,
                    file_path=str(hip_doc_path),
                    metadata={'doc_id': doc_id, 'url': url, 'type': 'doc_reference'}
                )
                index.add_chunk(chunk)
                
        except Exception as e:
            print(f"Error indexing hip_doc.json: {e}")
    
    # Index Triton documentation index
    triton_doc_path = jsons_dir / 'triton_doc.json'
    if triton_doc_path.exists():
        try:
            with open(triton_doc_path) as f:
                data = json.load(f)
            
            triton_docs = data.get('triton_docs_index', {})
            
            # Index documentation sets
            for doc_set in triton_docs.get('documentation_sets', []):
                doc_id = doc_set.get('id', '')
                name = doc_set.get('name', '')
                url = doc_set.get('url', '')
                topics = doc_set.get('topics', [])
                rag_value = doc_set.get('rag_value', '')
                
                content = f"""# {name}

**URL:** {url}

**Topics covered:**
{chr(10).join('- ' + t for t in topics)}

**RAG value:** {rag_value}
"""
                chunk = Chunk(
                    id=f"triton_doc_{doc_id}",
                    source='documentation',
                    category='triton',
                    title=name,
                    content=content,
                    file_path=str(triton_doc_path),
                    metadata={'doc_id': doc_id, 'url': url, 'type': 'doc_reference'}
                )
                index.add_chunk(chunk)
                
        except Exception as e:
            print(f"Error indexing triton_doc.json: {e}")
    
    # Index ROCm libraries reference
    rocm_json_path = jsons_dir / 'rocm.json'
    if rocm_json_path.exists():
        try:
            with open(rocm_json_path) as f:
                data = json.load(f)
            
            for lib in data.get('rocm_libraries', []):
                lib_name = lib.get('name', '')
                github = lib.get('github', '')
                description = lib.get('description', '')
                
                content = f"""# ROCm Library: {lib_name}

**GitHub:** {github}

**Description:** {description}

This is a ROCm library that can be used for GPU kernel optimization.
"""
                chunk = Chunk(
                    id=f"rocm_lib_{lib_name}",
                    source='documentation',
                    category='rocm',
                    title=f"ROCm Library: {lib_name}",
                    content=content,
                    file_path=str(rocm_json_path),
                    metadata={'library': lib_name, 'github': github, 'type': 'rocm_library'}
                )
                index.add_chunk(chunk)
                
        except Exception as e:
            print(f"Error indexing rocm.json: {e}")

def index_code_files():
    """Index relevant code files from ROCm repos."""
    rocm_dir = Path(config.rocm_dir)
    
    # Priority repos for kernel optimization
    # Priority repos for kernel optimization (from LIBRARY_KNOWLEDGE)
    priority_repos = [
        # Tier 1: Essential for kernel optimization
        'composable_kernel', 'rocWMMA', 'aiter', 'triton', 'sglang', 'vllm',
        # Tier 2: Commonly used libraries
        'MIOpen', 'rocBLAS', 'hipBLASLt', 'rocPRIM', 'hipCUB', 'rccl',
        # Tier 3: Specialized
        'rocThrust', 'hipTensor', 'hipBLAS', 'AMDMIGraphX'
    ]
    
    for repo_name in priority_repos:
        repo_path = rocm_dir / repo_name
        if not repo_path.exists():
            continue
            
        # Index key directories
        for subdir in ['library', 'include', 'src', 'samples', 'example', 'client_example']:
            target_dir = repo_path / subdir
            if not target_dir.exists():
                continue
                
            for ext in config.code_extensions:
                for code_file in target_dir.rglob(f"*{ext}"):
                    # Skip test files
                    if any(skip in str(code_file) for skip in config.skip_dirs):
                        continue
                        
                    try:
                        content = code_file.read_text(encoding='utf-8', errors='ignore')
                        
                        if len(content) < 50 or len(content) > 50000:
                            continue
                            
                        # Extract file header/docstring for title
                        lines = content.split('\n')[:20]
                        title = code_file.stem
                        for line in lines:
                            if line.strip().startswith('//') or line.strip().startswith('/*'):
                                comment = line.strip().lstrip('/').lstrip('*').strip()
                                if len(comment) > 10:
                                    title = comment[:80]
                                    break
                                    
                        # Chunk code files
                        code_chunks = chunk_text(content, config.max_chunk_size, config.chunk_overlap)
                        
                        for i, chunk_content in enumerate(code_chunks):
                            chunk = Chunk(
                                id=generate_chunk_id(chunk_content, str(code_file)),
                                source='code',
                                category=repo_name.lower(),
                                title=f"{repo_name}/{code_file.name}" + (f" (Part {i+1})" if len(code_chunks) > 1 else ""),
                                content=chunk_content,
                                file_path=str(code_file),
                                metadata={
                                    'repo': repo_name,
                                    'extension': ext,
                                    'relative_path': str(code_file.relative_to(repo_path))
                                }
                            )
                            index.add_chunk(chunk)
                            
                    except Exception as e:
                        pass  # Skip files that can't be read
                        
        # Also index README and docs
        for doc_file in repo_path.glob("*.md"):
            try:
                content = doc_file.read_text(encoding='utf-8')
                if len(content) > 100:
                    chunk = Chunk(
                        id=generate_chunk_id(content[:500], str(doc_file)),
                        source='docs',
                        category=repo_name.lower(),
                        title=f"{repo_name} - {doc_file.stem}",
                        content=content,
                        file_path=str(doc_file),
                        metadata={'repo': repo_name, 'type': 'readme'}
                    )
                    index.add_chunk(chunk)
            except:
                pass

def initialize_index():
    """Initialize the RAG index with all sources using parallel indexing."""
    if index.initialized:
        return
    
    start_time = time.time()
    print("Initializing RAG index...")
    
    # Try to load from cache first
    if load_index_cache():
        print("  Building BM25 search indices from cached data...")
        index.build_bm25_indices()
        index.initialized = True
        elapsed = time.time() - start_time
        print(f"\nIndex ready from cache: {len(index.chunks)} chunks ({elapsed:.1f}s)")
        return
    
    # Build fresh index using parallel execution
    print("  Building fresh index (parallel mode)...")
    
    # Phase 1: Index fast, independent sources in parallel
    # These don't have interdependencies
    fast_indexers = [
        ("Library knowledge", index_library_knowledge),
        ("Optimization snippets", index_optimization_snippets),
        ("Documentation JSONs", index_documentation_jsons),
        ("Documentation", index_documentation),
    ]
    
    print("  Phase 1: Indexing fast sources in parallel...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, func in fast_indexers:
            future = executor.submit(func)
            futures[future] = name
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"    ✓ {name} complete")
            except Exception as e:
                print(f"    ✗ {name} failed: {e}")
    
    # Phase 2: Index slower sources in parallel
    # PDF and code indexing can be slow but are independent
    slow_indexers = [
        ("PDF documents", index_pdf_documents),
        ("Code files", index_code_files),
    ]
    
    print("  Phase 2: Indexing heavy sources in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for name, func in slow_indexers:
            future = executor.submit(func)
            futures[future] = name
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"    ✓ {name} complete")
            except Exception as e:
                print(f"    ✗ {name} failed: {e}")
    
    print("  Building BM25 search indices...")
    index.build_bm25_indices()
    
    # Save to cache for next time
    save_index_cache()
    
    index.initialized = True
    elapsed = time.time() - start_time
    print(f"\nIndex ready: {len(index.chunks)} chunks indexed ({elapsed:.1f}s)")
    print(f"  - Knowledge: {len(index.by_source.get('knowledge', []))}")
    print(f"  - PDFs: {len(index.by_source.get('pdf', []))}")
    print(f"  - Documentation: {len(index.by_source.get('documentation', []))}")
    print(f"  - Docs: {len(index.by_source.get('docs', []))}")
    print(f"  - Snippets: {len(index.by_source.get('snippets', []))}")
    print(f"  - Code: {len(index.by_source.get('code', []))}")

# =============================================================================
# MCP Server
# =============================================================================

app = Server("kernel-optimization-rag")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG tools."""
    return [
        Tool(
            name="search_kernel_optimization",
            description="""Search for kernel optimization techniques, patterns, and best practices.
            
Use this for:
- Finding optimization patterns (memory coalescing, tiling, reduction, etc.)
- Getting code examples for specific optimization techniques
- Understanding performance guidelines for HIP/Triton

Args:
    query: What optimization technique or pattern to search for
    framework: Optional filter - 'hip', 'triton', or 'all' (default)
    source: Optional filter - 'docs', 'snippets', 'code', or 'all' (default)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'memory coalescing', 'shared memory tiling', 'warp shuffle reduction')"
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["hip", "triton", "all"],
                        "default": "all",
                        "description": "Filter by framework"
                    },
                    "source": {
                        "type": "string", 
                        "enum": ["docs", "snippets", "code", "all"],
                        "default": "all",
                        "description": "Filter by source type"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_optimization_snippet",
            description="""Get a specific optimization snippet with code example.
            
Use when you know the pattern name and want the full code example and guidance.

Args:
    pattern: The optimization pattern name
    framework: 'hip' or 'triton'""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern name (e.g., 'coalesced_linear_access', 'vectorized_float4', 'wave_shuffle_reduce', 'autotune_configs')"
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["hip", "triton"],
                        "description": "Framework"
                    }
                },
                "required": ["pattern", "framework"]
            }
        ),
        Tool(
            name="get_optimization_playbook",
            description="""Get optimization playbook/heuristics for a specific scenario.
            
Use when:
- You observe specific performance characteristics (memory-bound, compute-bound, low occupancy)
- You need a systematic approach to optimization
- You want to know what to try next

Args:
    scenario: The optimization scenario
    framework: 'hip' or 'triton'""",
            inputSchema={
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "enum": ["memory_bound", "compute_bound", "low_occupancy", "launch_overhead", "tiling", "autotune"],
                        "description": "The optimization scenario"
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["hip", "triton"],
                        "description": "Framework"
                    }
                },
                "required": ["scenario", "framework"]
            }
        ),
        Tool(
            name="search_library_code",
            description="""Search for code examples in ROCm libraries.
            
Use for:
- Finding implementation patterns in rocWMMA, composable_kernel, MIOpen, etc.
- Understanding how specific operations are implemented
- Getting API usage examples

Args:
    query: What to search for
    library: Optional specific library to search (rocwmma, composable_kernel, miopen, rocblas, etc.)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "library": {
                        "type": "string",
                        "description": "Specific library to search (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_kernel_for_optimization",
            description="""Analyze kernel code and suggest optimizations.
            
Provide kernel code and get optimization suggestions based on patterns in the knowledge base.

Args:
    code: The kernel code to analyze
    framework: 'hip' or 'triton'""",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Kernel code to analyze"
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["hip", "triton"],
                        "description": "Framework"
                    }
                },
                "required": ["code", "framework"]
            }
        ),
        Tool(
            name="list_available_patterns",
            description="""List all available optimization patterns.
            
Use to see what optimization patterns are available in the knowledge base.

Args:
    framework: 'hip', 'triton', or 'all'""",
            inputSchema={
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "enum": ["hip", "triton", "all"],
                        "default": "all"
                    }
                }
            }
        ),
        Tool(
            name="search_gpu_documentation",
            description="""Search AMD GPU reference documentation (PDFs).
            
Use for:
- GPU architecture details (CDNA3, CDNA4)
- ISA instructions (MFMA, vector ops, etc.)
- MI300X/MI325X/MI350X specifications
- FlashAttention implementation details
- FP8/Quantization specifications
- Kernel optimization techniques from papers

Args:
    query: What to search for in documentation
    doc_type: Optional filter - 'isa', 'architecture', 'datasheet', 'attention', 'quantization', 'optimization', 'all'""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'MFMA instruction', 'MI300X memory bandwidth', 'FlashAttention tiling')"
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["isa", "architecture", "datasheet", "attention", "quantization", "optimization", "triton", "rocm_docs", "all"],
                        "default": "all",
                        "description": "Filter by document type"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_library_guide",
            description="""Get guidance on which ROCm library to use for a specific task.
            
Use when:
- You need to choose between libraries (e.g., rocBLAS vs hipBLASLt)
- You want to understand what a library is for
- You need to find the right library for an operation

Args:
    task: Description of what you want to do
    library: Optional specific library to get info about""",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What you want to do (e.g., 'matrix multiplication', 'attention kernel', 'parallel reduction')"
                    },
                    "library": {
                        "type": "string",
                        "description": "Optional: specific library name (composable_kernel, rocWMMA, aiter, etc.)"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="get_gpu_specs",
            description="""Get specifications for AMD Instinct GPUs (MI300X, MI325X, MI350X).
            
Use for:
- Memory bandwidth, HBM capacity
- Compute units, wave size
- MFMA throughput
- Comparing GPU generations

Args:
    gpu: GPU model to query about
    spec_type: Type of spec needed""",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu": {
                        "type": "string",
                        "enum": ["MI300X", "MI325X", "MI350X", "all"],
                        "description": "GPU model"
                    },
                    "spec_type": {
                        "type": "string",
                        "enum": ["memory", "compute", "architecture", "all"],
                        "default": "all",
                        "description": "Type of specification"
                    }
                },
                "required": ["gpu"]
            }
        ),
        Tool(
            name="get_index_status",
            description="""Get the current status of the RAG index.

Shows what documents are indexed, counts by source/category, and cache status.
Use this to understand what knowledge is available for search.

Returns:
- Total chunks indexed
- Breakdown by source (docs, snippets, code, pdf, knowledge)
- Breakdown by category (hip, triton, rocm libraries, etc.)
- Cache status and age
- Available search categories""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    initialize_index()
    
    if name == "search_kernel_optimization":
        return await search_kernel_optimization(
            arguments.get("query", ""),
            arguments.get("framework", "all"),
            arguments.get("source", "all")
        )
    elif name == "get_optimization_snippet":
        return await get_optimization_snippet(
            arguments.get("pattern", ""),
            arguments.get("framework", "hip")
        )
    elif name == "get_optimization_playbook":
        return await get_optimization_playbook(
            arguments.get("scenario", ""),
            arguments.get("framework", "hip")
        )
    elif name == "search_library_code":
        return await search_library_code(
            arguments.get("query", ""),
            arguments.get("library")
        )
    elif name == "analyze_kernel_for_optimization":
        return await analyze_kernel_for_optimization(
            arguments.get("code", ""),
            arguments.get("framework", "hip")
        )
    elif name == "list_available_patterns":
        return await list_available_patterns(
            arguments.get("framework", "all")
        )
    elif name == "search_gpu_documentation":
        return await search_gpu_documentation(
            arguments.get("query", ""),
            arguments.get("doc_type", "all")
        )
    elif name == "get_library_guide":
        return await get_library_guide(
            arguments.get("task", ""),
            arguments.get("library")
        )
    elif name == "get_gpu_specs":
        return await get_gpu_specs(
            arguments.get("gpu", "all"),
            arguments.get("spec_type", "all")
        )
    elif name == "get_index_status":
        return await get_index_status()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# =============================================================================
# Tool Implementations
# =============================================================================

async def search_kernel_optimization(query: str, framework: str, source: str) -> list[TextContent]:
    """Search for optimization techniques."""
    category = None if framework == "all" else framework
    src = None if source == "all" else source
    
    results = index.search(query, source=src, category=category, max_results=10)
    
    if not results:
        return [TextContent(type="text", text=f"No results found for '{query}'. Try different search terms.")]
    
    output = f"# Search Results for: {query}\n\n"
    output += f"Found {len(results)} results:\n\n"
    
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        output += f"## {i}. {chunk.title}\n"
        output += f"**Source:** {chunk.source} | **Category:** {chunk.category}\n"
        output += f"**Score:** {result.score:.2f}\n\n"
        
        # Show content preview
        content_preview = chunk.content[:800] if len(chunk.content) > 800 else chunk.content
        output += f"{content_preview}\n\n"
        output += "---\n\n"
        
    return [TextContent(type="text", text=output)]

async def get_optimization_snippet(pattern: str, framework: str) -> list[TextContent]:
    """Get a specific optimization snippet with fuzzy matching support."""
    key = f"{framework}_{pattern}"
    
    # Try exact match first
    if key in index.snippets:
        snippet = index.snippets[key]
        
        output = f"# {snippet.get('title', pattern)}\n\n"
        output += f"**Category:** {snippet.get('category', '')}\n\n"
        output += f"**Clue:** {snippet.get('clue', '')}\n\n"
        
        watch_for = snippet.get('watch_for', [])
        if watch_for:
            output += "**Watch for (signs you need this optimization):**\n"
            for w in watch_for:
                output += f"- {w}\n"
            output += "\n"
            
        output += "**Code Example:**\n```\n"
        output += snippet.get('snippet', '')
        output += "\n```\n\n"
        
        rewrite_moves = snippet.get('rewrite_moves', [])
        if rewrite_moves:
            output += "**Optimization Moves:**\n"
            for m in rewrite_moves:
                output += f"- {m}\n"
                
        return [TextContent(type="text", text=output)]
    
    # Try fuzzy matching on pattern names
    available_patterns = [k.replace(f"{framework}_", "") 
                          for k in index.snippets.keys() 
                          if k.startswith(f"{framework}_")]
    
    fuzzy_matches = fuzzy_match_pattern(pattern, available_patterns, cutoff=0.4)
    
    if fuzzy_matches:
        # If there's a very close match (>0.8), return that snippet
        best_match, best_score = fuzzy_matches[0]
        if best_score >= 0.8:
            key = f"{framework}_{best_match}"
            snippet = index.snippets[key]
            
            output = f"# {snippet.get('title', best_match)}\n\n"
            output += f"*Note: Matched '{pattern}' to '{best_match}' (similarity: {best_score:.0%})*\n\n"
            output += f"**Category:** {snippet.get('category', '')}\n\n"
            output += f"**Clue:** {snippet.get('clue', '')}\n\n"
            
            watch_for = snippet.get('watch_for', [])
            if watch_for:
                output += "**Watch for (signs you need this optimization):**\n"
                for w in watch_for:
                    output += f"- {w}\n"
                output += "\n"
                
            output += "**Code Example:**\n```\n"
            output += snippet.get('snippet', '')
            output += "\n```\n\n"
            
            rewrite_moves = snippet.get('rewrite_moves', [])
            if rewrite_moves:
                output += "**Optimization Moves:**\n"
                for m in rewrite_moves:
                    output += f"- {m}\n"
                    
            return [TextContent(type="text", text=output)]
        
        # Otherwise suggest similar patterns
        output = f"Pattern '{pattern}' not found. Did you mean one of these?\n\n"
        for match_name, score in fuzzy_matches[:5]:
            snippet = index.snippets.get(f"{framework}_{match_name}", {})
            title = snippet.get('title', match_name)
            output += f"- **{match_name}**: {title} (similarity: {score:.0%})\n"
        output += f"\nUse: `get_optimization_snippet(pattern=\"<pattern_name>\", framework=\"{framework}\")`"
        return [TextContent(type="text", text=output)]
    
    # Fallback to search
    results = index.search(pattern, source='snippets', category=framework, max_results=3)
    
    if results:
        output = f"Pattern '{pattern}' not found exactly. Similar patterns from search:\n\n"
        for result in results:
            output += f"- **{result.chunk.title}** (ID: {result.chunk.metadata.get('snippet_id', '')})\n"
        return [TextContent(type="text", text=output)]
    
    # Show all available patterns
    output = f"No optimization snippet found for pattern '{pattern}' in {framework}.\n\n"
    output += f"**Available {framework.upper()} patterns:**\n"
    for p in sorted(available_patterns):
        output += f"- {p}\n"
    return [TextContent(type="text", text=output)]

async def get_optimization_playbook(scenario: str, framework: str) -> list[TextContent]:
    """Get optimization playbook for a scenario."""
    heuristics = index.heuristics.get(framework, [])
    
    # Map scenario to heuristic IDs
    scenario_map = {
        'memory_bound': ['memory_bound_playbook', 'memory_bound_playbook_triton'],
        'compute_bound': ['compute_bound_playbook'],
        'low_occupancy': ['low_occupancy_playbook'],
        'launch_overhead': ['launch_overhead_playbook'],
        'tiling': ['tiling_playbook_triton'],
        'autotune': ['autotune_playbook_triton']
    }
    
    target_ids = scenario_map.get(scenario, [])
    matching = [h for h in heuristics if h.get('id', '') in target_ids or scenario in h.get('id', '').lower()]
    
    if not matching:
        # Return all heuristics for the framework
        output = f"# Optimization Playbooks for {framework.upper()}\n\n"
        for h in heuristics:
            output += f"## {h.get('id', '').replace('_', ' ').title()}\n\n"
            output += "**If you observe:**\n"
            for s in h.get('if_signals', []):
                output += f"- {s}\n"
            output += "\n**Then try:**\n"
            for a in h.get('then_try', []):
                output += f"- {a}\n"
            output += "\n---\n\n"
        return [TextContent(type="text", text=output)]
    
    output = f"# Optimization Playbook: {scenario.replace('_', ' ').title()}\n\n"
    for h in matching:
        output += "**If you observe:**\n"
        for s in h.get('if_signals', []):
            output += f"- {s}\n"
        output += "\n**Then try:**\n"
        for a in h.get('then_try', []):
            output += f"- {a}\n"
        output += "\n"
        
        if 'doc_pointer' in h:
            dp = h['doc_pointer']
            output += f"\n**Reference:** [{dp.get('name', '')}]({dp.get('url', '')})\n"
            
    return [TextContent(type="text", text=output)]

async def search_library_code(query: str, library: Optional[str]) -> list[TextContent]:
    """Search code in ROCm libraries."""
    category = library.lower() if library else None
    
    results = index.search(query, source='code', category=category, max_results=8)
    
    if not results:
        return [TextContent(type="text", text=f"No code found for '{query}'" + (f" in {library}" if library else ""))]
    
    output = f"# Code Search: {query}\n\n"
    
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        repo = chunk.metadata.get('repo', chunk.category)
        rel_path = chunk.metadata.get('relative_path', chunk.file_path)
        
        output += f"## {i}. {repo}: {rel_path}\n\n"
        output += f"```cpp\n{chunk.content[:1500]}\n```\n\n"
        output += "---\n\n"
        
    return [TextContent(type="text", text=output)]

async def analyze_kernel_for_optimization(code: str, framework: str) -> list[TextContent]:
    """Analyze kernel code and suggest optimizations."""
    suggestions = []
    
    code_lower = code.lower()
    
    # HIP-specific patterns
    if framework == "hip":
        # Basic kernel structure check
        has_kernel = 'threadidx' in code_lower or '__global__' in code_lower
        
        if has_kernel:
            # Memory access patterns
            if '/' in code and 'threadidx' in code_lower:
                suggestions.append(("strength_reduce_index_math", "Integer division in indexing - consider using bit shifts for power-of-2"))
            if '%' in code and 'threadidx' in code_lower:
                suggestions.append(("strength_reduce_index_math", "Modulo operation in indexing - consider using bitwise AND for power-of-2"))
            
            # Divergence check
            if re.search(r'if\s*\([^)]*threadidx', code_lower):
                suggestions.append(("predication_reduce_divergence", "Thread-dependent branch detected - potential warp divergence"))
            
            # Shared memory usage
            if '__shared__' not in code_lower:
                if 'matmul' in code_lower or 'gemm' in code_lower or ('a[' in code_lower and 'b[' in code_lower):
                    suggestions.append(("shared_memory_tiling", "Matrix operation without shared memory - consider LDS tiling for data reuse"))
            
            # Reduction patterns
            if ('sum' in code_lower or 'reduce' in code_lower or '+=' in code) and '__shfl' not in code_lower:
                suggestions.append(("wave_shuffle_reduce", "Reduction without shuffle - consider warp shuffle for efficiency"))
            
            # Vectorized access
            if 'float4' not in code_lower and 'float2' not in code_lower and 'half2' not in code_lower:
                if 'float*' in code_lower or 'float *' in code_lower:
                    suggestions.append(("vectorized_float4", "Scalar float loads - consider float4/float2 for better memory bandwidth"))
            
            # Coalescing check
            if re.search(r'\[\s*\w+\s*\*\s*threadidx', code_lower):
                suggestions.append(("coalesced_linear_access", "Strided memory access pattern detected - ensure coalesced access"))
            
            # Async operations
            if 'hipmemcpy(' in code_lower and 'async' not in code_lower:
                suggestions.append(("async_overlap_streams", "Using synchronous memcpy - consider hipMemcpyAsync with streams"))
            
            # Atomic operations
            if 'atomicadd' in code_lower or 'atomic' in code_lower:
                suggestions.append(("wave_shuffle_reduce", "Atomic operations detected - consider local accumulation first to reduce contention"))
            
            # Loop unrolling
            if 'for' in code_lower and '#pragma unroll' not in code_lower:
                suggestions.append(("blocksize_autotune", "Loops without #pragma unroll - consider unrolling for small fixed-count loops"))
            
            # Grid stride loop
            if 'for' in code_lower and 'griddim' not in code_lower:
                suggestions.append(("grid_stride_loop", "Consider grid-stride loop pattern for better work distribution"))
                
    # Triton-specific patterns
    elif framework == "triton":
        # Mask check
        if 'tl.load' in code_lower:
            if 'mask=' not in code_lower and 'mask =' not in code_lower:
                suggestions.append(("vector_add_masked_load_store", "Missing mask in tl.load - always use masks for bounds safety"))
            if 'tl.multiple_of' not in code_lower and 'tl.max_contiguous' not in code_lower:
                suggestions.append(("compiler_hints_alignment", "No alignment hints - consider tl.multiple_of for better vectorization"))
        
        # Autotune check
        if '@triton.autotune' not in code_lower:
            if 'block' in code_lower or 'BLOCK' in code:
                suggestions.append(("autotune_configs", "No autotune decorator - add @triton.autotune for performance tuning"))
        
        # Reduction patterns
        if 'tl.sum' in code_lower or 'tl.max' in code_lower or 'tl.min' in code_lower:
            suggestions.append(("reduce_then_broadcast_pattern", "Reduction detected - ensure proper axis handling"))
        
        # Software pipelining
        if 'tl.dot' in code_lower or 'matmul' in code_lower:
            if 'num_stages' not in code_lower:
                suggestions.append(("software_pipelining_num_stages", "Matrix operation without num_stages - consider software pipelining"))
        
        # Block pointer for 2D access
        if '2d' in code_lower or 'matmul' in code_lower or 'gemm' in code_lower:
            if 'make_block_ptr' not in code_lower:
                suggestions.append(("2d_tiling_block_ptr", "2D operation without make_block_ptr - consider block pointers for cleaner code"))
        
        # Attention patterns
        if 'attention' in code_lower or 'softmax' in code_lower:
            if 'flash' not in code_lower:
                suggestions.append(("flash_attention_tiling", "Attention kernel detected - consider Flash Attention tiling to avoid O(N^2) memory"))
        
        # FP8 quantization
        if 'fp8' in code_lower or 'float8' in code_lower or 'quantize' in code_lower:
            suggestions.append(("fp8_mixed_precision", "FP8 operations detected - ensure proper scaling for accuracy"))
        
        # Atomic operations
        if 'tl.atomic' in code_lower:
            suggestions.append(("atomic_accumulation", "Atomic operations detected - minimize contention with local accumulation"))
        
        # Constexpr usage
        if ': tl.constexpr' not in code and 'tl.constexpr' not in code:
            if 'BLOCK' in code or 'block' in code_lower:
                suggestions.append(("autotune_configs", "Missing tl.constexpr annotations - use for compile-time constants"))
        
        # Fusion opportunity
        load_count = code_lower.count('tl.load')
        store_count = code_lower.count('tl.store')
        if load_count >= 3 and store_count >= 2:
            suggestions.append(("fuse_ops_to_reduce_bytes", "Multiple loads/stores detected - ensure operations are fused to reduce memory traffic"))
    
    if not suggestions:
        return [TextContent(type="text", text="No specific optimization opportunities detected. Consider profiling to identify bottlenecks.\n\nTips:\n- Use rocprof/omniperf for AMD GPUs\n- Check memory bandwidth utilization\n- Verify occupancy and register usage")]
    
    output = f"# Kernel Analysis ({framework.upper()})\n\n"
    output += f"Found {len(suggestions)} potential optimization opportunities:\n\n"
    
    for pattern_id, reason in suggestions:
        output += f"## {pattern_id.replace('_', ' ').title()}\n"
        output += f"**Observation:** {reason}\n\n"
        
        # Get the snippet details
        key = f"{framework}_{pattern_id}"
        if key in index.snippets:
            snippet = index.snippets[key]
            output += f"**Recommended Pattern:** {snippet.get('title', '')}\n"
            output += f"**Clue:** {snippet.get('clue', '')}\n\n"
            output += "**Example:**\n```\n"
            output += snippet.get('snippet', '')[:500]
            output += "\n```\n\n"
        output += "---\n\n"
        
    return [TextContent(type="text", text=output)]

async def list_available_patterns(framework: str) -> list[TextContent]:
    """List all available optimization patterns."""
    output = "# Available Optimization Patterns\n\n"
    
    frameworks = ['hip', 'triton'] if framework == 'all' else [framework]
    
    for fw in frameworks:
        output += f"## {fw.upper()} Patterns\n\n"
        
        patterns = [k for k in index.snippets.keys() if k.startswith(f"{fw}_")]
        if patterns:
            for pattern_key in patterns:
                snippet = index.snippets[pattern_key]
                pattern_id = pattern_key.replace(f"{fw}_", "")
                output += f"- **{pattern_id}**: {snippet.get('title', '')} ({snippet.get('category', '')})\n"
        else:
            output += "No patterns indexed yet.\n"
        output += "\n"
        
    return [TextContent(type="text", text=output)]

async def search_gpu_documentation(query: str, doc_type: str) -> list[TextContent]:
    """Search GPU reference documentation (PDFs)."""
    category = None if doc_type == "all" else doc_type
    
    results = index.search(query, source='pdf', category=category, max_results=10)
    
    if not results:
        return [TextContent(type="text", text=f"No documentation found for '{query}'. Try different search terms.")]
    
    output = f"# GPU Documentation Search: {query}\n\n"
    output += f"Found {len(results)} relevant documents:\n\n"
    
    for i, result in enumerate(results, 1):
        chunk = result.chunk
        doc_type_str = chunk.metadata.get('doc_type', chunk.category)
        
        output += f"## {i}. {chunk.title}\n"
        output += f"**Type:** {doc_type_str} | **Category:** {chunk.category}\n"
        output += f"**Source:** {chunk.metadata.get('filename', 'unknown')}\n\n"
        
        # Show content preview
        content_preview = chunk.content[:1200] if len(chunk.content) > 1200 else chunk.content
        output += f"{content_preview}\n\n"
        output += "---\n\n"
        
    return [TextContent(type="text", text=output)]

async def get_library_guide(task: str, library: Optional[str] = None) -> list[TextContent]:
    """Get guidance on which library to use."""
    
    if library:
        # Get info about specific library
        lib_key = library.lower().replace('-', '_')
        
        if lib_key in LIBRARY_KNOWLEDGE:
            lib_info = LIBRARY_KNOWLEDGE[lib_key]
            output = f"# {lib_info['name']}\n\n"
            output += f"**Purpose:** {lib_info['purpose']}\n\n"
            output += "## When to Use This Library\n\n"
            for use in lib_info['use_for']:
                output += f"- {use}\n"
            output += "\n## Key Concepts\n\n"
            for concept in lib_info['key_concepts']:
                output += f"- {concept}\n"
            output += f"\n**Priority for Kernel Optimization:** {lib_info['priority']}/3\n"
            
            # Search for relevant code examples
            code_results = index.search(task or lib_key, source='code', category=lib_key, max_results=3)
            if code_results:
                output += "\n## Code Examples in This Library\n\n"
                for result in code_results:
                    output += f"- `{result.chunk.metadata.get('relative_path', result.chunk.title)}`\n"
            
            return [TextContent(type="text", text=output)]
        else:
            # Search for it
            results = index.search(library, source='knowledge', max_results=3)
            if results:
                return [TextContent(type="text", text=results[0].chunk.content)]
            return [TextContent(type="text", text=f"Library '{library}' not found. Available: {', '.join(LIBRARY_KNOWLEDGE.keys())}")]
    
    # Find best library for the task
    task_lower = task.lower()
    recommendations = []
    
    for lib_id, lib_info in LIBRARY_KNOWLEDGE.items():
        score = 0
        matched_uses = []
        
        for use_case in lib_info['use_for']:
            use_lower = use_case.lower()
            for word in task_lower.split():
                if len(word) > 3 and word in use_lower:
                    score += 2
                    if use_case not in matched_uses:
                        matched_uses.append(use_case)
        
        # Check purpose
        if any(word in lib_info['purpose'].lower() for word in task_lower.split() if len(word) > 3):
            score += 1
            
        # Boost by priority (lower priority number = more important)
        score += (4 - lib_info['priority'])
        
        if score > 0:
            recommendations.append((lib_id, lib_info, score, matched_uses))
    
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    output = f"# Library Recommendations for: {task}\n\n"
    
    if not recommendations:
        output += "No specific library recommendations found. Here are the key libraries for kernel optimization:\n\n"
        for lib_id, lib_info in sorted(LIBRARY_KNOWLEDGE.items(), key=lambda x: x[1]['priority']):
            if lib_info['priority'] <= 2:
                output += f"- **{lib_info['name']}**: {lib_info['purpose']}\n"
    else:
        for lib_id, lib_info, score, matched_uses in recommendations[:5]:
            output += f"## {lib_info['name']}\n"
            output += f"**Purpose:** {lib_info['purpose']}\n\n"
            if matched_uses:
                output += "**Relevant for your task:**\n"
                for use in matched_uses[:3]:
                    output += f"- {use}\n"
            output += "\n"
    
    return [TextContent(type="text", text=output)]

async def get_gpu_specs(gpu: str, spec_type: str) -> list[TextContent]:
    """Get GPU specifications from structured database and indexed documentation."""
    
    output = ""
    
    # First, use structured GPU specs database
    gpus_to_show = list(GPU_SPECS.keys()) if gpu == "all" else [gpu]
    
    for gpu_name in gpus_to_show:
        if gpu_name not in GPU_SPECS:
            continue
            
        specs = GPU_SPECS[gpu_name]
        output += f"# {specs['name']} Specifications\n\n"
        output += f"**Generation:** {specs['generation']}\n\n"
        
        # Memory specs
        if spec_type in ["memory", "all"]:
            mem = specs["memory"]
            output += "## Memory\n\n"
            output += f"| Specification | Value |\n|---------------|-------|\n"
            output += f"| HBM Capacity | {mem['hbm_capacity_gb']} GB |\n"
            output += f"| HBM Bandwidth | {mem['hbm_bandwidth_tb_s']} TB/s |\n"
            output += f"| HBM Type | {mem['hbm_type']} |\n"
            output += f"| Memory Interface | {mem['memory_interface_bits']} bits |\n"
            output += f"| Infinity Cache | {mem['infinity_cache_mb']} MB |\n\n"
        
        # Compute specs
        if spec_type in ["compute", "all"]:
            comp = specs["compute"]
            output += "## Compute\n\n"
            output += f"| Specification | Value |\n|---------------|-------|\n"
            output += f"| Compute Units | {comp['compute_units']} |\n"
            output += f"| Stream Processors | {comp['stream_processors']} |\n"
            output += f"| Matrix Cores (MFMA) | {comp['matrix_cores']} |\n"
            output += f"| Peak Clock | {comp['peak_clock_mhz']} MHz |\n"
            output += f"| Peak FP64 | {comp['peak_fp64_tflops']} TFLOPS |\n"
            output += f"| Peak FP32 | {comp['peak_fp32_tflops']} TFLOPS |\n"
            output += f"| Peak FP16/BF16 | {comp['peak_fp16_tflops']} TFLOPS |\n"
            output += f"| Peak FP8 | {comp['peak_fp8_tflops']} TFLOPS |\n"
            if 'peak_fp4_tflops' in comp:
                output += f"| Peak FP4 | {comp['peak_fp4_tflops']} TFLOPS |\n"
            output += f"| Peak INT8 | {comp['peak_int8_tops']} TOPS |\n\n"
        
        # Architecture specs
        if spec_type in ["architecture", "all"]:
            arch = specs["architecture"]
            output += "## Architecture\n\n"
            output += f"| Specification | Value |\n|---------------|-------|\n"
            output += f"| Architecture | {arch['arch']} |\n"
            output += f"| Process Node | {arch['process_node']} |\n"
            output += f"| XCD Count | {arch['xcd_count']} |\n"
            output += f"| Wave Size | {arch['wave_size']} |\n"
            output += f"| LDS per CU | {arch['lds_per_cu_kb']} KB |\n"
            output += f"| VGPRs per SIMD | {arch['vgprs_per_simd']} |\n"
            output += f"| SGPRs per SIMD | {arch['sgprs_per_simd']} |\n"
            output += f"| Max Waves per CU | {arch['max_waves_per_cu']} |\n\n"
            
            # Interconnect
            inter = specs["interconnect"]
            output += "## Interconnect\n\n"
            output += f"| Specification | Value |\n|---------------|-------|\n"
            output += f"| Infinity Fabric Links | {inter['infinity_fabric_links']} |\n"
            output += f"| IF Bandwidth/Link | {inter['if_bandwidth_per_link_gb_s']} GB/s |\n"
            output += f"| PCIe Generation | Gen {inter['pcie_gen']} |\n"
            output += f"| PCIe Lanes | x{inter['pcie_lanes']} |\n\n"
            
            # Power
            pwr = specs["power"]
            output += "## Power\n\n"
            output += f"| Specification | Value |\n|---------------|-------|\n"
            output += f"| TDP | {pwr['tdp_w']} W |\n"
            output += f"| Form Factor | {pwr['form_factor']} |\n\n"
        
        output += "---\n\n"
    
    # Also search PDFs for additional details if specific GPU requested
    if gpu != "all" and spec_type != "all":
        queries = [gpu]
        if spec_type == "memory":
            queries.extend(["HBM", "memory bandwidth"])
        elif spec_type == "compute":
            queries.extend(["MFMA", "TFLOPS"])
        elif spec_type == "architecture":
            queries.extend(["CDNA", "architecture"])
        
        query = " ".join(queries)
        results = index.search(query, source='pdf', max_results=3)
        
        if results:
            output += "## Additional Details from Documentation\n\n"
            for result in results[:2]:
                output += f"**From:** {result.chunk.title}\n\n"
                output += f"{result.chunk.content[:800]}...\n\n"
    
    if not output:
        return [TextContent(type="text", text=f"No specifications found for {gpu}. Available GPUs: MI300X, MI325X, MI350X")]
    
    return [TextContent(type="text", text=output)]


async def get_index_status() -> list[TextContent]:
    """Get the current status of the RAG index."""
    
    output = "# RAG Index Status\n\n"
    
    # Basic stats
    output += "## Summary\n\n"
    output += f"**Total Chunks Indexed:** {len(index.chunks)}\n"
    output += f"**Index Initialized:** {'Yes' if index.initialized else 'No'}\n"
    output += f"**BM25 Index Built:** {'Yes' if index.bm25 is not None else 'No'}\n\n"
    
    # Breakdown by source
    output += "## Chunks by Source\n\n"
    output += "| Source | Count | Description |\n"
    output += "|--------|-------|-------------|\n"
    
    source_descriptions = {
        'knowledge': 'Library knowledge base (what each library is for)',
        'pdf': 'PDF documents (ISA, datasheets, papers)',
        'docs': 'Markdown documentation (HIP, Triton guides)',
        'snippets': 'Optimization snippets with code examples',
        'code': 'Source code from ROCm libraries',
        'documentation': 'Documentation references (URLs, topics)',
    }
    
    for source, chunks in sorted(index.by_source.items(), key=lambda x: len(x[1]), reverse=True):
        desc = source_descriptions.get(source, 'Other content')
        output += f"| {source} | {len(chunks)} | {desc} |\n"
    
    output += "\n"
    
    # Breakdown by category
    output += "## Chunks by Category\n\n"
    output += "| Category | Count |\n"
    output += "|----------|-------|\n"
    
    for category, chunks in sorted(index.by_category.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        output += f"| {category} | {len(chunks)} |\n"
    
    if len(index.by_category) > 20:
        output += f"| ... | ({len(index.by_category) - 20} more categories) |\n"
    
    output += "\n"
    
    # Snippets and heuristics
    output += "## Optimization Patterns\n\n"
    
    hip_snippets = [k for k in index.snippets.keys() if k.startswith('hip_')]
    triton_snippets = [k for k in index.snippets.keys() if k.startswith('triton_')]
    
    output += f"**HIP Optimization Patterns:** {len(hip_snippets)}\n"
    output += f"**Triton Optimization Patterns:** {len(triton_snippets)}\n"
    output += f"**HIP Decision Heuristics:** {len(index.heuristics.get('hip', []))}\n"
    output += f"**Triton Decision Heuristics:** {len(index.heuristics.get('triton', []))}\n\n"
    
    # Cache status
    output += "## Cache Status\n\n"
    cache_path = get_cache_path()
    
    if cache_path.exists():
        import time
        cache_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        output += f"**Cache File:** `{cache_path}`\n"
        output += f"**Cache Age:** {cache_age_hours:.1f} hours\n"
        output += f"**Cache Size:** {cache_size_mb:.1f} MB\n"
        output += f"**Cache Max Age:** {config.cache_max_age_hours} hours\n"
        output += f"**Cache Valid:** {'Yes' if cache_age_hours < config.cache_max_age_hours else 'No (will rebuild on next start)'}\n\n"
    else:
        output += "**Cache:** Not found (will build on first query)\n\n"
    
    # Source directories
    output += "## Source Directories\n\n"
    output += f"- **Docs:** `{config.docs_dir}`\n"
    output += f"- **ROCm Repos:** `{config.rocm_dir}`\n"
    output += f"- **JSON Sheets:** `{config.jsons_dir}`\n"
    output += f"- **PDFs:** `{config.pdf_dir}`\n\n"
    
    # Check which directories exist
    from pathlib import Path
    dirs_status = {
        "docs_dir": Path(config.docs_dir).exists(),
        "rocm_dir": Path(config.rocm_dir).exists(),
        "jsons_dir": Path(config.jsons_dir).exists(),
        "pdf_dir": Path(config.pdf_dir).exists(),
    }
    
    if not all(dirs_status.values()):
        output += "**Warning:** Some source directories don't exist:\n"
        for name, exists in dirs_status.items():
            if not exists:
                output += f"  - {name}: NOT FOUND\n"
        output += "\n"
    
    # Available search categories
    output += "## Available Search Categories\n\n"
    output += "**Sources you can filter by:** " + ", ".join(sorted(index.by_source.keys())) + "\n"
    output += "**Categories you can filter by:** " + ", ".join(sorted(list(index.by_category.keys())[:15])) + "...\n"
    
    return [TextContent(type="text", text=output)]


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
