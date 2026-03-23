#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Kernel Source Finder MCP Server
===============================
Find kernel source code from trace signatures and analyze kernel performance.

Features:
- Demangle C++ kernel names
- Search multiple source locations
- Identify kernel origin and find alternatives
- Handle binary/precompiled kernels
- Classify kernels by type and origin
- Identify optimization hotspots
- Suggest optimization approaches
- Find CK templates by operation signature

Tools:
- find_kernel_source: Search for kernel implementation
- demangle_kernel_name: Demangle C++ signatures
- identify_kernel_origin: Detect source library
- check_source_availability: Check if source exists
- find_library_alternative: Check for optimized library versions
- decode_tensile_kernel: Decode Tensile kernel names
- classify_kernel: Classify kernel by name pattern
- identify_hotspots: Find optimization hotspots from kernel list
- suggest_optimization_approach: Get optimization suggestions for kernel type
- find_ck_template: Find Composable Kernel templates matching operation signature
"""

import asyncio
import subprocess
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# =============================================================================
# Configuration
# =============================================================================

# Derive base paths from this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # mcp_migration_20260120_155504/
ROCM_DIR = PROJECT_ROOT / "rocm"

# Source search locations (relative to project root)
def get_search_paths() -> Dict[str, Path]:
    """Get search paths, resolving relative to project root."""
    paths = {
        # ROCm libraries (cloned repos)
        "composable_kernel": ROCM_DIR / "composable_kernel",
        "rocWMMA": ROCM_DIR / "rocWMMA",
        "MIOpen": ROCM_DIR / "MIOpen",
        "rocBLAS": ROCM_DIR / "rocBLAS",
        "hipBLAS": ROCM_DIR / "hipBLAS",
        "hipBLASLt": ROCM_DIR / "hipBLASLt",
        "rocPRIM": ROCM_DIR / "rocPRIM",
        "hipCUB": ROCM_DIR / "hipCUB",
        "hipTensor": ROCM_DIR / "hipTensor",
        "rocThrust": ROCM_DIR / "rocThrust",
        "rccl": ROCM_DIR / "rccl",
        
        # Inference frameworks
        "vllm": ROCM_DIR / "vllm",
        "sglang": ROCM_DIR / "sglang",
        "aiter": ROCM_DIR / "aiter",
        "triton": ROCM_DIR / "triton",
        
        # MIGraphX
        "AMDMIGraphX": ROCM_DIR / "AMDMIGraphX",
        
        # Tensile (in rocm-libraries)
        "tensile": ROCM_DIR / "rocm-libraries" / "projects" / "hipblas" / "tensilelite",
    }
    
    # Only return paths that exist
    return {k: str(v) for k, v in paths.items() if v.exists()}

# Lazily initialized
_SEARCH_PATHS: Optional[Dict[str, str]] = None

def get_paths() -> Dict[str, str]:
    """Get search paths, initializing if needed."""
    global _SEARCH_PATHS
    if _SEARCH_PATHS is None:
        _SEARCH_PATHS = get_search_paths()
        print(f"Initialized {len(_SEARCH_PATHS)} search paths from {ROCM_DIR}")
    return _SEARCH_PATHS

# For backwards compatibility
SEARCH_PATHS = property(lambda self: get_paths())

# Pattern to origin mapping
ORIGIN_PATTERNS = {
    r"^Cijk_": "tensile",
    r"^ck_tile::": "composable_kernel",
    r"^ck::": "composable_kernel",
    r"^at::native::": "pytorch",
    r"^at::cuda::": "pytorch",
    r"triton_": "triton",
    r"^aiter::": "aiter",
    r"fused_dynamic_mxfp4": "vllm",
    r"marlin_": "vllm",
    r"flash_attn": "flash_attention",
}

# =============================================================================
# Kernel Classification (merged from trace-analyzer)
# =============================================================================

KERNEL_PATTERNS = {
    # Pattern -> (origin, kernel_type, description)
    r"^Cijk_": ("tensile", "gemm", "Tensile-generated GEMM kernel"),
    r"^ck_tile::": ("composable_kernel", "various", "Composable Kernel tile-based"),
    r"^ck::": ("composable_kernel", "various", "Composable Kernel"),
    r"^at::native::": ("pytorch", "various", "PyTorch native kernel"),
    r"^at::cuda::": ("pytorch", "various", "PyTorch CUDA kernel"),
    r"triton_": ("triton", "various", "Triton-generated kernel"),
    r"^aiter::": ("aiter", "various", "AITER inference kernel"),
    r"fused_dynamic_mxfp4": ("vllm", "quantization", "vLLM MXFP4 quantization"),
    r"marlin_": ("vllm", "quantization", "vLLM Marlin quantized GEMM"),
    r"moe_": ("various", "moe", "Mixture of Experts kernel"),
    r"flash_": ("various", "attention", "Flash Attention kernel"),
    r"attention": ("various", "attention", "Attention kernel"),
    r"softmax": ("various", "softmax", "Softmax kernel"),
    r"layernorm|layer_norm|rmsnorm|rms_norm": ("various", "normalization", "Normalization kernel"),
    r"elementwise": ("pytorch", "elementwise", "Elementwise operation"),
    r"reduce|reduction": ("various", "reduction", "Reduction kernel"),
    r"gemm|matmul|dot": ("various", "gemm", "Matrix multiplication"),
    r"copy|memcpy": ("runtime", "memory", "Memory copy"),
}

KERNEL_TYPE_RECOMMENDATIONS = {
    "gemm": {
        "libraries": ["rocBLAS", "hipBLASLt", "rocWMMA", "composable_kernel"],
        "optimizations": [
            "Use MFMA instructions for matrix cores",
            "Tile for shared memory reuse",
            "Consider mixed precision (FP16/BF16/FP8)",
            "Use Tensile tuning for custom shapes"
        ]
    },
    "attention": {
        "libraries": ["composable_kernel (FlashAttention)", "Triton"],
        "optimizations": [
            "Use Flash Attention pattern",
            "Fuse QKV projection with attention",
            "Online softmax to avoid materializing attention matrix",
            "Use MFMA for matmuls"
        ]
    },
    "moe": {
        "libraries": ["composable_kernel (ck_tile MoE)", "vLLM MoE kernels"],
        "optimizations": [
            "Use grouped GEMM for experts",
            "Optimize token routing separately",
            "Consider quantized MoE (MXFP4/INT8)"
        ]
    },
    "quantization": {
        "libraries": ["hipBLASLt", "vLLM Marlin", "composable_kernel"],
        "optimizations": [
            "Use appropriate quantization format (MXFP4, INT8, FP8)",
            "Fuse dequantization with GEMM",
            "Consider block-wise quantization"
        ]
    },
    "normalization": {
        "libraries": ["composable_kernel", "Triton", "PyTorch native"],
        "optimizations": [
            "Fuse with adjacent operations",
            "Use online algorithms for numerics",
            "Wave-level reductions for stats"
        ]
    },
    "reduction": {
        "libraries": ["rocPRIM", "hipCUB", "Triton"],
        "optimizations": [
            "Use wave shuffle for intra-wavefront reduction",
            "Two-stage reduction for large arrays",
            "Consider atomics for small reductions"
        ]
    },
    "elementwise": {
        "libraries": ["Triton", "PyTorch JIT"],
        "optimizations": [
            "Fuse multiple elementwise ops",
            "Use vectorized loads/stores",
            "Memory-bound: maximize bandwidth utilization"
        ]
    },
    "softmax": {
        "libraries": ["composable_kernel", "Triton"],
        "optimizations": [
            "Online softmax (numerically stable)",
            "Fuse with attention if possible",
            "Wave-level reduction for row sums"
        ]
    }
}

@dataclass
class KernelInfo:
    """Information about a kernel from trace."""
    name: str
    time_us: float
    calls: int
    avg_time_us: float
    percentage: float
    origin: str
    kernel_type: str
    description: str
    
    def to_dict(self):
        return asdict(self)

def classify_kernel_by_name(name: str) -> tuple[str, str, str]:
    """Classify kernel by name pattern. Returns (origin, type, description)."""
    for pattern, (origin, ktype, desc) in KERNEL_PATTERNS.items():
        if re.search(pattern, name, re.IGNORECASE):
            return origin, ktype, desc
    return "unknown", "unknown", "Unknown kernel type"

def decode_tensile_name(name: str) -> dict:
    """Decode Tensile kernel naming convention."""
    info = {"raw_name": name, "type": "gemm"}
    
    # Extract macro tile sizes
    mt_match = re.search(r'MT(\d+)x(\d+)x(\d+)', name)
    if mt_match:
        info["tile_m"] = int(mt_match.group(1))
        info["tile_n"] = int(mt_match.group(2))
        info["tile_k"] = int(mt_match.group(3))
        
    # Extract MFMA instruction
    mi_match = re.search(r'MI(\d+)x(\d+)x(\d+)', name)
    if mi_match:
        info["mfma_m"] = int(mi_match.group(1))
        info["mfma_n"] = int(mi_match.group(2))
        info["mfma_k"] = int(mi_match.group(3))
        
    # Check for features
    info["has_bias"] = "Bias" in name
    info["batched"] = "BBS" in name or "Batch" in name
    info["high_accuracy"] = "HA" in name
    
    return info

def analyze_hotspots(kernels: List[Dict], top_n: int = 20) -> List[KernelInfo]:
    """Analyze kernels and identify hotspots."""
    if not kernels:
        return []
        
    total_time = sum(k.get("time_us", 0) for k in kernels)
    
    results = []
    for k in kernels:
        origin, ktype, desc = classify_kernel_by_name(k.get("name", ""))
        
        calls = k.get("calls", 1)
        time_us = k.get("time_us", 0)
        
        info = KernelInfo(
            name=k.get("name", ""),
            time_us=time_us,
            calls=calls,
            avg_time_us=time_us / calls if calls > 0 else 0,
            percentage=(time_us / total_time * 100) if total_time > 0 else 0,
            origin=origin,
            kernel_type=ktype,
            description=desc
        )
        results.append(info)
        
    # Sort by total time descending
    results.sort(key=lambda x: x.time_us, reverse=True)
    
    return results[:top_n]

# Library alternatives for kernel types
LIBRARY_ALTERNATIVES = {
    "gemm": [
        ("rocBLAS", "rocblas_gemm_*", "Optimized BLAS GEMM"),
        ("hipBLASLt", "hipblasLtMatmul", "Flexible GEMM with epilogue fusion"),
        ("composable_kernel", "ck::tensor_operation::device::DeviceGemm*", "Template-based GEMM"),
        ("rocWMMA", "rocwmma::mma_sync", "WMMA/MFMA-based GEMM")
    ],
    "attention": [
        ("composable_kernel", "ck_tile::FlashAttention*", "Flash Attention implementation"),
        ("Triton", "flash_attn_triton.py", "Triton Flash Attention")
    ],
    "moe": [
        ("composable_kernel", "ck_tile::MoeGemm*", "MoE GEMM kernels"),
        ("vllm", "csrc/moe/", "vLLM MoE implementations")
    ],
    "normalization": [
        ("composable_kernel", "ck_tile::*Norm*", "CK normalization kernels"),
        ("Triton", "layer_norm.py", "Triton LayerNorm")
    ],
    "reduction": [
        ("rocPRIM", "rocprim::reduce", "Optimized parallel reduction"),
        ("hipCUB", "hipcub::DeviceReduce", "CUB-style reduction")
    ],
    "quantization": [
        ("vllm", "csrc/quantization/", "vLLM quantization kernels"),
        ("hipBLASLt", "FP8/INT8 GEMM", "Quantized GEMM"),
        ("composable_kernel", "ck_tile::pk_fp4_t", "CK MXFP4 support")
    ]
}

# =============================================================================
# CK Template Database
# =============================================================================

CK_TEMPLATES = {
    "gemm": {
        "DeviceGemm": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp",
            "description": "Basic GEMM using XDL (MFMA) instructions",
            "dtypes": ["fp16", "bf16", "fp32", "fp8", "int8"],
            "layouts": ["RowMajor", "ColumnMajor"],
            "features": ["MFMA", "tile-based"],
            "use_case": "Standard matrix multiplication"
        },
        "DeviceGemmMultiD": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl.hpp",
            "description": "GEMM with multiple epilogue tensors (bias, scaling, etc.)",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["MFMA", "multi-D epilogue", "fused bias/activation"],
            "use_case": "GEMM + bias + activation fusion"
        },
        "DeviceBatchedGemm": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp",
            "description": "Batched GEMM for parallel matrix multiplications",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["MFMA", "batched", "strided"],
            "use_case": "Attention QKV projections, batched operations"
        },
        "DeviceGroupedGemm": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp",
            "description": "Grouped GEMM for variable-size batches",
            "dtypes": ["fp16", "bf16", "fp32", "fp8"],
            "features": ["MFMA", "grouped", "variable sizes"],
            "use_case": "MoE expert GEMMs, ragged batches"
        },
        "DeviceSplitKGemm": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk.hpp",
            "description": "Split-K GEMM for small M/N, large K",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["MFMA", "split-K", "parallel reduction"],
            "use_case": "When K >> M*N, e.g., attention output projection"
        },
    },
    "attention": {
        "DeviceFmha": {
            "path": "include/ck_tile/ops/fmha/pipeline/",
            "description": "Flash Multi-Head Attention (FMHA)",
            "dtypes": ["fp16", "bf16"],
            "features": ["flash attention", "online softmax", "memory efficient"],
            "use_case": "Standard multi-head attention"
        },
        "DeviceFmhaFwd": {
            "path": "include/ck_tile/ops/fmha/block/block_fmha_fwd.hpp",
            "description": "FMHA forward pass with causal masking",
            "dtypes": ["fp16", "bf16"],
            "features": ["causal mask", "ALiBi", "RoPE compatible"],
            "use_case": "Causal/decoder attention"
        },
        "DeviceFmhaBwd": {
            "path": "include/ck_tile/ops/fmha/block/block_fmha_bwd.hpp",
            "description": "FMHA backward pass for training",
            "dtypes": ["fp16", "bf16"],
            "features": ["gradient computation", "checkpointing"],
            "use_case": "Training attention layers"
        },
        "DevicePagedAttention": {
            "path": "include/ck_tile/ops/fmha/",
            "description": "Paged attention for KV cache",
            "dtypes": ["fp16", "bf16"],
            "features": ["paged KV", "block tables", "variable length"],
            "use_case": "LLM inference with KV cache"
        },
    },
    "moe": {
        "DeviceMoeGemm": {
            "path": "include/ck_tile/ops/moe_gemm/",
            "description": "Mixture of Experts GEMM",
            "dtypes": ["fp16", "bf16", "fp8"],
            "features": ["grouped GEMM", "token routing", "expert parallel"],
            "use_case": "MoE FFN layers"
        },
        "DeviceMoeSorting": {
            "path": "include/ck_tile/ops/moe_sorting/",
            "description": "MoE token sorting and dispatch",
            "dtypes": ["fp16", "bf16", "int32"],
            "features": ["topk", "sorting", "scatter/gather"],
            "use_case": "Token routing to experts"
        },
    },
    "normalization": {
        "DeviceLayernorm": {
            "path": "include/ck_tile/ops/layernorm/",
            "description": "Layer Normalization",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["online algorithm", "fused scale/bias"],
            "use_case": "Transformer layer normalization"
        },
        "DeviceRmsnorm": {
            "path": "include/ck_tile/ops/rmsnorm/",
            "description": "RMS Normalization",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["no mean computation", "faster than layernorm"],
            "use_case": "LLaMA-style models"
        },
        "DeviceGroupnorm": {
            "path": "include/ck_tile/ops/groupnorm/",
            "description": "Group Normalization",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["group-wise stats", "NHWC support"],
            "use_case": "Vision models, diffusion"
        },
    },
    "reduction": {
        "DeviceReduce": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_reduce.hpp",
            "description": "General reduction operations",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["sum", "max", "min", "prod"],
            "use_case": "Loss computation, statistics"
        },
        "DeviceSoftmax": {
            "path": "include/ck_tile/ops/softmax/",
            "description": "Softmax reduction",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["numerically stable", "online algorithm"],
            "use_case": "Attention softmax, classifier output"
        },
    },
    "elementwise": {
        "DeviceElementwise": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_elementwise.hpp",
            "description": "Fused elementwise operations",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["custom functors", "N-ary", "broadcast"],
            "use_case": "Fuse multiple elementwise ops"
        },
        "DeviceBinaryElementwise": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_binary_elementwise.hpp",
            "description": "Binary operations (add, mul, etc.)",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["broadcast", "in-place"],
            "use_case": "Residual add, scaling"
        },
    },
    "convolution": {
        "DeviceConv2dFwd": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl.hpp",
            "description": "2D Convolution forward",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["MFMA", "implicit GEMM", "NHWC"],
            "use_case": "Vision model convolutions"
        },
        "DeviceConv2dBwdData": {
            "path": "include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl.hpp",
            "description": "2D Convolution backward data",
            "dtypes": ["fp16", "bf16", "fp32"],
            "features": ["MFMA", "gradient wrt input"],
            "use_case": "Training convolutions"
        },
    },
    "quantization": {
        "DeviceQuantize": {
            "path": "include/ck_tile/ops/quantize/",
            "description": "Quantization kernels",
            "dtypes": ["fp8_e4m3", "fp8_e5m2", "int8", "mxfp4"],
            "features": ["per-tensor", "per-channel", "block-wise"],
            "use_case": "Weight/activation quantization"
        },
        "DeviceDequantizeGemm": {
            "path": "include/ck/tensor_operation/gpu/device/",
            "description": "Fused dequantize + GEMM",
            "dtypes": ["fp8", "int8", "mxfp4"],
            "features": ["fused dequant", "scaled GEMM"],
            "use_case": "Quantized inference GEMM"
        },
    },
}


def find_ck_templates(operation: str, dtype: str = None, features: List[str] = None) -> List[Dict]:
    """
    Find CK templates matching the operation signature.
    
    Args:
        operation: Operation type (gemm, attention, moe, normalization, reduction, elementwise)
        dtype: Optional data type filter (fp16, bf16, fp32, fp8, int8)
        features: Optional list of required features
        
    Returns:
        List of matching templates with paths and descriptions
    """
    results = []
    
    # Normalize operation name
    op_lower = operation.lower()
    op_aliases = {
        "matmul": "gemm",
        "linear": "gemm",
        "fmha": "attention",
        "flash_attention": "attention",
        "layernorm": "normalization",
        "rmsnorm": "normalization",
        "groupnorm": "normalization",
        "softmax": "reduction",
        "conv": "convolution",
        "conv2d": "convolution",
    }
    op_category = op_aliases.get(op_lower, op_lower)
    
    if op_category not in CK_TEMPLATES:
        # Try to find partial match
        for cat in CK_TEMPLATES:
            if op_lower in cat or cat in op_lower:
                op_category = cat
                break
    
    if op_category not in CK_TEMPLATES:
        return results
    
    templates = CK_TEMPLATES[op_category]
    
    for name, info in templates.items():
        match = True
        match_score = 0
        
        # Check dtype if specified
        if dtype:
            dtype_lower = dtype.lower()
            dtype_matches = any(dtype_lower in d.lower() for d in info.get("dtypes", []))
            if not dtype_matches:
                match = False
            else:
                match_score += 10
        
        # Check features if specified
        if features and match:
            template_features = [f.lower() for f in info.get("features", [])]
            for feat in features:
                if any(feat.lower() in tf for tf in template_features):
                    match_score += 5
                # Features are optional, don't fail match
        
        if match:
            results.append({
                "name": name,
                "category": op_category,
                "path": info["path"],
                "description": info["description"],
                "dtypes": info.get("dtypes", []),
                "features": info.get("features", []),
                "use_case": info.get("use_case", ""),
                "match_score": match_score,
            })
    
    # Sort by match score
    results.sort(key=lambda x: x["match_score"], reverse=True)
    
    return results


# =============================================================================
# Demangling
# =============================================================================

def demangle_cxx_name(mangled: str) -> str:
    """Demangle C++ name using c++filt."""
    if not mangled.startswith("_Z"):
        return mangled
        
    try:
        result = subprocess.run(
            ["c++filt", mangled],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
        
    return mangled

def extract_kernel_name(full_name: str) -> str:
    """Extract clean kernel name from potentially mangled name."""
    # Demangle if needed
    if full_name.startswith("_Z"):
        full_name = demangle_cxx_name(full_name)
        
    # Remove template parameters for searching
    clean = re.sub(r'<[^>]+>', '', full_name)
    
    # Get the last component (function name)
    parts = clean.split("::")
    return parts[-1] if parts else clean

# =============================================================================
# Source Search
# =============================================================================

def identify_kernel_origin(kernel_name: str) -> Tuple[str, str, bool]:
    """
    Identify kernel origin.
    Returns: (origin, kernel_type, source_available)
    """
    # Check patterns
    for pattern, origin in ORIGIN_PATTERNS.items():
        if re.search(pattern, kernel_name, re.IGNORECASE):
            # Determine kernel type
            ktype = "unknown"
            if "gemm" in kernel_name.lower() or kernel_name.startswith("Cijk_"):
                ktype = "gemm"
            elif "attention" in kernel_name.lower() or "flash" in kernel_name.lower():
                ktype = "attention"
            elif "moe" in kernel_name.lower():
                ktype = "moe"
            elif "norm" in kernel_name.lower():
                ktype = "normalization"
            elif "reduce" in kernel_name.lower():
                ktype = "reduction"
            elif "quant" in kernel_name.lower() or "mxfp" in kernel_name.lower():
                ktype = "quantization"
                
            # Check if source is typically available
            source_available = origin not in ["aiter", "binary"]
            if origin == "tensile":
                source_available = False  # Generated, not hand-written
                
            return origin, ktype, source_available
            
    return "unknown", "unknown", True

def search_source_files(kernel_name: str, search_path: str, max_results: int = 5) -> List[Dict]:
    """Search for kernel source in a directory."""
    results = []
    path = Path(search_path)
    
    if not path.exists():
        return results
        
    # Clean kernel name for searching
    search_term = extract_kernel_name(kernel_name)
    
    # Search patterns
    patterns = [search_term]
    if "::" in kernel_name:
        # Add namespace components
        parts = kernel_name.split("::")
        patterns.extend(parts)
        
    # Search in source files
    extensions = [".cpp", ".hpp", ".h", ".cu", ".hip", ".py"]
    
    try:
        for pattern in patterns[:3]:  # Limit patterns
            if len(results) >= max_results:
                break
                
            # Use grep for fast searching
            result = subprocess.run(
                ["grep", "-rl", "--include=*.cpp", "--include=*.hpp", 
                 "--include=*.h", "--include=*.py", "--include=*.cu",
                 pattern, str(path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            for file_path in result.stdout.strip().split('\n')[:max_results]:
                if file_path and file_path not in [r.get("path") for r in results]:
                    # Read first match context
                    try:
                        content = Path(file_path).read_text(errors='ignore')
                        # Find the line with the pattern
                        for i, line in enumerate(content.split('\n')):
                            if pattern in line:
                                start = max(0, i - 2)
                                end = min(len(content.split('\n')), i + 10)
                                context = '\n'.join(content.split('\n')[start:end])
                                results.append({
                                    "path": file_path,
                                    "line": i + 1,
                                    "context": context[:500]
                                })
                                break
                    except:
                        results.append({"path": file_path, "line": 1, "context": ""})
                        
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
        
    return results[:max_results]

def find_all_sources(kernel_name: str) -> Dict[str, List[Dict]]:
    """Search all known source locations for kernel."""
    origin, ktype, _ = identify_kernel_origin(kernel_name)
    search_paths = get_paths()
    
    all_results = {}
    
    # Prioritize likely locations based on origin
    priority_paths = []
    if origin == "composable_kernel":
        priority_paths = ["composable_kernel"]
    elif origin == "pytorch":
        priority_paths = ["pytorch"]
    elif origin == "vllm":
        priority_paths = ["vllm", "sglang"]
    elif origin == "aiter":
        priority_paths = ["aiter", "composable_kernel", "triton"]
    elif origin == "tensile":
        priority_paths = ["tensile", "rocBLAS", "hipBLAS"]
    elif origin == "triton":
        priority_paths = ["triton", "sglang", "vllm"]
    else:
        priority_paths = list(search_paths.keys())
        
    for name in priority_paths:
        if name in search_paths:
            path = search_paths[name]
            results = search_source_files(kernel_name, path)
            if results:
                all_results[name] = results
                
    return all_results

# =============================================================================
# MCP Server
# =============================================================================

app = Server("source-finder")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available source finder tools."""
    return [
        Tool(
            name="find_kernel_source",
            description="""Search for kernel source code implementation.

Given a kernel name (from profiler trace), searches:
- composable_kernel, rocWMMA, MIOpen
- vLLM, SGLang custom kernels
- PyTorch native kernels
- Tensile configs

Returns file paths, line numbers, and code context.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Kernel name from trace (can be mangled)"
                    },
                    "search_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: specific paths to search"
                    }
                },
                "required": ["kernel_name"]
            }
        ),
        Tool(
            name="demangle_kernel_name",
            description="""Demangle a C++ mangled kernel name.

Converts names like _ZN7ck_tile6kentryILi1E... to readable form.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "mangled_name": {
                        "type": "string",
                        "description": "Mangled C++ name"
                    }
                },
                "required": ["mangled_name"]
            }
        ),
        Tool(
            name="identify_kernel_origin",
            description="""Identify the origin/source library of a kernel.

Returns:
- Origin library (composable_kernel, pytorch, tensile, vllm, etc.)
- Kernel type (gemm, attention, moe, etc.)
- Whether source is available""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Kernel name"
                    }
                },
                "required": ["kernel_name"]
            }
        ),
        Tool(
            name="check_source_availability",
            description="""Check if source code is available for a kernel.

For binary/precompiled kernels, suggests alternatives.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Kernel name"
                    }
                },
                "required": ["kernel_name"]
            }
        ),
        Tool(
            name="find_library_alternative",
            description="""Find optimized library alternatives for a kernel operation.

Given a kernel type (gemm, attention, etc.), returns:
- Library alternatives (rocBLAS, CK, Triton, etc.)
- API/function names
- When to use each""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_type": {
                        "type": "string",
                        "enum": ["gemm", "attention", "moe", "normalization", 
                                "reduction", "quantization", "elementwise"],
                        "description": "Type of kernel operation"
                    },
                    "current_origin": {
                        "type": "string",
                        "description": "Optional: current library to find alternatives for"
                    }
                },
                "required": ["kernel_type"]
            }
        ),
        Tool(
            name="decode_tensile_kernel",
            description="""Decode a Tensile kernel name to extract configuration.

Returns tile sizes, MFMA config, data types, and features.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Tensile kernel name (e.g., Cijk_Alik_Bljk_...)"
                    }
                },
                "required": ["kernel_name"]
            }
        ),
        Tool(
            name="classify_kernel",
            description="""Classify a kernel by its name.

Returns:
- Origin (pytorch, composable_kernel, tensile, triton, vllm, etc.)
- Type (gemm, attention, reduction, elementwise, etc.)
- Description

Also decodes Tensile kernel names if applicable.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Kernel name to classify"
                    }
                },
                "required": ["kernel_name"]
            }
        ),
        Tool(
            name="identify_hotspots",
            description="""Identify optimization hotspots from a list of kernel names and times.

Input can be a simple list of kernel info or output from parse_* tools.

Returns ranked list with classification and recommendations.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "time_us": {"type": "number"},
                                "calls": {"type": "integer"}
                            }
                        },
                        "description": "List of kernel info objects"
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 10
                    }
                },
                "required": ["kernels"]
            }
        ),
        Tool(
            name="suggest_optimization_approach",
            description="""Get optimization suggestions for a kernel type.

Based on kernel classification, returns:
- Recommended libraries
- Optimization strategies
- Implementation hints""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_type": {
                        "type": "string",
                        "enum": ["gemm", "attention", "moe", "quantization", 
                                "normalization", "reduction", "elementwise", "softmax"],
                        "description": "Type of kernel"
                    },
                    "kernel_name": {
                        "type": "string",
                        "description": "Optional: specific kernel name for more targeted suggestions"
                    }
                },
                "required": ["kernel_type"]
            }
        ),
        Tool(
            name="find_ck_template",
            description="""Find Composable Kernel (CK) templates matching an operation signature.

Given an operation type, data type, and optional features, returns matching CK templates
with file paths, descriptions, and use cases.

USE THIS to find the right CK implementation for a kernel you want to optimize.

Examples:
- "gemm" + "fp16" -> DeviceGemm, DeviceGemmMultiD, DeviceBatchedGemm
- "attention" + "bf16" + ["causal"] -> DeviceFmhaFwd
- "moe" + "fp8" -> DeviceMoeGemm

Args:
    operation: Operation type (gemm, attention, moe, normalization, reduction, elementwise, convolution, quantization)
    dtype: Optional data type (fp16, bf16, fp32, fp8, int8, mxfp4)
    features: Optional list of features to match (e.g., ["batched", "causal", "fused bias"])""",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["gemm", "matmul", "attention", "fmha", "moe", 
                                "normalization", "layernorm", "rmsnorm", "reduction", 
                                "softmax", "elementwise", "convolution", "quantization"],
                        "description": "Operation type to find templates for"
                    },
                    "dtype": {
                        "type": "string",
                        "enum": ["fp16", "bf16", "fp32", "fp8", "int8", "mxfp4"],
                        "description": "Optional: data type filter"
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: required features (batched, causal, fused, split-k, grouped, etc.)"
                    }
                },
                "required": ["operation"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "find_kernel_source":
        return await handle_find_source(arguments)
    elif name == "demangle_kernel_name":
        return await handle_demangle(arguments)
    elif name == "identify_kernel_origin":
        return await handle_identify_origin(arguments)
    elif name == "check_source_availability":
        return await handle_check_availability(arguments)
    elif name == "find_library_alternative":
        return await handle_find_alternative(arguments)
    elif name == "decode_tensile_kernel":
        return await handle_decode_tensile(arguments)
    elif name == "classify_kernel":
        return await handle_classify_kernel(arguments)
    elif name == "identify_hotspots":
        return await handle_identify_hotspots(arguments)
    elif name == "suggest_optimization_approach":
        return await handle_suggest_optimization(arguments)
    elif name == "find_ck_template":
        return await handle_find_ck_template(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# =============================================================================
# Tool Implementations  
# =============================================================================

async def handle_find_source(args: dict) -> list[TextContent]:
    """Find kernel source."""
    kernel_name = args.get("kernel_name", "")
    if not kernel_name:
        return [TextContent(type="text", text="No kernel name provided")]
        
    # Demangle if needed
    demangled = demangle_cxx_name(kernel_name)
    origin, ktype, source_available = identify_kernel_origin(demangled)
    
    output = f"# Source Search: `{kernel_name[:60]}...`\n\n"
    
    if demangled != kernel_name:
        output += f"**Demangled:** `{demangled[:80]}...`\n\n"
        
    output += f"**Origin:** {origin}\n"
    output += f"**Type:** {ktype}\n"
    output += f"**Source Available:** {'Yes' if source_available else 'No (generated/binary)'}\n\n"
    
    if not source_available:
        output += "## Note\n\n"
        if origin == "tensile":
            output += "This is a **Tensile-generated** kernel. Source is generated from YAML configs.\n"
            output += "To optimize: tune Tensile configs or use composable_kernel alternative.\n\n"
        elif origin == "aiter":
            output += "This is an **AITER binary** kernel. Source may not be available.\n"
            output += "Consider using Triton or CK alternatives.\n\n"
            
    # Search for sources
    all_sources = find_all_sources(demangled)
    
    if all_sources:
        output += "## Source Files Found\n\n"
        for lib, sources in all_sources.items():
            output += f"### {lib}\n\n"
            for src in sources:
                output += f"**File:** `{src['path']}`\n"
                output += f"**Line:** {src['line']}\n"
                if src.get('context'):
                    output += f"```cpp\n{src['context']}\n```\n"
                output += "\n"
    else:
        output += "## No Source Found\n\n"
        output += "Source not found in indexed locations. Consider:\n"
        output += "- Kernel may be generated at runtime\n"
        output += "- Kernel may be in a non-indexed location\n"
        output += f"- Use `find_library_alternative` to find alternative implementations\n"
        
    return [TextContent(type="text", text=output)]

async def handle_demangle(args: dict) -> list[TextContent]:
    """Demangle C++ name."""
    mangled = args.get("mangled_name", "")
    demangled = demangle_cxx_name(mangled)
    
    output = f"# Demangled Name\n\n"
    output += f"**Mangled:** `{mangled}`\n\n"
    output += f"**Demangled:** `{demangled}`\n"
    
    return [TextContent(type="text", text=output)]

async def handle_identify_origin(args: dict) -> list[TextContent]:
    """Identify kernel origin."""
    kernel_name = args.get("kernel_name", "")
    demangled = demangle_cxx_name(kernel_name)
    origin, ktype, source_available = identify_kernel_origin(demangled)
    
    output = f"# Kernel Origin\n\n"
    output += f"**Kernel:** `{kernel_name[:60]}`\n\n"
    output += f"| Property | Value |\n"
    output += f"|----------|-------|\n"
    output += f"| Origin | {origin} |\n"
    output += f"| Type | {ktype} |\n"
    output += f"| Source Available | {'Yes' if source_available else 'No'} |\n"
    
    # Add origin-specific info
    search_paths = get_paths()
    if origin == "tensile":
        output += "\n## Tensile Kernel\n\n"
        output += "- Generated from YAML solution files\n"
        if "tensile" in search_paths:
            output += f"- Source in: `{search_paths['tensile']}`\n"
        output += "- Optimization: Tune via Tensile or use CK alternative\n"
    elif origin == "composable_kernel":
        output += "\n## Composable Kernel\n\n"
        output += "- Template-based high-performance kernels\n"
        if "composable_kernel" in search_paths:
            output += f"- Source in: `{search_paths['composable_kernel']}`\n"
        output += "- Optimization: Tune template parameters or tile sizes\n"
    elif origin == "vllm":
        output += "\n## vLLM Kernel\n\n"
        output += "- Custom inference kernels\n"
        if "vllm" in search_paths:
            output += f"- Source in: `{search_paths['vllm']}/csrc/`\n"
    elif origin == "aiter":
        output += "\n## AITER Kernel\n\n"
        output += "- AMD inference optimization kernels\n"
        if "aiter" in search_paths:
            output += f"- Source in: `{search_paths['aiter']}`\n"
        else:
            output += "- May be binary/precompiled\n"
        output += "- Consider Triton or CK alternatives if source not available\n"
    elif origin == "triton":
        output += "\n## Triton Kernel\n\n"
        output += "- Python-based GPU kernel\n"
        if "triton" in search_paths:
            output += f"- Source in: `{search_paths['triton']}`\n"
        output += "- Optimization: Tune autotuning configs or block sizes\n"
        
    return [TextContent(type="text", text=output)]

async def handle_check_availability(args: dict) -> list[TextContent]:
    """Check source availability."""
    kernel_name = args.get("kernel_name", "")
    origin, ktype, source_available = identify_kernel_origin(kernel_name)
    
    output = f"# Source Availability Check\n\n"
    output += f"**Kernel:** `{kernel_name[:60]}`\n\n"
    
    if source_available:
        output += "**Status:** Source code is available\n\n"
        sources = find_all_sources(kernel_name)
        if sources:
            output += "**Found in:**\n"
            for lib, files in sources.items():
                output += f"- {lib}: {len(files)} file(s)\n"
        else:
            output += "Note: Source not found in indexed locations but may exist elsewhere.\n"
    else:
        output += "**Status:** Source code NOT directly available\n\n"
        output += "**Reason:** "
        if origin == "tensile":
            output += "Tensile kernels are generated from YAML configs, not hand-written.\n"
        elif origin == "aiter":
            output += "AITER kernels may be precompiled binaries.\n"
        else:
            output += "Kernel may be binary or generated at runtime.\n"
            
        output += "\n## Alternatives\n\n"
        if ktype in LIBRARY_ALTERNATIVES:
            for lib, api, desc in LIBRARY_ALTERNATIVES[ktype]:
                output += f"- **{lib}**: `{api}` - {desc}\n"
                
    return [TextContent(type="text", text=output)]

async def handle_find_alternative(args: dict) -> list[TextContent]:
    """Find library alternatives."""
    ktype = args.get("kernel_type", "")
    current = args.get("current_origin", "")
    
    if ktype not in LIBRARY_ALTERNATIVES:
        available = list(LIBRARY_ALTERNATIVES.keys())
        return [TextContent(type="text", text=f"Unknown kernel type: {ktype}\nAvailable: {available}")]
        
    alternatives = LIBRARY_ALTERNATIVES[ktype]
    
    output = f"# Library Alternatives for {ktype.upper()}\n\n"
    
    if current:
        output += f"**Current:** {current}\n\n"
        
    output += "## Available Alternatives\n\n"
    
    search_paths = get_paths()
    for lib, api, desc in alternatives:
        if lib.lower() == current.lower():
            continue  # Skip current
        output += f"### {lib}\n\n"
        output += f"- **API/Pattern:** `{api}`\n"
        output += f"- **Description:** {desc}\n"
        
        # Add source location if known
        lib_key = lib.lower().replace(" ", "_")
        if lib_key in search_paths:
            output += f"- **Source:** `{search_paths[lib_key]}`\n"
        output += "\n"
        
    return [TextContent(type="text", text=output)]

async def handle_decode_tensile(args: dict) -> list[TextContent]:
    """Decode Tensile kernel name."""
    name = args.get("kernel_name", "")
    
    if not name.startswith("Cijk"):
        return [TextContent(type="text", text="Not a Tensile kernel (doesn't start with Cijk)")]
        
    output = f"# Tensile Kernel Decoding\n\n"
    output += f"**Kernel:** `{name}`\n\n"
    
    # Parse components
    info = {}
    
    # Matrix indexing
    idx_match = re.match(r'Cijk_([A-Za-z]+)_([A-Za-z]+)', name)
    if idx_match:
        info["A_indexing"] = idx_match.group(1)
        info["B_indexing"] = idx_match.group(2)
        
    # Macro tile sizes
    mt_match = re.search(r'MT(\d+)x(\d+)x(\d+)', name)
    if mt_match:
        info["tile_M"] = int(mt_match.group(1))
        info["tile_N"] = int(mt_match.group(2))
        info["tile_K"] = int(mt_match.group(3))
        
    # MFMA instruction
    mi_match = re.search(r'MI(\d+)x(\d+)x(\d+)', name)
    if mi_match:
        info["mfma_M"] = int(mi_match.group(1))
        info["mfma_N"] = int(mi_match.group(2))
        info["mfma_K"] = int(mi_match.group(3))
        
    # Features
    info["has_bias"] = "Bias" in name
    info["batched"] = "BBS" in name or "Batched" in name
    info["high_accuracy"] = "_HA_" in name or "HighAccuracy" in name
    info["strided"] = "_S_" in name
    info["atomic"] = "Atomic" in name
    
    output += "## Configuration\n\n"
    output += "| Parameter | Value |\n"
    output += "|-----------|-------|\n"
    for k, v in info.items():
        output += f"| {k} | {v} |\n"
        
    output += "\n## Optimization Notes\n\n"
    if info.get("tile_M"):
        output += f"- Macro tile: {info['tile_M']}x{info['tile_N']}x{info['tile_K']}\n"
    if info.get("mfma_M"):
        output += f"- Using MFMA: {info['mfma_M']}x{info['mfma_N']}x{info['mfma_K']}\n"
    if info.get("has_bias"):
        output += "- Bias fusion enabled\n"
    if info.get("batched"):
        output += "- Batched GEMM mode\n"
        
    output += "\n**To optimize:** Tune via Tensile YAML configs or consider composable_kernel alternative.\n"
    
    return [TextContent(type="text", text=output)]

# =============================================================================
# New Tool Handlers (merged from trace-analyzer)
# =============================================================================

async def handle_classify_kernel(args: dict) -> list[TextContent]:
    """Classify a single kernel."""
    name = args.get("kernel_name", "")
    if not name:
        return [TextContent(type="text", text="No kernel name provided")]
        
    origin, ktype, desc = classify_kernel_by_name(name)
    
    output = f"# Kernel Classification\n\n"
    output += f"**Kernel:** `{name}`\n\n"
    output += f"| Property | Value |\n"
    output += f"|----------|-------|\n"
    output += f"| Origin | {origin} |\n"
    output += f"| Type | {ktype} |\n"
    output += f"| Description | {desc} |\n\n"
    
    # Decode Tensile name if applicable
    if origin == "tensile" or name.startswith("Cijk_"):
        output += "## Tensile Kernel Decoding\n\n"
        info = decode_tensile_name(name)
        for key, val in info.items():
            if key != "raw_name":
                output += f"- **{key}:** {val}\n"
        output += "\n"
        
    # Add recommendations
    if ktype in KERNEL_TYPE_RECOMMENDATIONS:
        recs = KERNEL_TYPE_RECOMMENDATIONS[ktype]
        output += f"## Recommended Libraries\n\n"
        for lib in recs["libraries"]:
            output += f"- {lib}\n"
        output += f"\n## Optimization Strategies\n\n"
        for opt in recs["optimizations"]:
            output += f"- {opt}\n"
            
    return [TextContent(type="text", text=output)]

async def handle_identify_hotspots(args: dict) -> list[TextContent]:
    """Identify hotspots from kernel list."""
    kernels = args.get("kernels", [])
    if not kernels:
        return [TextContent(type="text", text="No kernels provided")]
        
    hotspots = analyze_hotspots(kernels, args.get("top_n", 10))
    return format_hotspots_output(hotspots)

async def handle_suggest_optimization(args: dict) -> list[TextContent]:
    """Suggest optimization approach."""
    ktype = args.get("kernel_type", "")
    kname = args.get("kernel_name", "")
    
    if ktype not in KERNEL_TYPE_RECOMMENDATIONS:
        available = list(KERNEL_TYPE_RECOMMENDATIONS.keys())
        return [TextContent(type="text", text=f"Unknown kernel type: {ktype}\nAvailable: {available}")]
        
    recs = KERNEL_TYPE_RECOMMENDATIONS[ktype]
    
    output = f"# Optimization Approach: {ktype.upper()}\n\n"
    
    if kname:
        output += f"**Kernel:** `{kname}`\n\n"
        
    output += "## Recommended Libraries\n\n"
    for lib in recs["libraries"]:
        output += f"- {lib}\n"
        
    output += "\n## Optimization Strategies\n\n"
    for i, opt in enumerate(recs["optimizations"], 1):
        output += f"{i}. {opt}\n"
        
    # Add kernel-specific hints
    if kname:
        if "Cijk_" in kname:
            output += "\n## Tensile-Specific\n\n"
            info = decode_tensile_name(kname)
            output += f"- Current tile sizes: M={info.get('tile_m')}, N={info.get('tile_n')}, K={info.get('tile_k')}\n"
            if info.get('mfma_m'):
                output += f"- Using MFMA: {info.get('mfma_m')}x{info.get('mfma_n')}x{info.get('mfma_k')}\n"
            output += "- To optimize: Tune via Tensile or consider CK alternative\n"
            
    return [TextContent(type="text", text=output)]

def format_hotspots_output(hotspots: List[KernelInfo]) -> list[TextContent]:
    """Format hotspots for output."""
    if not hotspots:
        return [TextContent(type="text", text="No kernels found")]
        
    total_time = sum(k.time_us for k in hotspots)
    
    output = f"# Kernel Hotspots\n\n"
    output += f"**Total time:** {total_time/1000:.2f} ms\n"
    output += f"**Kernels analyzed:** {len(hotspots)}\n\n"
    
    output += "## Top Kernels by Time\n\n"
    output += "| Rank | Kernel | Time (ms) | % | Calls | Type | Origin |\n"
    output += "|------|--------|-----------|---|-------|------|--------|\n"
    
    for i, k in enumerate(hotspots, 1):
        # Truncate long kernel names
        name = k.name[:50] + "..." if len(k.name) > 50 else k.name
        output += f"| {i} | `{name}` | {k.time_us/1000:.3f} | {k.percentage:.1f}% | {k.calls} | {k.kernel_type} | {k.origin} |\n"
        
    output += "\n## Optimization Priorities\n\n"
    
    # Group by type for recommendations
    by_type = defaultdict(list)
    for k in hotspots[:10]:
        by_type[k.kernel_type].append(k)
        
    for ktype, kernels in by_type.items():
        if ktype in KERNEL_TYPE_RECOMMENDATIONS:
            type_time = sum(k.time_us for k in kernels)
            type_pct = (type_time / total_time * 100) if total_time > 0 else 0
            output += f"### {ktype.upper()} ({type_pct:.1f}% of time)\n\n"
            recs = KERNEL_TYPE_RECOMMENDATIONS[ktype]
            output += f"**Libraries:** {', '.join(recs['libraries'][:2])}\n\n"
            output += "**Quick wins:**\n"
            for opt in recs["optimizations"][:2]:
                output += f"- {opt}\n"
            output += "\n"
            
    return [TextContent(type="text", text=output)]


async def handle_find_ck_template(args: dict) -> list[TextContent]:
    """Find CK templates matching operation signature."""
    operation = args.get("operation", "")
    dtype = args.get("dtype")
    features = args.get("features", [])
    
    if not operation:
        return [TextContent(type="text", text="No operation specified")]
    
    templates = find_ck_templates(operation, dtype, features)
    
    output = f"# Composable Kernel Templates for: {operation.upper()}\n\n"
    
    if dtype:
        output += f"**Data Type Filter:** {dtype}\n"
    if features:
        output += f"**Feature Filter:** {', '.join(features)}\n"
    output += "\n"
    
    if not templates:
        output += "**No matching templates found.**\n\n"
        output += "Available operation categories:\n"
        for cat in CK_TEMPLATES.keys():
            output += f"- {cat}\n"
        return [TextContent(type="text", text=output)]
    
    output += f"**Found {len(templates)} matching template(s)**\n\n"
    
    # Get CK path
    search_paths = get_paths()
    ck_base = search_paths.get("composable_kernel", "/path/to/composable_kernel")
    
    for i, tmpl in enumerate(templates, 1):
        output += f"## {i}. {tmpl['name']}\n\n"
        output += f"**Description:** {tmpl['description']}\n"
        output += f"**Use Case:** {tmpl['use_case']}\n"
        output += f"**Path:** `{ck_base}/{tmpl['path']}`\n"
        output += f"**Data Types:** {', '.join(tmpl['dtypes'])}\n"
        output += f"**Features:** {', '.join(tmpl['features'])}\n\n"
    
    # Add usage hints
    output += "---\n\n"
    output += "## Usage Tips\n\n"
    output += "1. Clone CK if not present: `git clone https://github.com/ROCm/composable_kernel.git`\n"
    output += "2. Check examples in `composable_kernel/example/` for usage patterns\n"
    output += "3. Template parameters (tile sizes, etc.) can be tuned for your shapes\n"
    
    if operation.lower() in ["gemm", "matmul"]:
        output += "\n**GEMM-specific tips:**\n"
        output += "- Use `DeviceGemmMultiD` for fused bias/activation\n"
        output += "- Use `DeviceSplitKGemm` when K >> M*N\n"
        output += "- Use `DeviceGroupedGemm` for MoE or variable-size batches\n"
    elif operation.lower() in ["attention", "fmha"]:
        output += "\n**Attention-specific tips:**\n"
        output += "- Use `DeviceFmhaFwd` with causal=true for decoder attention\n"
        output += "- Use `DevicePagedAttention` for inference with KV cache\n"
        output += "- Check ck_tile for latest FMHA implementations\n"
    
    return [TextContent(type="text", text=output)]


# =============================================================================
# Main
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
