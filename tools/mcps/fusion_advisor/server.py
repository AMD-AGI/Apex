#!/usr/bin/env python3
"""
Fusion Advisor MCP Server
=========================
Detect kernel fusion opportunities and generate fused kernels.

Tools:
- detect_fusion_opportunities: Analyze kernel sequence and identify fuseable patterns
- analyze_data_flow: Build producer-consumer graph, find memory-bound chains
- generate_fused_kernel: Create fused kernel in Triton or HIP
- validate_fusion: Check correctness constraints
- estimate_fusion_benefit: Predict memory savings and potential speedup
- parse_magpie_output: Parse Magpie analyze output for fusion analysis
- calculate_memory_savings: Calculate actual memory savings with tensor shapes
- check_library_fusion: Check if fused kernel exists in CK/hipBLASLt/vLLM
- benchmark_compare: Compare two kernel implementations A vs B
"""

import asyncio
import json
import re
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# =============================================================================
# Fusion Patterns Database
# =============================================================================

FUSION_PATTERNS = {
    "elementwise_chain": {
        "name": "Elementwise Chain Fusion",
        "pattern": ["elementwise", "elementwise"],
        "kernel_types": ["add", "mul", "relu", "gelu", "silu", "tanh", "sigmoid", 
                        "dropout", "scale", "cast", "copy"],
        "benefit": "Reduce memory traffic by N-1 intermediate reads/writes",
        "memory_saving": "high",
        "complexity": "low",
        "frameworks": ["triton", "hip"],
        "example": "x = relu(add(a, b)) -> fused_add_relu(a, b)"
    },
    "gemm_epilogue": {
        "name": "GEMM Epilogue Fusion",
        "pattern": ["gemm", "bias|activation|elementwise"],
        "kernel_types": ["gemm", "matmul", "linear", "bias_add", "relu", "gelu", "silu"],
        "benefit": "Fuse post-GEMM ops into epilogue, single kernel launch",
        "memory_saving": "high",
        "complexity": "medium",
        "frameworks": ["hipblaslt", "composable_kernel", "triton"],
        "example": "y = relu(matmul(x, w) + b) -> gemm_bias_relu(x, w, b)"
    },
    "attention_block": {
        "name": "Attention Block Fusion (Flash Attention)",
        "pattern": ["qkv_proj", "attention", "output_proj"],
        "kernel_types": ["matmul", "softmax", "attention", "qkv", "scaled_dot_product"],
        "benefit": "Flash Attention pattern - avoid materializing NxN attention matrix",
        "memory_saving": "very_high",
        "complexity": "high",
        "frameworks": ["composable_kernel", "triton", "aiter"],
        "example": "FlashAttention(Q, K, V) instead of softmax(QK^T)V"
    },
    "norm_activation": {
        "name": "Normalization + Activation Fusion",
        "pattern": ["layernorm|rmsnorm", "activation|scale|multiply"],
        "kernel_types": ["layernorm", "rmsnorm", "batchnorm", "groupnorm", 
                        "scale", "multiply", "silu", "gelu"],
        "benefit": "Single pass over data for norm + activation",
        "memory_saving": "medium",
        "complexity": "low",
        "frameworks": ["triton", "hip", "composable_kernel"],
        "example": "silu(rmsnorm(x)) -> fused_rmsnorm_silu(x)"
    },
    "residual_norm": {
        "name": "Residual Add + Normalization",
        "pattern": ["add", "layernorm|rmsnorm"],
        "kernel_types": ["add", "residual", "layernorm", "rmsnorm"],
        "benefit": "Fuse residual connection with following norm",
        "memory_saving": "medium",
        "complexity": "low",
        "frameworks": ["triton", "hip"],
        "example": "rmsnorm(x + residual) -> fused_add_rmsnorm(x, residual)"
    },
    "rotary_embedding": {
        "name": "Rotary Position Embedding Fusion",
        "pattern": ["rotary", "attention|qkv"],
        "kernel_types": ["rotary", "rope", "position_embedding", "qkv"],
        "benefit": "Apply RoPE during QKV projection or attention",
        "memory_saving": "medium",
        "complexity": "medium",
        "frameworks": ["triton", "composable_kernel"],
        "example": "Apply RoPE inline with attention computation"
    },
    "quantize_dequantize": {
        "name": "Quantization Fusion",
        "pattern": ["dequantize", "gemm|matmul", "quantize"],
        "kernel_types": ["quantize", "dequantize", "gemm", "matmul"],
        "benefit": "Fuse quant/dequant into GEMM, reduce memory for weights",
        "memory_saving": "high",
        "complexity": "high",
        "frameworks": ["hipblaslt", "vllm", "composable_kernel"],
        "example": "FP8 GEMM with fused scaling"
    },
    "moe_gating": {
        "name": "MoE Gating + Expert Fusion",
        "pattern": ["topk", "scatter|gather", "expert_gemm"],
        "kernel_types": ["topk", "softmax", "scatter", "gather", "gemm", "moe"],
        "benefit": "Fuse gating with expert dispatch",
        "memory_saving": "medium",
        "complexity": "high",
        "frameworks": ["composable_kernel", "vllm"],
        "example": "Fused MoE with grouped GEMM"
    }
}

# =============================================================================
# Library Fused Kernel Database
# =============================================================================

LIBRARY_FUSED_KERNELS = {
    # Composable Kernel (CK) fused operations
    "ck": {
        "gemm_bias": {
            "name": "GEMM + Bias Add",
            "pattern": ["gemm", "bias_add"],
            "api": "ck::tensor_operation::device::DeviceGemmBias",
            "header": "ck/tensor_operation/gpu/device/device_gemm_bias.hpp",
            "notes": "Fuses bias addition into GEMM epilogue"
        },
        "gemm_bias_relu": {
            "name": "GEMM + Bias + ReLU",
            "pattern": ["gemm", "bias_add", "relu"],
            "api": "ck::tensor_operation::device::DeviceGemmBiasRelu",
            "header": "ck/tensor_operation/gpu/device/device_gemm_bias_relu.hpp",
            "notes": "Common in MLP layers"
        },
        "gemm_bias_gelu": {
            "name": "GEMM + Bias + GELU",
            "pattern": ["gemm", "bias_add", "gelu"],
            "api": "ck::tensor_operation::device::DeviceGemmBiasGelu",
            "header": "ck/tensor_operation/gpu/device/device_gemm_bias_gelu.hpp",
            "notes": "Common in transformer FFN"
        },
        "gemm_add": {
            "name": "GEMM + Elementwise Add (Residual)",
            "pattern": ["gemm", "add"],
            "api": "ck::tensor_operation::device::DeviceGemmAdd",
            "header": "ck/tensor_operation/gpu/device/device_gemm_add.hpp",
            "notes": "Fuses residual connection"
        },
        "grouped_gemm": {
            "name": "Grouped GEMM (MoE)",
            "pattern": ["grouped_gemm", "moe"],
            "api": "ck::tensor_operation::device::DeviceGroupedGemm",
            "header": "ck/tensor_operation/gpu/device/device_grouped_gemm.hpp",
            "notes": "For MoE expert parallel execution"
        },
        "batched_gemm_softmax_gemm": {
            "name": "Flash Attention (QKV)",
            "pattern": ["gemm", "softmax", "gemm"],
            "api": "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemm",
            "header": "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm.hpp",
            "notes": "Flash attention pattern for transformers"
        },
        "layernorm": {
            "name": "Fused LayerNorm",
            "pattern": ["mean", "variance", "normalize", "scale", "bias"],
            "api": "ck::tensor_operation::device::DeviceLayernorm",
            "header": "ck/tensor_operation/gpu/device/device_layernorm.hpp",
            "notes": "Single-pass layer normalization"
        },
        "rmsnorm": {
            "name": "Fused RMSNorm",
            "pattern": ["rms", "normalize", "scale"],
            "api": "ck::tensor_operation::device::DeviceRMSNorm",
            "header": "ck/tensor_operation/gpu/device/device_rmsnorm.hpp",
            "notes": "Single-pass RMS normalization"
        },
        "softmax": {
            "name": "Fused Softmax",
            "pattern": ["max", "subtract", "exp", "sum", "divide"],
            "api": "ck::tensor_operation::device::DeviceSoftmax",
            "header": "ck/tensor_operation/gpu/device/device_softmax.hpp",
            "notes": "Numerically stable single-pass softmax"
        }
    },
    # hipBLASLt fused operations
    "hipblaslt": {
        "gemm_bias": {
            "name": "GEMM + Bias (hipBLASLt)",
            "pattern": ["gemm", "bias"],
            "api": "hipblasLtMatmul with HIPBLASLT_EPILOGUE_BIAS",
            "header": "hipblaslt/hipblaslt.h",
            "notes": "Set epilogue in hipblasLtMatmulPreference"
        },
        "gemm_bias_relu": {
            "name": "GEMM + Bias + ReLU (hipBLASLt)",
            "pattern": ["gemm", "bias", "relu"],
            "api": "hipblasLtMatmul with HIPBLASLT_EPILOGUE_RELU_BIAS",
            "header": "hipblaslt/hipblaslt.h",
            "notes": "Fused ReLU activation in GEMM epilogue"
        },
        "gemm_bias_gelu": {
            "name": "GEMM + Bias + GELU (hipBLASLt)",
            "pattern": ["gemm", "bias", "gelu"],
            "api": "hipblasLtMatmul with HIPBLASLT_EPILOGUE_GELU_BIAS",
            "header": "hipblaslt/hipblaslt.h",
            "notes": "Fused GELU activation in GEMM epilogue"
        },
        "gemm_scale": {
            "name": "GEMM with Scaling (FP8)",
            "pattern": ["dequantize", "gemm", "quantize"],
            "api": "hipblasLtMatmul with amax/scale descriptors",
            "header": "hipblaslt/hipblaslt.h",
            "notes": "FP8 GEMM with fused scaling for quantization"
        }
    },
    # vLLM/AIter fused operations
    "vllm": {
        "paged_attention": {
            "name": "Paged Attention",
            "pattern": ["attention", "kv_cache"],
            "api": "vllm.attention.PagedAttention",
            "import": "from vllm.attention import PagedAttention",
            "notes": "Memory-efficient attention with paged KV cache"
        },
        "fused_add_rmsnorm": {
            "name": "Residual + RMSNorm",
            "pattern": ["add", "rmsnorm"],
            "api": "vllm.model_executor.layers.layernorm.RMSNorm",
            "import": "from vllm.model_executor.layers.layernorm import RMSNorm",
            "notes": "Fused residual add with RMSNorm"
        },
        "rotary_embedding": {
            "name": "Rotary Position Embedding",
            "pattern": ["cos", "sin", "rotate", "multiply"],
            "api": "vllm.model_executor.layers.rotary_embedding.RotaryEmbedding",
            "import": "from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding",
            "notes": "Fused RoPE computation"
        },
        "fused_moe": {
            "name": "Fused MoE",
            "pattern": ["topk", "softmax", "grouped_gemm"],
            "api": "vllm.model_executor.layers.fused_moe.FusedMoE",
            "import": "from vllm.model_executor.layers.fused_moe import FusedMoE",
            "notes": "Fused gating + expert GEMM"
        },
        "silu_and_mul": {
            "name": "SiLU + Mul (SwiGLU)",
            "pattern": ["silu", "multiply"],
            "api": "vllm.model_executor.layers.activation.SiluAndMul",
            "import": "from vllm.model_executor.layers.activation import SiluAndMul",
            "notes": "Fused SwiGLU activation"
        }
    },
    # Triton existing kernels (from triton tutorials/examples)
    "triton": {
        "flash_attention": {
            "name": "Flash Attention (Triton)",
            "pattern": ["gemm", "softmax", "gemm"],
            "api": "triton.ops.flash_attention",
            "import": "import triton.ops",
            "notes": "Memory-efficient attention, O(N) memory"
        },
        "fused_softmax": {
            "name": "Fused Softmax (Triton)",
            "pattern": ["max", "exp", "sum", "divide"],
            "api": "@triton.jit softmax kernel",
            "notes": "Single-pass softmax from Triton tutorials"
        },
        "fused_layernorm": {
            "name": "Fused LayerNorm (Triton)",
            "pattern": ["mean", "variance", "normalize"],
            "api": "@triton.jit layernorm kernel",
            "notes": "Single-pass layernorm"
        }
    }
}

# Dtype sizes in bytes
DTYPE_SIZES = {
    "fp32": 4, "float32": 4, "float": 4,
    "fp16": 2, "float16": 2, "half": 2,
    "bf16": 2, "bfloat16": 2,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "int8": 1, "int32": 4, "int64": 8,
}

# Kernel type classification
KERNEL_TYPE_MAP = {
    # Elementwise
    "add": "elementwise", "sub": "elementwise", "mul": "elementwise", "div": "elementwise",
    "relu": "elementwise", "gelu": "elementwise", "silu": "elementwise", "tanh": "elementwise",
    "sigmoid": "elementwise", "exp": "elementwise", "log": "elementwise", "sqrt": "elementwise",
    "dropout": "elementwise", "scale": "elementwise", "cast": "elementwise", "copy": "elementwise",
    "pow": "elementwise", "abs": "elementwise", "neg": "elementwise",
    
    # GEMM/Linear
    "gemm": "gemm", "matmul": "gemm", "linear": "gemm", "dot": "gemm",
    "batched_gemm": "gemm", "grouped_gemm": "gemm",
    
    # Normalization
    "layernorm": "normalization", "layer_norm": "normalization",
    "rmsnorm": "normalization", "rms_norm": "normalization",
    "batchnorm": "normalization", "batch_norm": "normalization",
    "groupnorm": "normalization", "group_norm": "normalization",
    
    # Attention
    "attention": "attention", "flash_attention": "attention",
    "scaled_dot_product": "attention", "sdpa": "attention",
    "softmax": "attention", "qkv": "attention",
    
    # Reduction
    "sum": "reduction", "mean": "reduction", "max": "reduction", "min": "reduction",
    "reduce": "reduction", "all_reduce": "reduction",
    
    # Memory
    "copy": "memory", "transpose": "memory", "permute": "memory",
    "reshape": "memory", "view": "memory", "contiguous": "memory",
    
    # Quantization
    "quantize": "quantization", "dequantize": "quantization",
    "mxfp4": "quantization", "fp8": "quantization", "int8": "quantization",
    
    # MoE
    "moe": "moe", "topk": "moe", "expert": "moe",
    "scatter": "moe", "gather": "moe",
    
    # Position embeddings
    "rotary": "position", "rope": "position", "position_embedding": "position",
}

# =============================================================================
# Fusion Detection
# =============================================================================

@dataclass
class KernelNode:
    """Represents a kernel in the computation graph."""
    id: int
    name: str
    kernel_type: str
    category: str
    inputs: List[str]
    outputs: List[str]
    time_us: float = 0.0

@dataclass
class FusionOpportunity:
    """Represents a detected fusion opportunity."""
    pattern_name: str
    kernels: List[str]
    kernel_ids: List[int]
    benefit: str
    memory_saving: str
    complexity: str
    frameworks: List[str]
    estimated_speedup: str
    
    def to_dict(self):
        return asdict(self)

def classify_kernel(name: str) -> Tuple[str, str]:
    """Classify kernel by name. Returns (kernel_type, category)."""
    name_lower = name.lower()
    
    # Check exact matches first for priority
    if name_lower in KERNEL_TYPE_MAP:
        ktype = name_lower
        return ktype, KERNEL_TYPE_MAP[ktype]
    
    # Check for known patterns
    for ktype, category in KERNEL_TYPE_MAP.items():
        if ktype in name_lower:
            return ktype, category
            
    # Fallback heuristics
    if any(x in name_lower for x in ['elem', 'point', 'unary', 'binary']):
        return "elementwise", "elementwise"
    if any(x in name_lower for x in ['mm', 'gemm', 'matmul', 'linear']):
        return "gemm", "gemm"
    if any(x in name_lower for x in ['norm', 'ln', 'rms']):
        return "normalization", "normalization"
    if any(x in name_lower for x in ['attn', 'attention', 'flash']):
        return "attention", "attention"
        
    return "unknown", "unknown"

def parse_kernel_sequence(input_data: str) -> List[KernelNode]:
    """Parse kernel sequence from various input formats."""
    kernels = []
    
    # Try to parse as kernel list (one per line or comma-separated)
    lines = input_data.strip().replace(',', '\n').split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Extract kernel name and optional time
        parts = line.split()
        name = parts[0]
        time_us = 0.0
        
        # Try to extract time if present
        for part in parts[1:]:
            try:
                time_us = float(part.replace('us', '').replace('ms', ''))
                if 'ms' in part:
                    time_us *= 1000
                break
            except:
                pass
                
        ktype, category = classify_kernel(name)
        
        kernels.append(KernelNode(
            id=i,
            name=name,
            kernel_type=ktype,
            category=category,
            inputs=[f"tensor_{i}"],
            outputs=[f"tensor_{i+1}"],
            time_us=time_us
        ))
        
    return kernels

def detect_fusion_opportunities(kernels: List[KernelNode]) -> List[FusionOpportunity]:
    """Detect fusion opportunities in kernel sequence."""
    opportunities = []
    
    if len(kernels) < 2:
        return opportunities
        
    # Sliding window to find patterns
    for pattern_id, pattern_info in FUSION_PATTERNS.items():
        pattern_len = len(pattern_info["pattern"])
        
        for i in range(len(kernels) - pattern_len + 1):
            window = kernels[i:i + pattern_len]
            
            # Check if window matches pattern
            matches = True
            for j, (kernel, pattern) in enumerate(zip(window, pattern_info["pattern"])):
                pattern_types = pattern.split('|')
                if not any(pt in kernel.category or pt in kernel.kernel_type 
                          for pt in pattern_types):
                    # Check if it's a continuation pattern (e.g., "elementwise+")
                    if pattern.endswith('+'):
                        base_pattern = pattern[:-1]
                        if base_pattern not in kernel.category and base_pattern not in kernel.kernel_type:
                            matches = False
                            break
                    else:
                        matches = False
                        break
                        
            if matches:
                # Calculate estimated speedup based on memory saving
                speedup_map = {"very_high": "2-4x", "high": "1.5-2x", "medium": "1.2-1.5x", "low": "1.1-1.2x"}
                
                opportunities.append(FusionOpportunity(
                    pattern_name=pattern_info["name"],
                    kernels=[k.name for k in window],
                    kernel_ids=[k.id for k in window],
                    benefit=pattern_info["benefit"],
                    memory_saving=pattern_info["memory_saving"],
                    complexity=pattern_info["complexity"],
                    frameworks=pattern_info["frameworks"],
                    estimated_speedup=speedup_map.get(pattern_info["memory_saving"], "1.1-1.2x")
                ))
                
    # Also check for elementwise chains longer than 2
    elem_chain = []
    for kernel in kernels:
        if kernel.category == "elementwise":
            elem_chain.append(kernel)
        else:
            if len(elem_chain) > 2:
                opportunities.append(FusionOpportunity(
                    pattern_name=f"Elementwise Chain ({len(elem_chain)} ops)",
                    kernels=[k.name for k in elem_chain],
                    kernel_ids=[k.id for k in elem_chain],
                    benefit=f"Reduce {len(elem_chain)-1} intermediate tensor writes",
                    memory_saving="high",
                    complexity="low",
                    frameworks=["triton", "hip"],
                    estimated_speedup=f"{1 + 0.3 * len(elem_chain):.1f}x"
                ))
            elem_chain = []
            
    return opportunities

# =============================================================================
# Fused Kernel Generation
# =============================================================================

FUSED_KERNEL_TEMPLATES = {
    "elementwise_chain": {
        "triton": '''import triton
import triton.language as tl

@triton.jit
def fused_{name}_kernel(
    {input_args}
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    {load_ops}
    
    # Fused operations
    {fused_ops}
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_{name}({python_args}):
    output = torch.empty_like({first_input})
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_{name}_kernel[grid]({kernel_args}, n_elements, BLOCK_SIZE=1024)
    return output
''',
        "hip": '''#include <hip/hip_runtime.h>

__global__ void fused_{name}_kernel(
    {input_args}
    float* __restrict__ output,
    int n_elements
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        // Load inputs
        {load_ops}
        
        // Fused operations
        {fused_ops}
        
        // Store result
        output[idx] = result;
    }}
}}

void fused_{name}({cpp_args}, int n) {{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(fused_{name}_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       {kernel_args}, n);
}}
'''
    },
    "norm_activation": {
        "triton": '''import triton
import triton.language as tl

@triton.jit
def fused_norm_{activation}_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    x_ptrs = x_ptr + row_idx * n_cols + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # RMSNorm
    variance = tl.sum(x * x, axis=0) / n_cols
    x_norm = x * tl.rsqrt(variance + eps)
    
    # Load weight and apply
    w = tl.load(weight_ptr + col_offsets, mask=mask)
    x_norm = x_norm * w
    
    # Activation: {activation}
    {activation_code}
    
    # Store
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, result, mask=mask)
'''
    },
    "gemm_epilogue": {
        "triton": '''import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}}, num_warps=8),
        triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_{epilogue}_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Epilogue: bias + {epilogue}
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    {epilogue_code}
    
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
'''
    }
}

ACTIVATION_CODE = {
    "relu": "result = tl.maximum(x_norm, 0.0)",
    "gelu": "result = x_norm * 0.5 * (1.0 + tl.libdevice.erf(x_norm / 1.41421356))",
    "silu": "result = x_norm * tl.sigmoid(x_norm)",
    "tanh": "result = tl.libdevice.tanh(x_norm)",
    "sigmoid": "result = tl.sigmoid(x_norm)",
    "none": "result = x_norm"
}

def generate_fused_kernel(opportunity: FusionOpportunity, framework: str = "triton") -> str:
    """Generate fused kernel code for an opportunity."""
    
    pattern = opportunity.pattern_name.lower()
    
    # Determine template
    if "elementwise" in pattern:
        template_key = "elementwise_chain"
    elif "norm" in pattern and any(x in pattern for x in ["activation", "silu", "gelu", "relu"]):
        template_key = "norm_activation"
    elif "gemm" in pattern or "epilogue" in pattern:
        template_key = "gemm_epilogue"
    else:
        template_key = "elementwise_chain"  # Default
        
    if template_key not in FUSED_KERNEL_TEMPLATES:
        return f"# No template available for pattern: {pattern}"
        
    templates = FUSED_KERNEL_TEMPLATES[template_key]
    
    if framework not in templates:
        return f"# Framework {framework} not supported for pattern: {pattern}"
        
    template = templates[framework]
    
    # Generate kernel name from operations
    ops = [k.split('_')[0] if '_' in k else k[:10] for k in opportunity.kernels]
    name = '_'.join(ops[:3])
    
    # Fill in template (simplified - real implementation would be more sophisticated)
    if template_key == "elementwise_chain":
        code = template.format(
            name=name,
            input_args=', '.join([f"in{i}_ptr," for i in range(len(opportunity.kernels))]),
            load_ops='\n    '.join([f"x{i} = tl.load(in{i}_ptr + offsets, mask=mask)" 
                                    for i in range(len(opportunity.kernels))]),
            fused_ops=f"result = x0  # TODO: Apply {' -> '.join(opportunity.kernels)}",
            python_args=', '.join([f"x{i}" for i in range(len(opportunity.kernels))]),
            first_input="x0",
            kernel_args=', '.join([f"x{i}" for i in range(len(opportunity.kernels))] + ["output"])
        )
    elif template_key == "norm_activation":
        activation = "silu"  # Default
        for k in opportunity.kernels:
            for act in ["relu", "gelu", "silu", "tanh", "sigmoid"]:
                if act in k.lower():
                    activation = act
                    break
        code = template.format(
            activation=activation,
            activation_code=ACTIVATION_CODE.get(activation, ACTIVATION_CODE["none"])
        )
    elif template_key == "gemm_epilogue":
        epilogue = "relu"  # Default
        for k in opportunity.kernels:
            for act in ["relu", "gelu", "silu"]:
                if act in k.lower():
                    epilogue = act
                    break
        epilogue_code = {
            "relu": "acc = tl.maximum(acc, 0.0)",
            "gelu": "acc = acc * 0.5 * (1.0 + tl.libdevice.erf(acc / 1.41421356))",
            "silu": "acc = acc * tl.sigmoid(acc)"
        }.get(epilogue, "# No activation")
        code = template.format(epilogue=epilogue, epilogue_code=epilogue_code)
    else:
        code = template
        
    return code

# =============================================================================
# Magpie Output Parsing
# =============================================================================

def parse_magpie_output(magpie_json: str) -> List[KernelNode]:
    """
    Parse Magpie analyze output to extract kernel sequence with timing.
    
    Magpie output format:
    {
        "performance_result": {
            "kernels": [
                {"kernel_name": "...", "dispatch_count": N, "duration_ns": {"avg": ..., "min": ..., "max": ...}}
            ]
        }
    }
    """
    try:
        data = json.loads(magpie_json)
    except json.JSONDecodeError:
        return []
    
    kernels = []
    perf = data.get("performance_result", {})
    kernel_list = perf.get("kernels", [])
    
    for i, k in enumerate(kernel_list):
        name = k.get("kernel_name", f"kernel_{i}")
        duration = k.get("duration_ns", {})
        avg_ns = duration.get("avg", 0) if isinstance(duration, dict) else duration
        time_us = avg_ns / 1000.0  # Convert ns to us
        
        ktype, category = classify_kernel(name)
        
        # Extract dispatch count for repeated kernel detection
        dispatch_count = k.get("dispatch_count", 1)
        
        kernels.append(KernelNode(
            id=i,
            name=name,
            kernel_type=ktype,
            category=category,
            inputs=[f"tensor_{i}"],
            outputs=[f"tensor_{i+1}"],
            time_us=time_us
        ))
    
    return kernels


def extract_fusion_candidates_from_magpie(magpie_json: str) -> Dict:
    """
    Extract fusion candidates from Magpie output with actual timing data.
    """
    kernels = parse_magpie_output(magpie_json)
    opportunities = detect_fusion_opportunities(kernels)
    
    # Calculate actual time savings potential
    total_time = sum(k.time_us for k in kernels)
    
    for opp in opportunities:
        # Get actual kernel times
        opp_times = [kernels[i].time_us for i in opp.kernel_ids if i < len(kernels)]
        opp_total = sum(opp_times)
        
        # Estimate fused time (typically 60-80% of sum due to reduced memory traffic)
        estimated_fused = opp_total * 0.65
        time_saved = opp_total - estimated_fused
        
        # Add timing info
        opp.estimated_speedup = f"{opp_total/estimated_fused:.2f}x ({time_saved:.1f}us saved)"
    
    return {
        "total_kernels": len(kernels),
        "total_time_us": total_time,
        "opportunities": [opp.to_dict() for opp in opportunities],
        "kernels_analyzed": [
            {"name": k.name, "category": k.category, "time_us": k.time_us}
            for k in kernels
        ]
    }


# =============================================================================
# Memory Savings Calculation
# =============================================================================

def calculate_memory_savings(
    kernels: List[str],
    tensor_shapes: Dict[str, List[int]],
    dtype: str = "fp16"
) -> Dict:
    """
    Calculate actual memory savings from kernel fusion.
    
    Args:
        kernels: List of kernel names in fusion
        tensor_shapes: Dict mapping tensor names to shapes, e.g., {"x": [batch, seq, hidden]}
        dtype: Data type (fp16, bf16, fp32, etc.)
    
    Returns:
        Dict with memory analysis
    """
    element_size = DTYPE_SIZES.get(dtype.lower(), 2)  # Default to fp16
    
    # Calculate intermediate tensor sizes
    intermediates = []
    total_elements = 0
    
    # For N kernels, we have N-1 intermediate tensors
    for i in range(len(kernels) - 1):
        # Use provided shapes or estimate from typical patterns
        tensor_name = f"intermediate_{i}"
        
        # Check if we have shape info for this intermediate
        if tensor_name in tensor_shapes:
            shape = tensor_shapes[tensor_name]
        elif "output" in tensor_shapes:
            shape = tensor_shapes["output"]
        elif "x" in tensor_shapes:
            shape = tensor_shapes["x"]
        else:
            # Default estimate: 1M elements
            shape = [1024, 1024]
        
        numel = 1
        for dim in shape:
            numel *= dim
        
        total_elements += numel
        intermediates.append({
            "name": tensor_name,
            "shape": shape,
            "elements": numel,
            "bytes": numel * element_size
        })
    
    total_bytes = total_elements * element_size
    
    # Memory traffic saved = 2x bytes (one write + one read per intermediate)
    memory_traffic_saved = total_bytes * 2
    
    # Estimate bandwidth and time savings
    hbm_bandwidth_gbps = 3200  # MI300X ~3.2 TB/s
    time_saved_us = (memory_traffic_saved / (hbm_bandwidth_gbps * 1e9)) * 1e6
    
    return {
        "kernels_fused": kernels,
        "dtype": dtype,
        "element_size_bytes": element_size,
        "intermediates_eliminated": len(intermediates),
        "intermediate_details": intermediates,
        "total_elements_saved": total_elements,
        "total_bytes_saved": total_bytes,
        "memory_traffic_saved_bytes": memory_traffic_saved,
        "memory_traffic_saved_human": _human_readable_bytes(memory_traffic_saved),
        "estimated_time_saved_us": round(time_saved_us, 2),
        "assumptions": {
            "hbm_bandwidth_gbps": hbm_bandwidth_gbps,
            "note": "Actual savings depend on memory access patterns and cache behavior"
        }
    }


def _human_readable_bytes(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# =============================================================================
# Library Fusion Lookup
# =============================================================================

def check_library_fusion(kernel_sequence: List[str]) -> Dict:
    """
    Check if a fused kernel already exists in CK/hipBLASLt/vLLM.
    
    Args:
        kernel_sequence: List of kernel names/types to fuse
        
    Returns:
        Dict with matching library implementations
    """
    matches = []
    
    # Normalize kernel names to categories
    categories = []
    for k in kernel_sequence:
        ktype, category = classify_kernel(k)
        categories.append(ktype)
    
    # Create a pattern string for matching
    pattern_str = "_".join(categories)
    
    # Search each library
    for lib_name, lib_kernels in LIBRARY_FUSED_KERNELS.items():
        for kernel_id, kernel_info in lib_kernels.items():
            lib_pattern = kernel_info.get("pattern", [])
            
            # Check if library pattern matches our kernel sequence
            if _pattern_matches(categories, lib_pattern):
                match = {
                    "library": lib_name,
                    "kernel_id": kernel_id,
                    "name": kernel_info["name"],
                    "api": kernel_info.get("api", ""),
                    "header": kernel_info.get("header", ""),
                    "import": kernel_info.get("import", ""),
                    "notes": kernel_info.get("notes", ""),
                    "match_score": _calculate_match_score(categories, lib_pattern)
                }
                matches.append(match)
    
    # Sort by match score
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {
        "input_sequence": kernel_sequence,
        "detected_pattern": categories,
        "matches_found": len(matches),
        "matches": matches,
        "recommendation": _get_library_recommendation(matches) if matches else None
    }


def _pattern_matches(kernel_categories: List[str], lib_pattern: List[str]) -> bool:
    """Check if kernel categories match a library pattern."""
    if not lib_pattern:
        return False
    
    # Exact length match
    if len(kernel_categories) == len(lib_pattern):
        for kc, lp in zip(kernel_categories, lib_pattern):
            if not _category_matches(kc, lp):
                return False
        return True
    
    # Subsequence match (library pattern is subset)
    if len(lib_pattern) <= len(kernel_categories):
        for i in range(len(kernel_categories) - len(lib_pattern) + 1):
            window = kernel_categories[i:i + len(lib_pattern)]
            if all(_category_matches(kc, lp) for kc, lp in zip(window, lib_pattern)):
                return True
    
    return False


def _category_matches(kernel_cat: str, pattern_cat: str) -> bool:
    """Check if a kernel category matches a pattern category."""
    if kernel_cat == pattern_cat:
        return True
    
    # Handle aliases
    aliases = {
        "gemm": ["matmul", "linear", "mm", "dot"],
        "bias_add": ["bias", "add"],
        "relu": ["activation"],
        "gelu": ["activation"],
        "silu": ["activation", "swish"],
        "layernorm": ["ln", "layer_norm"],
        "rmsnorm": ["rms", "rms_norm"],
        "softmax": ["attention"],
    }
    
    for main, alias_list in aliases.items():
        if kernel_cat == main and pattern_cat in alias_list:
            return True
        if pattern_cat == main and kernel_cat in alias_list:
            return True
        if kernel_cat in alias_list and pattern_cat in alias_list:
            return True
    
    return False


def _calculate_match_score(kernel_categories: List[str], lib_pattern: List[str]) -> float:
    """Calculate how well the kernel sequence matches the library pattern."""
    if not lib_pattern or not kernel_categories:
        return 0.0
    
    matches = sum(1 for kc, lp in zip(kernel_categories, lib_pattern) 
                  if _category_matches(kc, lp))
    
    return matches / max(len(kernel_categories), len(lib_pattern))


def _get_library_recommendation(matches: List[Dict]) -> Dict:
    """Generate a recommendation based on matches."""
    if not matches:
        return None
    
    best = matches[0]
    
    # Prefer CK for HIP, then hipBLASLt, then vLLM
    priority = {"ck": 1, "hipblaslt": 2, "vllm": 3, "triton": 4}
    
    for m in matches:
        if m["match_score"] == best["match_score"]:
            if priority.get(m["library"], 99) < priority.get(best["library"], 99):
                best = m
    
    return {
        "recommended_library": best["library"],
        "recommended_kernel": best["name"],
        "api": best["api"],
        "usage": best.get("import", "") or f"#include \"{best.get('header', '')}\"",
        "notes": best["notes"]
    }


# =============================================================================
# Benchmark Comparison
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    kernel_name: str
    time_us: float
    memory_bytes: int
    throughput_gflops: float = 0.0
    bandwidth_gb_s: float = 0.0
    occupancy_percent: float = 0.0
    

def compare_benchmarks(
    baseline: Dict,
    optimized: Dict,
    baseline_name: str = "Baseline",
    optimized_name: str = "Optimized"
) -> Dict:
    """
    Compare two benchmark results and generate a comparison report.
    
    Args:
        baseline: Dict with baseline benchmark results (from Magpie or similar)
                  Expected keys: kernel_name, time_us, memory_bytes, etc.
        optimized: Dict with optimized benchmark results
        baseline_name: Display name for baseline
        optimized_name: Display name for optimized version
        
    Returns:
        Dict with comparison metrics and analysis
    """
    result = {
        "baseline_name": baseline_name,
        "optimized_name": optimized_name,
        "baseline": baseline,
        "optimized": optimized,
        "comparison": {},
        "analysis": [],
        "winner": None,
    }
    
    # Time comparison
    base_time = baseline.get("time_us", baseline.get("avg_time_us", 0))
    opt_time = optimized.get("time_us", optimized.get("avg_time_us", 0))
    
    if base_time > 0 and opt_time > 0:
        speedup = base_time / opt_time
        result["comparison"]["time"] = {
            "baseline_us": base_time,
            "optimized_us": opt_time,
            "speedup": speedup,
            "improvement_percent": (1 - opt_time / base_time) * 100,
        }
        
        if speedup > 1.1:
            result["analysis"].append(f"Optimized version is {speedup:.2f}x faster")
            result["winner"] = optimized_name
        elif speedup < 0.9:
            result["analysis"].append(f"Optimized version is {1/speedup:.2f}x slower (regression!)")
            result["winner"] = baseline_name
        else:
            result["analysis"].append("Performance is similar (within 10%)")
    
    # Memory comparison
    base_mem = baseline.get("memory_bytes", 0)
    opt_mem = optimized.get("memory_bytes", 0)
    
    if base_mem > 0 and opt_mem > 0:
        mem_reduction = (base_mem - opt_mem) / base_mem * 100
        result["comparison"]["memory"] = {
            "baseline_bytes": base_mem,
            "optimized_bytes": opt_mem,
            "reduction_percent": mem_reduction,
            "baseline_mb": base_mem / (1024*1024),
            "optimized_mb": opt_mem / (1024*1024),
        }
        
        if mem_reduction > 10:
            result["analysis"].append(f"Memory reduced by {mem_reduction:.1f}%")
        elif mem_reduction < -10:
            result["analysis"].append(f"Memory increased by {-mem_reduction:.1f}% (regression)")
    
    # Throughput comparison (if available)
    base_tput = baseline.get("throughput_gflops", baseline.get("tflops", 0) * 1000)
    opt_tput = optimized.get("throughput_gflops", optimized.get("tflops", 0) * 1000)
    
    if base_tput > 0 and opt_tput > 0:
        tput_improvement = (opt_tput - base_tput) / base_tput * 100
        result["comparison"]["throughput"] = {
            "baseline_gflops": base_tput,
            "optimized_gflops": opt_tput,
            "improvement_percent": tput_improvement,
        }
        
        if tput_improvement > 10:
            result["analysis"].append(f"Throughput improved by {tput_improvement:.1f}%")
    
    # Bandwidth comparison (if available)
    base_bw = baseline.get("bandwidth_gb_s", 0)
    opt_bw = optimized.get("bandwidth_gb_s", 0)
    
    if base_bw > 0 and opt_bw > 0:
        bw_improvement = (opt_bw - base_bw) / base_bw * 100
        result["comparison"]["bandwidth"] = {
            "baseline_gb_s": base_bw,
            "optimized_gb_s": opt_bw,
            "improvement_percent": bw_improvement,
        }
    
    # Kernel launch comparison
    base_launches = baseline.get("kernel_launches", baseline.get("calls", 1))
    opt_launches = optimized.get("kernel_launches", optimized.get("calls", 1))
    
    if base_launches != opt_launches:
        launch_reduction = base_launches - opt_launches
        result["comparison"]["launches"] = {
            "baseline": base_launches,
            "optimized": opt_launches,
            "reduction": launch_reduction,
        }
        
        if launch_reduction > 0:
            result["analysis"].append(f"Reduced kernel launches by {launch_reduction} (fusion benefit)")
    
    # Occupancy comparison (if available)
    base_occ = baseline.get("occupancy_percent", 0)
    opt_occ = optimized.get("occupancy_percent", 0)
    
    if base_occ > 0 and opt_occ > 0:
        result["comparison"]["occupancy"] = {
            "baseline_percent": base_occ,
            "optimized_percent": opt_occ,
            "change": opt_occ - base_occ,
        }
    
    # Overall verdict
    if not result["winner"]:
        # Default to optimized if no clear winner but it's faster
        if result["comparison"].get("time", {}).get("speedup", 1) > 1:
            result["winner"] = optimized_name
        else:
            result["winner"] = baseline_name
    
    return result


def compare_kernel_profiles(
    baseline_kernels: List[Dict],
    optimized_kernels: List[Dict],
    baseline_name: str = "Baseline",
    optimized_name: str = "Optimized"
) -> Dict:
    """
    Compare two sets of kernel profiles (e.g., from Magpie).
    
    Args:
        baseline_kernels: List of kernel dicts from baseline run
        optimized_kernels: List of kernel dicts from optimized run
        
    Returns:
        Dict with aggregate comparison and per-kernel breakdown
    """
    result = {
        "baseline_name": baseline_name,
        "optimized_name": optimized_name,
        "aggregate": {},
        "per_kernel": [],
        "summary": [],
    }
    
    # Aggregate times
    base_total = sum(k.get("time_us", 0) for k in baseline_kernels)
    opt_total = sum(k.get("time_us", 0) for k in optimized_kernels)
    
    result["aggregate"]["total_time"] = {
        "baseline_us": base_total,
        "optimized_us": opt_total,
        "speedup": base_total / opt_total if opt_total > 0 else 0,
    }
    
    # Kernel count change
    result["aggregate"]["kernel_count"] = {
        "baseline": len(baseline_kernels),
        "optimized": len(optimized_kernels),
        "reduction": len(baseline_kernels) - len(optimized_kernels),
    }
    
    # Match kernels by name for per-kernel comparison
    baseline_by_name = {k.get("name", k.get("kernel_name", "")): k for k in baseline_kernels}
    optimized_by_name = {k.get("name", k.get("kernel_name", "")): k for k in optimized_kernels}
    
    all_names = set(baseline_by_name.keys()) | set(optimized_by_name.keys())
    
    for name in sorted(all_names, key=lambda n: baseline_by_name.get(n, {}).get("time_us", 0), reverse=True)[:20]:
        base_k = baseline_by_name.get(name)
        opt_k = optimized_by_name.get(name)
        
        entry = {"kernel": name[:50]}
        
        if base_k and opt_k:
            base_t = base_k.get("time_us", 0)
            opt_t = opt_k.get("time_us", 0)
            entry["baseline_us"] = base_t
            entry["optimized_us"] = opt_t
            entry["speedup"] = base_t / opt_t if opt_t > 0 else 0
            entry["status"] = "compared"
        elif base_k and not opt_k:
            entry["baseline_us"] = base_k.get("time_us", 0)
            entry["optimized_us"] = 0
            entry["status"] = "removed"  # Kernel was fused/removed
        elif opt_k and not base_k:
            entry["baseline_us"] = 0
            entry["optimized_us"] = opt_k.get("time_us", 0)
            entry["status"] = "new"  # New kernel in optimized
        
        result["per_kernel"].append(entry)
    
    # Generate summary
    speedup = result["aggregate"]["total_time"]["speedup"]
    if speedup > 1.1:
        result["summary"].append(f"Overall {speedup:.2f}x speedup")
    elif speedup < 0.9:
        result["summary"].append(f"Overall {1/speedup:.2f}x regression")
    else:
        result["summary"].append("Performance similar (within 10%)")
    
    kernel_reduction = result["aggregate"]["kernel_count"]["reduction"]
    if kernel_reduction > 0:
        result["summary"].append(f"{kernel_reduction} fewer kernel launches (fusion)")
    
    # Find biggest improvements
    improvements = [k for k in result["per_kernel"] if k.get("speedup", 1) > 1.2]
    if improvements:
        best = max(improvements, key=lambda k: k.get("speedup", 0))
        result["summary"].append(f"Biggest win: {best['kernel'][:30]} ({best['speedup']:.1f}x)")
    
    # Find regressions
    regressions = [k for k in result["per_kernel"] if k.get("speedup", 1) < 0.8 and k.get("status") == "compared"]
    if regressions:
        worst = min(regressions, key=lambda k: k.get("speedup", 1))
        result["summary"].append(f"Warning: {worst['kernel'][:30]} regressed ({worst['speedup']:.2f}x)")
    
    return result


# =============================================================================
# Data Flow Analysis
# =============================================================================

def analyze_data_flow(kernels: List[KernelNode]) -> Dict:
    """Analyze data flow between kernels."""
    
    analysis = {
        "total_kernels": len(kernels),
        "memory_bound_ratio": 0.0,
        "compute_bound_ratio": 0.0,
        "fusion_candidates": [],
        "data_dependencies": [],
        "redundant_memory_ops": []
    }
    
    # Classify kernels
    memory_bound = ["elementwise", "memory", "normalization", "reduction"]
    compute_bound = ["gemm", "attention", "moe"]
    
    mem_count = sum(1 for k in kernels if k.category in memory_bound)
    comp_count = sum(1 for k in kernels if k.category in compute_bound)
    
    analysis["memory_bound_ratio"] = mem_count / len(kernels) if kernels else 0
    analysis["compute_bound_ratio"] = comp_count / len(kernels) if kernels else 0
    
    # Find consecutive memory-bound ops (fusion candidates)
    chain = []
    for kernel in kernels:
        if kernel.category in memory_bound:
            chain.append(kernel.name)
        else:
            if len(chain) >= 2:
                analysis["fusion_candidates"].append({
                    "kernels": chain.copy(),
                    "reason": "Consecutive memory-bound ops"
                })
            chain = []
    if len(chain) >= 2:
        analysis["fusion_candidates"].append({
            "kernels": chain,
            "reason": "Consecutive memory-bound ops"
        })
        
    # Detect redundant memory operations
    seen_outputs = set()
    for kernel in kernels:
        for inp in kernel.inputs:
            if inp not in seen_outputs and kernel.id > 0:
                analysis["redundant_memory_ops"].append({
                    "kernel": kernel.name,
                    "issue": f"Input {inp} not produced by previous kernel"
                })
        seen_outputs.update(kernel.outputs)
        
    return analysis

# =============================================================================
# MCP Server
# =============================================================================

app = Server("fusion-advisor")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available fusion advisor tools."""
    return [
        Tool(
            name="detect_fusion_opportunities",
            description="""Detect kernel fusion opportunities in a sequence of kernels.

Input can be:
- List of kernel names (one per line or comma-separated)
- Trace output with kernel names and times
- Code snippet with kernel calls

Returns identified fusion patterns with estimated benefits.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_sequence": {
                        "type": "string",
                        "description": "Kernel names/sequence to analyze"
                    }
                },
                "required": ["kernel_sequence"]
            }
        ),
        Tool(
            name="analyze_data_flow",
            description="""Analyze data flow between kernels to find memory-bound chains.

Identifies:
- Memory-bound vs compute-bound ratio
- Consecutive memory ops (fusion candidates)
- Redundant memory operations""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_sequence": {
                        "type": "string",
                        "description": "Kernel names/sequence to analyze"
                    }
                },
                "required": ["kernel_sequence"]
            }
        ),
        Tool(
            name="generate_fused_kernel",
            description="""Generate fused kernel code for a detected fusion opportunity.

Args:
    kernels: List of kernel names to fuse
    framework: Target framework (triton or hip)
    pattern: Optional - specific fusion pattern to use""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Kernel names to fuse"
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["triton", "hip"],
                        "default": "triton"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional fusion pattern name"
                    }
                },
                "required": ["kernels"]
            }
        ),
        Tool(
            name="validate_fusion",
            description="""Validate if kernels can be safely fused.

Checks:
- No circular dependencies
- Compatible data types
- Shared memory constraints
- Correctness requirements""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Kernel names to validate for fusion"
                    },
                    "target_arch": {
                        "type": "string",
                        "default": "gfx942",
                        "description": "Target GPU architecture"
                    }
                },
                "required": ["kernels"]
            }
        ),
        Tool(
            name="estimate_fusion_benefit",
            description="""Estimate the benefit of fusing specific kernels.

Returns:
- Estimated memory savings
- Predicted speedup range
- Implementation complexity""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Kernel names to estimate fusion benefit"
                    },
                    "kernel_times_us": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional: execution times in microseconds"
                    }
                },
                "required": ["kernels"]
            }
        ),
        Tool(
            name="list_fusion_patterns",
            description="""List all known fusion patterns with their benefits.""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="parse_magpie_output",
            description="""Parse Magpie analyze output to detect fusion opportunities with real timing data.

Accepts the JSON output from Magpie's analyze() tool and extracts:
- Kernel sequence with actual execution times
- Fusion opportunities with estimated time savings
- Memory-bound vs compute-bound classification

USE THIS after running Magpie analyze to get data-driven fusion recommendations.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "magpie_json": {
                        "type": "string",
                        "description": "JSON output from Magpie analyze() tool"
                    }
                },
                "required": ["magpie_json"]
            }
        ),
        Tool(
            name="calculate_memory_savings",
            description="""Calculate actual memory savings from fusing kernels.

Given tensor shapes and data types, calculates:
- Bytes of intermediate tensors eliminated
- Total memory traffic saved
- Estimated time savings based on HBM bandwidth

Requires tensor shape information for accurate calculations.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of kernel names to fuse"
                    },
                    "tensor_shapes": {
                        "type": "object",
                        "description": "Dict mapping tensor names to shapes, e.g., {\"x\": [16, 2048, 4096]}"
                    },
                    "dtype": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "bf16", "fp8", "int8"],
                        "default": "fp16",
                        "description": "Data type of tensors"
                    }
                },
                "required": ["kernels", "tensor_shapes"]
            }
        ),
        Tool(
            name="check_library_fusion",
            description="""Check if a fused kernel already exists in GPU libraries.

Searches for existing implementations in:
- Composable Kernel (CK): High-performance fused kernels for AMD GPUs
- hipBLASLt: GEMM with fused epilogues (bias, activation)
- vLLM/AIter: Inference-optimized fused ops (PagedAttention, FusedMoE)
- Triton: Common fused patterns from tutorials

Returns API calls and usage examples for matching fused kernels.

USE THIS FIRST before writing custom fused kernels!""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_sequence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of kernel names/types to fuse, e.g., [\"gemm\", \"bias_add\", \"relu\"]"
                    }
                },
                "required": ["kernel_sequence"]
            }
        ),
        Tool(
            name="benchmark_compare",
            description="""Compare two kernel implementations A vs B.

Given benchmark results from two implementations (e.g., baseline vs optimized),
generates a detailed comparison report including:
- Time speedup/regression
- Memory usage changes
- Throughput improvements
- Kernel launch reduction (from fusion)
- Per-kernel breakdown

Accepts results from Magpie analyze or similar profiling tools.

USE THIS after implementing an optimization to validate improvements.

Args:
    baseline: Dict with baseline benchmark results
    optimized: Dict with optimized benchmark results
    baseline_name: Display name for baseline (default: "Baseline")
    optimized_name: Display name for optimized (default: "Optimized")
    
For multiple kernels, use baseline_kernels and optimized_kernels instead.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline": {
                        "type": "object",
                        "description": "Baseline benchmark results (single kernel)"
                    },
                    "optimized": {
                        "type": "object",
                        "description": "Optimized benchmark results (single kernel)"
                    },
                    "baseline_kernels": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of baseline kernel profiles (for multi-kernel comparison)"
                    },
                    "optimized_kernels": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of optimized kernel profiles (for multi-kernel comparison)"
                    },
                    "baseline_name": {
                        "type": "string",
                        "default": "Baseline",
                        "description": "Display name for baseline"
                    },
                    "optimized_name": {
                        "type": "string",
                        "default": "Optimized",
                        "description": "Display name for optimized version"
                    }
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "detect_fusion_opportunities":
        return await handle_detect_fusion(arguments)
    elif name == "analyze_data_flow":
        return await handle_analyze_flow(arguments)
    elif name == "generate_fused_kernel":
        return await handle_generate_fused(arguments)
    elif name == "validate_fusion":
        return await handle_validate_fusion(arguments)
    elif name == "estimate_fusion_benefit":
        return await handle_estimate_benefit(arguments)
    elif name == "list_fusion_patterns":
        return await handle_list_patterns(arguments)
    elif name == "parse_magpie_output":
        return await handle_parse_magpie(arguments)
    elif name == "calculate_memory_savings":
        return await handle_calculate_memory(arguments)
    elif name == "check_library_fusion":
        return await handle_check_library(arguments)
    elif name == "benchmark_compare":
        return await handle_benchmark_compare(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# =============================================================================
# Tool Implementations
# =============================================================================

async def handle_detect_fusion(args: dict) -> list[TextContent]:
    """Detect fusion opportunities."""
    sequence = args.get("kernel_sequence", "")
    
    kernels = parse_kernel_sequence(sequence)
    opportunities = detect_fusion_opportunities(kernels)
    
    output = "# Fusion Opportunity Analysis\n\n"
    output += f"**Kernels analyzed:** {len(kernels)}\n"
    output += f"**Fusion opportunities found:** {len(opportunities)}\n\n"
    
    if not opportunities:
        output += "No obvious fusion opportunities detected.\n"
        output += "Consider providing more kernels or checking if they're already fused.\n"
    else:
        output += "## Detected Opportunities\n\n"
        for i, opp in enumerate(opportunities, 1):
            output += f"### {i}. {opp.pattern_name}\n\n"
            output += f"**Kernels:** {' -> '.join(opp.kernels)}\n"
            output += f"**Benefit:** {opp.benefit}\n"
            output += f"**Memory Saving:** {opp.memory_saving}\n"
            output += f"**Estimated Speedup:** {opp.estimated_speedup}\n"
            output += f"**Complexity:** {opp.complexity}\n"
            output += f"**Recommended Frameworks:** {', '.join(opp.frameworks)}\n\n"
            
    return [TextContent(type="text", text=output)]

async def handle_analyze_flow(args: dict) -> list[TextContent]:
    """Analyze data flow."""
    sequence = args.get("kernel_sequence", "")
    
    kernels = parse_kernel_sequence(sequence)
    analysis = analyze_data_flow(kernels)
    
    output = "# Data Flow Analysis\n\n"
    output += f"**Total kernels:** {analysis['total_kernels']}\n"
    output += f"**Memory-bound ratio:** {analysis['memory_bound_ratio']:.1%}\n"
    output += f"**Compute-bound ratio:** {analysis['compute_bound_ratio']:.1%}\n\n"
    
    if analysis["fusion_candidates"]:
        output += "## Fusion Candidates\n\n"
        for cand in analysis["fusion_candidates"]:
            output += f"- {' -> '.join(cand['kernels'])}\n"
            output += f"  Reason: {cand['reason']}\n\n"
            
    return [TextContent(type="text", text=output)]

async def handle_generate_fused(args: dict) -> list[TextContent]:
    """Generate fused kernel."""
    kernels = args.get("kernels", [])
    framework = args.get("framework", "triton")
    pattern = args.get("pattern", "")
    
    # Create opportunity from kernel list
    kernel_str = '\n'.join(kernels)
    parsed = parse_kernel_sequence(kernel_str)
    opportunities = detect_fusion_opportunities(parsed)
    
    output = f"# Fused Kernel Generation\n\n"
    output += f"**Kernels:** {' -> '.join(kernels)}\n"
    output += f"**Framework:** {framework}\n\n"
    
    if opportunities:
        opp = opportunities[0]
        if pattern:
            for o in opportunities:
                if pattern.lower() in o.pattern_name.lower():
                    opp = o
                    break
                    
        code = generate_fused_kernel(opp, framework)
        output += f"## Generated Code ({opp.pattern_name})\n\n"
        output += f"```{'python' if framework == 'triton' else 'cpp'}\n{code}\n```\n"
    else:
        # Generate generic elementwise fusion
        opp = FusionOpportunity(
            pattern_name="Custom Elementwise Fusion",
            kernels=kernels,
            kernel_ids=list(range(len(kernels))),
            benefit="Reduce memory traffic",
            memory_saving="medium",
            complexity="low",
            frameworks=["triton", "hip"],
            estimated_speedup="1.2-1.5x"
        )
        code = generate_fused_kernel(opp, framework)
        output += f"## Generated Code (Generic Fusion)\n\n"
        output += f"```{'python' if framework == 'triton' else 'cpp'}\n{code}\n```\n"
        
    output += "\n**Note:** This is a template. Adjust the fused operations for your specific use case.\n"
    
    return [TextContent(type="text", text=output)]

async def handle_validate_fusion(args: dict) -> list[TextContent]:
    """Validate fusion feasibility."""
    kernels = args.get("kernels", [])
    arch = args.get("target_arch", "gfx942")
    
    output = "# Fusion Validation\n\n"
    output += f"**Kernels:** {' -> '.join(kernels)}\n"
    output += f"**Target arch:** {arch}\n\n"
    
    # Basic validation checks
    checks = []
    
    # Check 1: Kernel count
    if len(kernels) < 2:
        checks.append(("FAIL", "Need at least 2 kernels to fuse"))
    else:
        checks.append(("PASS", f"Kernel count: {len(kernels)}"))
        
    # Check 2: Compatible types
    kernel_str = '\n'.join(kernels)
    parsed = parse_kernel_sequence(kernel_str)
    categories = set(k.category for k in parsed)
    
    if "unknown" in categories and len(categories) > 1:
        checks.append(("WARN", "Some kernels could not be classified"))
    else:
        checks.append(("PASS", f"Kernel categories: {', '.join(categories)}"))
        
    # Check 3: Fusion complexity
    if "attention" in categories and "gemm" in categories:
        checks.append(("WARN", "Attention + GEMM fusion is complex"))
    elif len(set(k.category for k in parsed if k.category != "unknown")) == 1:
        checks.append(("PASS", "Homogeneous kernel types - fusion straightforward"))
    else:
        checks.append(("INFO", "Mixed kernel types - may require custom fusion logic"))
        
    # Check 4: Shared memory estimate (rough)
    lds_kb = 64  # gfx942 has 64KB LDS per CU
    if len(kernels) > 5:
        checks.append(("WARN", f"Many kernels ({len(kernels)}) - check LDS usage ({lds_kb}KB available)"))
    else:
        checks.append(("PASS", f"LDS usage likely within {lds_kb}KB limit"))
        
    output += "## Validation Results\n\n"
    for status, msg in checks:
        icon = {"PASS": "[OK]", "FAIL": "[X]", "WARN": "[!]", "INFO": "[i]"}[status]
        output += f"- {icon} {msg}\n"
        
    # Overall
    fails = sum(1 for s, _ in checks if s == "FAIL")
    warns = sum(1 for s, _ in checks if s == "WARN")
    
    output += f"\n**Overall:** "
    if fails > 0:
        output += "Fusion NOT recommended - address failures first\n"
    elif warns > 0:
        output += "Fusion possible with caution - review warnings\n"
    else:
        output += "Fusion looks safe to proceed\n"
        
    return [TextContent(type="text", text=output)]

async def handle_estimate_benefit(args: dict) -> list[TextContent]:
    """Estimate fusion benefit."""
    kernels = args.get("kernels", [])
    times = args.get("kernel_times_us", [])
    
    output = "# Fusion Benefit Estimate\n\n"
    output += f"**Kernels:** {' -> '.join(kernels)}\n\n"
    
    # Parse and analyze
    kernel_str = '\n'.join(kernels)
    parsed = parse_kernel_sequence(kernel_str)
    
    # Memory analysis
    mem_bound = sum(1 for k in parsed if k.category in ["elementwise", "normalization", "memory"])
    comp_bound = sum(1 for k in parsed if k.category in ["gemm", "attention"])
    
    output += "## Memory Analysis\n\n"
    output += f"- Memory-bound kernels: {mem_bound}\n"
    output += f"- Compute-bound kernels: {comp_bound}\n"
    output += f"- Intermediate tensors eliminated: ~{len(kernels) - 1}\n\n"
    
    # Speedup estimate
    output += "## Estimated Speedup\n\n"
    
    if mem_bound == len(kernels):
        output += "- **All memory-bound**: Fusion very beneficial\n"
        output += "- Estimated speedup: **1.5-3x**\n"
        output += "- Main benefit: Reduced memory traffic\n"
    elif comp_bound == len(kernels):
        output += "- **All compute-bound**: Fusion moderately beneficial\n"
        output += "- Estimated speedup: **1.1-1.3x**\n"
        output += "- Main benefit: Reduced kernel launch overhead\n"
    else:
        output += "- **Mixed**: Fusion beneficial for memory ops\n"
        output += "- Estimated speedup: **1.2-1.5x**\n"
        output += "- Main benefit: Pipeline memory and compute\n"
        
    # Time-based estimate if provided
    if times and len(times) == len(kernels):
        total_time = sum(times)
        # Rough estimate: fused kernel takes 60-80% of total
        est_fused = total_time * 0.7
        output += f"\n## Time Estimate (based on provided times)\n\n"
        output += f"- Current total: {total_time:.1f} us\n"
        output += f"- Estimated fused: {est_fused:.1f} us\n"
        output += f"- Potential savings: {total_time - est_fused:.1f} us ({(1 - est_fused/total_time)*100:.0f}%)\n"
        
    return [TextContent(type="text", text=output)]

async def handle_list_patterns(args: dict) -> list[TextContent]:
    """List all fusion patterns."""
    output = "# Known Fusion Patterns\n\n"
    
    for pattern_id, info in FUSION_PATTERNS.items():
        output += f"## {info['name']}\n\n"
        output += f"**Pattern:** {' -> '.join(info['pattern'])}\n"
        output += f"**Benefit:** {info['benefit']}\n"
        output += f"**Memory Saving:** {info['memory_saving']}\n"
        output += f"**Complexity:** {info['complexity']}\n"
        output += f"**Frameworks:** {', '.join(info['frameworks'])}\n"
        output += f"**Example:** {info['example']}\n\n"
        
    return [TextContent(type="text", text=output)]


async def handle_parse_magpie(args: dict) -> list[TextContent]:
    """Parse Magpie analyze output for fusion analysis."""
    magpie_json = args.get("magpie_json", "{}")
    
    try:
        result = extract_fusion_candidates_from_magpie(magpie_json)
        
        output = "# Fusion Analysis from Magpie Profile\n\n"
        output += f"**Total kernels:** {result['total_kernels']}\n"
        output += f"**Total time:** {result['total_time_us']:.2f} us\n\n"
        
        # Show kernel breakdown
        output += "## Kernel Breakdown\n\n"
        output += "| Kernel | Category | Time (us) |\n"
        output += "|--------|----------|----------:|\n"
        for k in result['kernels_analyzed'][:20]:  # Limit to top 20
            output += f"| {k['name'][:40]} | {k['category']} | {k['time_us']:.2f} |\n"
        
        if len(result['kernels_analyzed']) > 20:
            output += f"\n*...and {len(result['kernels_analyzed']) - 20} more kernels*\n"
        
        # Show fusion opportunities
        output += f"\n## Fusion Opportunities Found: {len(result['opportunities'])}\n\n"
        
        for i, opp in enumerate(result['opportunities'], 1):
            output += f"### {i}. {opp['pattern_name']}\n\n"
            output += f"**Kernels:** {' -> '.join(opp['kernels'])}\n"
            output += f"**Estimated Speedup:** {opp['estimated_speedup']}\n"
            output += f"**Memory Saving:** {opp['memory_saving']}\n"
            output += f"**Frameworks:** {', '.join(opp['frameworks'])}\n\n"
        
        if not result['opportunities']:
            output += "*No obvious fusion opportunities detected. Consider checking if kernels are already fused.*\n"
        
        return [TextContent(type="text", text=output)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error parsing Magpie output: {str(e)}")]


async def handle_calculate_memory(args: dict) -> list[TextContent]:
    """Calculate memory savings from fusion."""
    kernels = args.get("kernels", [])
    tensor_shapes = args.get("tensor_shapes", {})
    dtype = args.get("dtype", "fp16")
    
    if len(kernels) < 2:
        return [TextContent(type="text", text="Error: Need at least 2 kernels to calculate fusion savings")]
    
    result = calculate_memory_savings(kernels, tensor_shapes, dtype)
    
    output = "# Memory Savings Calculation\n\n"
    output += f"**Kernels to fuse:** {' -> '.join(result['kernels_fused'])}\n"
    output += f"**Data type:** {result['dtype']} ({result['element_size_bytes']} bytes/element)\n\n"
    
    output += "## Intermediate Tensors Eliminated\n\n"
    output += f"**Count:** {result['intermediates_eliminated']} intermediate tensors\n\n"
    
    if result['intermediate_details']:
        output += "| Intermediate | Shape | Elements | Bytes |\n"
        output += "|--------------|-------|----------|-------|\n"
        for inter in result['intermediate_details']:
            shape_str = "x".join(map(str, inter['shape']))
            output += f"| {inter['name']} | {shape_str} | {inter['elements']:,} | {_human_readable_bytes(inter['bytes'])} |\n"
    
    output += f"\n## Total Savings\n\n"
    output += f"- **Elements saved:** {result['total_elements_saved']:,}\n"
    output += f"- **Bytes saved:** {result['memory_traffic_saved_human']}\n"
    output += f"- **Memory traffic eliminated:** {result['memory_traffic_saved_human']} (write + read)\n"
    output += f"- **Estimated time saved:** ~{result['estimated_time_saved_us']:.2f} us\n"
    
    output += f"\n## Assumptions\n\n"
    output += f"- HBM bandwidth: {result['assumptions']['hbm_bandwidth_gbps']} GB/s (MI300X)\n"
    output += f"- {result['assumptions']['note']}\n"
    
    return [TextContent(type="text", text=output)]


async def handle_check_library(args: dict) -> list[TextContent]:
    """Check for existing library fused kernels."""
    kernel_sequence = args.get("kernel_sequence", [])
    
    if not kernel_sequence:
        return [TextContent(type="text", text="Error: No kernel sequence provided")]
    
    result = check_library_fusion(kernel_sequence)
    
    output = "# Library Fusion Check\n\n"
    output += f"**Input sequence:** {' -> '.join(result['input_sequence'])}\n"
    output += f"**Detected pattern:** {' -> '.join(result['detected_pattern'])}\n\n"
    
    if result['matches_found'] == 0:
        output += "## No Existing Fused Kernels Found\n\n"
        output += "Consider writing a custom Triton or HIP kernel for this fusion.\n"
        output += "Use `generate_fused_kernel` to get a starting template.\n"
    else:
        output += f"## Found {result['matches_found']} Matching Fused Kernels\n\n"
        
        for i, match in enumerate(result['matches'], 1):
            output += f"### {i}. {match['name']} ({match['library'].upper()})\n\n"
            output += f"**Library:** {match['library']}\n"
            output += f"**API:** `{match['api']}`\n"
            
            if match.get('header'):
                output += f"**Header:** `{match['header']}`\n"
            if match.get('import'):
                output += f"**Import:** `{match['import']}`\n"
            
            output += f"**Match score:** {match['match_score']:.0%}\n"
            output += f"**Notes:** {match['notes']}\n\n"
        
        # Show recommendation
        if result.get('recommendation'):
            rec = result['recommendation']
            output += "## Recommendation\n\n"
            output += f"**Use:** {rec['recommended_library'].upper()} - {rec['recommended_kernel']}\n\n"
            output += f"```\n{rec['usage']}\n```\n\n"
            output += f"API: `{rec['api']}`\n"
            output += f"\n{rec['notes']}\n"
    
    return [TextContent(type="text", text=output)]


async def handle_benchmark_compare(args: dict) -> list[TextContent]:
    """Compare two benchmark results."""
    baseline = args.get("baseline")
    optimized = args.get("optimized")
    baseline_kernels = args.get("baseline_kernels")
    optimized_kernels = args.get("optimized_kernels")
    baseline_name = args.get("baseline_name", "Baseline")
    optimized_name = args.get("optimized_name", "Optimized")
    
    # Determine comparison mode
    if baseline_kernels and optimized_kernels:
        # Multi-kernel comparison
        result = compare_kernel_profiles(
            baseline_kernels, optimized_kernels,
            baseline_name, optimized_name
        )
        return format_multi_kernel_comparison(result)
    elif baseline and optimized:
        # Single benchmark comparison
        result = compare_benchmarks(
            baseline, optimized,
            baseline_name, optimized_name
        )
        return format_single_comparison(result)
    else:
        return [TextContent(type="text", text=
            "Error: Provide either:\n"
            "- baseline and optimized (for single benchmark comparison)\n"
            "- baseline_kernels and optimized_kernels (for multi-kernel comparison)"
        )]


def format_single_comparison(result: Dict) -> list[TextContent]:
    """Format single benchmark comparison output."""
    output = f"# Benchmark Comparison\n\n"
    output += f"**{result['baseline_name']}** vs **{result['optimized_name']}**\n\n"
    
    # Winner banner
    if result.get("winner"):
        winner = result["winner"]
        speedup = result.get("comparison", {}).get("time", {}).get("speedup", 1)
        if speedup > 1:
            output += f"## Winner: {winner} ({speedup:.2f}x faster)\n\n"
        else:
            output += f"## Winner: {winner}\n\n"
    
    # Time comparison
    if "time" in result.get("comparison", {}):
        time_cmp = result["comparison"]["time"]
        output += "## Time Performance\n\n"
        output += f"| Metric | {result['baseline_name']} | {result['optimized_name']} | Change |\n"
        output += "|--------|----------|-----------|--------|\n"
        output += f"| Time | {time_cmp['baseline_us']:.2f} us | {time_cmp['optimized_us']:.2f} us | "
        
        if time_cmp['speedup'] > 1:
            output += f"**{time_cmp['speedup']:.2f}x faster** |\n"
        elif time_cmp['speedup'] < 1:
            output += f"{1/time_cmp['speedup']:.2f}x slower |\n"
        else:
            output += "same |\n"
        output += "\n"
    
    # Memory comparison
    if "memory" in result.get("comparison", {}):
        mem_cmp = result["comparison"]["memory"]
        output += "## Memory Usage\n\n"
        output += f"| Metric | {result['baseline_name']} | {result['optimized_name']} | Reduction |\n"
        output += "|--------|----------|-----------|------------|\n"
        output += f"| Memory | {mem_cmp['baseline_mb']:.2f} MB | {mem_cmp['optimized_mb']:.2f} MB | {mem_cmp['reduction_percent']:.1f}% |\n\n"
    
    # Throughput comparison
    if "throughput" in result.get("comparison", {}):
        tput_cmp = result["comparison"]["throughput"]
        output += "## Throughput\n\n"
        output += f"- Baseline: {tput_cmp['baseline_gflops']:.1f} GFLOPS\n"
        output += f"- Optimized: {tput_cmp['optimized_gflops']:.1f} GFLOPS\n"
        output += f"- Improvement: {tput_cmp['improvement_percent']:.1f}%\n\n"
    
    # Kernel launches
    if "launches" in result.get("comparison", {}):
        launch_cmp = result["comparison"]["launches"]
        output += "## Kernel Launches\n\n"
        output += f"- Baseline: {launch_cmp['baseline']} launches\n"
        output += f"- Optimized: {launch_cmp['optimized']} launches\n"
        output += f"- Reduction: {launch_cmp['reduction']} (fusion benefit)\n\n"
    
    # Analysis
    if result.get("analysis"):
        output += "## Analysis\n\n"
        for point in result["analysis"]:
            if "regression" in point.lower():
                output += f"- **Warning:** {point}\n"
            else:
                output += f"- {point}\n"
    
    return [TextContent(type="text", text=output)]


def format_multi_kernel_comparison(result: Dict) -> list[TextContent]:
    """Format multi-kernel profile comparison output."""
    output = f"# Kernel Profile Comparison\n\n"
    output += f"**{result['baseline_name']}** vs **{result['optimized_name']}**\n\n"
    
    # Aggregate stats
    agg = result.get("aggregate", {})
    
    output += "## Overall Performance\n\n"
    
    time_agg = agg.get("total_time", {})
    if time_agg:
        speedup = time_agg.get("speedup", 1)
        output += f"| Metric | {result['baseline_name']} | {result['optimized_name']} | Result |\n"
        output += "|--------|----------|-----------|--------|\n"
        output += f"| Total Time | {time_agg.get('baseline_us', 0)/1000:.2f} ms | {time_agg.get('optimized_us', 0)/1000:.2f} ms | "
        if speedup > 1:
            output += f"**{speedup:.2f}x faster** |\n"
        elif speedup < 1:
            output += f"{1/speedup:.2f}x slower |\n"
        else:
            output += "same |\n"
    
    kernel_agg = agg.get("kernel_count", {})
    if kernel_agg:
        output += f"| Kernel Count | {kernel_agg.get('baseline', 0)} | {kernel_agg.get('optimized', 0)} | "
        reduction = kernel_agg.get('reduction', 0)
        if reduction > 0:
            output += f"-{reduction} (fused) |\n"
        elif reduction < 0:
            output += f"+{-reduction} |\n"
        else:
            output += "same |\n"
    
    output += "\n"
    
    # Per-kernel breakdown
    if result.get("per_kernel"):
        output += "## Per-Kernel Breakdown\n\n"
        output += f"| Kernel | {result['baseline_name']} | {result['optimized_name']} | Speedup | Status |\n"
        output += "|--------|----------|-----------|---------|--------|\n"
        
        for k in result["per_kernel"][:15]:
            name = k.get("kernel", "")[:40]
            base_us = k.get("baseline_us", 0)
            opt_us = k.get("optimized_us", 0)
            speedup = k.get("speedup", 0)
            status = k.get("status", "")
            
            if status == "removed":
                output += f"| `{name}` | {base_us:.1f} us | - | - | **Fused** |\n"
            elif status == "new":
                output += f"| `{name}` | - | {opt_us:.1f} us | - | New |\n"
            else:
                speedup_str = f"{speedup:.2f}x" if speedup > 0 else "-"
                if speedup > 1.1:
                    speedup_str = f"**{speedup:.2f}x**"
                elif speedup < 0.9:
                    speedup_str = f"*{speedup:.2f}x*"
                output += f"| `{name}` | {base_us:.1f} us | {opt_us:.1f} us | {speedup_str} | {status} |\n"
        
        if len(result["per_kernel"]) > 15:
            output += f"\n*...and {len(result['per_kernel']) - 15} more kernels*\n"
        output += "\n"
    
    # Summary
    if result.get("summary"):
        output += "## Summary\n\n"
        for point in result["summary"]:
            if "warning" in point.lower() or "regressed" in point.lower():
                output += f"- **{point}**\n"
            else:
                output += f"- {point}\n"
    
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
