#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
GPU Info MCP Server
===================
Auto-detect GPU architecture and provide hardware-specific optimization context.

Tools:
- get_gpu_info: Detect current GPU(s), return arch, specs, capabilities
- get_arch_optimization_hints: Return arch-specific optimization recommendations
- set_target_arch: Override target arch for cross-compilation
"""

import asyncio
import subprocess
import re
import json
from typing import Optional
from dataclasses import dataclass, asdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# =============================================================================
# GPU Architecture Database
# =============================================================================

ARCH_SPECS = {
    "gfx950": {
        "name": "AMD Instinct MI355X",
        "architecture": "CDNA4",
        "compute_units": 256,  # MI355X
        "stream_processors": 16384,
        "wavefront_size": 64,
        "lds_size_kb": 64,
        "l2_cache_mb": 256,
        "hbm_bandwidth_tb_s": 8.0,  # HBM3e
        "hbm_capacity_gb": 288,  # 288 GB HBM3e
        "fp64_tflops": 200.0,
        "fp32_tflops": 200.0,
        "fp16_tflops": 1600.0,
        "bf16_tflops": 1600.0,
        "fp8_tflops": 3200.0,
        "fp4_tflops": 6400.0,
        "mfma_support": True,
        "mfma_instructions": ["mfma_f32_32x32x8_f16", "mfma_f32_16x16x16_f16", 
                              "mfma_f32_32x32x16_bf16", "mfma_f32_16x16x32_bf16",
                              "mfma_f32_32x32x16_fp8", "mfma_f32_32x32x32_fp4",
                              "mfma_f64_16x16x4_f64"],
        "matrix_core_sizes": ["16x16", "32x32"],
        "optimal_tile_sizes": {
            "gemm_m": [64, 128, 256, 512],
            "gemm_n": [64, 128, 256, 512],
            "gemm_k": [16, 32, 64, 128]
        },
        "memory_coalescing_bytes": 128,
        "optimal_block_sizes": [64, 128, 256, 512],
        "max_workgroup_size": 1024,
        "optimization_priorities": [
            "Use MFMA instructions for matrix operations (CDNA4)",
            "Target 128-byte memory coalescing",
            "Use LDS for data reuse (64KB per CU)",
            "Leverage high HBM3e bandwidth (8.0 TB/s)",
            "Use FP8/FP4 for inference workloads",
            "Consider larger tile sizes for 256 CUs"
        ]
    },
    "gfx942": {
        "name": "AMD Instinct MI300X/MI300A",
        "architecture": "CDNA3",
        "compute_units": 304,  # MI300X
        "stream_processors": 19456,
        "wavefront_size": 64,
        "lds_size_kb": 64,
        "l2_cache_mb": 256,
        "hbm_bandwidth_tb_s": 5.3,
        "hbm_capacity_gb": 192,
        "fp64_tflops": 163.4,
        "fp32_tflops": 163.4,
        "fp16_tflops": 1307.4,
        "bf16_tflops": 1307.4,
        "fp8_tflops": 2614.9,
        "mfma_support": True,
        "mfma_instructions": ["mfma_f32_32x32x8_f16", "mfma_f32_16x16x16_f16", 
                              "mfma_f32_32x32x16_bf16", "mfma_f32_16x16x32_bf16",
                              "mfma_f32_32x32x16_fp8", "mfma_f64_16x16x4_f64"],
        "matrix_core_sizes": ["16x16", "32x32"],
        "optimal_tile_sizes": {
            "gemm_m": [32, 64, 128, 256],
            "gemm_n": [32, 64, 128, 256],
            "gemm_k": [8, 16, 32, 64]
        },
        "memory_coalescing_bytes": 128,
        "optimal_block_sizes": [64, 128, 256, 512],
        "max_workgroup_size": 1024,
        "optimization_priorities": [
            "Use MFMA instructions for matrix operations",
            "Target 128-byte memory coalescing",
            "Use LDS for data reuse (64KB per CU)",
            "Optimize for high HBM bandwidth (5.3 TB/s)",
            "Consider FP8/BF16 for inference workloads"
        ]
    },
    "gfx90a": {
        "name": "AMD Instinct MI210/MI250/MI250X",
        "architecture": "CDNA2",
        "compute_units": 110,  # MI250X per GCD
        "stream_processors": 7040,
        "wavefront_size": 64,
        "lds_size_kb": 64,
        "l2_cache_mb": 8,
        "hbm_bandwidth_tb_s": 3.2,
        "hbm_capacity_gb": 128,
        "fp64_tflops": 47.9,
        "fp32_tflops": 47.9,
        "fp16_tflops": 383.0,
        "bf16_tflops": 383.0,
        "mfma_support": True,
        "mfma_instructions": ["mfma_f32_32x32x8_f16", "mfma_f32_16x16x16_f16",
                              "mfma_f32_32x32x8_bf16", "mfma_f64_16x16x4_f64"],
        "matrix_core_sizes": ["16x16", "32x32"],
        "optimal_tile_sizes": {
            "gemm_m": [32, 64, 128],
            "gemm_n": [32, 64, 128],
            "gemm_k": [8, 16, 32]
        },
        "memory_coalescing_bytes": 128,
        "optimal_block_sizes": [64, 128, 256],
        "max_workgroup_size": 1024,
        "optimization_priorities": [
            "Use MFMA for matrix operations",
            "Optimize memory coalescing (128 bytes)",
            "Leverage LDS for shared data",
            "Consider multi-GCD for MI250X"
        ]
    },
    "gfx908": {
        "name": "AMD Instinct MI100",
        "architecture": "CDNA1",
        "compute_units": 120,
        "stream_processors": 7680,
        "wavefront_size": 64,
        "lds_size_kb": 64,
        "l2_cache_mb": 8,
        "hbm_bandwidth_tb_s": 1.23,
        "hbm_capacity_gb": 32,
        "fp64_tflops": 11.5,
        "fp32_tflops": 23.1,
        "fp16_tflops": 184.6,
        "mfma_support": True,
        "mfma_instructions": ["mfma_f32_32x32x4_f16", "mfma_f32_16x16x4_f16"],
        "matrix_core_sizes": ["16x16", "32x32"],
        "optimal_tile_sizes": {
            "gemm_m": [32, 64, 128],
            "gemm_n": [32, 64, 128],
            "gemm_k": [4, 8, 16]
        },
        "memory_coalescing_bytes": 128,
        "optimal_block_sizes": [64, 128, 256],
        "max_workgroup_size": 1024,
        "optimization_priorities": [
            "Use MFMA instructions",
            "Focus on memory bandwidth optimization",
            "Use LDS for data reuse"
        ]
    },
    "gfx1100": {
        "name": "AMD Radeon RX 7900 XTX",
        "architecture": "RDNA3",
        "compute_units": 96,
        "stream_processors": 6144,
        "wavefront_size": 32,  # RDNA uses wave32 by default
        "lds_size_kb": 128,
        "l2_cache_mb": 6,
        "memory_bandwidth_gb_s": 960,
        "memory_capacity_gb": 24,
        "fp32_tflops": 61.4,
        "fp16_tflops": 122.8,
        "wmma_support": True,
        "wmma_instructions": ["wmma_f32_16x16x16_f16"],
        "optimal_tile_sizes": {
            "gemm_m": [16, 32, 64],
            "gemm_n": [16, 32, 64],
            "gemm_k": [16, 32]
        },
        "memory_coalescing_bytes": 64,
        "optimal_block_sizes": [32, 64, 128, 256],
        "max_workgroup_size": 1024,
        "optimization_priorities": [
            "Use Wave32 for better occupancy on simple kernels",
            "Use WMMA for matrix operations",
            "Leverage larger LDS (128KB)",
            "Optimize for RDNA3 cache hierarchy"
        ]
    }
}

# Default arch for fallback
DEFAULT_ARCH = "gfx950"

# =============================================================================
# GPU Detection
# =============================================================================

@dataclass
class GPUInfo:
    """Detected GPU information."""
    arch: str
    name: str
    marketing_name: str
    compute_units: int
    memory_gb: float
    specs: dict
    
    def to_dict(self):
        return asdict(self)

class GPUDetector:
    """Detect and cache GPU information."""
    
    _cached_info: Optional[GPUInfo] = None
    _target_arch: Optional[str] = None
    
    @classmethod
    def detect_gpu(cls) -> GPUInfo:
        """Detect GPU using rocminfo."""
        if cls._cached_info is not None:
            return cls._cached_info
            
        try:
            result = subprocess.run(
                ["rocminfo"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            output = result.stdout
            
            # Parse rocminfo output
            arch = None
            marketing_name = None
            compute_units = None
            
            # Look for GPU agent
            in_gpu_section = False
            for line in output.split('\n'):
                # Look for GPU vendor (skip CPU agents)
                if 'Vendor Name:' in line and 'AMD' in line and 'CPU' not in line:
                    in_gpu_section = True
                elif 'Vendor Name:' in line and 'CPU' in line:
                    in_gpu_section = False
                elif in_gpu_section:
                    # Match architecture name like "gfx950", "gfx942", "gfx90a"
                    # Exclude ISA strings like "amdgcn-amd-amdhsa--gfx950:..."
                    if 'Name:' in line and 'gfx' in line and 'amdgcn' not in line:
                        match = re.search(r'gfx\d+[a-z]?', line)
                        if match:
                            arch = match.group()
                    elif 'Marketing Name:' in line:
                        marketing_name = line.split(':', 1)[1].strip()
                    elif 'Compute Unit:' in line:
                        cu_match = re.search(r'Compute Unit:\s*(\d+)', line)
                        if cu_match:
                            compute_units = int(cu_match.group(1))
                    elif 'Agent' in line and arch is not None:
                        # Found next agent, stop parsing
                        break
                        
            if arch is None:
                arch = DEFAULT_ARCH
                
            # Get specs from database
            specs = ARCH_SPECS.get(arch, ARCH_SPECS[DEFAULT_ARCH])
            
            # Use detected values or fall back to specs
            detected_cu = compute_units if compute_units else specs.get("compute_units", 0)
            detected_memory = specs.get("hbm_capacity_gb", specs.get("memory_capacity_gb", 0))
            
            # Update specs with detected values if they differ
            if compute_units and compute_units != specs.get("compute_units"):
                # Create a copy of specs with actual detected values
                specs = dict(specs)
                specs["compute_units"] = compute_units
                specs["stream_processors"] = compute_units * 64  # wavefront size * CUs
            
            cls._cached_info = GPUInfo(
                arch=arch,
                name=specs.get("name", arch),
                marketing_name=marketing_name or specs.get("name", arch),
                compute_units=detected_cu,
                memory_gb=detected_memory,
                specs=specs
            )
            
        except Exception as e:
            # Fallback to default
            specs = ARCH_SPECS[DEFAULT_ARCH]
            cls._cached_info = GPUInfo(
                arch=DEFAULT_ARCH,
                name=specs["name"],
                marketing_name=specs["name"],
                compute_units=specs["compute_units"],
                memory_gb=specs.get("hbm_capacity_gb", 0),
                specs=specs
            )
            
        return cls._cached_info
    
    @classmethod
    def get_target_arch(cls) -> str:
        """Get target arch (user override or detected)."""
        if cls._target_arch:
            return cls._target_arch
        return cls.detect_gpu().arch
    
    @classmethod
    def set_target_arch(cls, arch: str) -> bool:
        """Set target arch for cross-compilation."""
        if arch in ARCH_SPECS:
            cls._target_arch = arch
            return True
        return False
    
    @classmethod
    def clear_target_arch(cls):
        """Clear target arch override."""
        cls._target_arch = None

# =============================================================================
# MCP Server
# =============================================================================

app = Server("gpu-info")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available GPU info tools."""
    return [
        Tool(
            name="get_gpu_info",
            description="""Detect current GPU and return hardware specifications.

Returns:
- Architecture (e.g., gfx942)
- Name and marketing name
- Compute units, memory, bandwidth
- MFMA/WMMA support
- Optimal tile sizes and block sizes
- Optimization priorities for this arch

Use this at the start of any optimization task to understand the target hardware.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_full_specs": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include full hardware specs"
                    }
                }
            }
        ),
        Tool(
            name="get_arch_optimization_hints",
            description="""Get architecture-specific optimization hints.

Returns optimization recommendations for:
- Memory access patterns
- MFMA/WMMA usage
- Tile sizes for GEMM
- Block/workgroup sizes
- Data types to use

Args:
    kernel_type: Optional - type of kernel (gemm, reduction, elementwise, attention, moe)""",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_type": {
                        "type": "string",
                        "enum": ["gemm", "reduction", "elementwise", "attention", "moe", "quantization", "general"],
                        "default": "general",
                        "description": "Type of kernel for targeted hints"
                    }
                }
            }
        ),
        Tool(
            name="set_target_arch",
            description="""Override target architecture for cross-compilation.

Use this when optimizing for a different GPU than the current one.
Call with arch=None to reset to auto-detect.

Args:
    arch: Target architecture (e.g., gfx942, gfx90a, gfx1100) or null to reset""",
            inputSchema={
                "type": "object",
                "properties": {
                    "arch": {
                        "type": ["string", "null"],
                        "description": "Target arch or null to reset"
                    }
                },
                "required": ["arch"]
            }
        ),
        Tool(
            name="list_supported_architectures",
            description="""List all supported GPU architectures in the database.""",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "get_gpu_info":
        return await get_gpu_info(arguments.get("include_full_specs", True))
    elif name == "get_arch_optimization_hints":
        return await get_arch_optimization_hints(arguments.get("kernel_type", "general"))
    elif name == "set_target_arch":
        return await set_target_arch(arguments.get("arch"))
    elif name == "list_supported_architectures":
        return await list_supported_architectures()
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# =============================================================================
# Tool Implementations
# =============================================================================

async def get_gpu_info(include_full_specs: bool) -> list[TextContent]:
    """Get GPU information."""
    gpu = GPUDetector.detect_gpu()
    target = GPUDetector.get_target_arch()
    
    output = f"# GPU Information\n\n"
    output += f"**Detected GPU:** {gpu.marketing_name}\n"
    output += f"**Architecture:** {gpu.arch}\n"
    output += f"**Compute Units:** {gpu.compute_units}\n"
    output += f"**Memory:** {gpu.memory_gb} GB\n\n"
    
    if target != gpu.arch:
        output += f"**Target Architecture (override):** {target}\n\n"
        specs = ARCH_SPECS.get(target, gpu.specs)
    else:
        specs = gpu.specs
        
    if include_full_specs:
        output += "## Hardware Specifications\n\n"
        output += f"| Spec | Value |\n"
        output += f"|------|-------|\n"
        output += f"| Architecture | {specs.get('architecture', 'N/A')} |\n"
        output += f"| Wavefront Size | {specs.get('wavefront_size', 64)} |\n"
        output += f"| LDS per CU | {specs.get('lds_size_kb', 'N/A')} KB |\n"
        output += f"| L2 Cache | {specs.get('l2_cache_mb', 'N/A')} MB |\n"
        
        if 'hbm_bandwidth_tb_s' in specs:
            output += f"| HBM Bandwidth | {specs['hbm_bandwidth_tb_s']} TB/s |\n"
        elif 'memory_bandwidth_gb_s' in specs:
            output += f"| Memory Bandwidth | {specs['memory_bandwidth_gb_s']} GB/s |\n"
            
        output += f"| Max Workgroup Size | {specs.get('max_workgroup_size', 1024)} |\n"
        output += f"| Memory Coalescing | {specs.get('memory_coalescing_bytes', 128)} bytes |\n\n"
        
        # Performance specs
        output += "## Performance Specs\n\n"
        for dtype in ['fp64', 'fp32', 'fp16', 'bf16', 'fp8']:
            key = f"{dtype}_tflops"
            if key in specs:
                output += f"- **{dtype.upper()}:** {specs[key]} TFLOPS\n"
                
        # MFMA/WMMA support
        if specs.get('mfma_support'):
            output += f"\n## MFMA Instructions\n\n"
            for instr in specs.get('mfma_instructions', []):
                output += f"- `{instr}`\n"
        elif specs.get('wmma_support'):
            output += f"\n## WMMA Instructions\n\n"
            for instr in specs.get('wmma_instructions', []):
                output += f"- `{instr}`\n"
                
        # Optimization priorities
        output += f"\n## Optimization Priorities\n\n"
        for i, priority in enumerate(specs.get('optimization_priorities', []), 1):
            output += f"{i}. {priority}\n"
            
    return [TextContent(type="text", text=output)]

async def get_arch_optimization_hints(kernel_type: str) -> list[TextContent]:
    """Get arch-specific optimization hints."""
    target = GPUDetector.get_target_arch()
    specs = ARCH_SPECS.get(target, ARCH_SPECS[DEFAULT_ARCH])
    
    output = f"# Optimization Hints for {target}\n\n"
    output += f"**Kernel Type:** {kernel_type}\n\n"
    
    # General hints
    output += "## General Optimization Guidelines\n\n"
    for priority in specs.get('optimization_priorities', []):
        output += f"- {priority}\n"
    output += "\n"
    
    # Memory access
    output += "## Memory Access\n\n"
    output += f"- **Coalescing Size:** {specs.get('memory_coalescing_bytes', 128)} bytes\n"
    output += f"- **Wavefront Size:** {specs.get('wavefront_size', 64)} threads\n"
    output += f"- **LDS Size:** {specs.get('lds_size_kb', 64)} KB per CU\n"
    output += f"- For coalesced access, ensure consecutive threads access consecutive memory addresses\n\n"
    
    # Workgroup sizes
    output += "## Recommended Block Sizes\n\n"
    for size in specs.get('optimal_block_sizes', [64, 128, 256]):
        output += f"- {size}\n"
    output += "\n"
    
    # Kernel-specific hints
    if kernel_type == "gemm":
        output += "## GEMM-Specific Hints\n\n"
        tiles = specs.get('optimal_tile_sizes', {})
        output += f"- **Recommended M tile sizes:** {tiles.get('gemm_m', [32, 64, 128])}\n"
        output += f"- **Recommended N tile sizes:** {tiles.get('gemm_n', [32, 64, 128])}\n"
        output += f"- **Recommended K tile sizes:** {tiles.get('gemm_k', [8, 16, 32])}\n"
        if specs.get('mfma_support'):
            output += f"- **Use MFMA instructions:** {specs.get('mfma_instructions', [])[:3]}\n"
            output += f"- **Matrix core sizes:** {specs.get('matrix_core_sizes', [])}\n"
        output += "- Use shared memory tiling for A and B matrices\n"
        output += "- Consider double buffering for latency hiding\n\n"
        
    elif kernel_type == "reduction":
        wavefront = specs.get('wavefront_size', 64)
        output += "## Reduction-Specific Hints\n\n"
        output += f"- Use warp/wave shuffle for intra-wavefront reduction\n"
        output += f"- Wavefront size is {wavefront}, use `__shfl_down` for reduction\n"
        output += f"- For large reductions, use two-stage: wave shuffle + shared memory\n"
        output += f"- Consider `__ballot` and `__popc` for boolean reductions\n\n"
        
    elif kernel_type == "elementwise":
        output += "## Elementwise-Specific Hints\n\n"
        output += f"- These are typically memory-bound\n"
        output += f"- Use vectorized loads (float4, half4) for better bandwidth\n"
        output += f"- Fuse multiple elementwise ops into one kernel\n"
        output += f"- Consider grid-stride loops for arbitrary sizes\n\n"
        
    elif kernel_type == "attention":
        output += "## Attention-Specific Hints\n\n"
        output += f"- Use Flash Attention pattern to reduce memory traffic\n"
        output += f"- Tile Q, K, V matrices to fit in LDS ({specs.get('lds_size_kb', 64)} KB)\n"
        if specs.get('mfma_support'):
            output += f"- Use MFMA for QK^T and attention@V matmuls\n"
        output += f"- Online softmax to avoid storing full attention matrix\n"
        output += f"- Consider composable_kernel FlashAttention implementation\n\n"
        
    elif kernel_type == "moe":
        output += "## MoE-Specific Hints\n\n"
        output += f"- Use grouped GEMM for expert computation\n"
        output += f"- Consider ck_tile MoE implementations\n"
        output += f"- Optimize routing/gating separately from expert GEMM\n"
        output += f"- For quantized MoE, use MXFP4/INT8 with appropriate kernels\n\n"
        
    elif kernel_type == "quantization":
        output += "## Quantization-Specific Hints\n\n"
        output += f"- FP8/BF16 for inference on this architecture\n"
        if specs.get('mfma_support'):
            output += f"- MFMA supports FP8: {[i for i in specs.get('mfma_instructions', []) if 'fp8' in i]}\n"
        output += f"- Consider hipBLASLt for quantized GEMM\n"
        output += f"- For MXFP4, use vLLM Marlin or CK implementations\n\n"
        
    return [TextContent(type="text", text=output)]

async def set_target_arch(arch: Optional[str]) -> list[TextContent]:
    """Set target architecture."""
    if arch is None:
        GPUDetector.clear_target_arch()
        detected = GPUDetector.detect_gpu()
        return [TextContent(type="text", text=f"Target arch reset to auto-detect: {detected.arch}")]
        
    if GPUDetector.set_target_arch(arch):
        specs = ARCH_SPECS[arch]
        return [TextContent(type="text", text=f"Target arch set to: {arch} ({specs['name']})")]
    else:
        available = list(ARCH_SPECS.keys())
        return [TextContent(type="text", text=f"Unknown arch: {arch}\nAvailable: {available}")]

async def list_supported_architectures() -> list[TextContent]:
    """List all supported architectures."""
    output = "# Supported GPU Architectures\n\n"
    
    for arch, specs in ARCH_SPECS.items():
        output += f"## {arch}\n"
        output += f"- **Name:** {specs['name']}\n"
        output += f"- **Architecture:** {specs.get('architecture', 'N/A')}\n"
        output += f"- **Compute Units:** {specs.get('compute_units', 'N/A')}\n"
        if specs.get('mfma_support'):
            output += f"- **MFMA:** Supported\n"
        if specs.get('wmma_support'):
            output += f"- **WMMA:** Supported\n"
        output += "\n"
        
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
