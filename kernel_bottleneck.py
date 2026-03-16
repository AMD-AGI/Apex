# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
bottleneck.py — Kernel bottleneck extraction and classification from Magpie
benchmark results.

Parses benchmark_report.json output from Magpie's benchmark mode,
classifies kernel names into categories (triton, hip, ck, asm), and maps
them back to KERNEL_SPECS kernel types for optimization.

Categories:
  - triton: Triton JIT-compiled kernels
  - hip:    HIP/CUDA native kernels (PyTorch, custom CUDA/HIP)
  - ck:     Composable Kernel (CK) library kernels
  - asm:    Assembly-optimized aiter kernels (MFMA-heavy, hand-tuned)
  - unknown: Cannot classify
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "prompts"))

# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------

_TRITON_PATTERNS = [
    re.compile(r"^triton_", re.IGNORECASE),
    re.compile(r"triton::"),
    re.compile(r"^kernel_unified_attention"),
    re.compile(r"_gemm_a\d+_w\d+_kernel_BLOCK_SIZE"),
]

_CK_PATTERNS = [
    re.compile(r"ck_tile"),
    re.compile(r"ck::"),
    re.compile(r"CK_"),
    re.compile(r"composable_kernel"),
    re.compile(r"_ZN7ck_tile"),
    re.compile(r"DeviceGemm"),
    re.compile(r"Rmsnorm2dFwd"),
    re.compile(r"Layernorm2dFwd"),
    re.compile(r"batched_gemm_softmax"),
]

_ASM_PATTERNS = [
    re.compile(r"^_matmul_ogs_"),
    re.compile(r"^_topk_forward"),
    re.compile(r"^_combined_routing"),
    re.compile(r"^_sum_bitmatrix"),
    re.compile(r"^_finalize_matmul"),
    re.compile(r"_asm"),
    re.compile(r"fused_moe_bf16_asm"),
    re.compile(r"reduce_segments$"),
]

_HIP_PATTERNS = [
    re.compile(r"^void vllm::"),
    re.compile(r"^void sglang::"),
    re.compile(r"^void at::native::"),
    re.compile(r"rcclGenericKernel"),
    re.compile(r"^void hip"),
    re.compile(r"hipLaunchKernel"),
    re.compile(r"wvSplitKrc_"),
]

# Mapping profiler kernel names → KERNEL_SPECS kernel_type.
# Order matters: first match wins.
_SPEC_MAPPING: list[tuple[re.Pattern, str]] = [
    (re.compile(r"cross_device_reduce|rcclGenericKernel|allreduce|all_reduce", re.I), "all_reduce"),
    (re.compile(r"flash_attn|flash_attention|pa_prefill|fmha_fwd", re.I), "flash_attn_prefill"),
    (re.compile(r"paged_attn|pa_decode|paged_attention", re.I), "paged_attn_decode"),
    (re.compile(r"unified_attention", re.I), "paged_attn_decode"),
    (re.compile(r"mla_attn|mla_decode|multi_head_latent", re.I), "mla_attn"),
    (re.compile(r"fused_moe|topk_forward|combined_routing|sum_bitmatrix|finalize_matmul_scatter|moe_forward", re.I), "fused_moe"),
    (re.compile(r"matmul_ogs|gemm.*swiglu|gemm_a16_w16|gemm_bf16|gemm_a\d+_w\d+.*BLOCK", re.I), "gemm_bf16"),
    (re.compile(r"gemm.*fp8|gemm.*w8a8|gemm.*a8w8", re.I), "gemm_w8a8"),
    (re.compile(r"rmsnorm|rms_norm|Rmsnorm2dFwd", re.I), "rms_norm"),
    (re.compile(r"rope|rotary", re.I), "rope_embedding"),
    (re.compile(r"reshape_and_cache|cache_kernel|kv_cache", re.I), "kv_cache_ops"),
    (re.compile(r"fp8_quant|act_quant|activation_quant", re.I), "act_quant_fp8"),
    (re.compile(r"silu_mul|swiglu|SiLU", re.I), "silu_mul"),
]


@dataclass
class BottleneckKernel:
    """One kernel extracted from profiler traces with its classification."""
    name: str
    category: str = "unknown"
    total_time_us: float = 0.0
    calls: int = 0
    avg_time_us: float = 0.0
    percent_total: float = 0.0
    matched_kernel_spec: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "total_time_us": round(self.total_time_us, 2),
            "calls": self.calls,
            "avg_time_us": round(self.avg_time_us, 2),
            "percent_total": round(self.percent_total, 4),
            "matched_kernel_spec": self.matched_kernel_spec,
        }


def classify_kernel(name: str) -> str:
    """Classify a profiler kernel name into triton/hip/ck/asm/unknown.

    Check order: triton → ck → asm → hip → unknown.
    Triton is checked first so that Triton-compiled GEMM kernels (which
    contain BLOCK_SIZE in the name) are not mis-classified as CK DeviceGemm.
    """
    for pat in _TRITON_PATTERNS:
        if pat.search(name):
            return "triton"
    for pat in _CK_PATTERNS:
        if pat.search(name):
            return "ck"
    for pat in _ASM_PATTERNS:
        if pat.search(name):
            return "asm"
    for pat in _HIP_PATTERNS:
        if pat.search(name):
            return "hip"
    return "unknown"


def match_to_kernel_spec(kernel_name: str) -> Optional[str]:
    """Map a profiler kernel name to a KERNEL_SPECS kernel_type."""
    for pat, spec_type in _SPEC_MAPPING:
        if pat.search(kernel_name):
            return spec_type
    return None


def extract_bottlenecks(
    benchmark_result: dict,
    top_k: int = 20,
) -> list[BottleneckKernel]:
    """
    Parse a Magpie benchmark_report.json and return the top bottleneck kernels
    sorted by total GPU time, classified and mapped to KERNEL_SPECS.

    Sources checked (in priority order):
      1. gap_analysis.top_kernels  (richest data)
      2. kernel_summary            (aggregated from traces)
      3. top_bottlenecks           (names only, no timing)
    """
    kernels: list[BottleneckKernel] = []

    gap = benchmark_result.get("gap_analysis") or {}
    top_kernels = gap.get("top_kernels", [])

    if top_kernels:
        for entry in top_kernels:
            name = entry.get("name", "")
            if not name:
                continue
            bk = BottleneckKernel(
                name=name,
                total_time_us=float(entry.get("self_cuda_total_us", 0)),
                calls=int(entry.get("calls", 0)),
                avg_time_us=float(entry.get("avg_time_us", 0)),
                percent_total=float(entry.get("pct_total", 0)),
            )
            bk.category = classify_kernel(name)
            bk.matched_kernel_spec = match_to_kernel_spec(name)
            kernels.append(bk)

    elif benchmark_result.get("kernel_summary"):
        for entry in benchmark_result["kernel_summary"]:
            name = entry.get("name", "")
            if not name:
                continue
            time_ms = float(entry.get("time_ms", 0))
            bk = BottleneckKernel(
                name=name,
                total_time_us=time_ms * 1000.0,
                calls=int(entry.get("calls", 0)),
                avg_time_us=(time_ms * 1000.0 / max(int(entry.get("calls", 1)), 1)),
                percent_total=float(entry.get("percent", 0)),
            )
            bk.category = classify_kernel(name)
            bk.matched_kernel_spec = match_to_kernel_spec(name)
            kernels.append(bk)

    elif benchmark_result.get("top_bottlenecks"):
        for name in benchmark_result["top_bottlenecks"]:
            bk = BottleneckKernel(name=name)
            bk.category = classify_kernel(name)
            bk.matched_kernel_spec = match_to_kernel_spec(name)
            kernels.append(bk)

    kernels.sort(key=lambda k: k.total_time_us, reverse=True)
    return kernels[:top_k]


def filter_by_types(
    kernels: list[BottleneckKernel],
    types: list[str],
) -> list[BottleneckKernel]:
    """Filter bottleneck list by kernel category. 'all' keeps everything."""
    if not types or "all" in types:
        return list(kernels)
    allowed = {t.lower() for t in types}
    return [k for k in kernels if k.category in allowed]


def filter_by_names(
    kernels: list[BottleneckKernel],
    names: list[str],
) -> list[BottleneckKernel]:
    """Keep only kernels whose matched_kernel_spec is in the given names list."""
    if not names or "all" in names:
        return list(kernels)
    allowed = {n.lower() for n in names}
    return [
        k for k in kernels
        if k.matched_kernel_spec and k.matched_kernel_spec.lower() in allowed
    ]


def deduplicate_by_spec(
    kernels: list[BottleneckKernel],
) -> list[BottleneckKernel]:
    """
    Group bottleneck kernels by matched_kernel_spec, keeping the entry with
    the highest total_time_us per spec.  Kernels with no spec match are kept
    individually.
    """
    best: dict[str, BottleneckKernel] = {}
    unmatched: list[BottleneckKernel] = []

    for k in kernels:
        if k.matched_kernel_spec is None:
            unmatched.append(k)
            continue
        existing = best.get(k.matched_kernel_spec)
        if existing is None or k.total_time_us > existing.total_time_us:
            best[k.matched_kernel_spec] = k

    result = list(best.values())
    result.sort(key=lambda x: x.total_time_us, reverse=True)
    return result + unmatched


def format_bottleneck_table(kernels: list[BottleneckKernel]) -> str:
    """Pretty-print bottleneck kernels as a CLI table."""
    if not kernels:
        return "  (no bottleneck kernels found)"

    lines = [
        f"  {'#':<4} {'Category':<9} {'Spec':<22} {'Time%':<8} {'Calls':<10} {'Name'}",
        f"  {'─'*4} {'─'*9} {'─'*22} {'─'*8} {'─'*10} {'─'*40}",
    ]
    for i, k in enumerate(kernels, 1):
        spec = k.matched_kernel_spec or "—"
        short_name = k.name[:60] + ("…" if len(k.name) > 60 else "")
        lines.append(
            f"  {i:<4} {k.category:<9} {spec:<22} {k.percent_total:>6.2f}% {k.calls:<10} {short_name}"
        )
    return "\n".join(lines)
