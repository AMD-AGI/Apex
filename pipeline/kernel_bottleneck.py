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

import json
import logging
import os
import re
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))

# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------

_TRITON_PATTERNS = [
    re.compile(r"^triton_", re.IGNORECASE),
    re.compile(r"triton::"),
    re.compile(r"^kernel_unified_attention"),
    re.compile(r"_gemm_a\d+_w\d+_kernel_BLOCK_SIZE"),
    re.compile(r"^_attn_fwd_sparse$", re.IGNORECASE),
    re.compile(r"^attn_fwd$", re.IGNORECASE),
    re.compile(r"^map_to_index_kernel$", re.IGNORECASE),
    re.compile(r"^topk_index_to_map_kernel$", re.IGNORECASE),
    re.compile(r"^triton_sta_kernel$", re.IGNORECASE),
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
    (re.compile(r"_attn_fwd_sparse|VIDEO_SPARSE_ATTN|block_sparse_attn", re.I), "video_sparse_attn"),
    (re.compile(r"map_to_index_kernel|topk_index_to_map_kernel", re.I), "fastvideo_sparse_index"),
    (re.compile(r"^attn_fwd$|sla_triton", re.I), "fastvideo_linear_attn"),
    (re.compile(r"triton_sta_kernel|sliding_tile_attention", re.I), "fastvideo_sliding_tile_attn"),
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


_SPEC_TO_AITER_ENV: dict[str, str] = {
    "paged_attn_decode": "VLLM_ROCM_USE_AITER_PAGED_ATTN",
    "flash_attn_prefill": "VLLM_ROCM_USE_AITER_MHA",
    "fused_moe": "VLLM_ROCM_USE_AITER_MOE",
    "rms_norm": "VLLM_ROCM_USE_AITER_RMSNORM",
    "mla_attn": "VLLM_ROCM_USE_AITER_MLA",
    "rope_embedding": "VLLM_ROCM_USE_AITER_TRITON_ROPE",
    "gemm_bf16": "VLLM_ROCM_USE_AITER_LINEAR",
    "gemm_w8a8": "VLLM_ROCM_USE_AITER_LINEAR",
    "act_quant_fp8": "VLLM_ROCM_USE_AITER",
    "silu_mul": "VLLM_ROCM_USE_AITER",
    "kv_cache_ops": "VLLM_ROCM_USE_AITER",
    "all_reduce": "VLLM_ROCM_USE_AITER",
}


@dataclass
class BottleneckKernel:
    """One kernel extracted from profiler traces with its classification."""
    name: str
    category: str = "unknown"
    origin_library: str = "unknown"
    total_time_us: float = 0.0
    calls: int = 0
    avg_time_us: float = 0.0
    percent_total: float = 0.0
    matched_kernel_spec: Optional[str] = None
    bound_type: str = ""
    bandwidth_pct: float = 0.0
    compute_pct: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "category": self.category,
            "origin_library": self.origin_library,
            "total_time_us": round(self.total_time_us, 2),
            "calls": self.calls,
            "avg_time_us": round(self.avg_time_us, 2),
            "percent_total": round(self.percent_total, 4),
            "matched_kernel_spec": self.matched_kernel_spec,
        }
        if self.bound_type:
            d["bound_type"] = self.bound_type
            d["bandwidth_pct"] = round(self.bandwidth_pct, 1)
            d["compute_pct"] = round(self.compute_pct, 1)
        return d


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


def detect_origin_library(
    kernel_name: str,
    kernel_spec: Optional[str] = None,
    config_envs: Optional[dict] = None,
) -> str:
    """Detect which library provides a kernel at runtime.

    For HIP kernels, the profiler name reveals the library directly.
    For Triton kernels (ambiguous), checks VLLM_ROCM_USE_AITER_* env vars
    to determine if aiter or vLLM is providing the kernel.

    Args:
        config_envs: Benchmark config envs dict — checked before os.environ
                     so detection works even outside the benchmark subprocess.
    """
    if re.search(r"^void vllm::", kernel_name):
        return "vllm"
    if re.search(r"^void sglang::", kernel_name):
        return "sglang"
    if re.search(r"^void at::native::", kernel_name):
        return "pytorch"

    def _env(key: str, default: str = "0") -> str:
        if config_envs:
            val = config_envs.get(key)
            if val is not None:
                return str(val)
        return os.environ.get(key, default)

    aiter_global = _env("VLLM_ROCM_USE_AITER", "0")
    aiter_enabled = aiter_global in ("1", "true", "True")

    if kernel_spec and aiter_enabled:
        env_key = _SPEC_TO_AITER_ENV.get(kernel_spec, "VLLM_ROCM_USE_AITER")
        per_kernel = _env(env_key, "1")
        if per_kernel in ("1", "true", "True"):
            return "aiter"

    category = classify_kernel(kernel_name)
    if category == "triton":
        return "aiter" if aiter_enabled else "vllm"
    if category in ("ck", "asm"):
        return "aiter"
    if category == "hip":
        return "vllm"
    return "unknown"


def _extract_from_per_run_reports(
    per_run_dir: str,
    config_envs: Optional[dict] = None,
) -> list[BottleneckKernel]:
    """Fallback: scan per-run benchmark JSONs for kernel data when aggregated report is corrupted."""
    import glob
    kernels_by_name: dict[str, BottleneckKernel] = {}
    run_files = sorted(glob.glob(os.path.join(per_run_dir, "*.json")))
    if not run_files:
        return []

    print(f"  Fallback: scanning {len(run_files)} per-run report(s) in {per_run_dir}")
    for rf in run_files:
        try:
            with open(rf) as f:
                run_data = json.load(f)
        except Exception as e:
            _log.debug("bottleneck JSON parse failed for %s: %s", rf, e)
            continue

        found = False
        gap = run_data.get("gap_analysis") or {}
        for entry in gap.get("top_kernels", []):
            name = entry.get("name", "")
            if not name:
                continue
            found = True
            if name not in kernels_by_name:
                bk = BottleneckKernel(
                    name=name,
                    total_time_us=float(entry.get("self_cuda_total_us", 0)),
                    calls=int(entry.get("calls", 0)),
                    avg_time_us=float(entry.get("avg_time_us", 0)),
                    percent_total=float(entry.get("pct_total", 0)),
                )
                bk.category = classify_kernel(name)
                bk.matched_kernel_spec = match_to_kernel_spec(name)
                bk.origin_library = detect_origin_library(name, bk.matched_kernel_spec, config_envs)
                kernels_by_name[name] = bk

        if found:
            print(f"    Found {len(kernels_by_name)} unique kernels from {os.path.basename(rf)}")
            continue

        for entry in run_data.get("kernel_summary", []):
            name = entry.get("name", "")
            if not name or name in kernels_by_name:
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
            bk.origin_library = detect_origin_library(name, bk.matched_kernel_spec, config_envs)
            kernels_by_name[name] = bk

    result = list(kernels_by_name.values())
    if result:
        print(f"  Fallback: recovered {len(result)} kernel(s) from per-run reports")
    return result


def _extract_from_profiler_csv(
    profiler_csv: str,
    config_envs: Optional[dict] = None,
) -> list[BottleneckKernel]:
    """Parse a rocprof-style kernel stats CSV.

    Expected columns include:
      Name, Calls, TotalDurationNs, AverageNs, Percentage
    """
    path = Path(profiler_csv)
    if not path.exists():
        return []

    kernels: list[BottleneckKernel] = []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Name") or row.get("KernelName") or row.get("Kernel") or "").strip()
                if not name:
                    continue
                total_ns = float(row.get("TotalDurationNs") or row.get("TotalNs") or row.get("DurationNs") or 0.0)
                avg_ns = float(row.get("AverageNs") or row.get("AvgNs") or row.get("MeanNs") or 0.0)
                calls = int(float(row.get("Calls") or row.get("Count") or 0))
                pct = float(row.get("Percentage") or row.get("Pct") or row.get("Percent") or 0.0)
                bk = BottleneckKernel(
                    name=name,
                    total_time_us=total_ns / 1000.0,
                    calls=calls,
                    avg_time_us=avg_ns / 1000.0,
                    percent_total=pct,
                )
                bk.category = classify_kernel(name)
                bk.matched_kernel_spec = match_to_kernel_spec(name)
                bk.origin_library = detect_origin_library(name, bk.matched_kernel_spec, config_envs)
                kernels.append(bk)
    except Exception as e:
        _log.debug("profiler CSV parse failed for %s: %s", profiler_csv, e)
        return []

    kernels.sort(key=lambda k: k.total_time_us, reverse=True)
    if kernels:
        print(f"  Loaded {len(kernels)} kernel(s) from profiler CSV: {profiler_csv}")
    return kernels


def extract_bottlenecks(
    benchmark_result: dict,
    top_k: int = 20,
    config_envs: Optional[dict] = None,
    per_run_dir: Optional[str] = None,
    profiler_csv: Optional[str] = None,
) -> list[BottleneckKernel]:
    """
    Parse a Magpie benchmark_report.json and return the top bottleneck kernels
    sorted by total GPU time, classified and mapped to KERNEL_SPECS.

    Sources checked (in priority order):
      1. gap_analysis.top_kernels  (richest data)
      2. kernel_summary            (aggregated from traces)
      3. top_bottlenecks           (names only, no timing)
      4. profiler CSV (rocprof kernel stats for custom frameworks)
      5. per-run benchmark reports (fallback when aggregated data is corrupted)

    Args:
        config_envs: Benchmark config envs dict for library detection.
        per_run_dir: Path to benchmark_runs/ dir with per-run report JSONs.
    """
    kernels: list[BottleneckKernel] = []

    gap = benchmark_result.get("gap_analysis") or {}
    gap_errors = gap.get("errors", [])
    if gap_errors:
        print(f"  WARNING: gap_analysis has {len(gap_errors)} error(s):")
        for err in gap_errors[:3]:
            print(f"    {err[:120]}")

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
            bk.origin_library = detect_origin_library(name, bk.matched_kernel_spec, config_envs)
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
            bk.origin_library = detect_origin_library(name, bk.matched_kernel_spec, config_envs)
            kernels.append(bk)

    elif benchmark_result.get("top_bottlenecks"):
        for name in benchmark_result["top_bottlenecks"]:
            bk = BottleneckKernel(name=name)
            bk.category = classify_kernel(name)
            bk.matched_kernel_spec = match_to_kernel_spec(name)
            bk.origin_library = detect_origin_library(name, bk.matched_kernel_spec, config_envs)
            kernels.append(bk)

    if not kernels and profiler_csv:
        kernels = _extract_from_profiler_csv(profiler_csv, config_envs)

    if not kernels and per_run_dir:
        kernels = _extract_from_per_run_reports(per_run_dir, config_envs)

    # Warn about unmatched high-impact kernels
    for bk in kernels:
        if bk.matched_kernel_spec is None and bk.percent_total > 2.0:
            print(f"  WARNING: Unmatched kernel '{bk.name[:70]}' ({bk.percent_total:.1f}% GPU time, "
                  f"category={bk.category}) — add pattern to _SPEC_MAPPING to enable optimization")

    unmatched_pct = sum(k.percent_total for k in kernels if k.matched_kernel_spec is None)
    if unmatched_pct > 5.0:
        print(f"  WARNING: {unmatched_pct:.1f}% of GPU time is in kernels with no known spec mapping")

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
        f"  {'#':<4} {'Category':<9} {'Library':<9} {'Spec':<22} {'Time%':<8} {'Calls':<10} {'Name'}",
        f"  {'─'*4} {'─'*9} {'─'*9} {'─'*22} {'─'*8} {'─'*10} {'─'*40}",
    ]
    for i, k in enumerate(kernels, 1):
        spec = k.matched_kernel_spec or "—"
        lib = k.origin_library if k.origin_library != "unknown" else "—"
        short_name = k.name[:55] + ("…" if len(k.name) > 55 else "")
        lines.append(
            f"  {i:<4} {k.category:<9} {lib:<9} {spec:<22} {k.percent_total:>6.2f}% {k.calls:<10} {short_name}"
        )
    return "\n".join(lines)
