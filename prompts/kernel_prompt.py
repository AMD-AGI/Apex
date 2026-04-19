#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
kernel_prompt.py — Kernel-level prompt constructor.

Generates one prompt per (model, kernel_type) pair asking an agent to
optimize a specific GPU kernel for the target hardware (default: MI355X).

Usage:
    python3 kernel_prompt.py [--target gfx950] [--framework sglang] [--list]
    python3 kernel_prompt.py --task-id llama3-8b_flash_attn  # single prompt
    python3 kernel_prompt.py --all > all_kernel_prompts.jsonl

Each prompt is a JSON object with:
    task_id, model_id, kernel_type, framework, target_gpu, prompt
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from models import MODELS, ModelConfig, moe_models, by_attention

# ── GPU target ────────────────────────────────────────────────────────────────

DEFAULT_TARGET = "gfx950"   # MI355X (CDNA4)
DEFAULT_TARGET_NAME = "AMD Instinct MI355X"

ARCH_MAP = {
    "gfx942":  "AMD Instinct MI300X (CDNA3)",
    "gfx940":  "AMD Instinct MI300A (CDNA3)",
    "gfx950":  "AMD Instinct MI355X (CDNA4)",
    "gfx90a":  "AMD Instinct MI250X (CDNA2)",
}


def detect_gpu() -> str:
    """Try to detect the installed GPU via rocm-smi; fall back to default."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname"], text=True, timeout=5
        )
        if any(k in out for k in ["MI355", "MI350"]):
            return "gfx950"
        if "MI300X" in out:
            return "gfx942"
        if "MI300A" in out:
            return "gfx940"
        if "MI250" in out:
            return "gfx90a"
    except Exception as e:
        _log.debug("GPU detection failed, using default %s: %s", DEFAULT_TARGET, e)
    return DEFAULT_TARGET


# ── Kernel type registry ──────────────────────────────────────────────────────

@dataclass
class KernelSource:
    """A known implementation of a kernel in a specific library."""
    library:  str    # aiter | composable_kernel | rocBLAS | hipBLASLt | MIOpen | rccl | triton | vllm | sglang
    paths:    tuple[str, ...]   # relative to tools/rocm/
    role:     str    = "impl"   # impl | wrapper | reference


@dataclass
class KernelSpec:
    kernel_type:  str
    description:  str
    applies_to:   str   # "all" | attention type | mlp type
    triton:       bool  = False   # True = Triton kernel (else HIP/C++)
    # Known implementations across ROCm libraries.
    # Paths are relative to tools/rocm/.  Use source-finder MCP to discover more.
    sources:      tuple[KernelSource, ...] = ()


KERNEL_SPECS: list[KernelSpec] = [
    KernelSpec(
        kernel_type="flash_attn_prefill",
        description="Flash attention for the prefill (prompt) phase",
        applies_to="all",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/triton/attention/mha.py",
                "aiter/aiter/ops/triton/attention/pa_prefill.py",
            )),
            KernelSource("composable_kernel", (
                "composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp",
            ), role="reference"),
            KernelSource("vllm", (
                "vllm/vllm/v1/attention/backends/rocm_aiter_fa.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/layers/attention/vision.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="paged_attn_decode",
        description="Paged attention for autoregressive decoding (single token per step)",
        applies_to="all",
        triton=True,
        sources=(
            # Public aiter path — installed as `aiter.ops.triton.pa_decode`
            # in both the `vllm-rocm` docker image and a local
            # `pip install -e tools/rocm/aiter`. The previously-used private
            # `attention/pa_decode.py` location does not exist in either.
            KernelSource("aiter", (
                "aiter/aiter/ops/triton/pa_decode.py",
                "aiter/aiter/ops/triton/gluon/pa_decode_gluon.py",
            )),
            KernelSource("vllm", (
                "vllm/vllm/v1/attention/backends/rocm_aiter_fa.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="mla_attn",
        description="Multi-Head Latent Attention (MLA) — DeepSeek-specific compressed KV",
        applies_to="mla",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/mla.py",
            )),
            KernelSource("vllm", (
                "vllm/vllm/v1/attention/backends/mla/rocm_aiter_mla.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/layers/attention/nsa_backend.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="fused_moe",
        description="Fused Mixture-of-Experts gate + topk routing + expert GEMM",
        applies_to="moe",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/fused_moe.py",
                "aiter/aiter/fused_moe_bf16_asm.py",
            )),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="gemm_w8a8",
        description="FP8 weight × FP8 activation GEMM for linear layers (W8A8)",
        applies_to="all",
        triton=False,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/gemm_op_a8w8.py",
                "aiter/aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py",
            )),
            KernelSource("composable_kernel", (
                "composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp",
            ), role="reference"),
            KernelSource("hipBLASLt", (
                "hipBLASLt/library/src/",
            ), role="reference"),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/layers/quantization/fp8_utils.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="gemm_bf16",
        description="BF16 GEMM for linear (QKV proj, up/gate/down proj)",
        applies_to="all",
        triton=False,
        sources=(
            # Public aiter paths — installed as `aiter.ops.gemm_op_a16w16`
            # and `aiter.ops.triton.gemm.basic.gemm_a16w16` on both the
            # vllm-rocm docker image and a local `pip install -e aiter`.
            KernelSource("aiter", (
                "aiter/aiter/ops/gemm_op_a16w16.py",
                "aiter/aiter/ops/triton/gemm/basic/gemm_a16w16.py",
            )),
            KernelSource("rocBLAS", (
                "rocBLAS/library/src/blas3/",
            ), role="reference"),
            KernelSource("hipBLASLt", (
                "hipBLASLt/library/src/",
            ), role="reference"),
            KernelSource("composable_kernel", (
                "composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp",
            ), role="reference"),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/utils.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/layers/rocm_linear_utils.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="rms_norm",
        description="RMSNorm (pre/post attention and MLP)",
        applies_to="all",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/triton/normalization/rmsnorm.py",
                "aiter/csrc/include/rmsnorm.h",
            )),
            KernelSource("composable_kernel", (
                "composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp",
            ), role="reference"),
            KernelSource("MIOpen", (
                "MIOpen/src/kernels/",
            ), role="reference"),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/layernorm.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/layers/layernorm.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="rope_embedding",
        description="Rotary Position Embedding (RoPE) — applied to Q and K",
        applies_to="all",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/triton/rope/rope.py",
                "aiter/aiter/ops/triton/fusions/fused_qk_concat.py",
            )),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/rotary_embedding/__init__.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/layers/rotary_embedding/factory.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="kv_cache_ops",
        description="KV cache reshape, copy, and quantization ops (paged cache management)",
        applies_to="all",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/cache.py",
                "aiter/csrc/kernels/cache_kernels.cu",
            )),
        ),
    ),
    KernelSpec(
        kernel_type="all_reduce",
        description="Tensor-parallel all-reduce (RCCL + custom fused reduce-scatter kernels)",
        applies_to="all",
        triton=False,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/dist/device_communicators/custom_all_reduce.py",
                "aiter/csrc/kernels/custom_all_reduce.cu",
            )),
            KernelSource("rccl", (
                "rccl/src/",
            ), role="reference"),
            KernelSource("vllm", (
                "vllm/vllm/distributed/device_communicators/custom_all_reduce.py",
            ), role="wrapper"),
            KernelSource("sglang", (
                "sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="act_quant_fp8",
        description="Dynamic per-token FP8 activation quantization before GEMM",
        applies_to="all",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/quant.py",
                "aiter/aiter/ops/triton/quant/fused_fp8_quant.py",
            )),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/quantization/input_quant_fp8.py",
            ), role="wrapper"),
        ),
    ),
    KernelSpec(
        kernel_type="silu_mul",
        description="Fused SiLU × gate (SwiGLU) activation for MLP",
        applies_to="all",
        triton=True,
        sources=(
            KernelSource("aiter", (
                "aiter/aiter/ops/activation.py",
                "aiter/csrc/kernels/activation_kernels.cu",
            )),
            KernelSource("MIOpen", (
                "MIOpen/src/kernels/",
            ), role="reference"),
            KernelSource("vllm", (
                "vllm/vllm/model_executor/layers/activation.py",
            ), role="wrapper"),
        ),
    ),
    # ── OpenAI `triton_kernels` MoE pipeline ────────────────────────────────
    # Three model-agnostic specs targeting the OGS (outer-grouped-scattered)
    # MoE design used by gpt-oss today and by any future workload that adopts
    # the same package. These are real `@triton.jit` kernels (NOT asm) and are
    # the dominant GPU-time consumer for OGS-MoE LLMs (~50%+ of total GPU
    # time on gpt-oss-20b).
    KernelSpec(
        kernel_type="moe_ogs_matmul",
        description=(
            "OGS (outer-grouped-scattered) MoE matmul — bf16/fp16/fp8 "
            "activations × MXFP4/FP8/BF16 expert weights, fused SwiGLU. "
            "Per-launch shape (M, N, K, BS) and dtype are extracted from "
            "the trace name and supplied via the shape-hint addendum."
        ),
        applies_to="moe",
        triton=True,
        sources=(
            KernelSource("triton_kernels", (
                "triton_kernels/matmul_ogs.py",
                "triton_kernels/matmul_ogs_details/_matmul_ogs.py",
                "triton_kernels/matmul_ogs_details/_p_matmul_ogs.py",
                "triton_kernels/matmul_ogs_details/_common.py",
                "triton_kernels/matmul_ogs_details/opt_flags.py",
                "triton_kernels/numerics_details/mxfp.py",
            )),
            KernelSource("aiter", (
                "aiter/aiter/ops/triton/fused_moe.py",
            ), role="reference"),
        ),
    ),
    KernelSpec(
        kernel_type="moe_ogs_routing",
        description=(
            "OGS MoE routing pipeline — topk gate selection, combined "
            "routing-compute, and bitmatrix sum. Launched per-token-block; "
            "cost is dominated by launch overhead, not compute. Strong "
            "candidate for kernel fusion (one persistent CTA per token-block)."
        ),
        applies_to="moe",
        triton=True,
        sources=(
            KernelSource("triton_kernels", (
                "triton_kernels/topk.py",
                "triton_kernels/topk_details/_topk_forward.py",
                "triton_kernels/routing.py",
                "triton_kernels/routing_details/_routing_compute.py",
                "triton_kernels/reduction_details/reduce_bitmatrix.py",
            )),
        ),
    ),
    KernelSpec(
        kernel_type="moe_ogs_finalize",
        description=(
            "OGS MoE scatter-finalize — combines per-expert outputs back "
            "into the per-token activation buffer. Memory-bound; benefits "
            "from vectorized stores and Gluon BlockedLayouts that match "
            "the upstream matmul tile shape."
        ),
        applies_to="moe",
        triton=True,
        sources=(
            KernelSource("triton_kernels", (
                "triton_kernels/matmul_ogs_details/_finalize_matmul.py",
                "triton_kernels/matmul_ogs.py",
            )),
        ),
    ),
]

KERNEL_MAP = {k.kernel_type: k for k in KERNEL_SPECS}


def applicable_kernels(model: ModelConfig) -> list[KernelSpec]:
    """Return kernel specs relevant to this model's architecture."""
    out = []
    for k in KERNEL_SPECS:
        if k.applies_to == "all":
            out.append(k)
        elif k.applies_to == model.attention:
            out.append(k)
        elif k.applies_to == model.mlp_type:
            out.append(k)
        elif k.applies_to in ("moe", "moe_shared") and model.mlp_type in ("moe", "moe_shared"):
            out.append(k)
    return out


# ── Prompt template ───────────────────────────────────────────────────────────

KERNEL_PROMPT_TEMPLATE = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.

## Target hardware
{gpu_name} ({gpu_arch})
- Architecture: {cdna_gen}
- Wavefront size: 64 threads
- Matrix units: MFMA (v_mfma_* instructions)
- LDS: 64 KB per CU
- HBM bandwidth: ~6.5 TB/s aggregate (MI355X)
- Compile target: --offload-arch={gpu_arch}

## Task
Optimize the **{kernel_type}** kernel for **{model_id}** running in **{framework}**.

Kernel: {kernel_description}

Model architecture:
- Attention: {attention} ({num_heads} Q heads, {num_kv_heads} KV heads, head_dim={head_dim})
- MLP: {mlp_type}{moe_info}
- Layers: {num_layers}, hidden_dim: {hidden_dim}
- Context length: {context_len:,} tokens

## Kernel source locations

{library_context}

{sources_block}

IMPORTANT: The paths above are known starting points — they may be incomplete. \
Always use the **source-finder** MCP to search broadly across `tools/rocm/`:
  - `find_kernel_source("{kernel_type}")` — searches ALL cloned ROCm repos
  - `find_ck_template` — find composable_kernel tile/device templates
  - `identify_kernel_origin` — trace which library a kernel comes from
Also use **kernel-rag** `search_library_code` to find related patterns.

## Available MCP tools

You have access to the following MCP servers — use them throughout this task:

| MCP | Key tools | When to use |
|-----|-----------|-------------|
| **source-finder** | `find_kernel_source`, `classify_kernel`, `identify_kernel_origin`, `find_ck_template` | Locate kernel source files, find CK templates, understand kernel lineage |
| **kernel-rag** / **rag-server** | `search_kernel_optimization`, `get_optimization_snippet`, `analyze_kernel_for_optimization`, `get_optimization_playbook` | Get optimization patterns, code snippets, and analysis |
| **gpu-info** | `get_gpu_info`, `get_arch_optimization_hints` | Get target GPU specs and architecture-specific optimization hints |
| **fusion-advisor** | `detect_fusion_opportunities`, `generate_fused_kernel` | Find kernel fusion opportunities and generate fused implementations |
| **magpie** | `analyze`, `compare`, `benchmark` | Evaluate kernel correctness and performance |

## Available skills (Claude Code agent)

Read these skill files before starting. They contain domain-specific optimization
knowledge that directly applies to this task:

| Skill | Path | Use for |
|-------|------|---------|
| triton-kernel-optimization | `tools/skills/triton-kernel-optimization/SKILL.md` | Triton tiling, autotuning, MFMA, LDS usage |
| hip-kernel-optimization | `tools/skills/hip-kernel-optimization/SKILL.md` | HIP C++ patterns, memory coalescing |
| gpu-architecture-fundamentals | `tools/skills/gpu-architecture-fundamentals/SKILL.md` | CU layout, memory hierarchy, wavefronts |
| mi300-cdna3-architecture | `tools/skills/mi300-cdna3-architecture/SKILL.md` | MI300/MI355 CDNA3/4 arch specifics |
| aiter-reflection | `tools/skills/aiter-reflection/SKILL.md` | AMD AI Tensor Engine patterns |
| kernel-exp-history | `tools/skills/kernel-exp-history/SKILL.md` | Learning from prior experiments |
| rocprof-compute | `tools/skills/rocprof-compute/SKILL.md` | Profiling with rocprof |

For Triton kernels: read `triton-kernel-optimization` and `aiter-reflection` first.
For HIP kernels: read `hip-kernel-optimization` and `mi300-cdna3-architecture` first.

## Instructions

1. **Locate** the kernel source using the source-finder MCP:
   - `find_kernel_source("{kernel_type}")` to search ALL ROCm repos
   - Read the implementations listed above (aiter, CK, rocBLAS, etc.)
   - Use `identify_kernel_origin` to understand which library provides the hot path

2. **Analyze** the kernel using kernel-rag MCP:
   - `analyze_kernel_for_optimization` with the kernel code
   - `get_optimization_snippet` for relevant patterns
   - `get_arch_optimization_hints` from gpu-info for {gpu_arch}-specific tips

3. **Identify** the primary performance bottleneck(s):
   - Memory access pattern (coalescing, LDS bank conflicts)
   - Compute utilization (MFMA usage, register pressure)
   - Occupancy (wavefront count per CU)
   - Kernel launch overhead

4. **Check** for fusion opportunities using fusion-advisor:
   - `detect_fusion_opportunities` on the surrounding ops

5. **Write** your optimized implementation to:
   `output/{task_id}/solution{ext}`

6. **Do not** modify any files outside `output/{task_id}/`.
   **Do not** write config.yaml — it is generated by the evaluation pipeline.
   Your solution will be tested for correctness against the original baseline
   kernel and benchmarked for performance automatically.
7. **Do not** create new benchmark scripts — use Magpie MCP (analyze, compare) for evaluation.
8. **Speedup threshold**: only solutions with >5% speedup over baseline are kept.

## Mandatory Workflow Order
1. FIRST: Write a solution that passes correctness (mcp__magpie__compare with check_performance=false)
2. ONLY AFTER correctness passes: Re-run mcp__magpie__compare with check_performance=true to measure speedup. The speedup number from this call is what determines your score.
3. Optimize for speed, then re-verify correctness (check_performance=false) before measuring speedup again (check_performance=true)
4. Never sacrifice correctness for speed — incorrect solutions score 0 on speedup

## Anti-Tampering Rules (violations are automatically detected and penalized)
- Do NOT add `if __name__ == "__main__":` blocks that bypass evaluation
- Do NOT use `sys.exit()`, `SystemExit`, or `os._exit()`
- Do NOT print hardcoded "PASS" or fabricated BENCHMARK_MS values
- Do NOT add timing code that bypasses the evaluation framework
- These patterns are detected by AST analysis and result in score penalties
- Focus on REAL kernel optimizations: memory coalescing, tiling, MFMA, fusion

## Optimization hints for {gpu_arch}
- Use `tl.dot` / MFMA for any matrix multiply ≥ 16×16×16
- Tile to fit hot data in LDS (64 KB); pad rows by 1 to avoid 32-way bank conflicts
- Block size must be a multiple of 64 (wavefront width)
- FP8 GEMMs: use `__hip_atomic_fetch_add` with `__HIP_MEMORY_SCOPE_WORKGROUP`
- Profile with: `rocprof --stats --hip-trace python solution_bench.py`
{extra_hints}
"""


def make_task_id(model: ModelConfig, kernel: KernelSpec) -> str:
    model_slug  = model.hf_id.split("/")[-1].replace(".", "-").lower()
    return f"{model_slug}__{kernel.kernel_type}"


def _format_sources_block(kernel: KernelSpec, framework: str) -> str:
    """Format all known source locations grouped by role."""
    if not kernel.sources:
        return "  (no known paths — use source-finder MCP to locate)"

    impl_lines: list[str] = []
    ref_lines: list[str] = []
    wrapper_lines: list[str] = []

    for src in kernel.sources:
        paths_str = "\n".join(f"    tools/rocm/{p}" for p in src.paths)
        entry = f"  **{src.library}**:\n{paths_str}"
        if src.role == "impl":
            impl_lines.append(entry)
        elif src.role == "reference":
            ref_lines.append(entry)
        elif src.role == "wrapper":
            wrapper_lines.append(entry)

    blocks: list[str] = []
    if impl_lines:
        blocks.append("Primary implementations:\n" + "\n".join(impl_lines))
    if ref_lines:
        blocks.append("Reference / alternative implementations:\n" + "\n".join(ref_lines))

    fw_wrappers = [e for e in wrapper_lines
                   if f"**{framework}**" in e]
    other_wrappers = [e for e in wrapper_lines
                      if f"**{framework}**" not in e]
    if fw_wrappers:
        blocks.append(f"{framework} wrappers (call into ROCm libs):\n" + "\n".join(fw_wrappers))
    if other_wrappers:
        other_fw = "vllm" if framework == "sglang" else "sglang"
        blocks.append(f"{other_fw} wrappers:\n" + "\n".join(other_wrappers))

    return "\n\n".join(blocks)


def _build_library_context(origin_library: str = "aiter") -> str:
    """Generate library-specific context paragraph for the kernel prompt."""
    if origin_library == "vllm":
        return (
            "This kernel is implemented in **vLLM** (not aiter). "
            "The source is a vLLM Triton kernel under `vllm/v1/attention/ops/` or "
            "`vllm/model_executor/layers/`. Your solution MUST preserve the same "
            "function signatures and import style as the vLLM baseline. "
            "Do NOT use aiter imports.\n\n"
            "Other libraries such as **composable_kernel** (CK), **rocBLAS**, and **rccl** "
            "may still provide reference implementations under `tools/rocm/`."
        )
    elif origin_library == "sglang":
        return (
            "This kernel is from **SGLang**. Preserve SGLang import conventions and "
            "function signatures. The source is under `sglang/srt/layers/`. "
            "Do NOT use aiter imports.\n\n"
            "Other libraries under `tools/rocm/` may provide reference implementations."
        )
    else:
        return (
            "On AMD ROCm, kernel implementations are spread across multiple libraries under "
            "`tools/rocm/`.  The most common primary source is **aiter** (AMD AI Tensor Engine "
            "for ROCm), but other libraries such as **composable_kernel** (CK), **rocBLAS**, "
            "**hipBLASLt**, **MIOpen**, and **rccl** also provide kernels and reference "
            "implementations.  vLLM and SGLang typically provide thin wrappers that call into "
            "these libraries on ROCm."
        )


def _build_moe_ogs_hints(
    kernel_type: str, gpu_arch: str, shape_hint: dict | None
) -> str:
    """Render the model-agnostic Gluon-rewrite addendum for moe_ogs_* specs.

    Shape and dtype values come from the shape_hint dict (extracted from the
    real trace name by `pipeline.kernel_bottleneck._parse_kernel_shape_hint`).
    Numbers are NEVER hard-coded for any specific model — gpt-oss-20b,
    gpt-oss-120b, Mixtral, DeepSeek-V3, etc. all flow through the same
    template.
    """
    sh = shape_hint or {}

    def _fmt(key: str, fallback: str = "?") -> str:
        v = sh.get(key)
        return str(v) if v not in (None, "") else fallback

    M, N, K, BS = _fmt("M"), _fmt("N"), _fmt("K"), _fmt("BS")
    dt_a, dt_w = _fmt("dtype_a", "bf16"), _fmt("dtype_w", "bf16")
    swiglu = bool(sh.get("swiglu_fused"))
    n_exp = sh.get("num_experts")
    top_k = sh.get("top_k")
    model_id = sh.get("model_id")

    mfma = "gl.amd.cdna4.mfma" if gpu_arch == "gfx950" else "gl.amd.cdna3.mfma"
    wave = 64
    expert_blurb = (
        f"({n_exp} experts, top-{top_k})" if (n_exp and top_k) else
        "(experts/top_k from model card)"
    )
    model_blurb = f" on `{model_id}`" if model_id else ""

    if kernel_type == "moe_ogs_matmul":
        swiglu_note = (
            "- Keep the fused SwiGLU epilogue (the `_swiglu` suffix in the "
            "trace name indicates SiLU(gate) * up is folded into the matmul "
            "epilogue). Do NOT split it into a separate launch.\n"
        ) if swiglu else ""
        mxfp4_note = ""
        if "fp4" in dt_w.lower() or "mxfp4" in dt_w.lower():
            mxfp4_note = (
                "- Weights are MXFP4 (block-scaled FP4 with one fp8 scale per "
                "32 elements). Pre-stage the FP4 packed bytes AND their fp8 "
                "block-scales into LDS in the same async issue group, then use "
                f"`{mfma}_scaled(...)` for the inner K-tile.\n"
            )
        return (
            f"- TARGET: rewrite `triton_kernels/matmul_ogs_details/_matmul_ogs.py` "
            f"as `@gluon.jit` for tile (M={M}, N={N}, K={K}, BS={BS}), "
            f"activations={dt_a}, weights={dt_w}{model_blurb}.\n"
            f"- Workload: OGS MoE matmul {expert_blurb}; this kernel is the "
            f"single largest GPU-time consumer.\n"
            f"- Use explicit `gl.BlockedLayout(threads_per_warp=[{wave}, ...])` "
            f"for the {dt_a} activation tile and "
            f"`gl.MFMALayout(version=4, mDim=16, nDim=16, kDim=128)` for the "
            f"fp32 accumulator on {gpu_arch} (wavefront={wave}).\n"
            + mxfp4_note
            + swiglu_note
            + "- Keep the Python wrapper signature in `triton_kernels/matmul_ogs.py` "
              "byte-identical so existing call sites continue to work.\n"
            + "- Verify correctness via `triton_kernels.testing.assert_close` against "
              "the original kernel on the canonical shape above. Aim for >= 1.3x.\n"
            + "- Read `tools/skills/gluon-kernel-optimization/SKILL.md` and "
              "`tools/skills/mi300-cdna3-architecture/SKILL.md` first; the "
              "`tools/gluon_rag/` directory has working Gluon MFMA examples."
        )

    if kernel_type == "moe_ogs_routing":
        return (
            f"- TARGET: fuse `triton_kernels/topk_details/_topk_forward.py`, "
            f"`triton_kernels/routing_details/_routing_compute.py`, and "
            f"`triton_kernels/reduction_details/reduce_bitmatrix.py` into a "
            f"single `@gluon.jit` 'routing-fuse' kernel{model_blurb}.\n"
            f"- These three kernels each launch ~25k times per benchmark with "
            f"~5us per launch — their cost is launch-overhead dominated, not "
            f"compute. Goal: cut launch count by 3-4x.\n"
            f"- Use ONE persistent CTA per token-block of size 128 with "
            f"`gl.BlockedLayout(threads_per_warp=[{wave}, ...])`; loop over "
            f"experts {expert_blurb} inside the kernel.\n"
            f"- Validate against the three originals on the same dispatch "
            f"tensors used by `triton_kernels.routing.routing_from_bitmatrix`. "
            f"Aim for >= 1.5x on the bundle.\n"
            f"- Read `tools/skills/gluon-kernel-optimization/SKILL.md` first."
        )

    if kernel_type == "moe_ogs_finalize":
        return (
            f"- TARGET: rewrite "
            f"`triton_kernels/matmul_ogs_details/_finalize_matmul.py` "
            f"(`_finalize_matmul_scatter_*`) as `@gluon.jit`{model_blurb}.\n"
            f"- This kernel scatters per-expert outputs back into the per-token "
            f"activation buffer; it is memory-bound. Use vectorized stores "
            f"(`size_per_thread[innermost] >= 4`) and a `gl.BlockedLayout` "
            f"that matches the upstream matmul tile shape "
            f"(M={M}, N={N}, BS={BS}).\n"
            f"- Verify against the existing kernel via "
            f"`triton_kernels.testing.assert_close`. Aim for >= 1.2x.\n"
            f"- Read `tools/skills/gluon-kernel-optimization/SKILL.md` first."
        )

    return ""


def build_kernel_prompt(
    model:     ModelConfig,
    kernel:    KernelSpec,
    framework: str = "sglang",
    gpu_arch:  str = DEFAULT_TARGET,
    origin_library: str = "aiter",
    rewrite_as: str = "",
    shape_hint: dict | None = None,
) -> dict:
    gpu_name  = ARCH_MAP.get(gpu_arch, gpu_arch)
    cdna_gen  = "CDNA4" if gpu_arch == "gfx950" else "CDNA3" if gpu_arch in ("gfx942","gfx940") else "CDNA2"
    ext       = ".py" if kernel.triton else ".hip"
    task_id   = make_task_id(model, kernel)

    moe_info = ""
    if model.mlp_type in ("moe", "moe_shared"):
        moe_info = f" ({model.num_experts} experts, top-{model.active_experts})"
        if model.mlp_type == "moe_shared":
            moe_info += " + 1 shared"

    extra_hints = ""
    if kernel.kernel_type == "fused_moe":
        extra_hints = "- Sort tokens by expert ID before dispatch to coalesce expert GEMMs\n"
        extra_hints += "- Use persistent kernels to amortize routing overhead at high batch"
    elif kernel.kernel_type in ("flash_attn_prefill", "paged_attn_decode"):
        extra_hints = "- Prefer Triton over HIP C++ for portability; tune BLOCK_M/N/K\n"
        extra_hints += "- Use online softmax (safe_softmax) to avoid materialising full N×N"
    elif kernel.kernel_type == "mla_attn":
        extra_hints = "- MLA absorbs W_UK and W_UV into the projection; avoid re-expanding KV\n"
        extra_hints += "- Latent KV dim is 512 (R1/V3); full KV is 128*num_heads; keep latent in LDS"
    elif kernel.kernel_type.startswith("moe_ogs_"):
        extra_hints = _build_moe_ogs_hints(kernel.kernel_type, gpu_arch, shape_hint)

    # Gluon rewrite addendum — appended when --rewrite-as gluon is set, or when
    # the bottleneck classifier identifies the source kernel as Gluon.
    if rewrite_as.lower() == "gluon":
        extra_hints += (
            ("\n" if extra_hints else "")
            + "- REWRITE TARGET: Gluon. Use `from triton.experimental import gluon` "
              "and `from triton.experimental.gluon import language as gl`. "
              "Decorate the device kernel with `@gluon.jit`.\n"
            + "- Choose `gl.BlockedLayout(threads_per_warp=[64, ...])` on AMD MI300X/MI355X "
              "(wavefront size = 64). Pick `size_per_thread[innermost] >= 2` for vectorized loads.\n"
            + "- Read `tools/skills/gluon-kernel-optimization/SKILL.md` first; consult "
              "`tools/gluon_rag/` for working Gluon kernel patterns. For matmul on this arch "
              "use `gl.amd.cdna3.mfma` (gfx942) or `gl.amd.cdna4.mfma` (gfx950)."
        )

    library_context = _build_library_context(origin_library)

    prompt = KERNEL_PROMPT_TEMPLATE.format(
        gpu_name=gpu_name, gpu_arch=gpu_arch, cdna_gen=cdna_gen,
        kernel_type=kernel.kernel_type,
        kernel_description=kernel.description,
        model_id=model.hf_id, framework=framework,
        attention=model.attention,
        num_heads=model.num_heads, num_kv_heads=model.num_kv_heads, head_dim=model.head_dim,
        hidden_dim=model.hidden_dim, num_layers=model.num_layers,
        mlp_type=model.mlp_type, moe_info=moe_info,
        context_len=model.context_len,
        library_context=library_context,
        sources_block=_format_sources_block(kernel, framework),
        task_id=task_id, ext=ext,
        extra_hints=extra_hints,
    )

    return {
        "task_id":     task_id,
        "model_id":    model.hf_id,
        "kernel_type": kernel.kernel_type,
        "framework":   framework,
        "gpu_arch":    gpu_arch,
        "triton":      kernel.triton,
        "rewrite_as":  rewrite_as,
        "shape_hint":  shape_hint or {},
        "prompt":      prompt,
    }


def all_prompts(framework: str = "sglang", gpu_arch: str = DEFAULT_TARGET) -> Iterator[dict]:
    """Yield one prompt dict per (model, kernel) pair."""
    for model in MODELS:
        if framework not in model.frameworks and framework != "both":
            continue
        for kernel in applicable_kernels(model):
            yield build_kernel_prompt(model, kernel, framework=framework, gpu_arch=gpu_arch)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kernel-level prompt constructor")
    parser.add_argument("--target",    default=None,
                        help="GPU arch (e.g. gfx950). Default: auto-detect, else MI355X")
    parser.add_argument("--framework", default="sglang", choices=["sglang", "vllm", "both"])
    parser.add_argument("--task-id",   default=None,
                        help="Generate a single prompt for this task_id")
    parser.add_argument("--list",      action="store_true",
                        help="List all task IDs without printing prompts")
    parser.add_argument("--all",       action="store_true",
                        help="Print all prompts as JSONL to stdout")
    args = parser.parse_args()

    gpu_arch = args.target or detect_gpu()
    fw       = args.framework

    prompts = list(all_prompts(framework=fw, gpu_arch=gpu_arch))

    if args.list:
        for p in prompts:
            print(p["task_id"])
        print(f"\n{len(prompts)} total (model × kernel) pairs", file=sys.stderr)
        return

    if args.task_id:
        matches = [p for p in prompts if p["task_id"] == args.task_id]
        if not matches:
            print(f"task_id '{args.task_id}' not found", file=sys.stderr)
            sys.exit(1)
        print(matches[0]["prompt"])
        return

    if args.all:
        for p in prompts:
            print(json.dumps(p))
        return

    # Default: print summary
    print(f"Kernel-level prompts  (gpu={gpu_arch}, framework={fw})")
    print(f"  Models:  {len(MODELS)}")
    print(f"  Kernels: {len(KERNEL_SPECS)}")
    print(f"  Total (model × kernel) pairs: {len(prompts)}")
    print(f"\nRun with --list to see all task IDs")
    print(f"Run with --all to emit JSONL of all prompts")
    print(f"Run with --task-id <id> to print a single prompt")


if __name__ == "__main__":
    main()
