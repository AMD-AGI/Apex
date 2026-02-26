#!/usr/bin/env python3
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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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
        for arch, name in ARCH_MAP.items():
            if any(k in out for k in ["MI355", "MI350"]):
                return "gfx950"
            if "MI300X" in out:
                return "gfx942"
            if "MI300A" in out:
                return "gfx940"
            if "MI250" in out:
                return "gfx90a"
    except Exception:
        pass
    return DEFAULT_TARGET


# ── Kernel type registry ──────────────────────────────────────────────────────

@dataclass
class KernelSpec:
    kernel_type:  str
    description:  str
    applies_to:   str   # "all" | attention type | mlp type
    # Where this kernel lives in each framework
    vllm_path:    str   = ""
    sglang_path:  str   = ""
    triton:       bool  = False   # True = Triton kernel (else HIP/C++)


KERNEL_SPECS: list[KernelSpec] = [
    KernelSpec(
        kernel_type="flash_attn_prefill",
        description="Flash attention for the prefill (prompt) phase",
        applies_to="all",
        vllm_path="vllm/attention/backends/rocm_flash_attn.py",
        sglang_path="sglang/srt/layers/attention/triton_ops/prefill_attention.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="paged_attn_decode",
        description="Paged attention for autoregressive decoding (single token per step)",
        applies_to="all",
        vllm_path="vllm/attention/backends/rocm_flash_attn.py",
        sglang_path="sglang/srt/layers/attention/triton_ops/decode_attention.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="mla_attn",
        description="Multi-Head Latent Attention (MLA) — DeepSeek-specific compressed KV",
        applies_to="mla",
        vllm_path="vllm/attention/backends/mla/rocm_mla_attn.py",
        sglang_path="sglang/srt/layers/attention/mla_attn_backend.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="fused_moe",
        description="Fused Mixture-of-Experts gate + topk routing + expert GEMM",
        applies_to="moe",
        vllm_path="vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py",
        sglang_path="sglang/srt/layers/moe/fused_moe_triton.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="gemm_w8a8",
        description="FP8 weight × FP8 activation GEMM for linear layers (W8A8)",
        applies_to="all",
        vllm_path="vllm/model_executor/layers/quantization/utils/w8a8_utils.py",
        sglang_path="sglang/srt/layers/quantization/fp8_kernel.py",
        triton=False,
    ),
    KernelSpec(
        kernel_type="gemm_bf16",
        description="BF16 GEMM for linear (QKV proj, up/gate/down proj)",
        applies_to="all",
        vllm_path="vllm/model_executor/layers/linear.py",
        sglang_path="sglang/srt/layers/linear.py",
        triton=False,
    ),
    KernelSpec(
        kernel_type="rms_norm",
        description="RMSNorm (pre/post attention and MLP)",
        applies_to="all",
        vllm_path="vllm/model_executor/layers/layernorm.py",
        sglang_path="sglang/srt/layers/layernorm.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="rope_embedding",
        description="Rotary Position Embedding (RoPE) — applied to Q and K",
        applies_to="all",
        vllm_path="vllm/model_executor/layers/rotary_embedding.py",
        sglang_path="sglang/srt/layers/rotary_embedding.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="kv_cache_ops",
        description="KV cache reshape, copy, and quantization ops (paged cache management)",
        applies_to="all",
        vllm_path="vllm/attention/ops/paged_attn.py",
        sglang_path="sglang/srt/mem_cache/memory_pool.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="all_reduce",
        description="Tensor-parallel all-reduce (RCCL + custom fused reduce-scatter kernels)",
        applies_to="all",
        vllm_path="vllm/distributed/communication_op.py",
        sglang_path="sglang/srt/distributed/all_reduce.py",
        triton=False,
    ),
    KernelSpec(
        kernel_type="act_quant_fp8",
        description="Dynamic per-token FP8 activation quantization before GEMM",
        applies_to="all",
        vllm_path="vllm/model_executor/layers/quantization/fp8.py",
        sglang_path="sglang/srt/layers/quantization/fp8_kernel.py",
        triton=True,
    ),
    KernelSpec(
        kernel_type="silu_mul",
        description="Fused SiLU × gate (SwiGLU) activation for MLP",
        applies_to="all",
        vllm_path="vllm/model_executor/layers/activation.py",
        sglang_path="sglang/srt/layers/activation.py",
        triton=True,
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

Kernel source location in {framework}:
  {kernel_path}

## Instructions

1. **Read** the existing kernel implementation at the path above (it will be \
available in the sandbox after running `bash files/setup_files.sh`).

2. **Identify** the primary performance bottleneck(s):
   - Memory access pattern (coalescing, LDS bank conflicts)
   - Compute utilization (MFMA usage, register pressure)
   - Occupancy (wavefront count per CU)
   - Kernel launch overhead

3. **Write** your optimized implementation to:
   `output/{task_id}/solution{ext}`

4. **Write** a Magpie evaluation config to:
   `output/{task_id}/config.yaml`
   (set `baseline.path` to the original kernel, `optimized.path` to your solution)

5. **Do not** modify any files outside `output/{task_id}/`.

## output/{task_id}/config.yaml template
```yaml
gpu:
  device: 0
  arch: {gpu_arch}
baseline:
  path: <path-to-original-{framework}-kernel>
optimized:
  path: ./solution{ext}
correctness:
  command: "pytest tests/ -k {kernel_type} -x"
performance:
  command: "python bench_{kernel_type}.py --arch {gpu_arch}"
  warmup_iterations: 10
  iterations: 100
```

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


def build_kernel_prompt(
    model:     ModelConfig,
    kernel:    KernelSpec,
    framework: str = "sglang",
    gpu_arch:  str = DEFAULT_TARGET,
) -> dict:
    gpu_name  = ARCH_MAP.get(gpu_arch, gpu_arch)
    cdna_gen  = "CDNA4" if gpu_arch == "gfx950" else "CDNA3" if gpu_arch in ("gfx942","gfx940") else "CDNA2"
    ext       = ".py" if kernel.triton else ".hip"
    task_id   = make_task_id(model, kernel)
    kpath     = kernel.sglang_path if framework == "sglang" else kernel.vllm_path

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
        kernel_path=kpath or "(see framework source)",
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
