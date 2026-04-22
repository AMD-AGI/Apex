#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
model_prompt.py — Model-level prompt constructor.

Generates one prompt per (model, framework, inference_config) triple asking
an agent to optimize whichever kernel is the bottleneck for end-to-end
throughput in that serving scenario.

Usage:
    python3 model_prompt.py [--target gfx950] [--framework sglang] [--list]
    python3 model_prompt.py --task-id llama3-8b__mlperf-server-short
    python3 model_prompt.py --all > all_model_prompts.jsonl

Cross product: 19 models × 17 configs × 2 frameworks ≈ 646 pairs (before filtering).
Filtered by framework support → realistic set of ~300 unique RL tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent))
from models  import MODELS, VIDEO_MODELS, ModelConfig, VideoModelConfig
from configs import CONFIGS, VIDEO_CONFIGS, InferenceConfig, VideoInferenceConfig
from kernel_prompt import detect_gpu, ARCH_MAP, DEFAULT_TARGET

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_task_id(model: ModelConfig, cfg: InferenceConfig, framework: str) -> str:
    model_slug = model.hf_id.split("/")[-1].replace(".", "-").lower()
    return f"{model_slug}__{cfg.config_id}__{framework}"


def bottleneck_hint(model: ModelConfig, cfg: InferenceConfig) -> str:
    """Heuristic: identify the likely bottleneck kernel given (model, config)."""
    prefill_heavy = cfg.input_len >= cfg.output_len * 4
    decode_heavy  = cfg.output_len >= cfg.input_len * 4
    high_batch    = cfg.concurrency >= 64
    is_moe        = model.mlp_type in ("moe", "moe_shared")
    is_mla        = model.attention == "mla"

    hints = []
    if prefill_heavy:
        hints.append("prefill-dominant → flash_attn_prefill is the likely bottleneck")
    if decode_heavy:
        hints.append("decode-dominant → paged_attn_decode + GEMM are the likely bottlenecks")
    if high_batch:
        hints.append(f"high concurrency ({cfg.concurrency}) → GEMM throughput matters more than latency")
    if is_moe:
        hints.append(f"MoE model (top-{model.active_experts}/{model.num_experts}) → fused_moe dispatch is critical")
    if is_mla:
        hints.append("MLA attention → latent KV compression kernel is unique to this architecture")
    if cfg.precision == "fp8":
        hints.append("FP8 precision → activation quantization + W8A8 GEMM kernels on the critical path")
    if cfg.precision == "fp4":
        hints.append("FP4 precision → dequantization before each GEMM; fusing dequant into GEMM is key")
    if cfg.input_len >= 8192:
        hints.append(f"long input ({cfg.input_len} tokens) → FlashAttn O(N²) → chunked prefill + ring-attn")
    if not hints:
        hints.append("balanced prefill/decode → profile first with rocprof to identify actual bottleneck")
    return "\n".join(f"  - {h}" for h in hints)


def precision_notes(precision: str) -> str:
    return {
        "bf16": "BF16 (default ROCm precision; no quantization overhead)",
        "fp8":  "FP8 (W8A8; per-token dynamic activation quant; requires MI300+/MI355)",
        "fp4":  "FP4 (MXFP4/NF4; 2× memory savings over FP8; requires MI355 CDNA4)",
        "int8": "INT8 (weight-only quantization; AWQ/GPTQ style)",
    }.get(precision, precision)


def _frameworks_to_render(framework: str) -> list[str]:
    if framework == "both":
        return ["sglang", "vllm"]
    if framework == "all":
        return ["sglang", "vllm", "fastvideo"]
    return [framework]


# ── Prompt template ───────────────────────────────────────────────────────────

MODEL_PROMPT_TEMPLATE = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.

## Target hardware
{gpu_name} ({gpu_arch})
- Architecture: {cdna_gen}
- Wavefront size: 64 threads; MFMA matrix units
- HBM bandwidth: ~6.5 TB/s aggregate (MI355X)
- Compile target: --offload-arch={gpu_arch}

## Model
{model_id}
- Framework: {framework}
- Attention: {attention} ({num_heads}Q / {num_kv_heads}KV heads, head_dim={head_dim})
- MLP: {mlp_type}{moe_info}
- Layers: {num_layers}, hidden dim: {hidden_dim}
- Max context: {context_len:,} tokens

## Inference configuration
Config ID:    {config_id}  [{source}]
Scenario:     {scenario}
Input tokens: {input_len:,}
Output tokens:{output_len:,}
Concurrency:  {concurrency} simultaneous requests
Precision:    {precision_desc}
Target metric: tokens/second (throughput) — higher is better

## Bottleneck analysis
Based on the configuration above, the expected performance bottleneck(s) are:
{bottleneck}

## Task

1. **Identify** the bottleneck kernel(s) for this (model, config) pair by:
   - Running a profiling sweep: `rocprof --stats python {framework}_bench.py`
   - Checking compute vs memory roofline (Omniperf: `omniperf profile -- ...`)

2. **Optimize** the bottleneck kernel(s). Place ALL modified files in:
   `output/{task_id}/`
   Do NOT modify any files outside this directory.

3. **Write** two config files:
   - `output/{task_id}/config.yaml`      — Magpie kernel-level compare config
   - `output/{task_id}/benchmark.yaml`   — Magpie model-level benchmark config

## output/{task_id}/config.yaml template
```yaml
gpu:
  device: 0
  arch: {gpu_arch}
baseline:
  path: <original kernel path in {framework}>
optimized:
  path: ./solution.py   # or .hip
correctness:
  command: "pytest tests/ -k <kernel_name> -x"
performance:
  command: "python bench_kernel.py --arch {gpu_arch}"
  iterations: 100
```

## output/{task_id}/benchmark.yaml template
```yaml
framework: {framework}
model: {model_id}
gpu:
  device: 0
  arch: {gpu_arch}
baseline:
  framework_config: {{}}          # stock {framework}, no patch
optimized:
  patch: ./solution.py            # kernel override
workload:
  input_len:   {input_len}
  output_len:  {output_len}
  num_prompts: {num_prompts}
  concurrency: {concurrency}
precision: {precision}
```

## Key optimization strategies for this config
{strategy_notes}

## Available MCP tools

You have access to the following MCP servers — use them throughout this task:

| MCP | Key tools | When to use |
|-----|-----------|-------------|
| **source-finder** | `find_kernel_source`, `classify_kernel`, `find_ck_template`, `decode_tensile_kernel` | Locate kernel source files in ROCm repos, find CK templates |
| **kernel-rag** / **rag-server** | `search_kernel_optimization`, `get_optimization_snippet`, `analyze_kernel_for_optimization`, `search_library_code`, `get_optimization_playbook` | Get optimization patterns, snippets, and analysis for HIP/Triton |
| **gpu-info** | `get_gpu_info`, `get_arch_optimization_hints`, `get_gpu_specs` | Get target GPU specs and architecture-specific tips |
| **fusion-advisor** | `detect_fusion_opportunities`, `generate_fused_kernel`, `estimate_fusion_benefit` | Find and exploit kernel fusion opportunities |
| **magpie** | `analyze`, `compare`, `benchmark` | Evaluate kernel correctness and performance |

## Available skills (Claude Code agent)

The following skills are installed and should be read for domain-specific guidance:

| Skill | Path | Use for |
|-------|------|---------|
| triton-kernel-optimization | `tools/skills/triton-kernel-optimization/SKILL.md` | Triton tiling, autotuning, MFMA usage |
| hip-kernel-optimization | `tools/skills/hip-kernel-optimization/SKILL.md` | HIP C++ kernel patterns, LDS, coalescing |
| gpu-architecture-fundamentals | `tools/skills/gpu-architecture-fundamentals/SKILL.md` | CU layout, memory hierarchy, wavefront scheduling |
| mi300-cdna3-architecture | `tools/skills/mi300-cdna3-architecture/SKILL.md` | MI300/MI355 CDNA3/4 specifics |
| aiter-reflection | `tools/skills/aiter-reflection/SKILL.md` | AMD AI Tensor Engine patterns and best practices |
| triton-hip-reference-kernel-search | `tools/skills/triton-hip-reference-kernel-search/SKILL.md` | Finding reference implementations |
| kernel-exp-history | `tools/skills/kernel-exp-history/SKILL.md` | Learning from prior optimization experiments |
| triton-kernel-reflection-prompts | `tools/skills/triton-kernel-reflection-prompts/SKILL.md` | Self-reflection prompts for kernel quality |
| rocprof-compute | `tools/skills/rocprof-compute/SKILL.md` | Using rocprof for profiling and analysis |

Read relevant skills before starting optimization. Example: for Triton kernels, read
`triton-kernel-optimization` and `aiter-reflection` first.

## Top-K kernel reporting

After profiling, report the **top 10 Triton kernels** by GPU time percentage.
Format the report as a table:

```
| Rank | Kernel Name | Spec | GPU Time % | Calls | Category |
|------|-------------|------|------------|-------|----------|
| 1    | ...         | ...  | ...        | ...   | triton   |
```

Focus optimization on these top-10 kernels. For each, document:
- Why it's a bottleneck (compute vs memory bound)
- What optimization strategy was applied
- Measured speedup (baseline vs optimized)

## Pipeline constraints

- **No hardcoding**: kernel lists must be extracted dynamically from profiling, never pre-selected.
- **No new scripts**: do not create new benchmark/eval scripts. Use Magpie MCP (analyze, compare, benchmark) and workload_optimizer.py for everything.
- **Speedup threshold**: only integrate kernels with >5% speedup over baseline.
- **No cache**: test every step for correctness. Do not reuse stale intermediate results.
- **Leaderboard**: always append a new record to leaderboard.json after each run.
- **Trajectories**: persist full trajectory.json with kernel-level and model-level results.
- **Replication**: generate a step-by-step replication guide (replication_guide.md) alongside the report.
- **Deprecated**: disregard AgentKernelArena and any arena-related logic.

## Reference implementations

All AMD kernel implementations are in `tools/rocm/` (cloned ROCm repos).  \
Kernels are spread across **multiple libraries** — do NOT assume a single \
library contains everything.

| Library | Location | What it provides |
|---------|----------|-----------------|
| **aiter** | `tools/rocm/aiter/` | Primary Triton/HIP kernels for attention, MLA, MoE, GEMM, norm, activation (vLLM/SGLang import from here) |
| **composable_kernel** | `tools/rocm/composable_kernel/include/ck/` | Tile-based C++ templates for GEMM, batched-GEMM, grouped-convolution, normalization, MHA |
| **rocBLAS** | `tools/rocm/rocBLAS/library/src/` | Optimized BLAS routines (GEMM, TRSM, etc.) for ROCm |
| **hipBLASLt** | `tools/rocm/hipBLASLt/library/src/` | Lightweight GEMM library with epilogue fusion (bias, activation, scaling) |
| **MIOpen** | `tools/rocm/MIOpen/src/` | CNN/normalization/activation/pooling primitives |
| **rccl** | `tools/rocm/rccl/src/` | Collective communication (AllReduce, AllGather, ReduceScatter) |
| **triton** | `tools/rocm/triton/python/triton/` | Triton compiler and language library for AMD GPUs |
| **vLLM** | `tools/rocm/vllm/vllm/` | Framework wrappers that call into the above libraries on ROCm |
| **SGLang** | `tools/rocm/sglang/python/sglang/srt/layers/` | Framework wrappers that call into the above libraries on ROCm |

IMPORTANT: Always use **source-finder** MCP to search across ALL repos:
  - `find_kernel_source("<kernel_type>")` — searches every cloned ROCm repo
  - `find_ck_template` — find composable_kernel tile/device templates
  - `identify_kernel_origin` — trace which library provides the actual implementation
Also use **kernel-rag** MCP `search_library_code` for optimization patterns.
"""


FASTVIDEO_PROMPT_TEMPLATE = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.

## Target hardware
{gpu_name} ({gpu_arch})
- Architecture: {cdna_gen}
- Wavefront size: 64 threads; MFMA matrix units
- Compile target: --offload-arch={gpu_arch}

## Model
{model_id}
- Framework: {framework}
- Family: {family}
- Params: ~{params_b}B
- Latent channels: {latent_channels}
- Downsample: spatial/{spatial_downsample}, temporal/{temporal_downsample}
- Max frames: {max_frames}

## Generation configuration
Config ID:      {config_id} [{source}]
Scenario:       {scenario}
Resolution:     {height}x{width}
Video length:   {video_length} frames
Infer steps:    {infer_steps}
Concurrency:    {concurrency}
Precision:      {precision_desc}
Target metric:  mean end-to-end generation time (seconds) — lower is better

## Expected bottlenecks
Based on the configuration above, the expected performance bottleneck(s) are:
{bottleneck}

## Task

1. **Identify** the bottleneck kernel(s) in the FastVideo pipeline by profiling the existing E2E harness and collecting the top kernels by GPU time.
2. **Optimize** the bottleneck kernel(s). Place ALL modified files in `output/{task_id}/`.
3. **Write** two config files:
   - `output/{task_id}/config.yaml` — Magpie/kernel compare config for the individual kernel
   - `output/{task_id}/benchmark.yaml` — command-driven model benchmark config for FastVideo

## output/{task_id}/benchmark.yaml template
```yaml
benchmark:
  framework: fastvideo
  model: {model_id}
  kind: command
  metric: timed_mean_s
  goal: minimize
baseline:
  command: "<baseline_fastvideo_e2e_command>"
  result_json: /tmp/{task_id}_baseline.json
optimized:
  command: "<optimized_fastvideo_e2e_command>"
  result_json: /tmp/{task_id}_optimized.json
```

## Key optimization strategies for this config
{strategy_notes}

## Available MCP tools

Use the same MCP stack as other Apex model tasks:
- `source-finder` to locate Triton/HIP/CK/MIOpen implementations
- `kernel-rag` to find ROCm optimization patterns
- `gpu-info` for architecture-specific tuning guidance
- `fusion-advisor` to spot fusion opportunities
- `magpie` for kernel-level compare / analysis

## Framework guidance

- Do not assume FastVideo behaves like an autoregressive LLM serving stack.
- Focus on attention kernels, VAE decode kernels, tensor movement overhead, and any model-specific indexing/conversion kernels in the generation path.
- Use the command-driven benchmark config above for end-to-end scoring rather than Magpie's built-in `vllm` / `sglang` benchmark mode.
"""


def strategy_notes(model: ModelConfig, cfg: InferenceConfig) -> str:
    notes = []
    if cfg.precision == "fp8":
        notes.append("Fuse FP8 activation quantization into the previous op to avoid a round-trip to HBM")
        notes.append("Use per-token dynamic scaling (not per-tensor) for accuracy-throughput balance")
    if cfg.precision == "fp4":
        notes.append("MXFP4 requires block-scaled (32-element groups) dequantization before GEMM")
        notes.append("Consider fusing dequant + GEMM into a single kernel via CK or custom Triton")
    if model.mlp_type in ("moe", "moe_shared"):
        notes.append(f"Sort tokens by expert ID before dispatch to coalesce expert GEMMs")
        notes.append(f"Expert parallelism at {model.active_experts}/{model.num_experts} active — avoid load imbalance")
    if model.attention == "mla":
        notes.append("MLA: do not expand latent KV to full KV during decode — keep in compressed form in LDS")
        notes.append("Absorb W_UK/W_UV projections into QKV projection at compile time")
    if cfg.input_len >= 4096:
        notes.append(f"Chunked prefill recommended (chunk_size=2048) to interleave with decode")
        notes.append("RadixAttention (SGLang) or prefix caching (vLLM) can reuse KV for repeated prefixes")
    if cfg.concurrency >= 128:
        notes.append("High concurrency → batched GEMM (cuBLASLt / hipBLASLt) outperforms loop over sequences")
        notes.append("Consider continuous batching with max_batch_tokens tuning")
    if not notes:
        notes.append("Profile first; standard GEMM and attention tiling improvements apply")
    return "\n".join(f"- {n}" for n in notes)


def fastvideo_bottleneck_hint(model: VideoModelConfig, cfg: VideoInferenceConfig) -> str:
    hints = [
        "  - Video sparse attention forward/backward can dominate denoising time",
        "  - VAE decode path often exposes grouped-convolution, transpose, and upsample hotspots",
        "  - Dense mask/index conversion and tensor movement can create avoidable overhead in the sparse-attention path",
    ]
    if cfg.video_length >= 61:
        hints.append("  - Long clip length increases attention and decode pressure across temporal blocks")
    if cfg.infer_steps <= 1:
        hints.append("  - Very low infer-step runs amplify fixed per-call overhead and launch latency")
    return "\n".join(hints)


def fastvideo_strategy_notes(model: VideoModelConfig, cfg: VideoInferenceConfig) -> str:
    notes = [
        "Profile the existing FastVideo E2E harness before selecting kernels; do not pre-select from names alone",
        "Prioritize custom Triton/HIP kernels and Python-side format/index conversions before library kernels you do not own",
        "Validate with the same prompt, resolution, and video length used in the benchmark config",
    ]
    if cfg.video_length >= 61:
        notes.append("Check temporal tiling and block-sparse metadata overhead for long clips")
    if cfg.scenario == "interactive":
        notes.append("Favor latency reductions and launch-count reductions over purely throughput-oriented batching")
    return "\n".join(f"- {n}" for n in notes)


def num_prompts_for_config(cfg: InferenceConfig) -> int:
    """Reasonable prompt count for a benchmark run."""
    if cfg.scenario == "offline":
        return min(cfg.concurrency * 4, 2000)
    elif cfg.scenario == "interactive":
        return 50
    else:
        return min(cfg.concurrency * 8, 1000)


def build_fastvideo_prompt(
    model: VideoModelConfig,
    cfg: VideoInferenceConfig,
    framework: str,
    gpu_arch: str = DEFAULT_TARGET,
) -> dict:
    gpu_name = ARCH_MAP.get(gpu_arch, gpu_arch)
    cdna_gen = "CDNA4" if gpu_arch == "gfx950" else "CDNA3" if gpu_arch in ("gfx942", "gfx940") else "CDNA2"
    task_id = make_task_id(model, cfg, framework)
    prompt = FASTVIDEO_PROMPT_TEMPLATE.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        cdna_gen=cdna_gen,
        model_id=model.hf_id,
        framework=framework,
        family=model.family,
        params_b=model.params_b,
        latent_channels=model.latent_channels,
        spatial_downsample=model.spatial_downsample,
        temporal_downsample=model.temporal_downsample,
        max_frames=model.max_frames,
        config_id=cfg.config_id,
        source=cfg.source,
        scenario=cfg.scenario,
        height=cfg.height,
        width=cfg.width,
        video_length=cfg.video_length,
        infer_steps=cfg.infer_steps,
        concurrency=cfg.concurrency,
        precision_desc=precision_notes(cfg.precision),
        bottleneck=fastvideo_bottleneck_hint(model, cfg),
        task_id=task_id,
        strategy_notes=fastvideo_strategy_notes(model, cfg),
    )
    return {
        "task_id": task_id,
        "model_id": model.hf_id,
        "config_id": cfg.config_id,
        "framework": framework,
        "gpu_arch": gpu_arch,
        "precision": cfg.precision,
        "height": cfg.height,
        "width": cfg.width,
        "video_length": cfg.video_length,
        "infer_steps": cfg.infer_steps,
        "concurrency": cfg.concurrency,
        "scenario": cfg.scenario,
        "prompt": prompt,
    }


def build_model_prompt(
    model:     ModelConfig,
    cfg:       InferenceConfig,
    framework: str,
    gpu_arch:  str = DEFAULT_TARGET,
) -> dict:
    gpu_name = ARCH_MAP.get(gpu_arch, gpu_arch)
    cdna_gen = "CDNA4" if gpu_arch == "gfx950" else "CDNA3" if gpu_arch in ("gfx942","gfx940") else "CDNA2"
    task_id  = make_task_id(model, cfg, framework)

    moe_info = ""
    if model.mlp_type in ("moe", "moe_shared"):
        moe_info = f" ({model.num_experts} experts, top-{model.active_experts})"
        if model.mlp_type == "moe_shared":
            moe_info += " + 1 shared"

    prompt = MODEL_PROMPT_TEMPLATE.format(
        gpu_name=gpu_name, gpu_arch=gpu_arch, cdna_gen=cdna_gen,
        model_id=model.hf_id, framework=framework,
        attention=model.attention,
        num_heads=model.num_heads, num_kv_heads=model.num_kv_heads, head_dim=model.head_dim,
        hidden_dim=model.hidden_dim, num_layers=model.num_layers,
        mlp_type=model.mlp_type, moe_info=moe_info,
        context_len=model.context_len,
        config_id=cfg.config_id, source=cfg.source, scenario=cfg.scenario,
        input_len=cfg.input_len, output_len=cfg.output_len,
        concurrency=cfg.concurrency,
        precision=cfg.precision,
        precision_desc=precision_notes(cfg.precision),
        bottleneck=bottleneck_hint(model, cfg),
        task_id=task_id,
        num_prompts=num_prompts_for_config(cfg),
        strategy_notes=strategy_notes(model, cfg),
    )

    return {
        "task_id":    task_id,
        "model_id":   model.hf_id,
        "config_id":  cfg.config_id,
        "framework":  framework,
        "gpu_arch":   gpu_arch,
        "precision":  cfg.precision,
        "input_len":  cfg.input_len,
        "output_len": cfg.output_len,
        "concurrency":cfg.concurrency,
        "scenario":   cfg.scenario,
        "prompt":     prompt,
    }


def all_prompts(framework: str = "both", gpu_arch: str = DEFAULT_TARGET) -> Iterator[dict]:
    """Yield one prompt dict per (model, config, framework) triple."""
    frameworks = _frameworks_to_render(framework)
    for fw in frameworks:
        if fw == "fastvideo":
            for model in VIDEO_MODELS:
                if fw not in model.frameworks:
                    continue
                for cfg in VIDEO_CONFIGS:
                    if cfg.framework != fw:
                        continue
                    yield build_fastvideo_prompt(model, cfg, fw, gpu_arch=gpu_arch)
            continue
        for model in MODELS:
            if fw not in model.frameworks:
                continue
            for cfg in CONFIGS:
                if cfg.framework not in (fw, "both"):
                    continue
                yield build_model_prompt(model, cfg, fw, gpu_arch=gpu_arch)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model-level prompt constructor")
    parser.add_argument("--target",    default=None,
                        help="GPU arch (e.g. gfx950). Default: auto-detect, else MI355X")
    parser.add_argument("--framework", default="sglang", choices=["sglang", "vllm", "fastvideo", "both", "all"])
    parser.add_argument("--task-id",   default=None,
                        help="Print a single prompt for this task_id")
    parser.add_argument("--list",      action="store_true",
                        help="List all task IDs")
    parser.add_argument("--all",       action="store_true",
                        help="Print all prompts as JSONL to stdout")
    args = parser.parse_args()

    gpu_arch = args.target or detect_gpu()
    fw       = args.framework

    prompts = list(all_prompts(framework=fw, gpu_arch=gpu_arch))
    framework_models = len(VIDEO_MODELS) if fw == "fastvideo" else len(MODELS)
    framework_configs = len(VIDEO_CONFIGS) if fw == "fastvideo" else len(CONFIGS)

    if args.list:
        for p in prompts:
            print(p["task_id"])
        print(f"\n{len(prompts)} total (model × config × framework) triples", file=sys.stderr)
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

    # Default: summary
    print(f"Model-level prompts  (gpu={gpu_arch}, framework={fw})")
    print(f"  Models:  {framework_models}")
    print(f"  Configs: {framework_configs}")
    print(f"  Total (model × config × framework) triples: {len(prompts)}")
    by_prec = {}
    for p in prompts:
        by_prec[p["precision"]] = by_prec.get(p["precision"], 0) + 1
    for prec, cnt in sorted(by_prec.items()):
        print(f"    {prec}: {cnt}")
    print(f"\nRun with --list to see all task IDs")
    print(f"Run with --all  to emit JSONL of all prompts")
    print(f"Run with --task-id <id> to print a single prompt")


if __name__ == "__main__":
    main()
