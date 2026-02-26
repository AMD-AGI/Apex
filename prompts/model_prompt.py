#!/usr/bin/env python3
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
from models  import MODELS, ModelConfig
from configs import CONFIGS, InferenceConfig
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

## Reference implementations
- Flash attention (AMD): `files/code/aiter/aiter/ops/triton/`
- FusedMoE (AMD): `files/code/rocm/rocm-libraries/` or AITER
- Composable Kernel GEMM: `files/code/rocm/composable_kernel/include/ck/`
- Triton kernels: `files/code/triton/python/triton/ops/`
- vLLM AMD kernels: `files/code/vllm/vllm/attention/backends/rocm_flash_attn.py`
- SGLang AMD kernels: `files/code/sglang/python/sglang/srt/layers/attention/`

Use the RAG tool (`find_relevant_files`) to locate the most relevant reference code.
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


def num_prompts_for_config(cfg: InferenceConfig) -> int:
    """Reasonable prompt count for a benchmark run."""
    if cfg.scenario == "offline":
        return min(cfg.concurrency * 4, 2000)
    elif cfg.scenario == "interactive":
        return 50
    else:
        return min(cfg.concurrency * 8, 1000)


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
    frameworks = ["sglang", "vllm"] if framework == "both" else [framework]
    for model in MODELS:
        for fw in frameworks:
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
    parser.add_argument("--framework", default="sglang", choices=["sglang", "vllm", "both"])
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
    print(f"  Models:  {len(MODELS)}")
    print(f"  Configs: {len(CONFIGS)}")
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
