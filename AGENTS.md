# AGENTS.md

This file provides guidance to the Codex agent when working with code in this repository.

## Project Overview

Apex is an RL environment for GPU kernel optimization on AMD ROCm hardware. It trains LLM agents to optimize GPU kernels through a pipeline:

```
prompt constructor → LLM agent → output/ → grader (Magpie) → score
```

The agent receives a baseline kernel, writes an optimized version to `output/<task_id>/solution.{py,hip}`, and is scored on compilation (+20 pts), correctness (+100 pts), and speedup (×100 pts).

Default target: **MI355X / gfx950 (CDNA4)**. Also supports gfx942 (MI300X), gfx940 (MI300A), gfx90a (MI250X).

## Environment (always set first)

Ensure you are in the Apex repo root (the directory containing this AGENTS.md),
then activate the venv and set MAGPIE_ROOT:
```bash
cd "$(dirname "$(readlink -f AGENTS.md)" 2>/dev/null || pwd)"
source .venv/bin/activate
export MAGPIE_ROOT=$(cd ../Magpie && pwd)
```

All commands below assume this working directory (`APEX_ROOT`).

## CRITICAL: Read-Only Codebase

**Do NOT modify any files in this repository or in the Magpie directory (`$MAGPIE_ROOT`).** This includes:
- This repo (Apex) — all `.py`, `.sh`, `.md`, `.yaml` files
- The Magpie repo (`$MAGPIE_ROOT`) — all files including benchmark scripts
- The Python virtual environment

You may only write to:
- Your designated results directory (passed via `-r`)
- Temporary directories under `/tmp/`
- The `output/<task_id>/` directories inside your results directory

If a benchmark script or pipeline step fails, do NOT patch the source code. Instead, work around it via environment variables, command-line flags, or report the issue.

## GPU cleanup (only if needed)

Before launching a GPU workload, check if GPUs are free. **Skip cleanup entirely
if `rocm-smi` already shows ~0% VRAM on the target GPU(s).**

Only if GPUs are busy, run:
```bash
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null
sleep 3
```

If VRAM is still held after that, check `rocm-smi --showpids` for stale PIDs:
- PID exists (`test -d /proc/<PID>`): `kill -9 <PID>`
- PID gone (stale KFD entry): cannot free from userspace; pick a different free GPU:
  ```bash
  export ROCR_VISIBLE_DEVICES=<free_gpu_id>
  export HIP_VISIBLE_DEVICES=<free_gpu_id>
  ```

If a package is missing, install with:
```bash
uv pip install <package>
```

## Common Commands

```bash
# Setup
bash setup.sh                          # Full setup (interactive: choose Claude/Codex/both)
source .venv/bin/activate              # Activate venv

# Tests (no GPU required, all mocked)
pytest tests/ -v                       # All tests
pytest tests/test_prompts.py -v        # Single test file
pytest tests/test_graders.py -v -k "test_score"  # Single test by name

# Mini eval (CPU-only, exercises full pipeline)
python3 eval.py                        # Uses Claude API
python3 eval.py --dry-run              # No API call, writes trivial solution

# Prompt exploration
python3 prompts/kernel_prompt.py --list                    # List kernel task IDs
python3 prompts/kernel_prompt.py --task-id <id>            # Print single prompt
python3 prompts/kernel_prompt.py --target gfx942 --list    # Different GPU arch
python3 prompts/model_prompt.py --list                     # Model-level tasks

# Grading (requires AMD GPU + Magpie)
python3 graders/kernel_grader.py       # Grade kernel tasks in output/
python3 graders/model_grader.py        # Grade model throughput

# Full pipeline (example for GPT OSS 120B)
python3 workload_optimizer.py run -r <RESULTS_DIR> \
  -b $MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_120b.yaml \
  --kernel-types triton --top-k 10 --max-iterations 3 --leaderboard
```

## workload_optimizer.py — subcommands

### `run` — full pipeline (all steps sequentially)

```bash
python3 workload_optimizer.py run \
  -r <RESULTS_DIR> \
  -b $MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_120b.yaml \
  --kernel-types triton \
  --top-k 10 \
  --max-iterations 3 \
  --max-turns 25 \
  --leaderboard
```

### Estimated step timings

| Step | Description | Est. Time |
|------|-------------|-----------|
| benchmark | Initial E2E benchmark | ~25 min |
| identify | Find bottleneck kernels | < 10 sec |
| optimize | Agent optimization loop (3 kernels × 3 iters) | ~60-190 min |
| integrate | Re-inject successful kernels | < 5 sec |
| benchmark-final | Final E2E benchmark | ~25 min |
| score | Compute rewards + leaderboard | < 5 sec |
| report | Generate report.md + replication_guide.md | < 5 sec |
| **Total** | **Full pipeline** | **~2-4 hours** |

### `benchmark` — Step 1: run or load E2E benchmark

Runs Magpie benchmark on the model. Collects baseline TPS, profiling, gap analysis.

```bash
# Fresh benchmark
python3 workload_optimizer.py benchmark -r <RESULTS_DIR> -b <BENCH_CONFIG>

# Reuse existing benchmark (skip the slow run)
python3 workload_optimizer.py benchmark -r <RESULTS_DIR> -b <BENCH_CONFIG> \
  --skip-benchmark <path/to/benchmark_report.json>
```

Produces: `pipeline_state.json` with `baseline_tps`, `baseline_result`.

### `identify` — Step 2: find bottleneck kernels from profiling

Parses profiling data, classifies kernels, ranks by GPU time %.

```bash
# Top 10 triton kernels only
python3 workload_optimizer.py identify -r <RESULTS_DIR> --kernel-types triton --top-k 10

# Top 5 of any type
python3 workload_optimizer.py identify -r <RESULTS_DIR> --top-k 5
```

`--kernel-types`: comma-separated filter — `triton`, `hip`, `ck`, `asm`, or `all` (default: all).
`--top-k`: how many top kernels by GPU time % to keep (default: 10).

Produces: `identified_kernels` list in `pipeline_state.json`.

### `list-kernels` — inspect identified kernels

```bash
python3 workload_optimizer.py list-kernels -r <RESULTS_DIR>
```

Shows a table like:
```
  #    Category  Library   Spec                   Time%    Calls      Name
  1    triton    aiter     fused_moe                2.87% 296352     triton_poi_fused_...
  2    triton    aiter     paged_attn_decode        2.67% 148752     kernel_unified_...
  3    triton    vllm      gemm_bf16                2.33% 257292     _gemm_a16_w16_...
```

### `optimize` — Step 3: agent optimization loop

Spawns a Claude Code sub-agent per kernel. Each gets the rich prompt from
`prompts/kernel_prompt.py` (MCP tables, skills, source locations, arch hints)
plus actual baseline source code. The sub-agent has 5 bundled MCPs via `mcp_config.json` (plus 2 optional external servers: kernel-perf, asm-tools).

```bash
# Optimize ALL identified triton kernels (dynamic, no hardcoding)
python3 workload_optimizer.py optimize -r <RESULTS_DIR> \
  --kernel-types triton \
  --max-iterations 3 \
  --max-turns 25

# Optimize specific kernels only (use spec names from list-kernels)
python3 workload_optimizer.py optimize -r <RESULTS_DIR> \
  --kernels fused_moe,paged_attn_decode \
  --max-iterations 3 \
  --max-turns 25
```

`--max-iterations`: optimization attempts per kernel (default: 5).
`--max-turns`: agent turns per iteration (default: 25).
`--score-threshold`: stop early if kernel score exceeds this (default: 300).
`--agent-backend`: `claude` (default) or `codex`.

Each iteration: agent writes `solution.py` → Magpie grades (compile + correct + speedup)
→ reflector generates feedback → next iteration.

Produces: `optimization_results` in `pipeline_state.json`, per-kernel dirs under
`<RESULTS_DIR>/output/workload__vllm__<spec>/` with `baseline.py`, `solution.py`,
`config.yaml`.

### `grade` — re-grade existing solutions (no agent)

```bash
python3 workload_optimizer.py grade -r <RESULTS_DIR> --kernel-types triton
python3 workload_optimizer.py grade -r <RESULTS_DIR> --kernels fused_moe
```

### `integrate` — Step 4: re-inject optimized kernels

Only kernels with >5% speedup are re-injected for the final benchmark.

```bash
python3 workload_optimizer.py integrate -r <RESULTS_DIR>
python3 workload_optimizer.py integrate -r <RESULTS_DIR> --kernels fused_moe
```

Produces: `reinjected_kernels` in `pipeline_state.json`, patched files in
`<RESULTS_DIR>/output/reinjected/`.

### `benchmark-final` — Step 5: E2E benchmark with optimized kernels

```bash
python3 workload_optimizer.py benchmark-final -r <RESULTS_DIR> -b <BENCH_CONFIG>
```

Produces: `final_tps`, `final_result` in `pipeline_state.json`.

### `score` — Step 6: compute rewards and update leaderboard

```bash
python3 workload_optimizer.py score -r <RESULTS_DIR> --leaderboard
```

### `report` — Step 7: generate reports

```bash
python3 workload_optimizer.py report -r <RESULTS_DIR> -b <BENCH_CONFIG>
```

Produces: `report.md`, `replication_guide.md`.

### `export-rl` — Export trajectories to RL training dataset format

Converts scored trajectories into `tasks.json` (for GRPO training)
and optional `sft_warmstart.jsonl` (for SFT warm-start).

```bash
# Basic export (tasks.json only)
python3 workload_optimizer.py export-rl -r <RESULTS_DIR> --export-output-dir /path/to/rl_training/data/

# With SFT warm-start pairs from good trajectories
python3 workload_optimizer.py export-rl -r <RESULTS_DIR> --export-output-dir /path/to/output/ --sft --quality good

# Custom trajectories directory and minimum score filter
python3 workload_optimizer.py export-rl -r <RESULTS_DIR> --export-output-dir /path/to/output/ \
  --trajectories-dir trajectories/ --min-score 100 --gpu-arch gfx950
```

Also available as a standalone script:
```bash
python3 export_rl_dataset.py --trajectories-dir trajectories/ --results-dirs results_total_* --output-dir /output/ --sft
```

Or programmatically via `TrajectoryStore`:
```python
from trajectory import FileStore
store = FileStore(base_dir=Path("trajectories"))
store.export_for_rl(output_dir=Path("/output"), include_sft=True)
```

Produces: `tasks.json`, `export_metadata.json`, optionally `sft_warmstart.jsonl`.

### `optimize-kernel` — standalone kernel optimization with agent

Optimize any kernel from any library (aiter, vLLM, sglang, torch, MIOpen, CK, etc.)
without running the full E2E model pipeline. The agent receives the baseline kernel,
MCP tools, and correctness definition, then iterates to produce an optimized solution.

```bash
# Triton kernel with PyTorch reference correctness
python3 workload_optimizer.py optimize-kernel \
  --kernel /path/to/my_kernel.py \
  --kernel-type triton \
  --kernel-name rms_norm \
  --correctness-mode pytorch \
  --reference /path/to/reference.py \
  -r /path/to/results \
  --max-iterations 3 --max-turns 25

# Triton kernel with library test suite correctness
python3 workload_optimizer.py optimize-kernel \
  --kernel /path/to/fused_moe.py \
  --kernel-type triton \
  --kernel-name fused_moe \
  --correctness-mode library_test \
  --test-cmd "python -m pytest tests/test_fused_moe.py -x" \
  -r /path/to/results \
  --max-iterations 3

# HIP kernel with Accordo HSA-level correctness
python3 workload_optimizer.py optimize-kernel \
  --kernel /path/to/gemm.hip \
  --kernel-type hip \
  --kernel-name gemm_fp16 \
  --correctness-mode accordo \
  -r /path/to/results

# From a YAML spec file (all fields in one file)
python3 workload_optimizer.py optimize-kernel \
  --kernel-spec /path/to/kernel_spec.yaml \
  -r /path/to/results \
  --max-iterations 3

# Use Codex instead of Claude
python3 workload_optimizer.py optimize-kernel \
  --kernel /path/to/kernel.py \
  --kernel-name my_kernel \
  --correctness-mode pytorch \
  --reference /path/to/ref.py \
  -r /path/to/results \
  --agent-backend codex
```

**YAML spec file format:**

```yaml
task_id: my_custom_norm         # optional (auto-derived from kernel filename)
kernel_path: /path/to/baseline_norm.py
kernel_type: triton             # triton | hip | pytorch
kernel_name: rms_norm
description: "Optimize RMSNorm for MI355X"
gpu_arch: gfx950
framework: vllm                 # optional context
solution_path: /path/to/solution.py  # only for grade-kernel
ground_truth:
  mode: pytorch                 # pytorch | library_test | accordo
  pytorch_reference_code: |
    def baseline_fn(x, weight, eps=1e-6):
        import torch
        return (x.float() / torch.sqrt(torch.mean(x.float()**2, -1, True) + eps) * weight.float()).to(x.dtype)
  test_shapes_code: |
    def get_test_inputs():
        import torch
        return [(torch.randn(4, 64, dtype=torch.float16, device="cuda"),
                 torch.randn(64, dtype=torch.float16, device="cuda"))]
```

**Library test mode spec:**

```yaml
ground_truth:
  mode: library_test
  unit_test_command: "python -m pytest tests/test_fused_moe.py -x"
  repo_url: "https://github.com/ROCm/aiter"
```

**Accordo mode spec:**

```yaml
ground_truth:
  mode: accordo
  accordo_config:
    correctness:
      backend: accordo
      accordo:
        kernel_name: gemm_fp16
        reference_binary: build/ref_gemm
        optimized_binary: build/opt_gemm
        tolerance: 0.001
```

Produces: `<RESULTS_DIR>/standalone_result.json`, task dir with `baseline.py`,
`solution.py`, `solution_best.py`, `config.yaml`.

### `grade-kernel` — grade an existing baseline + solution pair

Grade a kernel optimization without running an agent. Useful for scoring your own
manual optimization or checking if a solution passes correctness.

```bash
# Grade with PyTorch reference
python3 workload_optimizer.py grade-kernel \
  --kernel /path/to/baseline.py \
  --solution /path/to/solution.py \
  --kernel-type triton \
  --correctness-mode pytorch \
  --reference /path/to/reference.py \
  -r /path/to/results --json

# Grade with library test
python3 workload_optimizer.py grade-kernel \
  --kernel /path/to/baseline.py \
  --solution /path/to/solution.py \
  --correctness-mode library_test \
  --test-cmd "python -m pytest tests/test_norm.py -x" \
  -r /path/to/results --json

# Grade from YAML spec (includes solution_path)
python3 workload_optimizer.py grade-kernel \
  --kernel-spec /path/to/spec.yaml \
  -r /path/to/results --json
```

`--json` prints the result as structured JSON:
```json
{
  "task_id": "standalone__baseline",
  "compiled": true,
  "correct": true,
  "speedup": 1.23,
  "score": 243.0,
  "correctness_mode": "pytorch"
}
```

Produces: `<RESULTS_DIR>/grade_result.json`.

### 3-mode correctness system

The pipeline supports three correctness verification modes for kernel grading:

| Mode | How it works | When to use |
|------|-------------|-------------|
| **pytorch** | Runs `torch.allclose(baseline_output, solution_output)` using a PyTorch reference implementation and test shapes | Default. Use when a pure-Python/PyTorch reference exists |
| **library_test** | Runs the library's own test suite (e.g. `pytest tests/test_fused_moe.py`) with the solution on `PYTHONPATH` | Use when the kernel belongs to a library with existing tests (aiter, vLLM, etc.) |
| **accordo** | Validates GPU buffer outputs at the HSA runtime level using the Accordo tool | Use for HIP/C++ kernels where source-level comparison is not feasible |

The mode is set via `--correctness-mode` (CLI) or `ground_truth.mode` (YAML spec).
The `graders/ground_truth.py` registry auto-detects the correct mode for known kernel
types (e.g. `fused_moe` → pytorch, `kv_cache_ops` → library_test).

## Key flags reference

| Flag | Used by | Purpose |
|------|---------|---------|
| `-r <DIR>` | all | Results directory (state, reports, leaderboard) |
| `-b <YAML>` | benchmark, benchmark-final, report, run | Magpie benchmark config |
| `--skip-benchmark <JSON>` | benchmark, run | Reuse existing benchmark report |
| `--kernel-types triton` | identify, optimize, grade, run | Filter to triton kernels only |
| `--top-k 10` | identify, run | Keep top N kernels by GPU time % |
| `--kernels name1,name2` | optimize, grade, integrate | Target specific kernel specs |
| `--max-iterations 3` | optimize, run | Optimization attempts per kernel |
| `--max-turns 25` | optimize, run | Agent turns per iteration |
| `--leaderboard` | score, run | Append to leaderboard.json |
| `--dry-run` | any | Simulate without GPU or API calls |
| `--export-output-dir <DIR>` | export-rl | Output directory for exported tasks.json |
| `--sft` | export-rl | Also emit SFT warm-start JSONL |
| `--quality good` | export-rl | Filter SFT trajectories by quality |
| `--min-score 100` | export-rl | Minimum kernel score to include |
| `--kernel-spec <YAML>` | optimize-kernel, grade-kernel | YAML file with full kernel definition |
| `--kernel <PATH>` | optimize-kernel, grade-kernel | Path to baseline kernel file |
| `--kernel-type triton` | optimize-kernel, grade-kernel | Kernel type: triton, hip, or pytorch |
| `--kernel-name <NAME>` | optimize-kernel, grade-kernel | Human-readable kernel name |
| `--correctness-mode pytorch` | optimize-kernel, grade-kernel | Correctness: pytorch, library_test, or accordo |
| `--reference <PATH>` | optimize-kernel, grade-kernel | PyTorch reference implementation (pytorch mode) |
| `--test-cmd <CMD>` | optimize-kernel, grade-kernel | Unit test command (library_test mode) |
| `--solution <PATH>` | grade-kernel | Path to the solution file to grade |
| `--json` | grade-kernel | Print result as JSON to stdout |

## Architecture

### Core Modules

- **`prompts/`** — Generates task prompts for (model, kernel, GPU arch) combinations
  - `models.py`: Registry of 20 open-source LLMs (Llama 3, DeepSeek, Qwen, Mistral, etc.; includes openai/gpt-oss-120b and  openai/gpt-oss-20b)
  - `configs.py`: Inference config presets (batch sizes, dtypes, tensor parallelism)
  - `kernel_prompt.py`: Constructs prompts for 12 kernel types (flash_attn, fused_moe, gemm_w8a8, rms_norm, etc.)
  - `model_prompt.py`: End-to-end model optimization prompts

- **`graders/`** — Evaluates agent output via Magpie
  - `score.py`: Scoring formula, Magpie result parsing, helper functions
  - `kernel_grader.py`: Grades individual kernel solutions
  - `model_grader.py`: Grades end-to-end model throughput

- **`agents/backends.py`** — Dual agent backend (Claude Code via `claude-agent-sdk`, Codex via `codex exec` CLI)

- **`workload_optimizer.py`** — Full pipeline orchestrator with subcommands: `benchmark`, `identify`, `optimize`, `grade`, `integrate`, `benchmark-final`, `score`, `report`, `run` (all-in-one), `export-rl` (RL dataset export), `optimize-kernel` (standalone kernel optimization), `grade-kernel` (standalone kernel grading)

- **`graders/ground_truth.py`** — 3-mode correctness registry (pytorch, library_test, accordo) with auto-discovery and manual specs for known kernels

- **`eval.py`** — Lightweight CPU-only eval that exercises the full pipeline without GPU or Magpie

- **`trajectory.py`** — Captures full agent runs (messages, tool calls, solutions, scores) with pluggable backends: FileStore, CouchDB, S3. Includes `export_for_rl()` on `TrajectoryStore` for direct RL-compatible export.

- **`export_rl_dataset.py`** — Converts scored trajectories into RL training dataset format (`tasks.json` + optional `sft_warmstart.jsonl`). Also available via `workload_optimizer.py export-rl`.

- **`leaderboard.py`** — Tracks agent performance across runs for RL comparison

- **`reflector.py`** — Generates structured reflection prompts after failed iterations (compilation hints, accuracy tips, optimization suggestions)

- **`bottleneck.py`** — Classifies kernel names from Magpie output (triton/hip/ck/asm) and maps to optimization types

### MCP Servers (tools/mcps/)

Seven MCP servers (configured in `mcp_config.json`) provide agents with domain-specific tools:

| Server | Purpose |
|--------|---------|
| `magpie` | Kernel evaluation: analyze, compare, benchmark |
| `gpu-info` | GPU specs, architecture optimization hints |
| `kernel-perf` | Profiling, roofline analysis |
| `source-finder` | Find kernel source across ROCm repos |
| `asm-tools` | ISA disassembly, instruction analysis |
| `fusion-advisor` | Kernel fusion opportunities |
| `rag-server` | Optimization patterns, snippets, playbooks |

### Skills (tools/skills/)

13 domain knowledge skills under `tools/skills/`, each with a `SKILL.md`. Read the relevant SKILL.md before starting an optimization task.

| Skill | Path | When to use |
|-------|------|-------------|
| aiter-reflection | `tools/skills/aiter-reflection/SKILL.md` | Optimizing AMD GPU kernels on MI300 using the aiter project |
| gpu-architecture-fundamentals | `tools/skills/gpu-architecture-fundamentals/SKILL.md` | Reasoning about GPU architecture to guide kernel optimization |
| hip-kernel-optimization | `tools/skills/hip-kernel-optimization/SKILL.md` | Writing or tuning HIP kernels (memory, tiling, warps, profiling) |
| kernel-exp-history | `tools/skills/kernel-exp-history/SKILL.md` | Consulting or recording past kernel optimization experiments |
| mi300-cdna3-architecture | `tools/skills/mi300-cdna3-architecture/SKILL.md` | MI300/CDNA3 architecture details (MFMA, register files, LDS) |
| mi300-hip-programming-insights | `tools/skills/mi300-hip-programming-insights/SKILL.md` | CDNA3/MI300 HIP programming (chiplet/cache, Infinity Cache, matrix cores) |
| mi300-hip-vs-nvidia | `tools/skills/mi300-hip-vs-nvidia/SKILL.md` | Understanding MI300 HIP differences vs NVIDIA GPUs |
| pytorch-kernel-optimization | `tools/skills/pytorch-kernel-optimization/SKILL.md` | Optimizing PyTorch models/kernels (torch.compile, autograd, mixed precision) |
| rocprof-compute | `tools/skills/rocprof-compute/SKILL.md` | Profiling AMD GPU kernels with rocprof-compute |
| skill-creator | `tools/skills/skill-creator/SKILL.md` | Creating or updating skills |
| triton-hip-reference-kernel-search | `tools/skills/triton-hip-reference-kernel-search/SKILL.md` | Searching Triton/HIP kernel patterns from reference corpus |
| triton-kernel-optimization | `tools/skills/triton-kernel-optimization/SKILL.md` | Writing or tuning Triton GPU kernels (autotuning, fused ops, profiling) |
| triton-kernel-reflection-prompts | `tools/skills/triton-kernel-reflection-prompts/SKILL.md` | Reflection prompts for reviewing/fixing AMD-targeted Triton kernels |

### Data Dependencies (tools/)

- `tools/rocm/` — Cloned ROCm repositories (aiter, composable_kernel, vLLM, SGLang, MIOpen, hipBLASLt, etc.)
- `tools/doc/` — AMD/ROCm documentation PDFs
- `tools/jsons/` — Optimization snippets and specs
- `tools/magpie/` — Magpie framework (GPU kernel evaluation)

### Agent Output Convention

Agents write solutions to `output/<task_id>/` containing:
- `solution.py` or `solution.hip` — optimized kernel
- `config.yaml` — Magpie evaluation config (GPU arch, baseline path, correctness/perf commands)

## Key Environment Variables

- `ANTHROPIC_API_KEY` — Required for Claude agent backend
- `OPENAI_API_KEY` — Required for Codex agent backend
- `MAGPIE_ROOT` — Path to Magpie installation (default: `tools/magpie`)
- `MAGPIE_RUN_MODE` — Force benchmark run mode: `local` or `docker` (default: auto-detect)
- `RESULTS_DIR` — Pipeline results output directory
- `VLLM_ROCM_USE_AITER` — Global toggle: `1` = aiter provides kernels, `0` = vLLM provides them
- `VLLM_ROCM_USE_AITER_PAGED_ATTN` — Per-kernel override for paged attention
- `VLLM_ROCM_USE_AITER_MHA` — Per-kernel override for flash attention prefill
- `VLLM_ROCM_USE_AITER_MOE` — Per-kernel override for fused MoE
- `VLLM_ROCM_USE_AITER_RMSNORM` — Per-kernel override for RMS normalization
- `VLLM_ROCM_USE_AITER_LINEAR` — Per-kernel override for GEMM (bf16 and w8a8)
- `VLLM_ROCM_USE_AITER_MLA` — Per-kernel override for multi-latent attention
- `VLLM_ROCM_USE_AITER_TRITON_ROPE` — Per-kernel override for rotary embedding

## Scoring Formula

Kernel-level:
```
score = compiled × 20 + correct × 100 + speedup_ratio × 100
```

Model-level:
```
score = 0.5 × normalized_kernel_score + 0.5 × (optimized_tps / baseline_tps − 1)
```

Produces: `results_summary.json`, `trajectory.json`, `leaderboard.json`.

## Pipeline rules

1. No hardcoding kernel names — always extract from profiling dynamically.
2. No new scripts — everything through `workload_optimizer.py`.
3. Only integrate kernels with >5% speedup over baseline.
4. Always `--leaderboard` to append results per run.
5. Always run `report` to generate `report.md` + `replication_guide.md`. The replication guide must contain full CLI commands to reproduce the entire run from scratch.
6. Disregard `AgentKernelArena` and any deprecated logic.
7. **No cache** — Do not reuse stale intermediate results. Always re-run benchmarks fresh unless `--skip-benchmark` is explicitly requested.
8. **Kernel extraction** — Dynamically extract 3-4 candidate Triton kernels from profiling. Never hardcode kernel names or pre-select kernels.
9. **Multiple runs for stats** — For both kernel-level and E2E benchmarks, use Magpie's iteration settings to collect statistics and p99/percentile timings.
10. **Benchmark run mode** — Run mode is auto-detected internally (no CLI flag).
   To override, set the env var before running:
   - User says "locally" or "local" → `export MAGPIE_RUN_MODE=local`
   - User says "docker" → `export MAGPIE_RUN_MODE=docker`
   - If not specified, auto-detects via `docker info`.

## Multi-library kernel support

Kernels can originate from different libraries, not just aiter. The pipeline
automatically detects the source library using profiler name patterns and
`VLLM_ROCM_USE_AITER_*` environment variables:

| Library | Detection | Patchable |
|---------|-----------|-----------|
| **aiter** | `VLLM_ROCM_USE_AITER=1` + per-kernel env var | Yes (Triton .py, standalone HIP .so) |
| **vllm** | Profiler name `void vllm::*` or Triton when AITER disabled | Yes (Triton .py only; HIP in monolithic `_C.so` — not patchable) |
| **sglang** | Profiler name `void sglang::*` | Yes (Triton .py only) |
| **pytorch** | Profiler name `void at::native::*` | No (monolithic `_C.so`) |

The `origin_library` field in `pipeline_state.json` identifies the source per kernel.
The pipeline adapts import fixups, module resolution, and prompt generation per library.

Use `--kernel-types triton` to target reliably patchable kernels.

## Session isolation

Each pipeline run starts with a guaranteed clean baseline:
- `_ensure_clean_baseline()` restores any leftover patches from previous sessions
- `atexit` and signal handlers guarantee cleanup even on crash/SIGTERM
- No patches from previous sessions affect the current run

## Pipeline robustness features

- **Bisection rollback**: When smoke test fails, the pipeline bisects to find the bad patch instead of rolling back everything. Good patches are kept. After bisection, surviving patches are re-verified as a group.
- **GPU health check**: Validates GPU temperature/clocks before benchmarks. Warns on throttling (>85°C).
- **Baseline drift detection**: Quick re-baseline at final benchmark time catches thermal drift (>15% warns).
- **Solution syntax validation**: `ast.parse` check before patching catches truncated/corrupt agent output.
- **Atomic state saves**: `pipeline_state.json` uses atomic writes (tempfile + os.replace) to prevent corruption.
- **Stale lock detection**: Patch lock includes PID; stale locks from crashed processes are auto-broken.
- **Unmatched kernel warnings**: Pipeline warns when high-impact kernels (>2% GPU time) have no spec mapping.
- **Environment snapshot**: All `VLLM_ROCM_USE_AITER_*` env vars and package versions captured in trajectory.
- **Library test verification**: After hot-patching, the library's own test suite (from `MANUAL_REGISTRY`) is run to catch subtle correctness issues beyond import checks.
- **Multi-file patching**: Solutions can be a directory with `manifest.json` mapping multiple files to their install targets, supporting kernel + dispatch table changes.
- **Benchmark caching**: `--benchmark-cache-hours N` caches baseline benchmark results to skip re-running.
- **Parallel kernel optimization**: `--parallel-kernels N` runs up to N agent sessions concurrently. GPU grading is serialized.
- **Smart iteration**: No-progress early termination (stall detection: delta <5% for 2 consecutive iterations) with budget reallocation.
- **Agent model routing**: `--agent-model-simple` / `--agent-model-complex` for per-kernel model override based on difficulty.
- **Knowledge base**: `knowledge_base.json` records optimization outcomes; past insights injected into agent prompts automatically.
- **Anti-tampering prompts**: Explicit anti-tampering rules in all prompt templates with AST-based detection.
- **Correctness-first workflow**: Mandatory correctness → speed optimization order enforced in prompts.
- **Speedup measurement reliability**: Multiple profiling runs with outlier rejection and high-variance warnings.
- **Structured profiling feedback**: rocprof metrics parsed into Performance Scorecard in reflection prompts.
- **Reference injection**: PyTorch reference code and library test function signatures injected inline in agent prompts.
- **Configurable tampering cap**: `--tampering-speedup-cap X` overrides default speedup cap when benchmark tampering detected.

## Kernel reintegration scope

Reintegration (hot-patching) replaces installed `.py` or `.so` files in site-packages with optimized solutions during the final benchmark. **Current support:**

| Library | Patchable | Method |
|---------|-----------|--------|
| **aiter** (Triton) | Yes | Replace `.py` in site-packages, Triton JIT re-compiles |
| **aiter** (HIP) | Partial | Only standalone `.so` files (not monolithic `_C.so`) |
| **vllm** (Triton) | Yes | Replace `.py`, resolve relative imports to absolute |
| **sglang** (Triton) | Yes | Replace `.py`, resolve relative imports to absolute |
| **vllm/sglang** (HIP) | No | Compiled into monolithic `_C.so` |
| **pytorch** (HIP) | No | Compiled into monolithic `_C.so` |

**System-level C/C++ libraries are NOT patchable:**

| Library | Why not patchable |
|---------|-------------------|
| **hipBLASLt** | System shared library (`libhipblaslt.so`), used by `torch._scaled_mm` |
| **rocBLAS** | System shared library (`librocblas.so`), used by `torch.mm` |
| **composable_kernel** | Compiled into aiter's C extensions |
| **MIOpen** | System shared library (`libMIOpen.so`) |
| **rccl** | Communication library, not a compute kernel |

These system libraries can be optimized standalone (via `optimize-kernel` + Accordo) but cannot be reinjected into E2E benchmarks without rebuilding from source. This is planned for future support.

## State management

All pipeline state is stored in `<RESULTS_DIR>/pipeline_state.json`. Steps are
idempotent — re-running a step overwrites its output in state. To start fresh,
delete the state file:
```bash
rm <RESULTS_DIR>/pipeline_state.json
```

Key state fields: `completed_steps`, `baseline_tps`, `identified_kernels`,
`optimization_results`, `reinjected_kernels`, `final_tps`, `reward`,
`environment_snapshot`, `origin_library` (per kernel).

## Output file layout

```
<RESULTS_DIR>/
├── pipeline_state.json        # Pipeline state (resume/rerun)
├── trajectory.json            # Full trajectory record
├── results_summary.json       # Summary of all results
├── leaderboard.json           # Aggregated leaderboard
├── leaderboard.jsonl          # Append-only leaderboard log
├── report.md                  # Optimization report
├── replication_guide.md       # Auto-generated replication guide
└── output/
    ├── workload__vllm__<spec>/
    │   ├── config.yaml        # Magpie eval config
    │   ├── baseline.py        # Baseline kernel
    │   └── solution.py        # Optimized kernel
    └── reinjected/            # Solutions for final benchmark
```

## Existing benchmark data

Existing benchmark reports that can be reused with `--skip-benchmark`:
- `$MAGPIE_ROOT/results/benchmark_vllm_20260227_105427/benchmark_report.json`

## Model → benchmark config lookup

When the user says a model name, resolve `-b` from this table:

| Model name | `-b` flag |
|------------|-----------|
| GPT OSS 20B | `$MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml` |
| GPT OSS 120B | `$MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_120b.yaml` |

Note: configs live in `$MAGPIE_ROOT/examples/benchmarks/`, **not** `$MAGPIE_ROOT/examples/`.

### GPT OSS 120B details

- `openai/gpt-oss-120b` — MoE 16 experts top-4, GQA 64Q/8KV, 48 layers
- vLLM, FP4, TP=8, MI355X (gfx950)
- Critical triton kernels: fused_moe, paged_attn_decode, gemm_bf16, flash_attn_prefill

### GPT OSS 20B details

- `openai/gpt-oss-20b` — MoE 32 experts top-4, GQA 64Q/8KV, 24 layers, hidden 2880
- 20.9B total params, 3.6B active per token
- vLLM, FP4, TP=1, MI355X (gfx950)

## Default workflow — what to do when user says "optimize <model>"

When the user gives a short instruction like "optimize GPT OSS 20B" or "run the
full pipeline for gptoss120b", follow these steps **without asking for more details**:

1. Set up environment (see above).
2. Pick a fresh results directory: `$HOME/results_total_<N>` (increment N
   from whatever already exists — run `ls -d $HOME/results_total_* 2>/dev/null` to check).
3. Look up the `-b` config from the table above.
4. Check GPU availability with `rocm-smi`. Clean up if needed (see GPU cleanup section).
5. Run the full pipeline step-by-step:
   ```bash
   RESULTS=$HOME/results_total_<N>
   BENCH_CONFIG=$MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_<size>.yaml

   python3 workload_optimizer.py benchmark -r $RESULTS -b $BENCH_CONFIG
   python3 workload_optimizer.py identify  -r $RESULTS --kernel-types triton --top-k 10
   python3 workload_optimizer.py list-kernels -r $RESULTS
   python3 workload_optimizer.py optimize  -r $RESULTS --kernel-types triton --max-iterations 3 --max-turns 25
   python3 workload_optimizer.py integrate -r $RESULTS
   python3 workload_optimizer.py benchmark-final -r $RESULTS -b $BENCH_CONFIG
   python3 workload_optimizer.py score     -r $RESULTS --leaderboard
   python3 workload_optimizer.py report    -r $RESULTS -b $BENCH_CONFIG
   ```
6. After each step, verify it succeeded before proceeding to the next.
7. Print a summary of final results (TPS improvement, kernel-level speedups, report location).

## Default workflow — kernel-level optimization

When the user says "optimize this kernel", "optimize my fused_moe", or provides a
kernel file to optimize — **without** a full model benchmark — follow these steps:

1. Set up environment (see above).
2. Identify what the user provided:
   - A kernel file path → use `--kernel`
   - A YAML spec → use `--kernel-spec`
   - A kernel name from a library → use `source-finder` MCP to locate the source file
3. Determine the correctness mode based on user instruction:
   - User says "find pytorch source" / "use pytorch reference" / "compare against torch" → `--correctness-mode pytorch`
     - Use `source-finder` MCP (`find_kernel_source`, `identify_kernel_origin`) to locate the PyTorch reference implementation
     - Pass the reference file via `--reference`
   - User says "use library tests" / "use pytest" / "run test suite" → `--correctness-mode library_test`
     - Pass the test command via `--test-cmd`
   - User says "use accordo" / "HSA-level validation" → `--correctness-mode accordo`
   - If user doesn't specify → default to `pytorch`
4. Run the standalone optimization:
   ```bash
   RESULTS=$HOME/kernel_results_<name>

   python3 workload_optimizer.py optimize-kernel \
     --kernel /path/to/kernel.py \
     --kernel-type triton \
     --kernel-name <name> \
     --correctness-mode <mode> \
     --reference /path/to/ref.py \  # if pytorch mode
     --test-cmd "pytest tests/..." \  # if library_test mode
     -r $RESULTS \
     --max-iterations 3 --max-turns 25
   ```
5. After completion, print the result from `$RESULTS/standalone_result.json`.

### Grading workflow

When the user says "grade my kernel optimization" or "score this solution":

```bash
python3 workload_optimizer.py grade-kernel \
  --kernel /path/to/baseline.py \
  --solution /path/to/solution.py \
  --kernel-type triton \
  --correctness-mode pytorch \
  --reference /path/to/ref.py \
  -r $RESULTS --json
```

### MCP tools to use during kernel optimization

The agent has 5 bundled MCP servers (plus 2 optional external: kernel-perf, asm-tools). Use them actively for kernel-level work:

| MCP | Tool | When to call |
|-----|------|-------------|
| **source-finder** | `find_kernel_source` | Find the baseline kernel source across all ROCm libraries (aiter, CK, vLLM, MIOpen, hipBLASLt, etc.) |
| **source-finder** | `identify_kernel_origin` | Determine which library a kernel comes from |
| **source-finder** | `find_ck_template` | Find Composable Kernel template implementations |
| **rag-server** | `get_optimization_playbook` | Get a step-by-step optimization playbook for the kernel type |
| **rag-server** | `search_kernel_optimization` | Search for optimization patterns (tiling, vectorization, etc.) |
| **rag-server** | `get_optimization_snippet` | Get a ready-to-use code snippet for a specific optimization |
| **gpu-info** | `get_arch_optimization_hints` | Get MI355X / CDNA4 specific optimization hints |
| **gpu-info** | `get_gpu_specs` | Get target GPU memory bandwidth, compute, cache sizes |
| **fusion-advisor** | `detect_fusion_opportunities` | Find fusion opportunities in the kernel |
| **fusion-advisor** | `generate_fused_kernel` | Generate a fused kernel implementation |
| **kernel-perf** | `roofline_analysis` | Determine if kernel is memory-bound or compute-bound |
| **kernel-perf** | `profile_kernel` | Profile execution time, occupancy, memory throughput |
| **magpie** | `compare` | Validate correctness and measure speedup vs baseline |

### Any kernel from any library

The standalone optimization supports kernels from **any** library:

| Library | Kernel types | Example spec names |
|---------|-------------|-------------------|
| **aiter** | Triton, HIP | fused_moe, paged_attn_decode, rms_norm, flash_attn_prefill |
| **vLLM** | Triton, HIP | gemm_bf16, activation, moe_align |
| **sglang** | Triton | fused_moe, flashinfer_attn |
| **torch/PyTorch** | PyTorch, C++ | scaled_dot_product_attention, layer_norm |
| **MIOpen** | HIP, C++ | conv_fwd, batchnorm |
| **composable_kernel** | C++ templates | gemm, batched_gemm, grouped_conv |
| **hipBLASLt** | HIP | gemm with epilogue fusion |
| **rccl** | HIP | allreduce, allgather |

Use `source-finder` MCP to locate the exact source files across these libraries.
