# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Apex is an RL environment for GPU kernel optimization on AMD ROCm hardware. It trains LLM agents to optimize GPU kernels through a pipeline:

```
prompt constructor → LLM agent → output/ → grader (Magpie) → score
```

The agent receives a baseline kernel, writes an optimized version to `output/<task_id>/solution.{py,hip}`, and is scored on compilation (+20 pts), correctness (+100 pts), and speedup (×100 pts).

Default target: **MI355X / gfx950 (CDNA4)**. Also supports gfx942 (MI300X), gfx940 (MI300A), gfx90a (MI250X).

## Environment (always set first)

Ensure you are in the Apex repo root (the directory containing this CLAUDE.md),
then activate the venv:
```bash
source ../.venv/bin/activate
export MAGPIE_ROOT=$(pwd)/tools/magpie
```

All commands below assume this working directory. If any command fails with
"activate: No such file", `cd` back to the Apex root and retry.

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
  -b $MAGPIE_ROOT/examples/benchmark_vllm_gptoss_120b.yaml \
  --kernel-types triton --top-k 10 --max-iterations 3 --leaderboard
```

## workload_optimizer.py — subcommands

### `run` — full pipeline (all steps sequentially)

```bash
python3 workload_optimizer.py run \
  -r <RESULTS_DIR> \
  -b $MAGPIE_ROOT/examples/benchmark_vllm_gptoss_120b.yaml \
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
  #    Category  Spec                   Time%    Calls      Name
  1    triton    fused_moe                2.87% 296352     triton_poi_fused_...
  2    triton    paged_attn_decode        2.67% 148752     kernel_unified_...
  3    triton    gemm_bf16                2.33% 257292     _gemm_a16_w16_...
```

### `optimize` — Step 3: agent optimization loop

Spawns a Claude Code sub-agent per kernel. Each gets the rich prompt from
`prompts/kernel_prompt.py` (MCP tables, skills, source locations, arch hints)
plus actual baseline source code. The sub-agent has 7 MCPs via `mcp_config.json`.

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

- **`workload_optimizer.py`** — Full pipeline orchestrator with subcommands: `benchmark`, `identify`, `optimize`, `grade`, `integrate`, `benchmark-final`, `score`, `report`, `run` (all-in-one)

- **`eval.py`** — Lightweight CPU-only eval that exercises the full pipeline without GPU or Magpie

- **`trajectory.py`** — Captures full agent runs (messages, tool calls, solutions, scores) with pluggable backends: FileStore, CouchDB, S3

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

## State management

All pipeline state is stored in `<RESULTS_DIR>/pipeline_state.json`. Steps are
idempotent — re-running a step overwrites its output in state. To start fresh,
delete the state file:
```bash
rm <RESULTS_DIR>/pipeline_state.json
```

Key state fields: `completed_steps`, `baseline_tps`, `identified_kernels`,
`optimization_results`, `reinjected_kernels`, `final_tps`, `reward`.

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
- `/home/sirafati/code_combine/Magpie/results/benchmark_vllm_20260227_105427/benchmark_report.json`

## Model → benchmark config lookup

When the user says a model name, resolve `-b` from this table:

| Model name | `-b` flag |
|------------|-----------|
| GPT OSS 20B | `$MAGPIE_ROOT/examples/benchmark_vllm_gptoss_20b.yaml` |
| GPT OSS 120B | `$MAGPIE_ROOT/examples/benchmark_vllm_gptoss_120b.yaml` |

### GPT OSS 120B details

- `openai/gpt-oss-120b` — MoE 16 experts top-4, GQA 64Q/8KV, 48 layers
- vLLM, FP4, TP=8, MI355X (gfx950)
- Critical triton kernels: fused_moe, paged_attn_decode, gemm_bf16, flash_attn_prefill

### GPT OSS 20B details

- `openai/gpt-oss-20b` — MoE 32 experts top-4, GQA 64Q/8KV, 24 layers, hidden 2880
- 20.9B total params, 3.6B active per token
- vLLM, FP4, TP=1, MI355X (gfx950)
