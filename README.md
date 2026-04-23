# Apex — GPU Kernel Optimization Pipeline

> **License Notice**
>
> Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
>
> This project is licensed under the **MIT License**. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the following conditions:
>
> - The above copyright notice and this license notice must be included in all copies or substantial portions of the software.
> - The software is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement.
>
> **Third-party dependencies** (ROCm libraries, vLLM, Triton, Magpie, etc.) are governed by their own respective licenses. End users are responsible for reviewing and complying with the licensing terms of any dependencies used in conjunction with this software.
>
> **AI Agent notice** — This software orchestrates third-party AI coding agents to perform kernel optimization:
>
> | Agent | Provider | What you need |
> |---|---|---|
> | **Claude Code** | Anthropic | An active [Anthropic account](https://console.anthropic.com/) and acceptance of [Anthropic's usage policies](https://www.anthropic.com/legal/usage-policy) |
> | **OpenAI Codex** | OpenAI | An active [OpenAI account](https://platform.openai.com/) and acceptance of [OpenAI's terms of service](https://openai.com/policies/terms-of-use) |
> | **Cursor Agent** | Cursor | An active [Cursor](https://cursor.com/) subscription with agent mode enabled |
>
> Each user must independently obtain their own credentials and comply with the respective provider's licensing and usage terms. **This project does not include, bundle, or sublicense access to any AI model or API.** Usage of these agents may incur costs billed directly by the provider.

---

An RL training environment that tasks an LLM agent with optimizing GPU kernels for AMD ROCm hardware. The agent receives a baseline kernel, a sandbox with relevant source code and documentation, and is scored on compilation, correctness, and runtime speedup.

## How It Works

```
baseline kernel  →  prompt constructor  →  LLM agent  →  grader (Magpie)  →  score + reinjection
```

1. **Benchmark** — profile the model end-to-end to identify bottleneck kernels
2. **Identify** — rank kernels by GPU time and select candidates
3. **Optimize** — an LLM agent writes an optimized kernel in `output/<task_id>/solution.{py,hip}`
4. **Grade** — [Magpie](https://github.com/AMD-AGI/Magpie) checks compilation, correctness, and measures speedup
5. **Integrate** — kernels exceeding the speedup threshold (>1.05×) are hot-patched into site-packages
6. **Benchmark (final)** — re-run E2E benchmark with patches to measure real throughput improvement
7. **Score & Report** — compute rewards, update leaderboard, generate report

## Quick Start

### Prerequisites

- **OS:** Linux (Ubuntu 22.04+ recommended)
- **Python:** 3.10+
- **Node.js:** 18+ (for agent CLIs)
- **System packages:** `git`, `curl`, `jq`
- **GPU (optional):** AMD Instinct GPU with ROCm 6.x+ (required for real kernel grading; not needed for CPU-only eval)

### 1. Clone the Repository

```bash
git clone <repo-url> Apex
cd Apex
```

### 2. Install at Least One Agent CLI

Install whichever agent(s) you plan to use:

```bash
# Claude Code
npm install -g @anthropic-ai/claude-code
claude login

# OpenAI Codex
npm install -g @openai/codex
codex login

# Cursor Agent (standalone CLI)
npm install -g cursor-agent
cursor-agent login

# Cursor IDE (alternative — open Apex folder in Cursor; MCP servers auto-configure via .mcp.json)
```

### 3. Run Setup

```bash
bash setup.sh
```

This single command handles everything:

1. **CLI selection** — choose Claude Code, Codex, Cursor Agent, or all
2. **Python venv** — creates `.venv/` (or reuses an existing one)
3. **Python dependencies** — installs numpy, PyYAML, pytest, MCP packages, SDKs, etc.
4. **PyTorch for ROCm** — installs `torch` + `torchvision` from the ROCm 7.2 wheel index
5. **Triton** — installs the Triton compiler
6. **ROCm source repos** — clones AMD kernel source code into `tools/rocm/` (optional, for source-finder & RAG)
7. **Documentation** — downloads AMD architecture PDFs for the RAG server (optional)
8. **MCP servers** — installs and registers 5 MCP servers with the selected CLI(s)
9. **Magpie** — clones and installs the kernel evaluation framework into `tools/magpie/`
10. **Skills** — makes 13 domain-specific optimization skills discoverable by agents
11. **`.mcp.json`** — generates workspace config so Cursor IDE auto-discovers MCP servers

#### Setup Flags

```bash
bash setup.sh                     # Interactive (prompts for choices)
bash setup.sh --non-interactive   # Auto-detect CLIs, accept all defaults
bash setup.sh --skip-downloads    # Skip ROCm repo cloning + doc downloads
bash setup.sh --skip-tools        # Skip MCP + Magpie installation
bash setup.sh --venv=/path/.venv  # Use a specific venv path
```

### 4. Activate and Run

```bash
source .venv/bin/activate
export MAGPIE_ROOT=$(pwd)/tools/magpie

# Interactive agent session
claude                # or: codex / cursor-agent

# Automated pipeline
python3 workload_optimizer.py run \
  -r ./results \
  -b $MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_120b.yaml \
  --kernel-types triton --top-k 10 \
  --max-iterations 3 --max-turns 25 --leaderboard
```

## Repository Structure

```
Apex/
├── workload_optimizer.py    # Main pipeline CLI
├── eval.py                  # Mini eval (CPU-only, no GPU required)
├── setup.sh                 # One-shot environment setup
├── examples/                # Example benchmark/task configs, including custom workloads
├── mcp_config.json          # MCP server configuration
├── .mcp.json                # Auto-generated by setup.sh (MCP config for Cursor IDE)
│
├── agents/
│   └── backends.py          # Claude Code SDK + Codex + Cursor Agent runner
│
├── pipeline/
│   ├── knowledge_base.py    # Cross-kernel/cross-run learning store
│   ├── reflector.py         # Agent self-reflection between iterations
│   ├── trajectory.py        # Trajectory recording (file / CouchDB / S3)
│   ├── leaderboard.py       # Leaderboard tracking (file / CouchDB)
│   ├── kernel_bottleneck.py # Profiling data parser, kernel classification
│   └── export_rl_dataset.py # RL/SFT dataset export from trajectories
│
├── prompts/
│   ├── models.py            # Registry of 21 open-source LLMs
│   ├── configs.py           # 17 inference configurations
│   ├── kernel_prompt.py     # Kernel-level prompt constructor
│   └── model_prompt.py      # Model-level prompt constructor
│
├── graders/
│   ├── score.py             # Scoring formula + Magpie helpers
│   ├── kernel_grader.py     # Grades kernel tasks via Magpie
│   ├── model_grader.py      # Grades E2E model throughput via Magpie
│   ├── ground_truth.py      # ROCm kernel discovery + ground truth specs
│   ├── config_generator.py  # Magpie config.yaml generation + validation
│   └── cache_manager.py     # Cache isolation for reproducible grading
│
├── tools/
│   ├── setup_tools.sh       # Installs Magpie, MCP servers, skills
│   ├── skills/              # 13 domain skills (SKILL.md files)
│   ├── mcps/                # MCP server source
│   └── jsons/               # ROCm metadata indexes
│
├── files/
│   ├── setup_files.sh       # Clones ROCm repos and downloads docs
│   ├── hip_best_practices.md
│   └── triton_best_practices.md
│
├── tests/                   # pytest suite
│
└── output/                  # Agent solutions (git-ignored)
    └── <task_id>/
        ├── solution.py / solution.hip
        ├── config.yaml
        └── …
```

## Usage

### Full Pipeline (Automated)

Run the entire optimization loop end-to-end:

```bash
source .venv/bin/activate
export MAGPIE_ROOT=$(pwd)/tools/magpie

RESULTS=./results
BENCH_CONFIG=$MAGPIE_ROOT/examples/benchmarks/benchmark_vllm_gptoss_120b.yaml

python3 workload_optimizer.py run \
  -r $RESULTS \
  -b $BENCH_CONFIG \
  --kernel-types triton \
  --top-k 10 \
  --max-iterations 3 \
  --max-turns 25 \
  --leaderboard
```

### Step-by-Step Pipeline

```bash
# 1. Benchmark (or --skip-benchmark <path-to-existing-report.json>)
python3 workload_optimizer.py benchmark -r $RESULTS -b $BENCH_CONFIG

# 2. Identify top bottleneck kernels
python3 workload_optimizer.py identify -r $RESULTS --kernel-types triton --top-k 10

# 3. List identified kernels
python3 workload_optimizer.py list-kernels -r $RESULTS

# 4. Optimize all identified kernels
python3 workload_optimizer.py optimize -r $RESULTS --max-iterations 3 --max-turns 25

# 5. Integrate winners (auto-filters to >5% speedup)
python3 workload_optimizer.py integrate -r $RESULTS

# 6. Final E2E benchmark with optimized kernels
python3 workload_optimizer.py benchmark-final -r $RESULTS -b $BENCH_CONFIG

# 7. Score + trajectory + leaderboard
python3 workload_optimizer.py score -r $RESULTS --leaderboard

# 8. Generate report + replication guide
python3 workload_optimizer.py report -r $RESULTS -b $BENCH_CONFIG
```

### Custom Model Workloads

Apex can also score non-LLM model workloads through a command-driven `benchmark.yaml`.
This is the supported path for models where Magpie's built-in `vllm` /
`sglang` benchmark mode is not the right abstraction.

The benchmark config can declare:

```yaml
framework: custom_model
model: org/example-model
benchmark:
  kind: command
  metric: timed_mean_s
  goal: minimize
baseline:
  command: "bash -lc '... --output-json /tmp/baseline.json'"
  result_json: /tmp/baseline.json
optimized:
  command: "bash -lc '... --output-json /tmp/optimized.json'"
  result_json: /tmp/optimized.json
```

For command benchmarks:

- `metric` is the JSON field to compare, for example `timed_mean_s`.
- `goal: minimize` means Apex computes improvement as `baseline / optimized`.
- `goal: maximize` means Apex computes improvement as `optimized / baseline`.
- Commands must start with a trusted prefix such as `python3`, `python3 -m`, or `bash -lc`.

See [examples/benchmark_command_template.yaml](/root/burak/Apex/examples/benchmark_command_template.yaml:1) for a generic command-benchmark template.

For a portable FastVideo WAN text-to-video template, see
[examples/benchmark_fastvideo_wan_t2v_template.yaml](/root/burak/Apex/examples/benchmark_fastvideo_wan_t2v_template.yaml:1).
It uses the command-benchmark path with `timed_mean_s` latency scoring and a rocprof
kernel-stats CSV for bottleneck identification, without machine-specific paths.

For a repo-local FastVideo E2E scaffold inside Apex, use:

- [examples/benchmark_fastvideo_wan_t2v_local_template.yaml](/root/burak/Apex/examples/benchmark_fastvideo_wan_t2v_local_template.yaml:1)
- [compare_e2e.py](/root/burak/Apex/files/fastvideo_e2e/compare_e2e.py)
- [profile_e2e.py](/root/burak/Apex/files/fastvideo_e2e/profile_e2e.py)
- [env.example](/root/burak/Apex/files/fastvideo_e2e/env.example)

This keeps the Apex-owned benchmark scripts and config in-repo. The remaining
external inputs are the FastVideo source checkouts, model assets, and runtime
profiling outputs.

FastVideo requirements:

- For standalone vendored kernel optimization in Apex:
  - Apex repo checkout
  - Apex Python environment (`.venv`) with working PyTorch + Triton + ROCm stack
  - MI300X or compatible ROCm GPU for real grading/benchmarking
  - An agent backend (`codex`, `claude`, or `cursor`) if you want Apex to generate optimizations automatically
- For FastVideo E2E optimization:
  - Everything above, plus:
  - A real `FASTVIDEO_REPO` checkout
  - A baseline `FASTVIDEO_BASELINE_REPO` checkout or clean comparison branch
  - Access to the FastVideo model assets, for example `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
  - A profiler CSV path for kernel identification (`FASTVIDEO_PROFILER_CSV`)
  - Sufficient local disk for model cache, generated outputs, and profiler traces

What Apex now provides in-repo:

- Repo-local standalone kernel snapshots under `files/fastvideo_snapshots/`
- Standalone kernel specs under `examples/fastvideo_kernel_specs/`
- Repo-local FastVideo E2E helper scripts under `files/fastvideo_e2e/`

What Apex still does not bundle:

- The full FastVideo repository
- Model weights or Hugging Face credentials
- ROCm, PyTorch, or Triton installation
- rocprof output generation by itself; you still need to collect the profiler CSV from a real FastVideo run

Recommended install flow for FastVideo users:

```bash
cd Apex
bash setup.sh --fastvideo
```

This does the normal Apex setup and then:

- creates `.env.fastvideo` from the repo template if it does not exist
- checks ROCm, PyTorch, and Triton availability
- verifies that the vendored FastVideo standalone snapshots and E2E template are present
- prints the next commands for standalone and E2E FastVideo workflows

Example FastVideo end-to-end optimization run:

```bash
source .venv/bin/activate

export FASTVIDEO_REPO=/path/to/FastVideo
export FASTVIDEO_BASELINE_REPO=/path/to/FastVideo_baseline
export FASTVIDEO_COMPARE_SCRIPT=/path/to/fastvideo_compare_e2e.py
export FASTVIDEO_PROFILER_CSV=/path/to/rocprof_fastvideo_kernel_stats.csv
export FASTVIDEO_BASELINE_RESULT_JSON=/tmp/fastvideo_baseline.json
export FASTVIDEO_OPTIMIZED_RESULT_JSON=/tmp/fastvideo_optimized.json
export APEX_VENV_ACTIVATE=$(pwd)/.venv/bin/activate

python3 workload_optimizer.py run \
  -r ./results_fastvideo \
  -b ./examples/benchmark_fastvideo_wan_t2v_template.yaml \
  --framework fastvideo \
  --gpu gfx942 \
  --kernel-types triton \
  --top-k 5 \
  --max-iterations 3 \
  --max-turns 25
```

Repo-local E2E setup:

```bash
cd Apex
bash setup.sh --fastvideo
source .venv/bin/activate

export APEX_ROOT=$(pwd)
source .env.fastvideo

# Replace these with your actual checkouts.
export FASTVIDEO_REPO=/path/to/FastVideo
export FASTVIDEO_BASELINE_REPO=/path/to/FastVideo_baseline

python3 workload_optimizer.py run \
  -r ./results_fastvideo \
  -b ./examples/benchmark_fastvideo_wan_t2v_local_template.yaml \
  --framework fastvideo \
  --gpu gfx942 \
  --kernel-types triton \
  --top-k 5 \
  --max-iterations 3 \
  --max-turns 25
```

To generate a repo-local profile summary before running Apex:

```bash
source .venv/bin/activate
python3 files/fastvideo_e2e/profile_e2e.py \
  --repo-root $FASTVIDEO_REPO \
  --model-name $FASTVIDEO_MODEL_NAME \
  --output-dir $FASTVIDEO_E2E_OUTPUT_DIR/profile
```

FastVideo Triton kernels currently recognized and patchable by Apex:

| Apex kernel spec | Primary FastVideo source | Description |
|---|---|---|
| `video_sparse_attn` | `files/fastvideo_snapshots/video_sparse_attn.py` | Block-sparse video attention forward kernel. |
| `fastvideo_sparse_index` | `files/fastvideo_snapshots/fastvideo_sparse_index.py` | Sparse attention index and map conversion kernels. |
| `fastvideo_linear_attn` | `files/fastvideo_snapshots/fastvideo_linear_attn.py` | Sparse linear attention Triton kernel. |
| `fastvideo_sliding_tile_attn` | `files/fastvideo_snapshots/fastvideo_sliding_tile_attn.py` | Sliding-tile attention Triton kernel. |
| `fastvideo_turbodiffusion_rmsnorm` | `files/fastvideo_snapshots/fastvideo_turbodiffusion_rmsnorm.py` | TurboDiffusion Triton RMSNorm kernel. |
| `fastvideo_turbodiffusion_layernorm` | `files/fastvideo_snapshots/fastvideo_turbodiffusion_layernorm.py` | TurboDiffusion Triton LayerNorm kernels. |
| `fastvideo_sla_preprocess` | `files/fastvideo_snapshots/fastvideo_sla_preprocess.py` | SLA preprocessing and routing Triton kernels such as `compress_kernel`. |
| `fastvideo_longcat_bsa` | `files/fastvideo_snapshots/fastvideo_longcat_bsa.py` | Vendored LongCat block-sparse attention Triton kernels. |

Notes:

- These are the FastVideo kernel specs Apex can map back to editable source files today.
- A given workload/profile may only activate a subset of these kernels.
- Some entries group multiple related Triton helper kernels under one Apex kernel spec.
- Self-contained standalone kernel specs for these vendored snapshots live under `examples/fastvideo_kernel_specs/`.

### Standalone Kernel Optimization

Optimize a single kernel without running the full pipeline:

```bash
python3 workload_optimizer.py optimize-kernel \
  -r ./results \
  --kernel path/to/baseline_kernel.py \
  --kernel-name rms_norm \
  --kernel-type triton \
  --agent-backend cursor \
  --max-iterations 3 --max-turns 25
```

Correctness modes for standalone optimization:

```bash
# PyTorch reference (default) — validates against a PyTorch implementation
--correctness-mode pytorch

# Library test — runs the original library's unit test suite
--correctness-mode library_test

# Accordo — HSA-level validation for HIP/C++ kernels
--correctness-mode accordo
```

### Interactive Agent Mode

Launch the agent directly for exploratory optimization:

```bash
# Claude Code
cd Apex && claude

# OpenAI Codex
cd Apex && codex

# Cursor Agent (standalone CLI)
cd Apex && cursor-agent

# Cursor IDE (open Apex folder — MCP servers auto-configure via .mcp.json)
cursor .
```

### Agent-Driven Kernel Optimization Examples

Give the agent a natural language prompt and let it run the pipeline:

```bash
# cursor-agent: optimize rms_norm with pytorch correctness
cursor-agent -p "Optimize the rms_norm Triton kernel from \
  tools/rocm/aiter/aiter/ops/triton/normalization/rmsnorm.py for MI355X. \
  Use workload_optimizer.py optimize-kernel with --correctness-mode pytorch, \
  --kernel-type triton, --max-iterations 1 --max-turns 10. \
  Save results to /tmp/rms_norm_results. Report the score and speedup."

# Automated (non-interactive): optimize with library_test validation
python3 workload_optimizer.py optimize-kernel \
  -r /tmp/silu_results \
  --kernel tools/rocm/aiter/aiter/ops/triton/activation.py \
  --kernel-name silu_mul --kernel-type triton \
  --agent-backend cursor \
  --correctness-mode library_test \
  --max-iterations 1 --max-turns 10
```

### Mini Eval (No GPU Required)

Exercises the full pipeline on a CPU-only task (naive Python RMSNorm → NumPy):

```bash
pip install -r requirements-eval.txt

python3 eval.py              # Uses Claude API
python3 eval.py --dry-run    # Skip API call, grade a trivial solution
python3 eval.py --model claude-opus-4-6 --max-turns 12
```

### Explore Prompts

```bash
python3 prompts/kernel_prompt.py --list                          # List all kernel task IDs
python3 prompts/kernel_prompt.py --task-id llama-3-1-8b-instruct__rms_norm  # Print a single prompt
python3 prompts/kernel_prompt.py --all > prompts.jsonl             # Dump all as JSONL
python3 prompts/kernel_prompt.py --target gfx942 --list            # Target a specific GPU
```

### Grade Output Tasks

```bash
python3 graders/kernel_grader.py    # Grade all kernel tasks in output/
python3 graders/model_grader.py     # Grade model-level tasks
```

### Export RL/SFT Datasets

```bash
# Export trajectory data as RL training tasks
python3 workload_optimizer.py export-rl -r ./results --export-output-dir ./datasets

# Include SFT warm-start pairs
python3 workload_optimizer.py export-rl -r ./results --export-output-dir ./datasets --sft
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/test_prompts.py -v     # Prompt tests only
pytest tests/test_graders.py -v     # Grader tests only
```

## Pipeline Options

### Agent Backend

```bash
--agent-backend claude         # Use Claude Code (default)
--agent-backend codex          # Use OpenAI Codex
--agent-backend cursor         # Use Cursor Agent
```

### Docker Image Override

The pipeline runs E2E benchmarks inside Docker containers. Override the vLLM image:

```bash
--docker-image vllm/vllm-openai-rocm:v0.19.0
```

Or set the environment variable:

```bash
export APEX_VLLM_ROCM_IMAGE=vllm/vllm-openai-rocm:v0.19.0
```

### Benchmark Caching

Cache E2E baseline results to skip the ~30-minute benchmark on repeat runs:

```bash
--benchmark-cache-hours 4
```

### Parallel Kernel Optimization

Optimize up to N kernels simultaneously (agent reasoning is API-bound; GPU grading is serialized):

```bash
--parallel-kernels 2
```

### Agent Model Routing

Assign different models based on kernel difficulty:

```bash
--agent-model-simple claude-sonnet-4-20250514 \
--agent-model-complex claude-opus-4-6
```

### Anti-Tampering

AST-based detection penalizes solutions that fake benchmark results (`sys.exit()`, hardcoded `PASS`, fabricated timings). Configure the speedup cap:

```bash
--tampering-speedup-cap 1.0
```

## Scoring

```
score = compiled × 20  +  correct × 100  +  speedup_score(S)
```

Where `S = baseline_time / optimized_time`. Only compiled + correct solutions earn the speedup component.

- **Compiled** (+20 pts): solution imports and defines the expected function
- **Correct** (+100 pts): passes all unit tests against the baseline
- **Speedup** (piecewise):
  - S ≥ 1.0: `100 + (S − 1) × 200` pts (e.g. 1.2× → 140, 2× → 300, 3× → 500)
  - S < 1.0: `max(0, 100 × S − 50)` pts (regression penalty)

### Model-Level Reward

```
reward = 0.7 × normalized_kernel_score  +  0.3 × (optimized_tps / baseline_tps − 1)
```

Kernel score is normalized to [0, 1] against a reference of 420 pts (compile + correct + 2× speedup). Model-level grading requires a full AMD GPU environment.

## Target Hardware

Default target: **AMD Instinct MI355X (gfx950 / CDNA4)**.

| `--target` | Hardware |
|---|---|
| `gfx950` | AMD Instinct MI355X (CDNA4) — default |
| `gfx942` | AMD Instinct MI300X (CDNA3) |
| `gfx940` | AMD Instinct MI300A (CDNA3) |
| `gfx90a` | AMD Instinct MI250X (CDNA2) |

The GPU is auto-detected via `rocm-smi` if available; otherwise falls back to gfx950.

## Kernel Reintegration (Hot-Patching)

When the pipeline integrates optimized kernels for the final E2E benchmark, it hot-patches installed Python modules in site-packages. All patches are restored after benchmarking.

**Supported (hot-patching):**

- **aiter, vllm, sglang** — Python/Triton `.py` kernels can be replaced in site-packages. Triton JIT re-compiles automatically on next invocation.
- **aiter HIP** — Standalone `.so` files can be recompiled with `hipcc` and swapped.

**Not supported (requires source rebuild):**

- **System C/C++ libraries** — hipBLASLt, rocBLAS, composable_kernel (CK), MIOpen, rccl are system-level shared libraries that cannot be individually hot-patched.
- **Monolithic `_C.so`** — vLLM, sglang, and PyTorch HIP kernels compile into a single binary and cannot be individually replaced.

## MCP Servers

Apex ships with 5 MCP servers that give agents access to domain-specific tools:

| MCP | Key tools | Purpose |
|-----|-----------|---------|
| **source-finder** | `find_kernel_source`, `classify_kernel` | Search kernel implementations across ROCm repos |
| **kernel-rag** | `search_kernel_optimization`, `get_optimization_playbook` | Optimization patterns, snippets, domain analysis |
| **gpu-info** | `get_gpu_info`, `get_arch_optimization_hints` | MI355X / CDNA4 specs and optimization hints |
| **fusion-advisor** | `detect_fusion_opportunities`, `generate_fused_kernel` | Kernel fusion detection and code generation |
| **magpie** | `analyze`, `compare`, `benchmark` | Kernel correctness/performance evaluation |

MCP servers are auto-configured for:
- **Claude Code** — registered via `claude mcp add` during setup
- **Codex** — registered via `codex mcp add` during setup
- **Cursor IDE** — auto-discovered from `.mcp.json` in the project root (no manual registration needed)

## Prompt Architecture

Apex uses a two-layer prompt system when agents optimize kernels:

**System prompt** (`SYSTEM_PROMPT` in `workload_optimizer.py`) defines the agent's role and constraints:
- GPU kernel engineer persona with AMD ROCm specialization
- MCP tool inventory and when to use each tool
- Skill paths (13 `SKILL.md` files the agent can read for domain knowledge)
- Mandatory compare-before-submit workflow via Magpie
- Speedup calibration guidance and anti-tampering rules

**Task prompt** (`KERNEL_PROMPT_TEMPLATE` in `prompts/kernel_prompt.py`) is built per-kernel with:

| Section | Content |
|---------|---------|
| Target hardware | GPU arch, wavefront size, MFMA units, LDS, HBM bandwidth |
| Task definition | Kernel type, model architecture, framework (vLLM/SGLang) |
| Source locations | Paths to baseline implementations in `tools/rocm/` (aiter, CK, etc.) |
| MCP tools table | Available tools for source search, RAG, GPU info, fusion, Magpie |
| Skills table | 13 domain-specific optimization skills the agent can read |
| Instructions | Step-by-step: locate baseline, analyze bottlenecks, write `solution.py` |
| Optimization hints | Architecture-specific tips (e.g. MFMA usage, LDS tiling for CDNA4) |

When running via the full pipeline, the prompt is further enriched with:
- **Baseline source code** inlined as markdown
- **Profiling data** (GPU time %, bound type, bandwidth/compute utilization)
- **Knowledge base insights** from prior optimization runs
- **Correctness reference** (PyTorch reference code or library test commands)

Preview any task prompt:

```bash
python3 prompts/kernel_prompt.py --task-id llama-3-1-8b-instruct__rms_norm
```

## Kernels with Library Test Coverage

These kernels have explicit library test commands in the ground truth registry, enabling `--correctness-mode library_test` for validation against aiter's own pytest suite:

| Kernel | Test command | Type |
|--------|-------------|------|
| `silu_mul` | `pytest aiter/op_tests/triton_tests/test_activation.py` | memory-bound |
| `gemm_bf16` | `pytest aiter/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py` | compute-bound |
| `gemm_w8a8` | `pytest aiter/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py` | compute-bound |
| `act_quant_fp8` | `pytest aiter/op_tests/triton_tests/quant/test_quant.py` | memory-bound |
| `kv_cache_ops` | `pytest aiter/op_tests/triton_tests/fusions/test_fused_kv_cache.py` | memory-bound |
| `all_reduce` | `pytest aiter/op_tests/multigpu_tests/test_quick_all_reduce.py` | comms |

## Model Registry

21 models covering a range of architectures:

| Family | Models | Attention | MLP |
|---|---|---|---|
| Llama 3 | 1B, 8B, 70B (×2) | GQA | Dense |
| Mistral / Mixtral | 7B, 8×7B, 8×22B | GQA | Dense / MoE |
| Qwen 2.5 | 7B, 32B, 72B, Coder-32B | GQA | Dense |
| Gemma 2 | 9B, 27B | GQA | Dense |
| DeepSeek | R1 (671B), V3 (671B), R1-Distill-70B | MLA / GQA | MoE / Dense |
| Kimi | K2-Thinking | MLA | MoE |
| GPT OSS | 120B | GQA | MoE |
| Phi | 3.5-mini, phi-4 | GQA | Dense |
| Falcon | 7B | MQA | Dense |

## Kernel Types

12 kernel types are defined, applicable to models based on their architecture:

| Kernel | Framework | Notes |
|---|---|---|
| `flash_attn_prefill` | Triton | Flash attention for prompt (prefill) phase |
| `paged_attn_decode` | Triton | Paged attention for autoregressive decoding |
| `mla_attn` | Triton | Multi-Head Latent Attention (DeepSeek MLA) |
| `fused_moe` | Triton | Fused MoE gate + routing + expert GEMM |
| `gemm_w8a8` | HIP | FP8 × FP8 GEMM for quantized linear layers |
| `gemm_bf16` | HIP | BF16 GEMM for QKV/up/gate/down projections |
| `rms_norm` | Triton | Pre/post-attention and MLP normalization |
| `rope_embedding` | Triton | Rotary position embedding (Q and K) |
| `kv_cache_ops` | Triton | KV cache reshape, copy, and quantization |
| `all_reduce` | HIP | Tensor-parallel all-reduce (RCCL + fused kernels) |
| `act_quant_fp8` | Triton | Dynamic per-token FP8 activation quantization |
| `silu_mul` | Triton | Fused SiLU × gate (SwiGLU) for MLP |

## Troubleshooting

### PyTorch ROCm installation fails

```bash
# Install manually with the correct ROCm version
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

# For older ROCm:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

### Agent CLI not found

```bash
# Verify Node.js 18+
node --version

# Claude Code
npm install -g @anthropic-ai/claude-code && claude login

# Codex
npm install -g @openai/codex && codex login

# Cursor Agent
npm install -g cursor-agent && cursor-agent login
```

### MCP servers not working in Cursor IDE

Cursor auto-discovers MCP servers from `.mcp.json`. If MCPs aren't loading:

1. Verify `.mcp.json` exists in the Apex root (generated by `setup.sh`)
2. Restart Cursor or reload the window
3. Check that the Python path in `.mcp.json` matches your venv

### No GPU detected

The pipeline auto-detects GPUs via `rocm-smi`. For CPU-only development:

```bash
# Use the mini eval (no GPU needed)
python3 eval.py --dry-run

# Use --skip-benchmark with pre-recorded profiling data
python3 workload_optimizer.py run -r ./results --skip-benchmark report.json ...
```
