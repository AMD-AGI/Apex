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

## Repository Structure

```
Apex/
├── workload_optimizer.py    # Main pipeline CLI
├── eval.py                  # Mini eval (CPU-only, no GPU required)
├── setup.sh                 # One-shot environment setup
├── mcp_config.json          # MCP server configuration
│
├── agents/
│   └── backends.py          # Claude Code SDK + Codex CLI agent runner
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
│   ├── models.py            # Registry of 20 open-source LLMs
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

## Setup

**Requirements:** Python 3.10+, `git`, `curl`. AMD GPU with ROCm is optional (needed only for real kernel grading).

```bash
# Full setup: venv, dependencies, ROCm repos, Magpie + RAG tool
bash setup.sh

# Skip cloning repos and downloading docs (faster for development)
bash setup.sh --skip-downloads

# Skip Magpie + RAG tool install
bash setup.sh --skip-tools

# Custom venv path
bash setup.sh --venv=/path/to/.venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Agent CLI Installation

Install at least one agent CLI before running `setup.sh`:

```bash
# Claude Code
npm install -g @anthropic-ai/claude-code
claude login

# OpenAI Codex
npm install -g @openai/codex
codex login

# Cursor Agent
npm install -g cursor-agent
cursor-agent login
```

### What `setup.sh` Does

1. **CLI selection** — prompts you to choose Claude Code, Codex, or both
2. **Prerequisite checks** — verifies CLI(s) are installed, locates or creates a Python venv
3. **ROCm repos & docs** (optional) — clones AMD ROCm source repos into `tools/rocm/` and downloads architecture documentation
4. **MCP dependencies** — installs each MCP server's Python packages (including Magpie)
5. **MCP registration** — registers servers (`source-finder`, `kernel-rag`, `gpu-info`, `fusion-advisor`, `magpie`) with the selected CLI(s)
6. **Skill installation** — makes 13 domain skills discoverable by the agent
7. **Verification** — lists registered MCPs, prints configured paths, and shows ready-to-run commands

## Usage

### Full Pipeline (Automated)

Run the entire optimization loop end-to-end:

```bash
source .venv/bin/activate
export MAGPIE_ROOT=/path/to/Magpie

RESULTS=/path/to/results
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

### Interactive Agent Mode

Launch the agent directly for exploratory optimization:

```bash
# Claude Code
cd Apex && claude

# OpenAI Codex
cd Apex && codex

# Cursor Agent
cd Apex && cursor-agent
```

### Mini Eval (No GPU Required)

Exercises the full pipeline on a CPU-only task (naive Python RMSNorm → NumPy):

```bash
uv pip install -r requirements-eval.txt

python3 eval.py              # Uses Claude API
python3 eval.py --dry-run    # Skip API call, grade a trivial solution
python3 eval.py --model claude-opus-4-6 --max-turns 12
```

### Explore Prompts

```bash
python3 prompts/kernel_prompt.py --list                          # List all kernel task IDs
python3 prompts/kernel_prompt.py --task-id llama-3-1-8b__rms_norm  # Print a single prompt
python3 prompts/kernel_prompt.py --all > prompts.jsonl             # Dump all as JSONL
python3 prompts/kernel_prompt.py --target gfx942 --list            # Target a specific GPU
```

### Grade Output Tasks

```bash
python3 graders/kernel_grader.py    # Grade all kernel tasks in output/
python3 graders/model_grader.py     # Grade model-level tasks
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
score = compiled × 20  +  correct × 100  +  speedup × 100
```

Where `speedup = baseline_ms / optimized_ms`. Only correct solutions count the speedup component.

- **Compiled** (+20 pts): solution imports and defines the expected function
- **Correct** (+100 pts): passes all unit tests against the baseline
- **Speedup** (+speedup × 100 pts): e.g. 3× speedup → +300 pts

### Model-Level Reward

```
reward = 0.5 × normalized_kernel_score  +  0.5 × (optimized_tps / baseline_tps − 1)
```

Kernel score is normalized to [0, 1] against a reference of 320 pts. Model-level grading requires a full AMD GPU environment.

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

## Model Registry

19 open-source models covering a range of architectures:

| Family | Models | Attention | MLP |
|---|---|---|---|
| Llama 3 | 1B, 8B, 70B (×2) | GQA | Dense |
| Mistral / Mixtral | 7B, 8×7B, 8×22B | GQA | Dense / MoE |
| Qwen 2.5 | 7B, 32B, 72B, Coder-32B | GQA | Dense |
| Gemma 2 | 9B, 27B | MHA | Dense |
| DeepSeek | R1 (671B), V3 (671B), R1-Distill-70B | MLA / GQA | MoE / Dense |
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

## Building CK Examples (Optional)

Composable Kernel (CK) examples provide compiled HIP binaries that Accordo can use for HSA-level correctness validation. Only needed for `--correctness-mode accordo`.

```bash
bash tools/build_ck.sh --gpu-targets gfx950
```

Options:
- `--gpu-targets <arch>` — GPU architecture (default: auto-detect or `gfx950`)
- `--jobs N` / `-j N` — Parallel build jobs (default: `nproc`)
