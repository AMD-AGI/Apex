# RL Environment for Kernel Optimization

An RL training environment that tasks an LLM agent with optimizing GPU kernels for AMD ROCm hardware. The agent is given a baseline kernel, a sandbox with relevant source code and documentation, and is scored on compilation, correctness, and runtime speedup.

## Overview

```
prompt constructor  →  LLM agent  →  output/  →  grader  →  score
```

1. **Prompt constructor** — generates a task prompt for a specific (model, kernel) pair targeting a particular GPU architecture (default: MI355X / gfx950)
2. **LLM agent** — reads the existing kernel, writes an optimized version to `output/<task_id>/solution.{py,hip}`, and produces a Magpie evaluation config
3. **Grader** — calls [Magpie](https://github.com/AMD-AGI/Magpie) to check compilation, correctness (unit tests), and measure speedup
4. **Score** — `+20 pts` compiled, `+100 pts` correct, `+speedup × 100 pts` performance

## Repository Structure

```
rl-env-kernel-optimization/
├── eval.py                  # End-to-end mini eval (CPU, no GPU required)
├── setup.sh                 # One-shot environment setup
│
├── prompts/
│   ├── models.py            # Registry of 19 open-source LLMs (Llama, DeepSeek, Qwen, …)
│   ├── configs.py           # Inference configurations (batch size, dtype, …)
│   ├── kernel_prompt.py     # Kernel-level prompt constructor (model × kernel pairs)
│   └── model_prompt.py      # Model-level prompt constructor (end-to-end eval)
│
├── graders/
│   ├── score.py             # Scoring formula + Magpie helpers
│   ├── kernel_grader.py     # Grades kernel-level output/ tasks via Magpie
│   └── model_grader.py      # Grades end-to-end model throughput via Magpie
│
├── tools/
│   └── setup_tools.sh       # Installs Magpie and RAG tool
│
├── files/
│   ├── setup_files.sh       # Clones ROCm repos and downloads documentation
│   ├── hip_best_practices.md
│   └── triton_best_practices.md
│
├── tests/                   # pytest suite for all components
│   ├── test_prompts.py
│   ├── test_graders.py
│   ├── test_tools.py
│   └── test_files.py
│
└── output/                  # Agent writes all solutions here (git-ignored)
    └── <task_id>/
        ├── solution.py / solution.hip
        ├── config.yaml       # Magpie evaluation config
        └── …
```

## Setup

**Requirements:** Python 3.10+, `git`, `curl`. AMD GPU with ROCm optional (needed only for real kernel grading).

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

**Environment variable required for the agent:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

### Run the mini eval (no GPU required)

Exercises the full pipeline on a CPU-only task (naive Python RMSNorm → NumPy):

```bash
# Uses the Claude API to write an optimized solution
python3 eval.py

# Skip the API call; write a trivial numpy solution and grade it
python3 eval.py --dry-run

# Use a different model or increase turn budget
python3 eval.py --model claude-opus-4-6 --max-turns 12
```

### Run the test suite

```bash
pytest tests/ -v
```

### Explore the prompt space

```bash
# List all kernel task IDs
python3 prompts/kernel_prompt.py --list

# Print a single kernel prompt
python3 prompts/kernel_prompt.py --task-id llama-3-1-8b-instruct__flash_attn_prefill

# Dump all kernel prompts as JSONL
python3 prompts/kernel_prompt.py --all > all_kernel_prompts.jsonl

# Target a specific GPU arch (default: gfx950 / MI355X)
python3 prompts/kernel_prompt.py --target gfx942 --framework vllm --list

# Same for model-level prompts
python3 prompts/model_prompt.py --list
```

### Grade output tasks

```bash
# Grade all kernel tasks in output/
python3 graders/kernel_grader.py

# Grade model-level tasks
python3 graders/model_grader.py
```

## Target Hardware

The default target is the **AMD Instinct MI355X (gfx950 / CDNA4)**. Other supported architectures:

| `--target` | Hardware |
|---|---|
| `gfx950` | AMD Instinct MI355X (CDNA4) — default |
| `gfx942` | AMD Instinct MI300X (CDNA3) |
| `gfx940` | AMD Instinct MI300A (CDNA3) |
| `gfx90a` | AMD Instinct MI250X (CDNA2) |

The GPU is auto-detected via `rocm-smi` if available; otherwise falls back to gfx950.

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
| `flash_attn_prefill` | Triton | Flash attention for the prompt (prefill) phase |
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

## Scoring

### Kernel-level (AgentKernelArena formula)

```
score = compiled × 20  +  correct × 100  +  (baseline_ms / optimized_ms) × 100
```

- **Compiled** (+20 pts): solution imports and defines the expected function
- **Correct** (+100 pts): passes all unit tests against the baseline
- **Speedup** (+speedup × 100 pts): e.g. 3× speedup → +300 pts

### Model-level

```
score = 0.5 × normalized_kernel_score  +  0.5 × (optimized_tps / baseline_tps − 1)
```

Kernel score is normalized to [0, 1] against a reference of 320 pts (compile + correct + 3× speedup). Model-level grading requires a full AMD GPU environment.

## Grading Architecture

Both graders call [Magpie](https://github.com/AMD-AGI/Magpie) with the `config.yaml` the agent writes to `output/<task_id>/`:

```yaml
gpu:
  device: 0
  arch: gfx950
baseline:
  path: <path-to-original-kernel>
optimized:
  path: ./solution.py
correctness:
  command: "pytest tests/ -k rms_norm -x"
performance:
  command: "python bench_rms_norm.py --arch gfx950"
  iterations: 100
```

The mini eval (`eval.py`) uses a lightweight local grader that runs without Magpie or AMD hardware, suitable for CI and development.

## Files and Documentation

`files/setup_files.sh` populates the agent sandbox with:

- **ROCm source repos** — rocm-libraries, hipBLASLt, hipCUB, composable_kernel, RCCL, and more
- **Inference framework source** — SGLang, vLLM, AITER, Triton
- **Documentation** — ROCm docs, MI355X architecture references, HIP/Triton tutorials
- **Best-practice guides** — `hip_best_practices.md`, `triton_best_practices.md`

## Development

```bash
# Run only prompt tests
pytest tests/test_prompts.py -v

# Run only grader tests
pytest tests/test_graders.py -v

# Check prompt counts
python3 prompts/models.py
python3 prompts/kernel_prompt.py
```
