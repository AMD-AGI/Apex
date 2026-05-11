# Contributing to Apex

Thanks for your interest in Apex! This guide explains how to contribute, report issues, and submit changes.

## Before You Start

- Read `README.md` to understand the project scope, pipeline stages (benchmark → identify → optimize → grade → integrate → score), and usage.
- Review `AGENTS.md` and `CLAUDE.md` for agent-specific conventions if you plan to work on the LLM-agent integration.
- Ensure you have a supported GPU environment (AMD Instinct GPU with ROCm 6.x+ for real kernel grading), or run CPU-only eval.
- Confirm you have access to at least one supported agent CLI (Claude Code, OpenAI Codex, or Cursor Agent) if your change touches the agent pipeline.

## Development Setup

```bash
# Interactive setup — creates .venv, installs Python deps, ROCm PyTorch, Triton, MCP servers, Magpie, etc.
bash setup.sh

# Non-interactive (auto-detect CLIs, accept defaults)
bash setup.sh --non-interactive

# Activate the venv and export Magpie root
source .venv/bin/activate
export MAGPIE_ROOT=$(pwd)/tools/magpie
```

## Workflow

1. Create a new branch from `main`.
2. Keep changes focused and scoped.
3. Run tests before submitting:

```bash
pytest tests/ -x
```

4. Open a Pull Request with motivation, impact, and verification steps.

## Code Style and Quality

- Follow PEP 8 for Python code.
- Keep functions small and single-purpose; prefer composition over deeply nested logic.
- Add documentation or comments when intent is non-obvious.
- Update `README.md`, `AGENTS.md`, or skill prompts under `prompts/` when behavior visible to agents or users changes.

## Testing and Verification

This project depends on GPU hardware/drivers and orchestrates external LLM agent CLIs. In your PR, include:

- Test environment (GPU model, ROCm version, Python version, Node.js version)
- Agent CLI(s) used (Claude Code, Codex, Cursor Agent) and their versions
- Execution mode (local or Docker; see `docker/`)
- Key commands and output summary, e.g.:

```bash
python3 workload_optimizer.py run -r ./results --task <task_id>
pytest tests/test_gpu_kernel_grader.py -v
```

- For changes touching the grader, include a Magpie compilation + correctness + speedup result for at least one baseline kernel.

## Filing Issues

Please include:

- Reproduction steps (exact commands and config flags)
- Expected vs actual behavior
- Environment (OS, GPU, ROCm version, Python version, Node.js version, agent CLI version)
- Relevant logs from `results/<task_id>/` or a minimal repro

## Security

If you discover a security issue, do not open a public issue. Contact maintainers through a private channel.

This project orchestrates third-party AI agents and downloads model weights, ROCm source, and AMD documentation at setup time — flag any credential leakage, supply-chain, or sandbox-escape concerns privately.

## Suggested Contributions

- Add new optimization skills under `prompts/` or `tools/`
- Improve the grader (`graders/`) — new correctness checks, additional speedup metrics
- Add baseline kernels or workloads to expand the benchmark surface
- Improve MCP server reliability or add new MCP tools under `tools/`
- Improve docs, examples, and tests

## License

By contributing, you agree that your contributions are licensed under the repository `LICENSE` (MIT).
