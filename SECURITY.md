# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **GitHub Private Vulnerability Reporting:** [Report a vulnerability](../../security/advisories/new)
- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository — the Apex RL training environment, kernel-optimization pipeline (`workload_optimizer.py`, `eval.py`), MCP servers and tools under `tools/`, agent integrations under `agents/`, graders under `graders/`, and the `setup.sh` installer.

Because Apex orchestrates third-party LLM coding agents (Claude Code, OpenAI Codex, Cursor Agent), runs MCP servers, downloads ROCm source code and AMD documentation at setup time, and executes the kernel code that agents produce, please flag any of the following privately:

- Sandbox escape from agent execution or the kernel grader
- Credential leakage through prompts, logs, MCP server traffic, or generated code
- Supply-chain risk from `setup.sh` downloads (ROCm source repos, AMD docs, agent CLIs, Magpie clone)
- Hot-patching of `site-packages` performing unexpected mutations outside the intended kernel files
- Resource-exhaustion or denial-of-service against the host runner

For issues in the upstream agent CLIs (Claude Code, Codex, Cursor Agent), model providers (Anthropic, OpenAI, Cursor), or third-party dependencies (Magpie, vLLM, Triton, ROCm libraries), report to those vendors directly.

For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).
