#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""env_policy_prompt.py — Prompt template for the env-policy agent task.

The env-policy task is a model-agnostic, prompt-driven companion to the
existing kernel-rewrite loop. Instead of asking an agent to write a faster
kernel, we ask the same cursor / codex / claude agent to propose a minimal
diff to the `VLLM_ROCM_USE_AITER_*` environment-flag set so that unmatched
HIP/CK kernels in the bottleneck list are routed through their aiter Triton
equivalents.

This module is intentionally workload-agnostic: it carries NO hard-coded
knowledge about gpt-oss, Mixtral, DeepSeek, etc. Every per-workload value
(framework, gpu_arch, kernel breakdown, available flags) is rendered into
the prompt at call time so the same code works for any future workload.
"""

from __future__ import annotations

import textwrap
from typing import Iterable, Mapping


def render_env_policy_prompt(
    *,
    framework: str,
    gpu_arch: str,
    model_id: str,
    bottlenecks: Iterable[Mapping],
    available_flags: Iterable[Mapping],
    current_envs: Mapping[str, str] | None = None,
    benchmark_summary: str = "",
) -> str:
    """Render the env-policy prompt for the agent.

    Args:
        framework: "vllm" or "sglang".
        gpu_arch: "gfx942" / "gfx950" / etc.
        model_id: HuggingFace model id of the workload (free-form string).
        bottlenecks: iterable of dicts with keys
            {name, category, percent_total, total_time_us, calls,
             matched_kernel_spec, origin_library}.
        available_flags: iterable of dicts with keys
            {name, default, doc}.
        current_envs: currently set VLLM_ROCM_USE_AITER_* values (host env).
        benchmark_summary: free-form one-liner like
            "baseline TPS=440.0, top10 GPU%=87".

    Returns:
        A single str ready to hand to `_run_agent_iteration`.
    """
    bk_lines = []
    for b in bottlenecks:
        name = b.get("name", "?")
        spec = b.get("matched_kernel_spec") or "(unmatched)"
        cat = b.get("category", "?")
        pct = b.get("percent_total", 0.0)
        ms = b.get("total_time_us", 0.0) / 1000.0
        calls = b.get("calls", 0)
        origin = b.get("origin_library", "?")
        bk_lines.append(
            f"- `{name}`  spec={spec}  category={cat}  origin={origin}  "
            f"{pct:.1f}% ({ms:.1f} ms over {calls} calls)"
        )
    bk_block = "\n".join(bk_lines) if bk_lines else "(no bottlenecks supplied)"

    flag_lines = []
    for f in available_flags:
        n = f.get("name", "?")
        d = f.get("default", "?")
        doc = (f.get("doc", "") or "").strip().replace("\n", " ")[:160]
        flag_lines.append(f"- `{n}`  default=`{d}`  — {doc}")
    flag_block = "\n".join(flag_lines) if flag_lines else "(no flags discovered)"

    cur_lines = []
    for k, v in (current_envs or {}).items():
        cur_lines.append(f"- `{k}` = `{v}`")
    cur_block = "\n".join(cur_lines) if cur_lines else "(no aiter env vars currently set)"

    summary = benchmark_summary or "(none provided)"

    return textwrap.dedent(f"""\
        # Apex env-policy task — propose a minimal aiter env-flag diff

        You are an AMD ROCm performance engineer. Your job is **not** to
        write a kernel today. Your job is to propose the smallest possible
        diff to the `VLLM_ROCM_USE_AITER_*` environment flags that will
        re-route the largest unmatched HIP / CK kernels in the bottleneck
        list below to their aiter Triton equivalents.

        ## Workload context
        - Framework: `{framework}`
        - GPU arch: `{gpu_arch}`
        - Model: `{model_id}`
        - Benchmark summary: {summary}

        ## Bottleneck list (from rocprof; sorted by GPU time)
        {bk_block}

        ## Available `VLLM_ROCM_USE_AITER_*` flags (discovered on this install)
        {flag_block}

        ## Currently set aiter env vars (host environment)
        {cur_block}

        ## What to produce

        Reply with **exactly one** YAML code block, no other prose
        outside the code block. The pipeline parses the block directly;
        any extra text outside the block is silently dropped. Schema:

        ```yaml
        # env_policy: v1
        env_diff:
          # KEY: VALUE pairs. KEYs must come from the available-flags list
          # above. VALUEs are strings ("0" or "1" for booleans). Omit
          # flags you do not want to change.
          VLLM_ROCM_USE_AITER_PAGED_ATTN: "1"

        rationale:
          # Per-flag, one short sentence tied to a specific kernel from
          # the bottleneck list (cite by its `name` field).
          VLLM_ROCM_USE_AITER_PAGED_ATTN: "Re-routes wvSplitKrc paged-attn HIP launches to aiter Triton."

        risk:
          # Per-flag, one short sentence describing a possible regression.
          VLLM_ROCM_USE_AITER_PAGED_ATTN: "Aiter PA kernel may be slower at long context (>32k); revert if final TPS drops."
        ```

        ## Hard constraints
        - Output ONE yaml code block. No surrounding markdown.
        - Only propose flags that exist in the available-flags list above.
        - Do NOT touch flags that are already at the value you want — leave
          them out of `env_diff`.
        - Aim for the minimum diff. Three flags is plenty; ten is too many.
        - Each `rationale` MUST cite at least one kernel name verbatim.
        - If no flag flip is justified by the bottleneck list, return an
          empty `env_diff:` mapping and write a single sentence in
          `rationale` explaining why.

        ## Background skills
        - `tools/skills/aiter-reflection/SKILL.md` — aiter design and which
          ops have aiter Triton equivalents.
        - `tools/skills/mi300-cdna3-architecture/SKILL.md` — when aiter
          beats vllm on CDNA3 / CDNA4.
        """)


__all__ = ["render_env_policy_prompt"]
