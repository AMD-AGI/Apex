# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
knowledge_base.py — Persistent knowledge base for kernel optimization outcomes.

Records what strategies worked (or failed) across optimization runs, enabling
cross-kernel and cross-run learning.  Stored as a single JSON file with
fcntl-based locking for concurrent-safe updates.
"""

from __future__ import annotations

import fcntl
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OptimizationOutcome:
    kernel_spec: str = ""
    kernel_type: str = ""  # "triton" | "hip"
    gpu_arch: str = ""
    strategy_used: str = ""
    strategy_description: str = ""
    speedup: float = 0.0
    correct: bool = False
    score: float = 0.0
    agent_model: str = ""
    timestamp: str = ""
    solution_snippet: str = ""
    key_insight: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OptimizationOutcome:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# Heuristic patterns to detect optimization strategies from solution code
_STRATEGY_PATTERNS: list[tuple[str, str, list[str]]] = [
    (r"tl\.load.*,\s*mask.*,\s*other", "vectorized_loads",
     ["memory_bound", "coalescing"]),
    (r"tl\.dot\(", "mfma_matmul",
     ["compute_bound", "mfma"]),
    (r"torch\.(mm|addmm|matmul|bmm)\(", "library_dispatch",
     ["library_call", "cublas_hipblas"]),
    (r"@triton\.autotune", "autotuned_triton",
     ["autotuning", "triton"]),
    (r"tl\.store.*\.to\(tl\.float16\)|\.to\(tl\.bfloat16\)", "mixed_precision",
     ["precision", "fp16_bf16"]),
    (r"BLOCK_SIZE.*=.*\d{3,}", "large_tile",
     ["tiling", "occupancy"]),
    (r"fused|Fused|FUSED", "kernel_fusion",
     ["fusion"]),
    (r"__shared__|shared_mem|LDS|tl\.zeros\(.*BLOCK", "shared_memory",
     ["lds", "shared_memory"]),
]


def _infer_strategy(solution_code: str) -> tuple[str, str, list[str]]:
    """Infer the optimization strategy from solution source code."""
    strategies_found: list[tuple[str, list[str]]] = []
    for pattern, name, tags in _STRATEGY_PATTERNS:
        if re.search(pattern, solution_code):
            strategies_found.append((name, tags))

    if not strategies_found:
        return "custom_optimization", "Custom kernel optimization", ["custom"]

    primary = strategies_found[0]
    all_tags = []
    for _, t in strategies_found:
        all_tags.extend(t)

    descriptions = {
        "vectorized_loads": "Used vectorized loads for better memory coalescing",
        "mfma_matmul": "Leveraged MFMA matrix instructions via tl.dot",
        "library_dispatch": "Dispatched to optimized library calls (torch.mm/addmm)",
        "autotuned_triton": "Used Triton autotuning for block/tile size selection",
        "mixed_precision": "Applied mixed precision (FP16/BF16) for throughput",
        "large_tile": "Used large tile/block sizes for better occupancy",
        "kernel_fusion": "Fused multiple operations into a single kernel",
        "shared_memory": "Used shared memory (LDS) for data reuse",
    }

    name = primary[0]
    if len(strategies_found) > 1:
        name = "+".join(s[0] for s in strategies_found[:3])

    desc = descriptions.get(primary[0], f"Applied {primary[0]} optimization")
    if len(strategies_found) > 1:
        desc += " combined with " + ", ".join(
            descriptions.get(s[0], s[0]) for s in strategies_found[1:3]
        )

    return name, desc, list(dict.fromkeys(all_tags))


class KnowledgeBase:
    """Persistent store of kernel optimization outcomes."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).parent.parent / "knowledge_base.json"
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, data: list[dict]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(self._path)

    def record(self, outcome: OptimizationOutcome) -> None:
        """Append an optimization outcome to the knowledge base.

        Uses a single fd for the full read-modify-write cycle so the fcntl
        lock actually guards against concurrent writers.
        """
        self._path.touch(exist_ok=True)
        fd = open(self._path, "r+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            fd.seek(0)
            raw = fd.read()
            try:
                data = json.loads(raw) if raw.strip() else []
            except json.JSONDecodeError:
                data = []
            data.append(outcome.to_dict())
            fd.seek(0)
            fd.truncate()
            json.dump(data, fd, indent=2, default=str)
            fd.flush()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    def query(
        self,
        kernel_spec: str = "",
        kernel_type: str = "",
        gpu_arch: str = "",
        min_speedup: float = 0.0,
        top_k: int = 5,
    ) -> list[OptimizationOutcome]:
        """Query the knowledge base for matching outcomes, sorted by speedup."""
        data = self._load()
        results = []
        for d in data:
            if kernel_spec and d.get("kernel_spec") != kernel_spec:
                continue
            if kernel_type and d.get("kernel_type") != kernel_type:
                continue
            if gpu_arch and d.get("gpu_arch") != gpu_arch:
                continue
            if d.get("speedup", 0) < min_speedup:
                continue
            results.append(OptimizationOutcome.from_dict(d))

        results.sort(key=lambda o: o.speedup, reverse=True)
        return results[:top_k]

    def summarize_for_prompt(self, kernel_spec: str, kernel_type: str) -> str:
        """Generate a markdown section summarizing past optimization insights.

        Includes both successful strategies and failed attempts so agents
        avoid repeating mistakes.
        """
        exact = self.query(kernel_spec=kernel_spec, min_speedup=0.0, top_k=5)
        similar = self.query(kernel_type=kernel_type, min_speedup=0.0, top_k=5)

        exact_success = [o for o in exact if o.correct and o.speedup >= 1.0]
        exact_failed = [o for o in exact if not o.correct or o.speedup < 1.0]
        similar_success = [
            o for o in similar
            if o.kernel_spec != kernel_spec and o.correct and o.speedup >= 1.0
        ]

        if not exact_success and not exact_failed and not similar_success:
            return ""

        lines = ["\n## Optimization History (Knowledge Base)\n"]

        if exact_success:
            lines.append(f"### What worked for `{kernel_spec}`:")
            for o in exact_success[:3]:
                lines.append(
                    f"- **{o.speedup:.2f}x** using {o.strategy_used}: "
                    f"{o.strategy_description} ({o.agent_model}, {o.timestamp[:10]})"
                )
                if o.key_insight:
                    lines.append(f"  - Insight: {o.key_insight}")

        if exact_failed:
            lines.append(f"\n### What FAILED for `{kernel_spec}` (avoid these):")
            for o in exact_failed[:3]:
                lines.append(
                    f"- {o.strategy_used}: {o.strategy_description} "
                    f"({o.agent_model}, {o.timestamp[:10]})"
                )

        if similar_success:
            lines.append(f"\n### Insights from other `{kernel_type}` kernels:")
            for o in similar_success[:3]:
                lines.append(
                    f"- `{o.kernel_spec}` achieved **{o.speedup:.2f}x** "
                    f"using {o.strategy_used}: {o.strategy_description}"
                )
                if o.key_insight:
                    lines.append(f"  - Insight: {o.key_insight}")

        return "\n".join(lines) + "\n"

    def cross_kernel_insights(
        self,
        current_kernel: str,
        completed_results: list[dict],
    ) -> str:
        """Summarize what worked on other kernels in the current run."""
        if not completed_results:
            return ""

        good = [
            r for r in completed_results
            if r.get("correct") and r.get("speedup", 0) > 1.0
            and r.get("kernel_spec") != current_kernel
        ]
        if not good:
            return ""

        lines = ["\n## Cross-Kernel Insights (This Run)\n"]
        lines.append("The following kernels were already optimized in this run:\n")
        for r in good:
            lines.append(
                f"- `{r.get('kernel_spec', '?')}`: {r.get('speedup', 0):.2f}x speedup"
            )
            if r.get("strategy_used"):
                lines.append(f"  - Strategy: {r['strategy_used']}")
        lines.append(
            "\nConsider applying similar strategies if this kernel has "
            "comparable characteristics.\n"
        )
        return "\n".join(lines)

    def record_from_opt_result(
        self,
        opt_result_dict: dict,
        kernel_type: str = "triton",
        gpu_arch: str = "gfx950",
        agent_model: str = "",
        solution_code: str = "",
    ) -> None:
        """Record an optimization outcome from a KernelOptResult dict.

        Records both successful and failed attempts so future runs can learn
        from what didn't work.
        """
        correct = opt_result_dict.get("correct", False)
        speedup = opt_result_dict.get("speedup", 0)

        if not correct and not solution_code:
            return

        strategy, desc, tags = _infer_strategy(solution_code) if solution_code else (
            "unknown", "No solution produced", ["failed"]
        )

        if not correct:
            desc = f"FAILED ({opt_result_dict.get('error', 'incorrect results')[:100]}): {desc}"
            tags = ["failed"] + tags

        outcome = OptimizationOutcome(
            kernel_spec=opt_result_dict.get("kernel_spec", ""),
            kernel_type=kernel_type,
            gpu_arch=gpu_arch,
            strategy_used=strategy,
            strategy_description=desc,
            speedup=speedup if correct else 0.0,
            correct=correct,
            score=opt_result_dict.get("score", 0),
            agent_model=agent_model,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            solution_snippet=solution_code[:500] if solution_code else "",
            key_insight=desc,
            tags=tags,
        )
        self.record(outcome)
