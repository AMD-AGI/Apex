"""
score.py — Shared scoring logic and Magpie helpers for the RL graders.

AgentKernelArena scoring formula (kernel-level):
  compiled    → +20 pts
  correct     → +100 pts
  speedup S   → +S × 100 pts  (S = baseline_time / optimized_time ≥ 1.0)

  Total max (uncapped): 220+ pts per task
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── scoring constants (AgentKernelArena) ─────────────────────────────────────

PTS_COMPILED  = 20
PTS_CORRECT   = 100


def speedup_score(speedup: float) -> float:
    """Points awarded for performance improvement. speedup = baseline/optimized."""
    return max(0.0, speedup) * 100.0


def total_score(compiled: bool, correct: bool, speedup: float) -> float:
    return (PTS_COMPILED if compiled else 0) + \
           (PTS_CORRECT  if correct  else 0) + \
           (speedup_score(speedup) if (compiled and correct) else 0)


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class KernelResult:
    task_id:   str
    compiled:  bool           = False
    correct:   bool           = False
    speedup:   float          = 0.0     # baseline_time / optimized_time
    score:     float          = field(init=False)
    raw:       dict           = field(default_factory=dict)
    error:     Optional[str]  = None

    def __post_init__(self):
        self.score = total_score(self.compiled, self.correct, self.speedup)

    def to_dict(self) -> dict:
        return {
            "task_id":  self.task_id,
            "compiled": self.compiled,
            "correct":  self.correct,
            "speedup":  round(self.speedup, 4),
            "score":    round(self.score,   2),
            "error":    self.error,
        }


@dataclass
class ModelResult:
    model_id:             str
    kernel_score:         float  = 0.0
    e2e_throughput_ratio: float  = 0.0   # optimized / baseline tokens-per-second
    score:                float  = field(init=False)
    raw:                  dict   = field(default_factory=dict)
    error:                Optional[str] = None

    def __post_init__(self):
        # Weight: 50% kernel score (normalised to 0-1), 50% e2e improvement
        k_norm = min(self.kernel_score / 320.0, 1.0)   # 320 = compile+correct+3× speedup
        e_norm = max(0.0, self.e2e_throughput_ratio - 1.0)  # improvement over baseline
        self.score = round((k_norm + e_norm) * 100.0, 2)

    def to_dict(self) -> dict:
        return {
            "model_id":             self.model_id,
            "kernel_score":         round(self.kernel_score,         2),
            "e2e_throughput_ratio": round(self.e2e_throughput_ratio, 4),
            "score":                self.score,
            "error":                self.error,
        }


# ── Magpie helpers ────────────────────────────────────────────────────────────

def _magpie_bin() -> str:
    """Return path to the magpie executable."""
    p = shutil.which("magpie")
    if p:
        return p
    # Fall back to the locally-cloned copy
    local = Path(__file__).parent.parent / "tools" / "magpie" / "main.py"
    if local.exists():
        return f"python3 {local}"
    raise FileNotFoundError(
        "magpie not found. Run tools/setup_tools.sh first."
    )


def run_magpie(args: list[str], timeout: int = 300) -> dict:
    """
    Run a magpie command and return parsed JSON output.

    magpie always exits 0 on partial results; errors appear in the JSON.
    """
    cmd = _magpie_bin().split() + args + ["--output-format", "json"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        stdout = proc.stdout.strip()
        # Magpie may print log lines before the JSON blob; find the last {...}
        json_start = stdout.rfind("{")
        if json_start != -1:
            return json.loads(stdout[json_start:])
        return {"error": f"no JSON in output: {stdout[:200]}",
                "stderr": proc.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"error": "magpie timed out"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except FileNotFoundError as e:
        return {"error": str(e)}


def parse_compare_result(raw: dict) -> tuple[bool, bool, float]:
    """
    Extract (compiled, correct, speedup) from a `magpie compare` JSON result.

    Magpie compare output schema (best-effort — adjust if schema changes):
      {
        "optimized": {
          "compilation": {"success": true/false},
          "correctness": {"passed": true/false},
          "performance": {
            "baseline_ms": 10.0,
            "optimized_ms": 6.5
          }
        }
      }
    """
    opt = raw.get("optimized", raw)

    compiled = bool(
        opt.get("compilation", {}).get("success") or
        opt.get("compiled") or
        raw.get("compiled")
    )
    correct = bool(
        opt.get("correctness", {}).get("passed") or
        opt.get("correct") or
        raw.get("correct")
    )

    perf = opt.get("performance", {})
    baseline_ms  = float(perf.get("baseline_ms",  0) or perf.get("baseline_time_ms",  0))
    optimized_ms = float(perf.get("optimized_ms", 0) or perf.get("optimized_time_ms", 0))
    speedup = (baseline_ms / optimized_ms) if optimized_ms > 0 else 0.0

    return compiled, correct, speedup


def parse_benchmark_result(raw: dict) -> float:
    """
    Extract throughput ratio (optimized / baseline) from `magpie benchmark` JSON.

    Schema (best-effort):
      {
        "benchmark": {
          "baseline_tps":  1234.5,
          "optimized_tps": 1856.7
        }
      }
    """
    bench = raw.get("benchmark", raw)
    baseline_tps  = float(bench.get("baseline_tps",  0) or bench.get("baseline_tokens_per_sec",  0))
    optimized_tps = float(bench.get("optimized_tps", 0) or bench.get("optimized_tokens_per_sec", 0))
    return (optimized_tps / baseline_tps) if baseline_tps > 0 else 0.0
