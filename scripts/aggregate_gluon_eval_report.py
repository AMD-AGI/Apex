#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
aggregate_gluon_eval_report.py — Walk results/gluon_eval/<agent>/<scenario>/
and emit a single results/gluon_eval/SUMMARY.md.

Per (agent, scenario) we collect:
  • Standalone scenarios (A/B): standalone_result.json
  • Pipeline scenario  (C):     report.md + leaderboard.json
  • Saved prompts (per-iteration):  output/<task_id>/prompts/iter_NNN_user.md
  • Best solution:                  output/<task_id>/solution_best.py
  • Run log:                        run.log

The summary lists per-scenario tables (compiled / correct / speedup / score)
plus relative links to every saved prompt so a human can audit exactly what
the agent received.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCENARIO_LABEL = {
    "scenario_a_gluon_baseline":    "Scenario A — Optimize a Gluon baseline (vector_add)",
    "scenario_b_triton_to_gluon":   "Scenario B — Triton -> Gluon rewrite (rms_norm)",
    "scenario_c_gptoss20b_gluon":   "Scenario C — GPT-OSS-20B e2e with --rewrite-as gluon",
}
SCENARIO_ORDER = list(SCENARIO_LABEL.keys())


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _collect_standalone(rdir: Path) -> dict:
    out = {
        "compiled": "?", "correct": "?", "speedup": "?", "score": "?",
        "iters": "?", "task_dir": None, "prompts": [], "solution": None,
        "log": None, "error": "",
    }
    log = rdir / "run.log"
    if log.exists():
        out["log"] = log
    res = _read_json(rdir / "standalone_result.json")
    if res:
        out["compiled"] = bool(res.get("compiled", False))
        out["correct"]  = bool(res.get("correct", False))
        out["speedup"]  = f"{res.get('speedup', 0.0):.2f}x"
        out["score"]    = f"{res.get('score', 0):.0f}"
        out["iters"]    = res.get("iterations", "?")
        out["error"]    = res.get("error", "")[:300]
    output_dir = rdir / "output"
    if output_dir.exists():
        for task_dir in sorted(output_dir.iterdir()):
            if task_dir.is_dir():
                out["task_dir"] = task_dir
                pdir = task_dir / "prompts"
                if pdir.exists():
                    out["prompts"] = sorted(pdir.glob("iter_*_user.md"))
                sol = task_dir / "solution_best.py"
                if sol.exists():
                    out["solution"] = sol
                break
    return out


_TPS_RE = __import__("re").compile(r"(Baseline|Final)\s+TPS:\s*([0-9.]+)")
_IMPROV_RE = __import__("re").compile(r"Improvement:\s*([0-9.]+)x")
_REWARD_RE = __import__("re").compile(r"Model reward:\s*([0-9.]+)")
_KERNEL_RES_RE = __import__("re").compile(
    r"\[(\d+)/\d+\]\s*Optimizing:\s*(\S+).*?Result:\s*compiled=(True|False)\s+correct=(True|False)\s+speedup=([0-9.]+)x\s+score=([0-9]+)",
    __import__("re").DOTALL,
)
# Phase 4 honest-reporting markers. The pipeline emits these exact strings
# (see workload_optimizer.py:_is_improvement_significant and the patches
# log) so we can reliably tell apart real improvements, statistical noise,
# and runs where no patch was actually applied.
_TWO_STD_RE = __import__("re").compile(
    r"Improvement\s+([0-9.]+)\s+tok/s\s+<\s+2\*std\s+([0-9.]+)\s+tok/s"
)
_NOT_SIG_RE = __import__("re").compile(
    r"(throughput\s+(change|improvement)\s+(is\s+)?not\s+statistically\s+significant)",
    __import__("re").IGNORECASE,
)
_NO_PATCHES_RE = __import__("re").compile(
    r"No kernel patches were applied", __import__("re").IGNORECASE
)
_ENV_POLICY_TPS_RE = __import__("re").compile(
    r"\[env-policy\]\s+env-policy TPS:\s*([0-9.]+)"
)
_ENV_POLICY_DIFF_RE = __import__("re").compile(
    r"\[env-policy\]\s+agent proposed (\d+) flag\(s\):\s*\[(.*?)\]"
)


def _parse_pipeline_log(log: Path) -> dict:
    """Extract baseline/final TPS, improvement, reward, and per-kernel results.

    Also extracts the Phase 4 honesty markers (`stat_sig`,
    `patches_applied`, `env_policy_applied`, `env_policy_tps`) so the
    aggregator can hide noise / unapplied results from any "best agent"
    column.
    """
    out = {
        "baseline_tps": None,
        "final_tps": None,
        "improvement": None,
        "model_reward": None,
        "per_kernel": [],
        "stat_sig": None,            # True / False / None (unknown)
        "patches_applied": None,     # True / False / None
        "two_std": None,             # float tok/s, useful for tooltip
        "env_policy_applied": False,
        "env_policy_tps": None,
        "env_policy_flags": [],
    }
    if not log or not log.exists():
        return out
    text = log.read_text(errors="replace")
    for m in _TPS_RE.finditer(text):
        if m.group(1) == "Baseline" and out["baseline_tps"] is None:
            out["baseline_tps"] = float(m.group(2))
        elif m.group(1) == "Final":
            out["final_tps"] = float(m.group(2))
    m = _IMPROV_RE.search(text)
    if m:
        out["improvement"] = float(m.group(1))
    m = _REWARD_RE.search(text)
    if m:
        out["model_reward"] = float(m.group(1))
    seen = set()
    for m in _KERNEL_RES_RE.finditer(text):
        idx, name, comp, corr, sp, sc = m.groups()
        key = (idx, name)
        if key in seen:
            continue
        seen.add(key)
        out["per_kernel"].append({
            "kernel": name, "compiled": comp == "True", "correct": corr == "True",
            "speedup": float(sp), "score": int(sc),
        })

    if _NOT_SIG_RE.search(text):
        out["stat_sig"] = False
    elif (out["baseline_tps"] is not None
          and out["final_tps"] is not None
          and out["final_tps"] > out["baseline_tps"]):
        out["stat_sig"] = True
    m = _TWO_STD_RE.search(text)
    if m:
        out["two_std"] = float(m.group(2))

    if _NO_PATCHES_RE.search(text):
        out["patches_applied"] = False
    elif out["per_kernel"]:
        out["patches_applied"] = any(
            k["correct"] and k["speedup"] > 1.0 for k in out["per_kernel"]
        )

    m = _ENV_POLICY_TPS_RE.search(text)
    if m:
        out["env_policy_applied"] = True
        out["env_policy_tps"] = float(m.group(1))
    m = _ENV_POLICY_DIFF_RE.search(text)
    if m:
        flags_str = m.group(2)
        out["env_policy_flags"] = [
            f.strip().strip("'\"") for f in flags_str.split(",") if f.strip()
        ]

    return out


def _collect_pipeline(rdir: Path) -> dict:
    out = {
        "report": None, "leaderboard": None, "log": None,
        "task_dirs": [], "prompts_count": 0, "solutions": [],
        "kernels_done": "?", "per_kernel": [],
        "baseline_tps": None, "final_tps": None,
        "improvement": None, "model_reward": None,
        "stat_sig": None, "patches_applied": None, "two_std": None,
        "env_policy_applied": False, "env_policy_tps": None,
        "env_policy_flags": [],
    }
    log = rdir / "run.log"
    if log.exists():
        out["log"] = log
    if (rdir / "report.md").exists():
        out["report"] = rdir / "report.md"
    if (rdir / "leaderboard.json").exists():
        out["leaderboard"] = rdir / "leaderboard.json"
    output_dir = rdir / "output"
    if output_dir.exists():
        for task_dir in sorted(output_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            out["task_dirs"].append(task_dir)
            pdir = task_dir / "prompts"
            if pdir.exists():
                out["prompts_count"] += len(list(pdir.glob("iter_*_user.md")))
            sol = task_dir / "solution_best.py"
            if sol.exists():
                out["solutions"].append(sol)
        out["kernels_done"] = len(out["task_dirs"])
    parsed = _parse_pipeline_log(log)
    out.update(parsed)
    return out


def _rel(p: Path | None, root: Path) -> str:
    if p is None:
        return "—"
    try:
        return f"[`{p.relative_to(root)}`]({p.relative_to(root)})"
    except ValueError:
        return f"`{p}`"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_root", help="results/gluon_eval directory")
    args = ap.parse_args()

    root = Path(args.results_root).resolve()
    if not root.exists():
        print(f"results root not found: {root}", file=sys.stderr)
        return 1

    agents = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not agents:
        print(f"no agent subdirs under {root}", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("# Apex multi-agent Gluon evaluation — summary")
    lines.append("")
    lines.append(f"Results root: `{root}`")
    lines.append(f"Agents discovered: {', '.join(agents)}")
    lines.append("")
    lines.append("Each agent receives the same natural-language prompt for the "
                 "same scenario; only the saved per-iteration prompts under "
                 "`<task_dir>/prompts/iter_NNN_user.md` are the source of truth "
                 "for what was actually shown to the model.")
    lines.append("")
    lines.append("**Reading scenario-C verdict column**: `real` means at least "
                 "one kernel patch was injected (or the env-policy diff was "
                 "applied) AND the e2e TPS delta cleared the 2*std noise floor. "
                 "`noise` means the delta was inside the noise floor. "
                 "`noise / not injected` means no kernel patch was applied AND "
                 "no env-policy diff was applied — the reported speedup is "
                 "purely measurement noise and should NOT be attributed to "
                 "Apex. Rows tagged `noise` or `noise / not injected` must be "
                 "excluded from any 'best agent' tally.")
    lines.append("")

    prompts_appendix: list[str] = []

    for scenario in SCENARIO_ORDER:
        lines.append(f"## {SCENARIO_LABEL[scenario]}")
        lines.append("")
        if scenario.startswith("scenario_c"):
            lines.append(
                "| agent | baseline TPS | final TPS | e2e speedup | stat_sig | "
                "patches_applied | env_policy_applied | env-policy TPS | "
                "env-policy flags | model reward | per-kernel speedups | "
                "verdict | prompts | report | leaderboard | run log |"
            )
            lines.append(
                "|-------|--------------|-----------|-------------|----------|"
                "-----------------|--------------------|----------------|"
                "------------------|--------------|----------------------|"
                "---------|----------|--------|-------------|---------|"
            )
        else:
            lines.append("| agent | compiled | correct | speedup | score | iters | best solution | prompts | run log |")
            lines.append("|-------|----------|---------|---------|-------|-------|----------------|----------|----------|")
        any_data = False
        for agent in agents:
            rdir = root / agent / scenario
            if not rdir.exists():
                continue
            any_data = True
            if scenario.startswith("scenario_c"):
                data = _collect_pipeline(rdir)
                pk = ", ".join(
                    f"{k['kernel']}={k['speedup']:.2f}x"
                    + ("" if k["correct"] else "(✗corr)")
                    for k in data["per_kernel"]
                ) or "—"
                bt = f"{data['baseline_tps']:.1f}" if data['baseline_tps'] else "—"
                ft = f"{data['final_tps']:.1f}" if data['final_tps'] else "—"
                imp = f"{data['improvement']:.4f}x" if data['improvement'] else "—"
                rwd = f"{data['model_reward']:.3f}" if data['model_reward'] is not None else "—"

                # Phase 4 honesty: classify the row so a reader can tell
                # apart real e2e gains from measurement noise or runs that
                # never actually patched anything.
                stat_sig = data["stat_sig"]
                patches_applied = data["patches_applied"]
                env_policy_applied = data["env_policy_applied"]
                if stat_sig is True:
                    stat_str = "yes"
                elif stat_sig is False:
                    stat_str = f"no (Δ < 2*std≈{data['two_std']:.2f})" if data["two_std"] else "no"
                else:
                    stat_str = "—"
                pa_str = (
                    "yes" if patches_applied is True else
                    "no"  if patches_applied is False else
                    "—"
                )
                ep_str = "yes" if env_policy_applied else "no"
                ep_tps = f"{data['env_policy_tps']:.1f}" if data["env_policy_tps"] else "—"
                ep_flags = ", ".join(data["env_policy_flags"]) or "—"
                if patches_applied is False and not env_policy_applied:
                    verdict = "noise / not injected"
                elif stat_sig is False:
                    verdict = "noise"
                elif patches_applied is True or env_policy_applied:
                    verdict = "real"
                else:
                    verdict = "unknown"
                lines.append(
                    f"| {agent} | {bt} | {ft} | {imp} | {stat_str} | "
                    f"{pa_str} | {ep_str} | {ep_tps} | {ep_flags} | "
                    f"{rwd} | {pk} | {verdict} | "
                    f"{data['prompts_count']} | "
                    f"{_rel(data['report'], root)} | "
                    f"{_rel(data['leaderboard'], root)} | "
                    f"{_rel(data['log'], root)} |"
                )
                if data["task_dirs"]:
                    prompts_appendix.append(f"### {agent} / {scenario}")
                    for td in data["task_dirs"]:
                        for p in sorted((td / 'prompts').glob('iter_*_user.md')) if (td / 'prompts').exists() else []:
                            prompts_appendix.append(f"- {_rel(p, root)}")
                    prompts_appendix.append("")
            else:
                data = _collect_standalone(rdir)
                prompt_links = ", ".join(_rel(p, root) for p in data["prompts"]) or "—"
                lines.append(
                    f"| {agent} | {data['compiled']} | {data['correct']} | "
                    f"{data['speedup']} | {data['score']} | {data['iters']} | "
                    f"{_rel(data['solution'], root)} | {prompt_links} | "
                    f"{_rel(data['log'], root)} |"
                )
                if data["error"]:
                    lines.append(f"| {agent} | error | {data['error']} | | | | | | |")
        if not any_data:
            if scenario.startswith("scenario_c"):
                lines.append("| _no runs found_ | | | | | | | | | | | | | | | |")
            else:
                lines.append("| _no runs found_ | | | | | | | | |")
        lines.append("")

    if prompts_appendix:
        lines.append("## Pipeline (scenario C) prompt archive")
        lines.append("")
        lines.extend(prompts_appendix)

    out_path = root / "SUMMARY.md"
    out_path.write_text("\n".join(lines))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
