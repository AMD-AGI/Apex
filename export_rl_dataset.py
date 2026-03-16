#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
export_rl_dataset.py — Convert Apex scored trajectories to keystone-rl-training format.

Reads WorkloadTrajectoryRecord JSON files from Apex's trajectories/ directory,
extracts per-kernel optimization tasks, generates ground truth via templates,
and outputs a keystone-compatible tasks.json (+ optional SFT warm-start JSONL).

Usage:
    python export_rl_dataset.py \\
        --trajectories-dir trajectories/ \\
        --results-dirs results_total_2 results_total_3 \\
        --output-dir /path/to/keystone/data/ \\
        [--sft] [--quality good] [--min-score 0] [--gpu-arch gfx950]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from ground_truth_templates import (
    OP_TYPE_MAP,
    get_ground_truth,
    get_instruction,
)
from trajectory import WorkloadTrajectoryRecord


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ExtractedTask:
    """An extracted kernel optimization task ready for keystone export."""
    task_id: str
    kernel_spec: str
    instruction: str
    base_gpu_kernel_code: str
    difficulty_level: int
    op_type: str
    ground_truth: dict  # {"cpu_baseline_code": str, "test_shapes_code": str}
    # Provenance
    source_trajectory_id: str = ""
    gpu_arch: str = "gfx950"
    framework: str = "vllm"
    model_id: str = ""
    score: float = 0.0


@dataclass
class SFTPair:
    """A supervised fine-tuning example from a scored trajectory."""
    task_id: str
    source_trajectory_id: str
    trajectory_quality: str
    reward: float
    messages: list[dict]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_trajectories(traj_dir: Path) -> list[WorkloadTrajectoryRecord]:
    """Load all WorkloadTrajectoryRecord JSON files from a directory."""
    records = []
    for path in sorted(traj_dir.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            if "workload_id" in data:
                records.append(WorkloadTrajectoryRecord.from_dict(data))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  [warn] skipping {path.name}: {e}", file=sys.stderr)
    return records


def _find_baseline_code(
    kernel_spec: str,
    results_dirs: list[Path],
) -> str | None:
    """Search results directories for baseline.py for a given kernel_spec."""
    patterns = [
        f"output/workload__vllm__{kernel_spec}/baseline.py",
        f"output/workload__sglang__{kernel_spec}/baseline.py",
        f"output/kernel_{kernel_spec}/baseline.py",
        f"output/*__{kernel_spec}/baseline.py",
    ]
    for rdir in results_dirs:
        for pattern in patterns:
            matches = list(rdir.glob(pattern))
            if matches:
                return matches[0].read_text(encoding="utf-8")
    return None


def _find_solution_code(
    kernel_spec: str,
    results_dirs: list[Path],
) -> str | None:
    """Search results directories for solution.py for a given kernel_spec."""
    patterns = [
        f"output/workload__vllm__{kernel_spec}/solution.py",
        f"output/workload__sglang__{kernel_spec}/solution.py",
        f"output/kernel_{kernel_spec}/solution.py",
        f"output/*__{kernel_spec}/solution.py",
    ]
    for rdir in results_dirs:
        for pattern in patterns:
            matches = list(rdir.glob(pattern))
            if matches:
                code = matches[0].read_text(encoding="utf-8")
                if code.strip():
                    return code
    return None


# ── Difficulty assignment ────────────────────────────────────────────────────

def assign_difficulty(scores: list[float]) -> int:
    """Map historical kernel optimization scores to difficulty level 1-3.

    Uses the best score across all trajectories for this kernel_spec.
    """
    if not scores:
        return 3
    best = max(scores)
    if best >= 260:
        return 1
    if best >= 200:
        return 2
    return 3


def classify_op_type(kernel_spec: str) -> str:
    return OP_TYPE_MAP.get(kernel_spec, "memory_bound")


# ── Task extraction ──────────────────────────────────────────────────────────

def extract_kernel_tasks(
    trajectories: list[WorkloadTrajectoryRecord],
    results_dirs: list[Path],
    gpu_arch: str = "gfx950",
    min_score: float = 0.0,
) -> list[ExtractedTask]:
    """Extract unique kernel tasks from trajectories, deduplicated by kernel_spec.

    For each unique kernel_spec found across all trajectories:
    1. Collect all scores from all trajectories
    2. Find the baseline code from results directories
    3. Generate ground truth from templates
    4. Assign difficulty from historical scores
    """
    # Collect per-kernel-spec data across all trajectories
    kernel_data: dict[str, dict] = {}
    for traj in trajectories:
        for kopt in traj.kernel_optimizations:
            spec = kopt.get("kernel_spec", "")
            if not spec:
                continue
            if spec not in kernel_data:
                kernel_data[spec] = {
                    "scores": [],
                    "best_traj": None,
                    "best_score": -1,
                    "model_id": traj.model_id,
                    "framework": traj.framework,
                    "gpu_arch": traj.gpu_arch or gpu_arch,
                }
            score = kopt.get("score", 0.0)
            kernel_data[spec]["scores"].append(score)
            if score > kernel_data[spec]["best_score"]:
                kernel_data[spec]["best_score"] = score
                kernel_data[spec]["best_traj"] = traj

    tasks = []
    for spec, data in sorted(kernel_data.items()):
        if data["best_score"] < min_score:
            continue

        baseline_code = _find_baseline_code(spec, results_dirs)
        if not baseline_code:
            print(f"  [warn] no baseline.py found for {spec}, using template fallback",
                  file=sys.stderr)

        try:
            gt = get_ground_truth(spec, baseline_code)
        except ValueError as e:
            print(f"  [warn] skipping {spec}: {e}", file=sys.stderr)
            continue

        model_slug = data["model_id"].split("/")[-1].replace(".", "-").lower() if data["model_id"] else "unknown"
        task_id = f"{spec}_{model_slug}_v1"
        difficulty = assign_difficulty(data["scores"])
        instruction = get_instruction(spec, data["gpu_arch"])

        task = ExtractedTask(
            task_id=task_id,
            kernel_spec=spec,
            instruction=instruction,
            base_gpu_kernel_code=baseline_code or "",
            difficulty_level=difficulty,
            op_type=classify_op_type(spec),
            ground_truth=gt,
            source_trajectory_id=data["best_traj"].trajectory_id if data["best_traj"] else "",
            gpu_arch=data["gpu_arch"],
            framework=data["framework"],
            model_id=data["model_id"],
            score=data["best_score"],
        )
        tasks.append(task)

    return tasks


# ── Keystone formatting ──────────────────────────────────────────────────────

def format_keystone_tasks(tasks: list[ExtractedTask]) -> list[dict]:
    """Format tasks as keystone TaskRow-compatible JSON dicts."""
    rows = []
    for task in tasks:
        rows.append({
            "task_id": task.task_id,
            "instruction": task.instruction,
            "base_gpu_kernel_code": task.base_gpu_kernel_code,
            "difficulty_level": task.difficulty_level,
            "op_type": task.op_type,
            "ground_truth": {
                "cpu_baseline_code": task.ground_truth["cpu_baseline_code"],
                "test_shapes_code": task.ground_truth["test_shapes_code"],
            },
        })
    return rows


# ── SFT warm-start extraction ───────────────────────────────────────────────

def extract_sft_pairs(
    trajectories: list[WorkloadTrajectoryRecord],
    results_dirs: list[Path],
    quality_filter: str | None = None,
    gpu_arch: str = "gfx950",
) -> list[SFTPair]:
    """Extract (prompt, solution) pairs from scored trajectories for SFT.

    Each pair is formatted with <think>/<answer> tags matching keystone's
    expected model output format.
    """
    pairs = []
    for traj in trajectories:
        if quality_filter and traj.trajectory_quality != quality_filter:
            continue

        for kopt in traj.kernel_optimizations:
            spec = kopt.get("kernel_spec", "")
            if not spec:
                continue
            if not kopt.get("compiled") or not kopt.get("correct"):
                continue
            if kopt.get("speedup", 0) <= 1.0:
                continue

            solution_code = _find_solution_code(spec, results_dirs)
            if not solution_code:
                continue

            instruction = get_instruction(spec, traj.gpu_arch or gpu_arch)

            baseline_code = _find_baseline_code(spec, results_dirs)
            prompt_text = instruction
            if baseline_code:
                prompt_text += f"\n\nBaseline kernel:\n```python\n{baseline_code}\n```"

            speedup = kopt.get("speedup", 1.0)
            think_content = (
                f"The baseline {spec} kernel can be optimized for AMD MI355X. "
                f"Key optimizations: tiling for LDS, MFMA utilization, "
                f"memory coalescing, and occupancy tuning. "
                f"This achieved {speedup:.2f}x speedup."
            )
            answer_content = solution_code

            assistant_msg = (
                f"<think>{think_content}</think>\n"
                f"<answer>{answer_content}</answer>"
            )

            model_slug = traj.model_id.split("/")[-1].replace(".", "-").lower() if traj.model_id else "unknown"
            task_id = f"{spec}_{model_slug}_v1"

            pairs.append(SFTPair(
                task_id=task_id,
                source_trajectory_id=traj.trajectory_id,
                trajectory_quality=traj.trajectory_quality,
                reward=traj.model_reward,
                messages=[
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": assistant_msg},
                ],
            ))

    return pairs


# ── Main export ──────────────────────────────────────────────────────────────

def export(
    trajectories_dir: Path,
    results_dirs: list[Path],
    output_dir: Path,
    include_sft: bool = False,
    quality_filter: str | None = None,
    min_score: float = 0.0,
    gpu_arch: str = "gfx950",
) -> dict:
    """Run the full export pipeline.

    Returns a summary dict with counts of exported items.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectories from {trajectories_dir} ...")
    trajectories = load_trajectories(trajectories_dir)
    print(f"  Loaded {len(trajectories)} workload trajectories")

    if not trajectories:
        print("  [error] no trajectories found", file=sys.stderr)
        return {"tasks_exported": 0, "sft_pairs_exported": 0}

    print(f"Extracting kernel tasks (min_score={min_score}) ...")
    tasks = extract_kernel_tasks(trajectories, results_dirs, gpu_arch, min_score)
    print(f"  Extracted {len(tasks)} unique kernel tasks")

    task_rows = format_keystone_tasks(tasks)
    tasks_path = output_dir / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(task_rows, f, indent=2)
    print(f"  Wrote {len(task_rows)} tasks to {tasks_path}")

    summary = {"tasks_exported": len(task_rows), "sft_pairs_exported": 0}

    if include_sft:
        print(f"Extracting SFT pairs (quality={quality_filter or 'all'}) ...")
        pairs = extract_sft_pairs(trajectories, results_dirs, quality_filter, gpu_arch)
        print(f"  Extracted {len(pairs)} SFT pairs")
        if pairs:
            sft_path = output_dir / "sft_warmstart.jsonl"
            with open(sft_path, "w") as f:
                for pair in pairs:
                    row = {
                        "task_id": pair.task_id,
                        "source_trajectory_id": pair.source_trajectory_id,
                        "trajectory_quality": pair.trajectory_quality,
                        "reward": pair.reward,
                        "messages": pair.messages,
                    }
                    f.write(json.dumps(row, default=str) + "\n")
            print(f"  Wrote {len(pairs)} SFT pairs to {sft_path}")
            summary["sft_pairs_exported"] = len(pairs)

    # Write a metadata file for provenance
    meta = {
        "source_trajectories_dir": str(trajectories_dir),
        "source_results_dirs": [str(d) for d in results_dirs],
        "num_trajectories_loaded": len(trajectories),
        "gpu_arch": gpu_arch,
        "min_score": min_score,
        "quality_filter": quality_filter,
        **summary,
    }
    meta_path = output_dir / "export_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export Apex trajectories to keystone-rl-training format"
    )
    parser.add_argument(
        "--trajectories-dir", type=Path,
        default=REPO_ROOT / "trajectories",
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--results-dirs", type=str, nargs="+",
        default=[],
        help="Result directories containing output/<task_id>/ dirs (supports globs)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        required=True,
        help="Output directory for tasks.json and optional sft_warmstart.jsonl",
    )
    parser.add_argument("--sft", action="store_true", help="Also emit SFT warm-start JSONL")
    parser.add_argument("--quality", type=str, default=None,
                        help="Filter trajectories by quality (good|mediocre|bad)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum kernel score to include as a task")
    parser.add_argument("--gpu-arch", type=str, default="gfx950",
                        help="Target GPU architecture")
    parser.add_argument("--framework", type=str, default="vllm",
                        help="Target framework")
    args = parser.parse_args()

    # Expand globs in results-dirs
    expanded_dirs = []
    for pattern in args.results_dirs:
        matches = glob.glob(pattern)
        for m in matches:
            p = Path(m)
            if p.is_dir():
                expanded_dirs.append(p)
    if not expanded_dirs:
        expanded_dirs = [REPO_ROOT / "output"]

    summary = export(
        trajectories_dir=args.trajectories_dir,
        results_dirs=expanded_dirs,
        output_dir=args.output_dir,
        include_sft=args.sft,
        quality_filter=args.quality,
        min_score=args.min_score,
        gpu_arch=args.gpu_arch,
    )
    print(f"\nExport complete: {summary}")


if __name__ == "__main__":
    main()
