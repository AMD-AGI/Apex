"""
export_rl_dataset.py — Export Apex trajectories to RL fine-tuning dataset.

Converts Apex pipeline trajectories into a TaskRow schema with 3 ground
truth modes (pytorch, library_test, accordo).

Also supports standalone mode: generate tasks directly from discovered
kernel implementations + ground truth (no trajectories needed).

Output files:
  datasets/tasks.json          — Task definitions for RL training
  datasets/sft_warmstart.jsonl — SFT warm-start pairs from successful runs
  datasets/export_metadata.json — Provenance metadata
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

from ground_truth import (
    GroundTruthSpec,
    MANUAL_REGISTRY,
    ROCM_DIR,
    discover_all,
    get_spec,
)
from pipeline.trajectory import (
    FileStore,
    TrajectoryRecord,
    WorkloadTrajectoryRecord,
    _record_from_dict,
)


_SPEC_CACHE: dict[str, GroundTruthSpec | None] = {}


def _get_spec_cached(kernel_type: str) -> GroundTruthSpec | None:
    """Cache-through wrapper for get_spec to avoid re-scanning for each kernel."""
    if kernel_type in _SPEC_CACHE:
        return _SPEC_CACHE[kernel_type]
    spec = get_spec(kernel_type)
    _SPEC_CACHE[kernel_type] = spec
    return spec


# ── Kernel type -> instruction templates ─────────────────────────────────────

_INSTRUCTION_TEMPLATES: dict[str, str] = {
    "rms_norm": (
        "You are an expert GPU AI Compiler Engineer specializing in Triton kernels. "
        "Optimize the RMSNorm kernel for AMD MI355X (CDNA4, gfx950). "
        "Focus on vectorized memory access, fused residual add, and dynamic quantization fusion."
    ),
    "silu_mul": (
        "You are an expert GPU AI Compiler Engineer specializing in Triton kernels. "
        "Optimize the fused SiLU-and-Multiply (SwiGLU) activation kernel for AMD MI355X. "
        "Focus on memory coalescing, vectorized loads, and potential quantization fusion."
    ),
    "fused_moe": (
        "You are an expert GPU AI Compiler Engineer specializing in Triton kernels. "
        "Optimize the Fused Mixture-of-Experts kernel for AMD MI355X (CDNA4, gfx950). "
        "Focus on expert routing efficiency, MFMA utilization, tiled GEMM, and memory coalescing."
    ),
    "flash_attn_prefill": (
        "You are an expert GPU AI Compiler Engineer specializing in attention kernels. "
        "Optimize the Flash Attention prefill kernel for AMD MI355X (CDNA4, gfx950). "
        "Focus on tiled QKV processing, online softmax, LDS utilization, and MFMA instructions."
    ),
    "paged_attn_decode": (
        "You are an expert GPU AI Compiler Engineer specializing in attention kernels. "
        "Optimize the Paged Attention decode kernel for AMD MI355X (CDNA4, gfx950). "
        "Focus on KV cache paging efficiency, memory coalescing, and reduction optimization."
    ),
    "gemm_bf16": (
        "You are an expert GPU AI Compiler Engineer specializing in GEMM kernels. "
        "Optimize the BF16 GEMM kernel for AMD MI355X (CDNA4, gfx950). "
        "Focus on MFMA tile shapes, double buffering, LDS bank conflict avoidance, and occupancy."
    ),
    "gemm_w8a8": (
        "You are an expert GPU AI Compiler Engineer specializing in quantized GEMM kernels. "
        "Optimize the W8A8 (INT8/FP8) quantized GEMM for AMD MI355X (CDNA4, gfx950). "
        "Focus on quantization-aware tiling, scale factor handling, and MFMA FP8 support."
    ),
    "rope_embedding": (
        "You are an expert GPU AI Compiler Engineer specializing in Triton kernels. "
        "Optimize the Rotary Position Embedding (RoPE) kernel for AMD MI355X. "
        "Focus on vectorized sin/cos computation, memory coalescing, and fusion opportunities."
    ),
    "act_quant_fp8": (
        "You are an expert GPU AI Compiler Engineer specializing in quantization kernels. "
        "Optimize the dynamic FP8 activation quantization kernel for AMD MI355X. "
        "Focus on per-token scaling, efficient max reduction, and fused quantize-dequantize."
    ),
    "mla_attn": (
        "You are an expert GPU AI Compiler Engineer specializing in attention kernels. "
        "Optimize the Multi-Head Latent Attention (MLA) kernel for AMD MI355X. "
        "Focus on latent compression handling, sparse computation, and memory efficiency."
    ),
    "kv_cache_ops": (
        "You are an expert GPU AI Compiler Engineer specializing in memory kernels. "
        "Optimize the KV cache operations kernel for AMD MI355X. "
        "Focus on paged memory management, copy efficiency, and quantized cache support."
    ),
    "all_reduce": (
        "You are an expert GPU AI Compiler Engineer specializing in collective operations. "
        "Optimize the tensor-parallel all-reduce kernel for AMD MI355X. "
        "Focus on cross-XCD communication, bandwidth utilization, and overlap with compute."
    ),
}

_DEFAULT_INSTRUCTION = (
    "You are an expert GPU AI Compiler Engineer specializing in HIP/Triton kernels. "
    "Optimize the {kernel_type} kernel for AMD MI355X (CDNA4, gfx950). "
    "Focus on memory coalescing, tiling, MFMA utilization, and occupancy optimization."
)


def _get_instruction(kernel_type: str) -> str:
    return _INSTRUCTION_TEMPLATES.get(
        kernel_type,
        _DEFAULT_INSTRUCTION.format(kernel_type=kernel_type),
    )


# ── Baseline code extraction ────────────────────────────────────────────────

def _read_baseline_code(kernel_type: str, framework: str = "vllm") -> str:
    """Try to read the baseline kernel source code from tools/rocm/."""
    try:
        from kernel_prompt import KERNEL_MAP
    except ImportError:
        KERNEL_MAP = {}

    spec = KERNEL_MAP.get(kernel_type)
    if spec is None:
        return ""

    path_attr = f"{framework}_path"
    rel_path = getattr(spec, path_attr, "") or ""
    if not rel_path:
        return ""

    candidates = [
        ROCM_DIR / framework / rel_path,
        ROCM_DIR / rel_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                _log.debug("source read failed for %s: %s", candidate, e)
    return ""


# ── Task generation ─────────────────────────────────────────────────────────

def _trajectory_to_tasks(
    record: WorkloadTrajectoryRecord | TrajectoryRecord,
    quality_filter: str | None = None,
    min_score: float = 0.0,
) -> list[dict]:
    """Convert a trajectory record into dataset task dicts."""
    tasks = []

    if isinstance(record, WorkloadTrajectoryRecord):
        if quality_filter and record.trajectory_quality != quality_filter:
            return []

        framework = record.framework or "vllm"
        model_id = record.model_id or ""

        for ko in record.kernel_optimizations:
            kernel_name = ko.get("kernel_name", "")
            if not kernel_name:
                continue

            score = ko.get("kernel_score", 0.0)
            if score < min_score:
                continue

            gt_spec = _get_spec_cached(kernel_name)
            base_code = _read_baseline_code(kernel_name, framework)

            if gt_spec:
                ground_truth = gt_spec.to_ground_truth_dict()
                difficulty = gt_spec.difficulty_level
                op_type = gt_spec.op_type
            else:
                ground_truth = {
                    "pytorch_reference_code": "",
                    "test_shapes_code": "",
                    "repo_url": "",
                    "unit_test_command": "",
                    "accordo_config": {},
                }
                difficulty = 2
                op_type = "memory_bound"

            task_id = f"{kernel_name}_{framework}_{model_id.replace('/', '_')}"

            tasks.append({
                "task_id": task_id,
                "instruction": _get_instruction(kernel_name),
                "base_gpu_kernel_code": base_code,
                "difficulty_level": difficulty,
                "op_type": op_type,
                "ground_truth": ground_truth,
            })

    elif isinstance(record, TrajectoryRecord):
        if quality_filter and record.trajectory_quality != quality_filter:
            return []
        if record.final_score < min_score:
            return []

        kernel_type = record.kernel_type or ""
        if not kernel_type:
            return []

        gt_spec = _get_spec_cached(kernel_type)
        base_code = _read_baseline_code(kernel_type, record.framework or "vllm")

        if gt_spec:
            ground_truth = gt_spec.to_ground_truth_dict()
            difficulty = gt_spec.difficulty_level
            op_type = gt_spec.op_type
        else:
            ground_truth = {
                "pytorch_reference_code": "",
                "test_shapes_code": "",
                "repo_url": "",
                "unit_test_command": "",
                "accordo_config": {},
            }
            difficulty = 2
            op_type = "memory_bound"

        tasks.append({
            "task_id": record.task_id or f"{kernel_type}_{record.framework}",
            "instruction": _get_instruction(kernel_type),
            "base_gpu_kernel_code": base_code,
            "difficulty_level": difficulty,
            "op_type": op_type,
            "ground_truth": ground_truth,
        })

    return tasks


def _trajectory_to_sft_pairs(
    record: WorkloadTrajectoryRecord | TrajectoryRecord,
) -> list[dict]:
    """Extract SFT warm-start pairs from successful trajectory iterations."""
    pairs = []

    if isinstance(record, TrajectoryRecord):
        for iteration in record.iterations:
            solution = iteration.get("solution_code", "")
            score = iteration.get("score", 0.0)
            if solution and score > 0:
                pairs.append({
                    "prompt": record.prompt,
                    "response": solution,
                    "score": score,
                    "kernel_type": record.kernel_type,
                    "task_id": record.task_id,
                })

    elif isinstance(record, WorkloadTrajectoryRecord):
        for ko in record.kernel_optimizations:
            if ko.get("correct") and ko.get("kernel_score", 0) > 0:
                kernel_name = ko.get("kernel_name", "unknown")
                pairs.append({
                    "prompt": _get_instruction(kernel_name),
                    "response": f"# Optimized {kernel_name} kernel (speedup: {ko.get('speedup', 0):.2f}x)",
                    "score": ko.get("kernel_score", 0),
                    "kernel_type": kernel_name,
                    "task_id": f"{kernel_name}_{record.framework}",
                })

    return pairs


# ── Standalone mode ──────────────────────────────────────────────────────────

def generate_standalone_tasks(
    max_specs: int = 200,
    rocm_dir: Path | None = None,
) -> list[dict]:
    """Generate tasks directly from discovered ground truth (no trajectories).

    Useful for creating training data before any optimization runs.
    """
    specs = discover_all(rocm_dir=rocm_dir, max_files=2000)
    tasks = []
    seen_types: set[str] = set()

    for spec in specs[:max_specs]:
        if spec.kernel_type in seen_types:
            continue
        seen_types.add(spec.kernel_type)

        base_code = _read_baseline_code(spec.kernel_type)

        tasks.append({
            "task_id": f"{spec.kernel_type}_{spec.source_library}",
            "instruction": _get_instruction(spec.kernel_type),
            "base_gpu_kernel_code": base_code,
            "difficulty_level": spec.difficulty_level,
            "op_type": spec.op_type,
            "ground_truth": spec.to_ground_truth_dict(),
        })

    return tasks


# ── Main export function ─────────────────────────────────────────────────────

def export(
    trajectories_dir: Path,
    results_dirs: list[Path],
    output_dir: Path,
    include_sft: bool = False,
    quality_filter: str | None = None,
    min_score: float = 0.0,
    gpu_arch: str = "gfx950",
    extra_files: list[Path] | None = None,
    standalone: bool = False,
) -> dict:
    """Export trajectories to RL training dataset format.

    Returns dict with counts: tasks_exported, sft_pairs_exported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload spec cache to avoid repeated scans
    _SPEC_CACHE.clear()
    for spec in discover_all(max_files=2000):
        if spec.kernel_type not in _SPEC_CACHE:
            _SPEC_CACHE[spec.kernel_type] = spec

    all_tasks: list[dict] = []
    all_sft: list[dict] = []
    trajectory_count = 0

    # Load trajectories from FileStore
    if trajectories_dir.exists():
        store = FileStore(base_dir=trajectories_dir)
        for tid in store.list_ids():
            record = store.load(tid)
            if record is None:
                continue
            trajectory_count += 1
            all_tasks.extend(_trajectory_to_tasks(
                record, quality_filter=quality_filter, min_score=min_score,
            ))
            if include_sft:
                all_sft.extend(_trajectory_to_sft_pairs(record))

    # Load trajectories from results dirs
    for rdir in (results_dirs or []):
        traj_file = rdir / "trajectory.json"
        if traj_file.exists():
            try:
                with open(traj_file) as f:
                    data = json.load(f)
                record = _record_from_dict(data)
                trajectory_count += 1
                all_tasks.extend(_trajectory_to_tasks(
                    record, quality_filter=quality_filter, min_score=min_score,
                ))
                if include_sft:
                    all_sft.extend(_trajectory_to_sft_pairs(record))
            except Exception as e:
                _log.debug("trajectory parse failed for %s: %s", traj_file, e)

    # Load extra trajectory files
    for extra in (extra_files or []):
        if extra.exists():
            try:
                with open(extra) as f:
                    data = json.load(f)
                record = _record_from_dict(data)
                trajectory_count += 1
                all_tasks.extend(_trajectory_to_tasks(
                    record, quality_filter=quality_filter, min_score=min_score,
                ))
                if include_sft:
                    all_sft.extend(_trajectory_to_sft_pairs(record))
            except Exception as e:
                _log.debug("trajectory parse failed for %s: %s", extra, e)

    # Standalone mode: supplement with auto-discovered tasks
    if standalone or not all_tasks:
        standalone_tasks = generate_standalone_tasks()
        existing_types = {t["task_id"] for t in all_tasks}
        for t in standalone_tasks:
            if t["task_id"] not in existing_types:
                all_tasks.append(t)

    # Deduplicate by task_id
    seen: set[str] = set()
    deduped: list[dict] = []
    for task in all_tasks:
        if task["task_id"] not in seen:
            seen.add(task["task_id"])
            deduped.append(task)
    all_tasks = deduped

    # Write tasks.json
    tasks_path = output_dir / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(all_tasks, f, indent=2, default=str)

    # Write SFT warm-start
    sft_count = 0
    if include_sft and all_sft:
        sft_path = output_dir / "sft_warmstart.jsonl"
        with open(sft_path, "w") as f:
            for pair in all_sft:
                f.write(json.dumps(pair, default=str) + "\n")
                sft_count += 1

    # Write metadata
    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "trajectories_dir": str(trajectories_dir),
        "results_dirs": [str(d) for d in (results_dirs or [])],
        "trajectory_count": trajectory_count,
        "tasks_exported": len(all_tasks),
        "sft_pairs_exported": sft_count,
        "quality_filter": quality_filter,
        "min_score": min_score,
        "gpu_arch": gpu_arch,
        "standalone_mode": standalone,
    }
    meta_path = output_dir / "export_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "tasks_exported": len(all_tasks),
        "sft_pairs_exported": sft_count,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Apex trajectories to RL fine-tuning dataset."
    )
    parser.add_argument(
        "--trajectories-dir",
        default=str(REPO_ROOT / "trajectories"),
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--results-dirs",
        nargs="*",
        default=[],
        help="Additional results directories to scan for trajectory.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "datasets"),
        help="Output directory for exported dataset",
    )
    parser.add_argument(
        "--include-sft",
        action="store_true",
        help="Include SFT warm-start pairs",
    )
    parser.add_argument(
        "--quality",
        choices=["good", "mediocre", "bad"],
        default=None,
        help="Filter trajectories by quality",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum kernel score to include",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Generate tasks from ground truth even without trajectories",
    )
    args = parser.parse_args()

    result = export(
        trajectories_dir=Path(args.trajectories_dir),
        results_dirs=[Path(d) for d in args.results_dirs],
        output_dir=Path(args.output_dir),
        include_sft=args.include_sft,
        quality_filter=args.quality,
        min_score=args.min_score,
        standalone=args.standalone,
    )

    print(f"[export_rl_dataset] Exported {result['tasks_exported']} tasks")
    if result["sft_pairs_exported"]:
        print(f"[export_rl_dataset] Exported {result['sft_pairs_exported']} SFT pairs")
    print(f"[export_rl_dataset] Output: {args.output_dir}")


if __name__ == "__main__":
    main()
