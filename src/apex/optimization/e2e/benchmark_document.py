"""Canonical JSON projection for one normalized Magpie benchmark result."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from apex.benchmark import NormalizedBenchmarkResult


def benchmark_document(result: NormalizedBenchmarkResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "run_id": result.run_id,
        "pass_type": result.pass_type.value,
        "succeeded": result.succeeded,
        "framework": result.framework,
        "model": result.model,
        "workspace_path": str(result.workspace_path),
        "report_path": str(result.report_path) if result.report_path else None,
        "throughput": asdict(result.throughput),
        "latency": asdict(result.latency),
        "quality": {
            **asdict(result.quality),
            "source_paths": [str(path) for path in result.quality.source_paths],
            "raw_artifact_paths": [
                str(path) for path in result.quality.raw_artifact_paths
            ],
        },
        "profiling_enabled": result.profiling_enabled,
        "run_kind": result.run_kind,
        "reward_eligible": result.reward_eligible,
        "model_revision": {
            **asdict(result.model_revision),
            "source_path": (
                str(result.model_revision.source_path)
                if result.model_revision.source_path
                else None
            ),
        },
        "inferencex_runtime": {
            **asdict(result.inferencex_runtime),
            "source_root": (
                str(result.inferencex_runtime.source_root)
                if result.inferencex_runtime.source_root
                else None
            ),
            "runtime_path": (
                str(result.inferencex_runtime.runtime_path)
                if result.inferencex_runtime.runtime_path
                else None
            ),
            "receipt_path": (
                str(result.inferencex_runtime.receipt_path)
                if result.inferencex_runtime.receipt_path
                else None
            ),
        },
        "lm_eval_runtime": {
            **asdict(result.lm_eval_runtime),
            "manifest_path": (
                str(result.lm_eval_runtime.manifest_path)
                if result.lm_eval_runtime.manifest_path
                else None
            ),
            "receipt_path": (
                str(result.lm_eval_runtime.receipt_path)
                if result.lm_eval_runtime.receipt_path
                else None
            ),
        },
        "serving_runtime": asdict(result.serving_runtime),
        "artifacts": [str(path) for path in result.artifacts],
        "errors": list(result.errors),
        "command_exit_code": result.command_exit_code,
        "timed_out": result.timed_out,
    }


__all__ = ["benchmark_document"]
