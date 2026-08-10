"""Build self-consistent Apex Magpie config contracts for isolated tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.core import sha256_file, sha256_json
from apex.runtime import (
    DependencyReceipt,
    MagpieConfigContract,
)


_PROFILERS = (
    "torch_profiler",
    "system_profiler",
    "tracelens",
    "gpu_monitor",
    "targeted_trace",
)
_SERVING = {"atom", "sglang", "vllm"}


def resolved_contract(
    config: Path,
    receipt: DependencyReceipt,
    *,
    status: str = "config_compatible",
    blockers: tuple[str, ...] = (),
) -> MagpieConfigContract:
    """Return a test double with the production contract's content bindings."""

    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw = document["benchmark"]
    scoring = _scoring_view(raw)
    identity = _identity(scoring)
    lifecycle = _lifecycle(scoring.get("server_lifecycle"))
    requested_components = _requested_components(scoring)
    phase_views = {
        "requested": copy.deepcopy(scoring),
        "scoring_measurement": copy.deepcopy(scoring),
    }
    plan: dict[str, Any] = {
        "schema": "apex.magpie-main-resolved-plan/v1",
        "raw_config_sha256": sha256_file(config),
        "effective_config_sha256": sha256_json(scoring),
        "scoring_config_sha256": sha256_json(scoring),
        "phase_views_sha256": sha256_json(phase_views),
        "identity": identity,
        "lifecycle": lifecycle,
        "phase_views": phase_views,
        "diagnostics": {},
        "objective": {},
        "requirements": {"gpu_count": 1},
        "source_runtime": {
            "requested_image": scoring.get("docker_image"),
            "benchmark_script": scoring.get("benchmark_script"),
            "requested_components": requested_components,
            "potential_components": requested_components,
            "observation": "required_at_runtime",
            "build_engagement_hooks": "adapter_required",
        },
        "expected_result": {
            "schema": "apex.magpie-main-result-contract/v1",
            "official_report_artifact": "benchmark_report.json",
            "official_report_required_fields": [
                "success",
                "framework",
                "model",
                "profiling_enabled",
                "throughput.total_token_throughput",
                "latency.ttft.p99_ms",
                "latency.tpot.p99_ms",
                "workspace_dir",
                "errors",
            ],
            "execution_attestation_artifact": "evaluator/execution_attestation.json",
            "execution_attestation_schema": "apex.magpie-execution-attestation/v1",
            "execution_attestation_required_fields": [
                "schema",
                "authority",
                "report_sha256",
                "official_report_path",
                "official_report_size_bytes",
                "config_sha256",
                "run_id",
                "pass_type",
                "lane_verified",
                "reward_eligible",
                "profiling_enabled",
                "process",
                "dependencies",
                "runtime",
                "gpu_engagement",
                "quality_gate",
                "errors",
            ],
            "authority": "apex_evaluator",
        },
        "extensions": {"owner": "apex"},
        "redactions": {"policy_id": "secret_name_markers_v2", "paths": []},
    }
    plan["plan_sha256"] = sha256_json(plan)
    capability: dict[str, Any] = {
        "schema": "apex.magpie-main-capability-receipt/v1",
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "framework": identity["framework"],
        "run_mode": identity["run_mode"],
        "lifecycle": lifecycle,
        "optimization_applicable": lifecycle != "cleanup",
        "reward_contract": {
            "owner": "apex",
            "policy_id": "e2e_throughput_qos_v1",
            "measurement_view_sha256": plan["scoring_config_sha256"],
            "required_metrics": [
                "total_token_throughput", "ttft_p99", "tpot_p99", "quality"
            ],
        },
        "capabilities": {
            "benchmark_execution": "published_magpie_main",
            "quality_evaluation": "runtime_required",
            "source_observation": "runtime_required",
            "source_build_engagement": "adapter_required",
        },
        "blockers": list(blockers),
    }
    capability["receipt_sha256"] = sha256_json(capability)
    return MagpieConfigContract(
        config.resolve(),
        sha256_file(config),
        receipt.commits["magpie"],
        receipt.lock_sha256,
        sha256_json(["test-magpie-main-resolution", str(config.resolve())]),
        plan,
        capability,
    )


class ResolvedPlanStub:
    """Resolve arbitrary fixture configs against one dependency receipt."""

    def __init__(
        self,
        receipt: DependencyReceipt,
        *,
        status: str = "config_compatible",
        blockers: tuple[str, ...] = (),
    ) -> None:
        self.receipt = receipt
        self.status = status
        self.blockers = blockers

    def resolve(self, config: Path) -> MagpieConfigContract:
        return resolved_contract(
            config,
            self.receipt,
            status=self.status,
            blockers=self.blockers,
        )


def _scoring_view(value: Mapping[str, Any]) -> dict[str, Any]:
    scoring = copy.deepcopy(dict(value))
    framework = str(scoring.get("framework", ""))
    scoring.setdefault("precision", "fp8")
    scoring.setdefault("run_mode", "docker")
    envs = scoring.setdefault("envs", {})
    if framework in _SERVING:
        if "RUN_EVAL" not in envs or _enabled(envs["RUN_EVAL"]):
            envs["RUN_EVAL"] = "true"
        envs.setdefault("MAGPIE_EVAL_TASKS", "gsm8k")
        envs.setdefault(
            "MAGPIE_EVAL_TASK_DEFINITION_PATH", "utils/evals/gsm8k.yaml"
        )
        envs.setdefault(
            "MAGPIE_EVAL_TASK_DEFINITION_SHA256", "c0e109ed6dc356e082aea80cd775c12d64dada787b88c602408a3b960e0b04a1"
        )
        envs.setdefault("MAGPIE_EVAL_DATASET_PATH", "openai/gsm8k")
        envs.setdefault("MAGPIE_EVAL_DATASET_NAME", "main")
        envs.setdefault(
            "MAGPIE_EVAL_DATASET_REVISION", "740312add88f781978c0658806c59bc2815b9866"
        )
        envs.setdefault("EVAL_TASKS_DIR", "utils/evals/gsm8k.yaml")
        envs.setdefault("MAGPIE_EVAL_BATCH_SIZE", "auto")
        max_length = int(envs.get("MAX_MODEL_LEN", 2248))
        envs.setdefault("MAGPIE_EVAL_POLICY_ID", "apex-lm-eval-gsm8k-v2")
        envs.setdefault(
            "MAGPIE_EVAL_PRIMARY_METRIC", "exact_match,strict-match"
        )
        envs.setdefault("MAGPIE_EVAL_MAX_LENGTH", str(max_length))
        envs.setdefault(
            "MAGPIE_EVAL_MAX_GEN_TOKENS", str(min(480, max_length - 1))
        )
    profiler = scoring.setdefault("profiler", {})
    for name in _PROFILERS:
        entry = profiler.setdefault(name, {})
        entry["enabled"] = False
    gap = scoring.setdefault("gap_analysis", {})
    gap["enabled"] = False
    scoring["run_kind"] = "measurement"
    return scoring


def _identity(scoring: Mapping[str, Any]) -> dict[str, str]:
    framework = str(scoring.get("framework", ""))
    model = str(scoring.get("model", ""))
    return {
        "framework": framework,
        "model": model,
        "model_sha256": sha256_json({"model": model}),
        "precision": str(scoring.get("precision", "fp8")),
        "run_mode": str(scoring.get("run_mode", "docker")),
        "workload_kind": "serving",
    }


def _lifecycle(value: object) -> str:
    if not isinstance(value, Mapping) or not _enabled(value.get("enabled")):
        return "one_shot"
    return "cleanup" if _enabled(value.get("cleanup")) else "reuse"


def _enabled(value: object) -> bool:
    return value is True or value == 1 or (
        isinstance(value, str)
        and value.strip().lower() in {"1", "true", "yes", "on"}
    )


def _requested_components(scoring: Mapping[str, Any]) -> list[str]:
    framework = str(scoring.get("framework", ""))
    components = [framework]
    envs = scoring.get("envs")
    if isinstance(envs, Mapping) and _enabled(envs.get("VLLM_ROCM_USE_AITER")):
        components.append("aiter")
    extra = envs.get("EXTRA_VLLM_ARGS") if isinstance(envs, Mapping) else None
    if isinstance(extra, str) and "--moe-backend flydsl" in extra:
        components.append("flydsl")
    return components


__all__ = ["ResolvedPlanStub", "resolved_contract"]
