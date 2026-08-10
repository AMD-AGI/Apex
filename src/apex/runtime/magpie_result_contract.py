"""Apex-owned result contract over unchanged published Magpie reports."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ConfigurationError


RESULT_SCHEMA = "apex.magpie-main-result-contract/v1"
EXECUTION_ATTESTATION_SCHEMA = "apex.magpie-execution-attestation/v1"


def build_magpie_result_contract() -> dict[str, Any]:
    """Describe the separate official-report and evaluator-attestation inputs."""

    return {
        "schema": RESULT_SCHEMA,
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
        "execution_attestation_schema": EXECUTION_ATTESTATION_SCHEMA,
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
    }


def validate_magpie_result_contract(value: object) -> None:
    """Reject drift back to private Magpie-branch report fields."""

    expected = build_magpie_result_contract()
    valid = (
        isinstance(value, Mapping)
        and dict(value) == expected
        and isinstance(value.get("official_report_required_fields"), list)
        and isinstance(value.get("execution_attestation_required_fields"), list)
    )
    if not valid:
        raise ConfigurationError(
            "Unsupported Apex benchmark-result expectation",
            "unsupported_magpie_result_schema",
        )


__all__ = [
    "EXECUTION_ATTESTATION_SCHEMA",
    "RESULT_SCHEMA",
    "build_magpie_result_contract",
    "validate_magpie_result_contract",
]
