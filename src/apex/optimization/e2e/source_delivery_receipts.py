"""Strict I/O for evaluator-owned E2E source-delivery receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError, canonical_json_bytes
from apex.evaluation import E2EAcceptancePolicy, E2EObservation
from apex.intake import RegressionGates


PRIMARY_BENCHMARK_SCHEMA = "apex.e2e-primary-source-benchmark/v1"


def load_primary_benchmark(path: Path | None) -> Mapping[str, Any]:
    if path is None or path.is_symlink() or not path.is_file():
        raise IntegrityError(
            "Primary benchmark receipt is missing", "missing_primary_receipt"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Primary benchmark receipt is invalid", "missing_primary_receipt"
        ) from error
    if not isinstance(value, Mapping) or value.get("schema") != PRIMARY_BENCHMARK_SCHEMA:
        raise IntegrityError(
            "Primary benchmark schema is invalid", "missing_primary_receipt"
        )
    return value


def measurement_from_mapping(value: object) -> E2EObservation:
    if not isinstance(value, Mapping):
        raise IntegrityError(
            "Stored E2E measurement is invalid", "missing_primary_receipt"
        )
    try:
        return E2EObservation(**dict(value))
    except (TypeError, ValueError) as error:
        raise IntegrityError(
            "Stored E2E measurement is invalid", "missing_primary_receipt"
        ) from error


def acceptance_policy_from_mapping(value: object) -> E2EAcceptancePolicy:
    if not isinstance(value, Mapping):
        raise IntegrityError("Acceptance policy is missing", "missing_primary_receipt")
    gates = value.get("gates")
    if not isinstance(gates, Mapping):
        raise IntegrityError("Acceptance gates are missing", "missing_primary_receipt")
    try:
        policy = E2EAcceptancePolicy(
            gates=RegressionGates(**dict(gates)),
            min_throughput_gain_pct=float(value["min_throughput_gain_pct"]),
            policy_id=str(value["policy_id"]),
            min_paired_windows=int(value["min_paired_windows"]),
            bootstrap_seed=int(value["bootstrap_seed"]),
            bootstrap_repetitions=int(value["bootstrap_repetitions"]),
            bootstrap_confidence_level=float(value["bootstrap_confidence_level"]),
            aa_envelope_pct=float(value["aa_envelope_pct"]),
            outlier_policy_id=str(value["outlier_policy_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError(
            "Acceptance policy is invalid", "missing_primary_receipt"
        ) from error
    if policy.to_dict() != dict(value):
        raise IntegrityError(
            "Acceptance policy is noncanonical", "missing_primary_receipt"
        )
    return policy


def primary_runtime_identity(value: Mapping[str, Any]) -> str:
    identity = value.get("runtime_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise IntegrityError(
            "Primary runtime identity is missing", "missing_primary_receipt"
        )
    return identity


def write_primary_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError(
            "Primary receipt already exists", "immutable_delivery_artifact"
        )
    with path.open("xb") as output:
        output.write(canonical_json_bytes(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())


__all__ = [
    "PRIMARY_BENCHMARK_SCHEMA",
    "acceptance_policy_from_mapping",
    "load_primary_benchmark",
    "measurement_from_mapping",
    "primary_runtime_identity",
    "write_primary_receipt",
]
