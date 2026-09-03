"""Strict parser for evaluator-owned invocation-level kernel timing reports."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, sha256_file

from .kernel import CaseTiming, GradeAggregation
from .statistics import (
    GpuHealthSnapshot,
    MeasurementBlock,
    MeasurementPolicy,
    PairedTimingUnit,
    SampleSeries,
    TimingProtocol,
)


REPORT_SCHEMA = "apex.kernel-measurement/v1"
_MAX_REPORT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class KernelMeasurementArtifact:
    """Validated raw timings plus their immutable file identity."""

    path: Path
    sha256: str
    timing_method: str
    warmup_samples: int
    policy: MeasurementPolicy
    protocol: TimingProtocol
    aggregation: GradeAggregation
    cases: tuple[CaseTiming, ...]
    blocks: tuple[MeasurementBlock, ...]


def load_kernel_measurement_report(
    path: Path,
    *,
    aggregation: GradeAggregation = GradeAggregation.EQUAL_CASE,
    measurement_policy: MeasurementPolicy | None = None,
) -> KernelMeasurementArtifact:
    """Load raw samples without accepting summaries, NaN, duplicates, or links."""

    source, document = _load_document(Path(path))
    policy, protocol = _policy_protocol(document, measurement_policy)
    raw_cases = document["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("Measurement cases must be a non-empty list", "invalid_measurement_report")
    digest = sha256_file(source)
    parsed = tuple(_case(item, digest, policy, protocol) for item in raw_cases)
    cases = tuple(item[0] for item in parsed)
    identities = [item.case_id for item in cases]
    if len(identities) != len(set(identities)):
        raise ContractError("Measurement case IDs are duplicated", "duplicate_case_id")
    return KernelMeasurementArtifact(
        source,
        digest,
        protocol.timer,
        policy.warmup_samples,
        policy,
        protocol,
        aggregation,
        cases,
        tuple(block for item in parsed for block in item[1]),
    )


def _load_document(requested: Path) -> tuple[Path, Mapping[str, Any]]:
    if requested.is_symlink():
        raise IntegrityError("Kernel measurement report is not a regular file", "unsafe_measurement_report")
    try:
        source = requested.resolve(strict=True)
        metadata = os.lstat(source)
    except OSError as error:
        raise IntegrityError(
            "Kernel measurement report cannot be resolved",
            "measurement_report_missing",
        ) from error
    if not source.is_file() or metadata.st_nlink != 1:
        raise IntegrityError("Kernel measurement report is not a regular file", "unsafe_measurement_report")
    if metadata.st_size <= 0 or metadata.st_size > _MAX_REPORT_BYTES:
        raise ContractError("Kernel measurement report size is invalid", "invalid_measurement_report")
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError("Kernel measurement report cannot be decoded", "invalid_measurement_report") from error
    if not isinstance(document, Mapping):
        raise ContractError("Kernel measurement report must be an object", "invalid_measurement_report")
    return source, document


def _policy_protocol(
    document: Mapping[str, Any],
    trusted_policy: MeasurementPolicy | None,
) -> tuple[MeasurementPolicy, TimingProtocol]:
    _exact_keys(
        document,
        required={
            "schema",
            "policy_id",
            "sample_unit",
            "quantile_method",
            "timer",
            "timer_resolution_ns",
            "inner_repeats",
            "measurement_method_sha256",
            "abba_seed",
            "warmup_samples",
            "cases",
        },
        optional={"metadata"},
    )
    if document["schema"] != REPORT_SCHEMA:
        raise ContractError("Unsupported kernel measurement report", "unsupported_measurement_report")
    declared_policy = MeasurementPolicy(
        policy_id=str(document["policy_id"]),
        sample_unit=str(document["sample_unit"]),
        quantile_method=str(document["quantile_method"]),
        warmup_samples=_integer(document["warmup_samples"], "warmup_samples", minimum=0),
    )
    policy = trusted_policy or declared_policy
    if (
        policy.policy_id != declared_policy.policy_id
        or policy.sample_unit != declared_policy.sample_unit
        or policy.quantile_method != declared_policy.quantile_method
        or policy.warmup_samples != declared_policy.warmup_samples
    ):
        raise ContractError(
            "Measurement report disagrees with the frozen task policy",
            "measurement_policy_mismatch",
        )
    return policy, TimingProtocol(
        timer=str(document["timer"]).strip(),
        timer_resolution_ns=_positive_number(
            document["timer_resolution_ns"], "timer_resolution_ns"
        ),
        inner_repeats=_integer(document["inner_repeats"], "inner_repeats", minimum=1),
        measurement_method_sha256=str(document["measurement_method_sha256"]),
        abba_seed=_integer(document["abba_seed"], "abba_seed", minimum=0),
    )


def _case(
    value: object,
    digest: str,
    policy: MeasurementPolicy,
    protocol: TimingProtocol,
) -> tuple[CaseTiming, tuple[MeasurementBlock, ...]]:
    if not isinstance(value, Mapping):
        raise ContractError("Measurement case must be an object", "invalid_measurement_report")
    _exact_keys(
        value,
        required={"case_id", "blocks"},
        optional={"workload_count"},
    )
    case_id = str(value["case_id"]).strip()
    if not case_id:
        raise ContractError("Measurement case ID is empty", "invalid_measurement_report")
    blocks = _blocks(value["blocks"], case_id)
    reference = _series(blocks, "reference", digest, policy, protocol)
    optimized = _series(blocks, "optimized", digest, policy, protocol)
    return CaseTiming(
        case_id=case_id,
        reference=reference,
        optimized=optimized,
        workload_count=_integer(value.get("workload_count", 1), "workload_count", minimum=1),
        paired_units=_paired_units(blocks),
    ), blocks


def _paired_units(blocks: tuple[MeasurementBlock, ...]) -> tuple[PairedTimingUnit, ...]:
    units: list[PairedTimingUnit] = []
    for start in range(0, len(blocks), 4):
        quartet = blocks[start : start + 4]
        units.append(
            PairedTimingUnit(
                unit_id=start // 4,
                reference_samples_ms=tuple(
                    sample
                    for block in quartet
                    if block.implementation == "reference"
                    for sample in block.samples_ms
                ),
                optimized_samples_ms=tuple(
                    sample
                    for block in quartet
                    if block.implementation == "optimized"
                    for sample in block.samples_ms
                ),
            )
        )
    return tuple(units)


def _blocks(value: object, case_id: str) -> tuple[MeasurementBlock, ...]:
    if not isinstance(value, list) or not value or len(value) % 4:
        raise ContractError("ABBA blocks must be a non-empty multiple of four", "invalid_abba_blocks")
    parsed = tuple(_block(item, case_id) for item in value)
    positions = tuple(item.order_position for item in parsed)
    identifiers = tuple(item.block_id for item in parsed)
    if positions != tuple(range(len(parsed))) or len(set(identifiers)) != len(identifiers):
        raise ContractError("ABBA block positions or IDs are invalid", "invalid_abba_blocks")
    for start in range(0, len(parsed), 4):
        order = tuple(item.implementation for item in parsed[start : start + 4])
        if order not in {
            ("reference", "optimized", "optimized", "reference"),
            ("optimized", "reference", "reference", "optimized"),
        }:
            raise ContractError("Timing blocks are not paired ABBA", "invalid_abba_blocks")
    return parsed


def _block(value: object, case_id: str) -> MeasurementBlock:
    if not isinstance(value, Mapping):
        raise ContractError("Measurement block must be an object", "invalid_abba_blocks")
    _exact_keys(
        value,
        required={
            "block_id",
            "order_position",
            "implementation",
            "samples_ms",
            "invalid_sample_counts",
            "gpu_health_before",
            "gpu_health_after",
        },
        optional=set(),
    )
    before = value["gpu_health_before"]
    after = value["gpu_health_after"]
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ContractError("GPU health evidence is missing", "invalid_gpu_health")
    return MeasurementBlock(
        case_id=case_id,
        block_id=_integer(value["block_id"], "block_id", minimum=0),
        order_position=_integer(value["order_position"], "order_position", minimum=0),
        implementation=str(value["implementation"]),
        samples_ms=_samples(value["samples_ms"]),
        invalid_sample_counts=_counts(value["invalid_sample_counts"]),
        gpu_health_before=GpuHealthSnapshot.from_mapping(before),
        gpu_health_after=GpuHealthSnapshot.from_mapping(after),
    )


def _series(
    blocks: tuple[MeasurementBlock, ...],
    implementation: str,
    digest: str,
    policy: MeasurementPolicy,
    protocol: TimingProtocol,
) -> SampleSeries:
    selected = tuple(item for item in blocks if item.implementation == implementation)
    counts: dict[str, int] = {}
    for block in selected:
        for reason, count in block.invalid_sample_counts.items():
            counts[reason] = counts.get(reason, 0) + count
    return SampleSeries(
        tuple(sample for block in selected for sample in block.samples_ms),
        sample_unit=policy.sample_unit,
        invalid_sample_counts=counts,
        artifact_sha256=digest,
        timer=protocol.timer,
        timer_resolution_ns=protocol.timer_resolution_ns,
        inner_repeats=protocol.inner_repeats,
        measurement_method_sha256=protocol.measurement_method_sha256,
    )


def _samples(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError("Raw timing samples must be a non-empty list", "invalid_measurement_report")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ContractError("Raw timing samples must be numbers", "invalid_measurement_report")
    return tuple(float(item) for item in value)


def _counts(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractError("Invalid-sample counts must be an object", "invalid_measurement_report")
    result = {str(key): _integer(item, "invalid_sample_count", minimum=0) for key, item in value.items()}
    if any(not key.strip() for key in result):
        raise ContractError("Invalid-sample reason is empty", "invalid_measurement_report")
    return result


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{field} is invalid", "invalid_measurement_report")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} is invalid", "invalid_measurement_report")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ContractError(f"{field} is invalid", "invalid_measurement_report")
    return result


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
) -> None:
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise ContractError("Measurement report fields are invalid", "invalid_measurement_report")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = ["KernelMeasurementArtifact", "REPORT_SCHEMA", "load_kernel_measurement_report"]
