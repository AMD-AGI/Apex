"""Canonical raw-sample and quantile contract for kernel and E2E evidence."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from apex.core import ContractError, sha256_json


_IMPLEMENTATIONS = frozenset({"reference", "optimized"})
_TIMERS = frozenset({"hip_event", "torch_cuda_event", "external"})
_INVALID_REASONS = frozenset(
    {"warmup", "timer_error", "process_interruption", "gpu_health_violation"}
)


class MeasurementStatus(str, Enum):
    NOT_RUN_DUE_TO_GATE = "not_run_due_to_gate"
    NOT_RUN_DUE_TO_SAFETY = "not_run_due_to_safety"
    VALID = "valid"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MeasurementPolicy:
    """Versioned policy for invocation-level percentile evidence."""

    policy_id: str = "kernel_invocation_nearest_rank_v1"
    min_valid_samples: int = 300
    min_tail_observations: int = 3
    sample_unit: str = "kernel_invocation"
    quantile_method: str = "nearest_rank_v1"
    warmup_samples: int = 20
    keep_srobust_threshold: float = 1.05
    confidence_srobust_floor: float = 1.0
    worst_case_srobust_floor: float = 1.0
    max_cv: float = 0.10
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 1729
    bootstrap_repetitions: int = 1000
    min_bootstrap_units: int = 2

    def __post_init__(self) -> None:
        if self.policy_id != "kernel_invocation_nearest_rank_v1":
            raise ContractError(
                "unsupported canonical measurement policy identity",
                "unsupported_measurement_policy",
            )
        if self.min_valid_samples < 300 or self.min_tail_observations < 1:
            raise ContractError(
                "canonical measurement requires at least 300 valid samples",
                "invalid_measurement_policy",
            )
        if math.ceil(self.min_valid_samples * 0.01) < self.min_tail_observations:
            raise ContractError("sample minimum does not provide enough tail observations", "invalid_measurement_policy")
        if self.sample_unit != "kernel_invocation" or self.quantile_method != "nearest_rank_v1":
            raise ContractError("unsupported canonical measurement policy", "unsupported_measurement_policy")
        finite_positive = (
            self.keep_srobust_threshold,
            self.confidence_srobust_floor,
            self.worst_case_srobust_floor,
            self.max_cv,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
            raise ContractError("promotion thresholds must be positive", "invalid_measurement_policy")
        if (
            self.keep_srobust_threshold < 1.05
            or self.confidence_srobust_floor < 1.0
            or self.worst_case_srobust_floor < 1.0
            or self.max_cv > 0.10
        ):
            raise ContractError(
                "measurement policy cannot weaken canonical promotion gates",
                "invalid_measurement_policy",
            )
        if not 0.95 <= self.bootstrap_confidence_level < 1:
            raise ContractError("bootstrap confidence must be at least 0.95 and below one", "invalid_measurement_policy")
        if (
            self.warmup_samples < 0
            or self.bootstrap_seed < 0
            or self.bootstrap_repetitions < 100
            or self.min_bootstrap_units < 2
        ):
            raise ContractError("bootstrap policy is invalid", "invalid_measurement_policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "min_valid_samples": self.min_valid_samples,
            "min_tail_observations": self.min_tail_observations,
            "sample_unit": self.sample_unit,
            "quantile_method": self.quantile_method,
            "warmup_samples": self.warmup_samples,
            "keep_srobust_threshold": self.keep_srobust_threshold,
            "confidence_srobust_floor": self.confidence_srobust_floor,
            "worst_case_srobust_floor": self.worst_case_srobust_floor,
            "max_cv": self.max_cv,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "min_bootstrap_units": self.min_bootstrap_units,
        }


@dataclass(frozen=True, slots=True)
class TimingProtocol:
    """One method identity shared by both implementations in every case."""

    timer: str
    timer_resolution_ns: float
    inner_repeats: int
    measurement_method_sha256: str
    abba_seed: int

    def __post_init__(self) -> None:
        digest = self.measurement_method_sha256.removeprefix("sha256:")
        if self.timer not in _TIMERS:
            raise ContractError("unsupported kernel timer", "unsupported_timer")
        if not math.isfinite(self.timer_resolution_ns) or self.timer_resolution_ns <= 0:
            raise ContractError("timer resolution must be positive", "invalid_measurement_protocol")
        if self.inner_repeats != 1:
            raise ContractError(
                "batched inner repeats are not invocation-level p99",
                "unsupported_sample_unit",
            )
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContractError("measurement method hash is invalid", "invalid_measurement_protocol")
        if self.abba_seed < 0:
            raise ContractError("ABBA seed must be non-negative", "invalid_measurement_protocol")


@dataclass(frozen=True, slots=True)
class GpuHealthSnapshot:
    """Minimal evaluator health evidence surrounding one timing block."""

    device: str
    healthy: bool
    temperature_c: float
    clock_mhz: float

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ContractError("GPU health device is missing", "invalid_gpu_health")
        if not self.healthy:
            raise ContractError("GPU health gate failed", "gpu_health_violation")
        for field_name, value in (
            ("temperature_c", self.temperature_c),
            ("clock_mhz", self.clock_mhz),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ContractError(f"{field_name} is invalid", "invalid_gpu_health")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GpuHealthSnapshot":
        try:
            healthy = value["healthy"]
            if not isinstance(healthy, bool):
                raise TypeError("healthy")
            return cls(
                device=str(value["device"]),
                healthy=healthy,
                temperature_c=float(value["temperature_c"]),
                clock_mhz=float(value["clock_mhz"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("GPU health snapshot is malformed", "invalid_gpu_health") from error


@dataclass(frozen=True, slots=True)
class MeasurementBlock:
    """One raw, health-bracketed member of a seeded paired ABBA sequence."""

    case_id: str
    block_id: int
    order_position: int
    implementation: str
    samples_ms: tuple[float, ...]
    invalid_sample_counts: Mapping[str, int]
    gpu_health_before: GpuHealthSnapshot
    gpu_health_after: GpuHealthSnapshot

    def __post_init__(self) -> None:
        if not self.case_id or self.block_id < 0 or self.order_position < 0:
            raise ContractError("measurement block identity is invalid", "invalid_abba_blocks")
        if self.implementation not in _IMPLEMENTATIONS or not self.samples_ms:
            raise ContractError("measurement block implementation is invalid", "invalid_abba_blocks")
        if any(not math.isfinite(value) or value <= 0 for value in self.samples_ms):
            raise ContractError("latency samples must be finite and positive", "invalid_latency_sample")
        if any(
            key not in _INVALID_REASONS or not isinstance(count, int) or count < 0
            for key, count in self.invalid_sample_counts.items()
        ):
            raise ContractError("invalid sample accounting is unsupported", "invalid_sample_count")


@dataclass(frozen=True, slots=True)
class PairedTimingUnit:
    """One ABBA quartet collapsed into a paired block-bootstrap unit."""

    unit_id: int
    reference_samples_ms: tuple[float, ...]
    optimized_samples_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.unit_id < 0 or not self.reference_samples_ms:
            raise ContractError("paired timing unit is empty", "invalid_paired_timing_unit")
        if len(self.reference_samples_ms) != len(self.optimized_samples_ms):
            raise ContractError("paired timing unit counts differ", "invalid_paired_timing_unit")
        values = (*self.reference_samples_ms, *self.optimized_samples_ms)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ContractError("paired timing samples are invalid", "invalid_paired_timing_unit")


@dataclass(frozen=True, slots=True)
class BootstrapDistribution:
    """Seeded paired/block-bootstrap percentile samples for one timing case."""

    reference_p50_ms: tuple[float, ...]
    optimized_p50_ms: tuple[float, ...]
    reference_p99_ms: tuple[float, ...]
    optimized_p99_ms: tuple[float, ...]
    unit_count: int

    @property
    def repetitions(self) -> int:
        return len(self.reference_p50_ms)

    @property
    def s50(self) -> tuple[float, ...]:
        return tuple(
            reference / optimized
            for reference, optimized in zip(
                self.reference_p50_ms, self.optimized_p50_ms, strict=True
            )
        )

    @property
    def s99(self) -> tuple[float, ...]:
        return tuple(
            reference / optimized
            for reference, optimized in zip(
                self.reference_p99_ms, self.optimized_p99_ms, strict=True
            )
        )

    @property
    def srobust(self) -> tuple[float, ...]:
        return tuple(
            min(s50, s99)
            for s50, s99 in zip(self.s50, self.s99, strict=True)
        )


@dataclass(frozen=True, slots=True)
class SampleSeries:
    """Evaluator-owned valid samples and explicit invalid-reason counts."""

    values_ms: tuple[float, ...]
    sample_unit: str = "kernel_invocation"
    invalid_sample_counts: Mapping[str, int] = field(default_factory=dict)
    artifact_sha256: str | None = None
    timer: str | None = None
    timer_resolution_ns: float | None = None
    inner_repeats: int = 1
    measurement_method_sha256: str | None = None

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in self.values_ms):
            raise ContractError("latency samples must be finite and positive", "invalid_latency_sample")
        if any(count < 0 for count in self.invalid_sample_counts.values()):
            raise ContractError("invalid sample counts cannot be negative", "invalid_sample_count")
        if self.artifact_sha256 is not None and len(self.artifact_sha256.removeprefix("sha256:")) != 64:
            raise ContractError("sample artifact digest is invalid", "invalid_sample_digest")
        if self.inner_repeats != 1:
            raise ContractError("inner repeats do not represent invocations", "unsupported_sample_unit")
        if self.timer_resolution_ns is not None and (
            not math.isfinite(self.timer_resolution_ns) or self.timer_resolution_ns <= 0
        ):
            raise ContractError("timer resolution is invalid", "invalid_measurement_protocol")

    @property
    def method_identity(self) -> tuple[object, ...]:
        return (
            self.sample_unit,
            self.timer,
            self.timer_resolution_ns,
            self.inner_repeats,
            self.measurement_method_sha256,
        )

    @property
    def digest(self) -> str:
        return self.artifact_sha256 or sha256_json(
            {
                "sample_unit": self.sample_unit,
                "values_ms": list(self.values_ms),
                "invalid_sample_counts": dict(sorted(self.invalid_sample_counts.items())),
                "method_identity": list(self.method_identity),
            }
        )


@dataclass(frozen=True, slots=True)
class Quantiles:
    p50_ms: float
    p99_ms: float
    sample_count: int
    sample_unit: str
    quantile_method: str
    artifact_sha256: str


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return population CV without deleting legitimate tail samples."""

    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ContractError("CV input is invalid", "invalid_latency_sample")
    mean = statistics.fmean(values)
    return float(statistics.pstdev(values) / mean)


def paired_block_bootstrap(
    units: Sequence[PairedTimingUnit],
    policy: MeasurementPolicy,
    *,
    seed: int,
) -> BootstrapDistribution | None:
    """Resample whole paired ABBA units; never resample or trim tail points alone."""

    frozen = tuple(units)
    if len(frozen) < policy.min_bootstrap_units:
        return None
    generator = random.Random(seed)
    ref50: list[float] = []
    opt50: list[float] = []
    ref99: list[float] = []
    opt99: list[float] = []
    for _ in range(policy.bootstrap_repetitions):
        selected = tuple(frozen[generator.randrange(len(frozen))] for _ in frozen)
        reference = tuple(value for unit in selected for value in unit.reference_samples_ms)
        optimized = tuple(value for unit in selected for value in unit.optimized_samples_ms)
        reference_p50, reference_p99 = _raw_quantiles(reference)
        optimized_p50, optimized_p99 = _raw_quantiles(optimized)
        ref50.append(reference_p50)
        opt50.append(optimized_p50)
        ref99.append(reference_p99)
        opt99.append(optimized_p99)
    return BootstrapDistribution(
        tuple(ref50), tuple(opt50), tuple(ref99), tuple(opt99), len(frozen)
    )


def bootstrap_interval(
    values: Sequence[float], policy: MeasurementPolicy
) -> tuple[float, float]:
    """Return the canonical nearest-rank percentile bootstrap interval."""

    tail = (1.0 - policy.bootstrap_confidence_level) / 2.0
    return nearest_rank(values, tail), nearest_rank(values, 1.0 - tail)


def _raw_quantiles(values: Sequence[float]) -> tuple[float, float]:
    return float(statistics.median(values)), nearest_rank(values, 0.99)


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Return the 1-indexed nearest-rank percentile from sorted values."""

    if not values or not 0 < percentile <= 1:
        raise ContractError("nearest-rank input is invalid", "invalid_quantile_input")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return float(ordered[rank - 1])


def quantiles(series: SampleSeries, policy: MeasurementPolicy) -> Quantiles:
    """Validate a raw series and compute the canonical p50 and p99."""

    if series.sample_unit != policy.sample_unit:
        raise ContractError("sample unit is not reward eligible", "unsupported_sample_unit")
    if len(series.values_ms) < policy.min_valid_samples:
        raise ContractError(
            f"Need at least {policy.min_valid_samples} valid samples",
            "insufficient_samples",
            {"actual": len(series.values_ms), "required": policy.min_valid_samples},
        )
    if series.timer_resolution_ns is not None:
        fastest_ns = min(series.values_ms) * 1_000_000.0
        if fastest_ns < 10.0 * series.timer_resolution_ns:
            raise ContractError(
                "Timer resolution is insufficient for invocation-level p99",
                "needs_better_timer",
            )
    return Quantiles(
        p50_ms=float(statistics.median(series.values_ms)),
        p99_ms=nearest_rank(series.values_ms, 0.99),
        sample_count=len(series.values_ms),
        sample_unit=series.sample_unit,
        quantile_method=policy.quantile_method,
        artifact_sha256=series.digest,
    )
