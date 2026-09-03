"""Paired raw-observation contract and point/confidence estimators for E2E."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ContractError, sha256_json

from .e2e import E2EAcceptancePolicy, E2EObservation


ESTIMATOR_ID = "paired_log_ratio_geomean_v1"


@dataclass(frozen=True, slots=True)
class E2EPairedWindow:
    """One complete counterbalanced A(current)-B-B-A observation window."""

    window_id: str
    anchor_before: E2EObservation
    candidate_forward: E2EObservation
    candidate_reverse: E2EObservation
    anchor_after: E2EObservation

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ContractError("Paired E2E window ID is empty", "invalid_e2e_paired_window")
        protocols = {item.protocol_hash for item in self.observations}
        if len(protocols) != 1:
            raise ContractError(
                "Paired E2E window protocols differ", "measurement_protocol_mismatch"
            )

    @property
    def observations(self) -> tuple[E2EObservation, ...]:
        return (
            self.anchor_before,
            self.candidate_forward,
            self.candidate_reverse,
            self.anchor_after,
        )

    @property
    def pairs(self) -> tuple[tuple[E2EObservation, E2EObservation], ...]:
        return (
            (self.anchor_before, self.candidate_forward),
            (self.anchor_after, self.candidate_reverse),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "order": ["anchor", "candidate", "candidate", "anchor"],
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass(frozen=True, slots=True)
class E2EPairedMeasurement:
    """The sole formal reward-bearing E2E measurement contract."""

    windows: tuple[E2EPairedWindow, ...]
    acceptance_policy_digest: str
    minimum_window_count: int
    estimator_id: str = ESTIMATOR_ID

    def __post_init__(self) -> None:
        if (
            self.estimator_id != ESTIMATOR_ID
            or self.minimum_window_count < 3
            or not _digest(self.acceptance_policy_digest)
        ):
            raise ContractError(
                "Paired E2E estimator policy is invalid", "invalid_e2e_paired_measurement"
            )
        if len(self.windows) < self.minimum_window_count:
            raise ContractError(
                "Paired E2E measurement has too few complete windows",
                "insufficient_e2e_paired_windows",
            )
        ids = tuple(item.window_id for item in self.windows)
        if len(set(ids)) != len(ids):
            raise ContractError(
                "Paired E2E window IDs are duplicated", "invalid_e2e_paired_measurement"
            )
        protocols = {item.anchor_before.protocol_hash for item in self.windows}
        if len(protocols) != 1:
            raise ContractError(
                "Paired E2E measurement protocols differ", "measurement_protocol_mismatch"
            )
        receipts = self.raw_measurement_receipts
        if len(set(receipts)) != len(receipts):
            raise ContractError(
                "Paired E2E measurement receipts are duplicated",
                "invalid_e2e_paired_measurement",
            )

    @property
    def protocol_hash(self) -> str:
        return self.windows[0].anchor_before.protocol_hash

    @property
    def raw_measurement_receipts(self) -> tuple[str, ...]:
        return tuple(
            observation.measurement_receipt
            for window in self.windows
            for observation in window.observations
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-paired-measurement/v1",
            "estimator_id": self.estimator_id,
            "minimum_window_count": self.minimum_window_count,
            "acceptance_policy_digest": self.acceptance_policy_digest,
            "protocol_hash": self.protocol_hash,
            "windows": [item.to_dict() for item in self.windows],
            "raw_measurement_receipts": list(self.raw_measurement_receipts),
        }


@dataclass(frozen=True, slots=True)
class E2EPairedEstimate:
    """Evaluator-recomputed ratios and promotion confidence."""

    throughput_ratio: float
    ttft_p99_ratio: float
    tpot_p99_ratio: float
    throughput_gain_pct: float
    accuracy_regression_pct: float
    ttft_p99_regression_pct: float
    tpot_p99_regression_pct: float
    throughput_confidence_lower_pct: float
    promotion_threshold_pct: float
    confidence_passed: bool
    paired_window_count: int
    paired_ratio_count: int

    def __post_init__(self) -> None:
        numbers = (
            self.throughput_ratio,
            self.ttft_p99_ratio,
            self.tpot_p99_ratio,
            self.throughput_gain_pct,
            self.accuracy_regression_pct,
            self.ttft_p99_regression_pct,
            self.tpot_p99_regression_pct,
            self.throughput_confidence_lower_pct,
            self.promotion_threshold_pct,
        )
        if (
            any(not math.isfinite(item) for item in numbers)
            or min(self.throughput_ratio, self.ttft_p99_ratio, self.tpot_p99_ratio) <= 0
            or self.paired_window_count < 3
            or self.paired_ratio_count != 2 * self.paired_window_count
        ):
            raise ContractError("Paired E2E estimate is invalid", "invalid_e2e_paired_estimate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_id": ESTIMATOR_ID,
            "ratios": {
                "total_token_throughput": self.throughput_ratio,
                "ttft_p99": self.ttft_p99_ratio,
                "tpot_p99": self.tpot_p99_ratio,
            },
            "metrics": {
                "throughput_gain_pct": self.throughput_gain_pct,
                "accuracy_regression_pct": self.accuracy_regression_pct,
                "ttft_p99_regression_pct": self.ttft_p99_regression_pct,
                "tpot_p99_regression_pct": self.tpot_p99_regression_pct,
            },
            "confidence": {
                "throughput_lower_pct": self.throughput_confidence_lower_pct,
                "promotion_threshold_pct": self.promotion_threshold_pct,
                "passed": self.confidence_passed,
            },
            "paired_window_count": self.paired_window_count,
            "paired_ratio_count": self.paired_ratio_count,
        }


@dataclass(frozen=True, slots=True)
class E2EPairedVerdict:
    """Promotion decision kept separate from the point reward."""

    keep: bool
    reason_code: str
    estimate: E2EPairedEstimate
    measurement_id: str

    @property
    def throughput_gain_pct(self) -> float:
        return self.estimate.throughput_gain_pct

    @property
    def accuracy_regression_pct(self) -> float:
        return self.estimate.accuracy_regression_pct

    @property
    def ttft_p99_regression_pct(self) -> float:
        return self.estimate.ttft_p99_regression_pct

    @property
    def tpot_p99_regression_pct(self) -> float:
        return self.estimate.tpot_p99_regression_pct

    @property
    def anchor_measurement_id(self) -> str:
        return sha256_json({"measurement": self.measurement_id, "side": "anchor"})

    @property
    def candidate_measurement_id(self) -> str:
        return sha256_json({"measurement": self.measurement_id, "side": "candidate"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep": self.keep,
            "reason_code": self.reason_code,
            "measurement_id": self.measurement_id,
            "anchor_measurement_id": self.anchor_measurement_id,
            "candidate_measurement_id": self.candidate_measurement_id,
            **self.estimate.to_dict(),
        }


def estimate_e2e_paired(
    measurement: E2EPairedMeasurement,
    policy: E2EAcceptancePolicy | None = None,
) -> E2EPairedEstimate:
    """Recompute log-ratio point estimates and a window-bootstrap lower bound."""

    chosen = policy or E2EAcceptancePolicy()
    if measurement.acceptance_policy_digest != chosen.digest:
        raise ContractError(
            "Paired E2E acceptance policy differs",
            "e2e_acceptance_policy_mismatch",
        )
    if measurement.minimum_window_count != chosen.min_paired_windows:
        raise ContractError(
            "Paired E2E minimum window policy differs",
            "e2e_acceptance_policy_mismatch",
        )
    if len(measurement.windows) < chosen.min_paired_windows:
        raise ContractError(
            "Paired E2E measurement has too few policy windows",
            "insufficient_e2e_paired_windows",
        )
    pairs = tuple(pair for window in measurement.windows for pair in window.pairs)
    if any(
        min(anchor.ttft_p99_ms, candidate.ttft_p99_ms, anchor.tpot_p99_ms, candidate.tpot_p99_ms)
        <= 0
        for anchor, candidate in pairs
    ):
        raise ContractError(
            "Paired E2E latency denominator is zero", "invalid_e2e_paired_measurement"
        )
    throughput = tuple(candidate.throughput / anchor.throughput for anchor, candidate in pairs)
    ttft = tuple(anchor.ttft_p99_ms / candidate.ttft_p99_ms for anchor, candidate in pairs)
    tpot = tuple(anchor.tpot_p99_ms / candidate.tpot_p99_ms for anchor, candidate in pairs)
    accuracy = tuple(_regression(candidate.accuracy, anchor.accuracy) for anchor, candidate in pairs)
    ratios = (_geomean(throughput), _geomean(ttft), _geomean(tpot))
    lower = _throughput_confidence_lower(measurement, chosen)
    threshold = max(0.5, chosen.min_throughput_gain_pct, chosen.aa_envelope_pct)
    return E2EPairedEstimate(
        ratios[0],
        ratios[1],
        ratios[2],
        (ratios[0] - 1.0) * 100.0,
        max(accuracy),
        (1.0 / ratios[1] - 1.0) * 100.0,
        (1.0 / ratios[2] - 1.0) * 100.0,
        lower,
        threshold,
        lower > threshold,
        len(measurement.windows),
        len(pairs),
    )


def evaluate_paired_current_anchor(
    measurement: E2EPairedMeasurement,
    policy: E2EAcceptancePolicy | None = None,
) -> E2EPairedVerdict:
    """Apply hard QoS gates and separate confidence-based KEEP policy."""

    chosen = policy or E2EAcceptancePolicy()
    estimate = estimate_e2e_paired(measurement, chosen)
    reason = "accepted"
    if _exceeds(estimate.accuracy_regression_pct, chosen.gates.accuracy_regression_pct):
        reason = "accuracy_regression"
    elif _exceeds(estimate.ttft_p99_regression_pct, chosen.gates.ttft_p99_regression_pct):
        reason = "ttft_p99_regression"
    elif _exceeds(estimate.tpot_p99_regression_pct, chosen.gates.tpot_p99_regression_pct):
        reason = "tpot_p99_regression"
    elif estimate.throughput_gain_pct < chosen.min_throughput_gain_pct:
        reason = "insufficient_throughput_gain"
    elif not estimate.confidence_passed:
        reason = "insufficient_throughput_confidence"
    return E2EPairedVerdict(reason == "accepted", reason, estimate, measurement.digest)


def load_e2e_paired_measurement(value: Mapping[str, Any]) -> E2EPairedMeasurement:
    """Parse the sole formal paired schema and reject summary-only substitutes."""

    try:
        if set(value) != {
            "schema",
            "estimator_id",
            "minimum_window_count",
            "acceptance_policy_digest",
            "protocol_hash",
            "windows",
            "raw_measurement_receipts",
        } or value.get("schema") != "apex.e2e-paired-measurement/v1":
            raise TypeError
        raw_windows = value["windows"]
        if not isinstance(raw_windows, list):
            raise TypeError
        windows = tuple(_load_window(item) for item in raw_windows)
        measurement = E2EPairedMeasurement(
            windows,
            str(value["acceptance_policy_digest"]),
            int(value["minimum_window_count"]),
            str(value["estimator_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(
            "Paired E2E measurement document is invalid",
            "invalid_e2e_paired_measurement",
        ) from error
    if measurement.to_dict() != dict(value):
        raise ContractError(
            "Paired E2E measurement document differs from raw evidence",
            "invalid_e2e_paired_measurement",
        )
    return measurement


def _throughput_confidence_lower(
    measurement: E2EPairedMeasurement, policy: E2EAcceptancePolicy
) -> float:
    window_ratios = tuple(
        _geomean(
            tuple(candidate.throughput / anchor.throughput for anchor, candidate in window.pairs)
        )
        for window in measurement.windows
    )
    generator = random.Random(policy.bootstrap_seed)
    gains: list[float] = []
    for _ in range(policy.bootstrap_repetitions):
        selected = tuple(window_ratios[generator.randrange(len(window_ratios))] for _ in window_ratios)
        gains.append((_geomean(selected) - 1.0) * 100.0)
    return _nearest_rank(gains, 1.0 - policy.bootstrap_confidence_level)


def _load_window(value: object) -> E2EPairedWindow:
    if not isinstance(value, Mapping) or set(value) != {
        "window_id",
        "order",
        "observations",
    }:
        raise TypeError
    observations = value.get("observations")
    if value.get("order") != list(("anchor", "candidate", "candidate", "anchor")):
        raise TypeError
    if not isinstance(observations, list) or len(observations) != 4:
        raise TypeError
    return E2EPairedWindow(
        str(value["window_id"]),
        *(_load_observation(item) for item in observations),
    )


def _load_observation(value: object) -> E2EObservation:
    if not isinstance(value, Mapping) or set(value) != {
        "throughput",
        "ttft_p99_ms",
        "tpot_p99_ms",
        "accuracy",
        "completed_requests",
        "protocol_hash",
        "quality_receipt",
        "measurement_receipt",
        "pass_type",
    }:
        raise TypeError
    return E2EObservation(**dict(value))


def _geomean(values: tuple[float, ...]) -> float:
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise ContractError("Paired E2E ratio is invalid", "invalid_e2e_paired_measurement")
    return math.exp(sum(math.log(item) for item in values) / len(values))


def _regression(candidate: float, anchor: float) -> float:
    if anchor == 0:
        return 0.0 if candidate == 0 else math.inf
    return (1.0 - candidate / anchor) * 100.0


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise ContractError("Bootstrap percentile is invalid", "invalid_acceptance_policy")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def _exceeds(value: float, threshold: float) -> bool:
    return value - threshold > 1e-9


def _digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ESTIMATOR_ID",
    "E2EPairedEstimate",
    "E2EPairedMeasurement",
    "E2EPairedVerdict",
    "E2EPairedWindow",
    "estimate_e2e_paired",
    "evaluate_paired_current_anchor",
    "load_e2e_paired_measurement",
]
