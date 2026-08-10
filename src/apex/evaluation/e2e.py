"""Current-anchor E2E acceptance and regression gates."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from apex.core import ContractError, sha256_json
from apex.intake import RegressionGates


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class E2EObservation:
    """Profiler-off serving measurement bound to one frozen protocol."""

    throughput: float
    ttft_p99_ms: float
    tpot_p99_ms: float
    accuracy: float
    completed_requests: int
    protocol_hash: str
    quality_receipt: str
    measurement_receipt: str
    pass_type: str = "measurement"

    def __post_init__(self) -> None:
        values = (self.throughput, self.ttft_p99_ms, self.tpot_p99_ms, self.accuracy)
        if any(not math.isfinite(value) for value in values):
            raise ContractError("E2E metrics must be finite", "invalid_e2e_measurement")
        if self.throughput <= 0 or self.ttft_p99_ms < 0 or self.tpot_p99_ms < 0:
            raise ContractError("E2E timing metrics are invalid", "invalid_e2e_measurement")
        if self.completed_requests < 1:
            raise ContractError("No completed requests in E2E measurement", "no_completed_requests")
        if self.pass_type != "measurement":
            raise ContractError("Profiler-on diagnostics cannot be scored", "diagnostic_not_rewardable")
        if not _DIGEST.fullmatch(self.protocol_hash):
            raise ContractError("Measurement protocol hash is invalid", "invalid_protocol_hash")
        if not self.quality_receipt or not self.measurement_receipt:
            raise ContractError("Measurement receipts are required", "missing_measurement_receipt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class E2EAcceptancePolicy:
    """Frozen kernel-only promotion policy."""

    gates: RegressionGates = RegressionGates()
    min_throughput_gain_pct: float = 0.5
    policy_id: str = "current_anchor_throughput_v1"
    min_paired_windows: int = 3
    bootstrap_seed: int = 20260810
    bootstrap_repetitions: int = 2000
    bootstrap_confidence_level: float = 0.95
    aa_envelope_pct: float = 0.5
    outlier_policy_id: str = "retain_all_complete_windows_v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.min_throughput_gain_pct)
            or self.min_throughput_gain_pct < -100
            or self.min_paired_windows < 3
            or self.bootstrap_seed < 0
            or self.bootstrap_repetitions < 100
            or not 0.5 <= self.bootstrap_confidence_level < 1.0
            or not math.isfinite(self.aa_envelope_pct)
            or self.aa_envelope_pct < 0
            or self.outlier_policy_id != "retain_all_complete_windows_v1"
        ):
            raise ContractError("Minimum throughput gain is invalid", "invalid_acceptance_policy")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-acceptance-policy/v1",
            "policy_id": self.policy_id,
            "min_throughput_gain_pct": self.min_throughput_gain_pct,
            "min_paired_windows": self.min_paired_windows,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "aa_envelope_pct": self.aa_envelope_pct,
            "outlier_policy_id": self.outlier_policy_id,
            "gates": asdict(self.gates),
        }


@dataclass(frozen=True, slots=True)
class E2EVerdict:
    keep: bool
    reason_code: str
    throughput_gain_pct: float
    accuracy_regression_pct: float
    ttft_p99_regression_pct: float
    tpot_p99_regression_pct: float
    anchor_measurement_id: str
    candidate_measurement_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_no_regression(
    baseline: E2EObservation,
    replay: E2EObservation,
    policy: E2EAcceptancePolicy | None = None,
    *,
    throughput_noise_pct: float = 1.0,
) -> E2EVerdict:
    """Verify an unchanged/no-winner replay without pretending it is a KEEP."""

    if throughput_noise_pct < 0 or not math.isfinite(throughput_noise_pct):
        raise ContractError("Throughput noise bound is invalid", "invalid_acceptance_policy")
    chosen = policy or E2EAcceptancePolicy()
    verdict = evaluate_current_anchor(
        baseline,
        replay,
        E2EAcceptancePolicy(chosen.gates, -throughput_noise_pct, "no_regression_replay_v1"),
    )
    if verdict.keep:
        return E2EVerdict(
            keep=True,
            reason_code="no_regression",
            throughput_gain_pct=verdict.throughput_gain_pct,
            accuracy_regression_pct=verdict.accuracy_regression_pct,
            ttft_p99_regression_pct=verdict.ttft_p99_regression_pct,
            tpot_p99_regression_pct=verdict.tpot_p99_regression_pct,
            anchor_measurement_id=verdict.anchor_measurement_id,
            candidate_measurement_id=verdict.candidate_measurement_id,
        )
    return verdict


def evaluate_current_anchor(
    anchor: E2EObservation,
    candidate: E2EObservation,
    policy: E2EAcceptancePolicy | None = None,
) -> E2EVerdict:
    """KEEP only a source candidate that improves the current live anchor."""

    chosen = policy or E2EAcceptancePolicy()
    if anchor.protocol_hash != candidate.protocol_hash:
        raise ContractError("E2E measurement protocols differ", "measurement_protocol_mismatch")
    throughput_gain = _change_pct(candidate.throughput, anchor.throughput)
    accuracy_regression = -_change_pct(candidate.accuracy, anchor.accuracy)
    ttft_regression = _change_pct(candidate.ttft_p99_ms, anchor.ttft_p99_ms)
    tpot_regression = _change_pct(candidate.tpot_p99_ms, anchor.tpot_p99_ms)
    reason = "accepted"
    if accuracy_regression > chosen.gates.accuracy_regression_pct:
        reason = "accuracy_regression"
    elif ttft_regression > chosen.gates.ttft_p99_regression_pct:
        reason = "ttft_p99_regression"
    elif tpot_regression > chosen.gates.tpot_p99_regression_pct:
        reason = "tpot_p99_regression"
    elif throughput_gain < chosen.min_throughput_gain_pct:
        reason = "insufficient_throughput_gain"
    return E2EVerdict(
        keep=reason == "accepted",
        reason_code=reason,
        throughput_gain_pct=throughput_gain,
        accuracy_regression_pct=accuracy_regression,
        ttft_p99_regression_pct=ttft_regression,
        tpot_p99_regression_pct=tpot_regression,
        anchor_measurement_id=anchor.digest,
        candidate_measurement_id=candidate.digest,
    )


def validate_baseline_measurement(measurement: E2EObservation) -> None:
    """Explicit semantic hook used before any agent or diagnostic work."""

    if not measurement.quality_receipt:
        raise ContractError("Baseline quality evidence is missing", "baseline_quality_missing")


def _change_pct(value: float, baseline: float) -> float:
    if baseline == 0:
        if value == 0:
            return 0.0
        return math.inf if value > 0 else -math.inf
    return (value / baseline - 1.0) * 100.0


__all__ = [
    "E2EAcceptancePolicy",
    "E2EObservation",
    "E2EVerdict",
    "evaluate_current_anchor",
    "evaluate_no_regression",
    "validate_baseline_measurement",
]
