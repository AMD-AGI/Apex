"""Evaluator-owned workload reward for one E2E kernel-candidate attempt."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ContractError, sha256_json

from .e2e_paired import E2EPairedVerdict


_VERDICTS = frozenset({"keep", "revert", "reject", "needs_more_measurement"})
_HARD_GATE_REASONS = frozenset(
    {
        "accuracy_regression",
        "quality_gate_failed",
        "ttft_p99_regression",
        "tpot_p99_regression",
    }
)
_NO_SOURCE_REASON = "agent_made_no_source_change"


@dataclass(frozen=True, slots=True)
class E2ERewardPolicy:
    """Frozen `e2e_throughput_qos_v1` scalarization."""

    policy_id: str = "e2e_throughput_qos_v1"
    runtime_reward: float = 20.0
    eligible_reward: float = 100.0
    gain_scale: float = 200.0
    throughput_weight: float = 0.80
    ttft_p99_weight: float = 0.10
    tpot_p99_weight: float = 0.10
    clip_lower: float = -0.25
    clip_upper: float = 1.00

    def __post_init__(self) -> None:
        values = (
            self.runtime_reward,
            self.eligible_reward,
            self.gain_scale,
            self.throughput_weight,
            self.ttft_p99_weight,
            self.tpot_p99_weight,
            self.clip_lower,
            self.clip_upper,
        )
        if self.policy_id != "e2e_throughput_qos_v1" or any(
            not math.isfinite(item) for item in values
        ):
            raise ContractError("E2E reward policy is invalid", "invalid_e2e_reward_policy")
        if (
            min(self.runtime_reward, self.eligible_reward, self.gain_scale) < 0
            or min(
                self.throughput_weight,
                self.ttft_p99_weight,
                self.tpot_p99_weight,
            ) < 0
            or not math.isclose(
                self.throughput_weight + self.ttft_p99_weight + self.tpot_p99_weight,
                1.0,
            )
            or self.clip_lower >= self.clip_upper
        ):
            raise ContractError("E2E reward policy is invalid", "invalid_e2e_reward_policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-reward-policy/v1",
            "policy_id": self.policy_id,
            "runtime_reward": self.runtime_reward,
            "eligible_reward": self.eligible_reward,
            "gain_scale": self.gain_scale,
            "weights": {
                "total_token_throughput": self.throughput_weight,
                "ttft_p99": self.ttft_p99_weight,
                "tpot_p99": self.tpot_p99_weight,
            },
            "ratio_gain_clip": [self.clip_lower, self.clip_upper],
            "reward_bounds": [0.0, 320.0],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class E2ERewardVector:
    """Replayable point reward plus the evaluator facts that produced it."""

    scope: str
    policy_id: str
    policy_digest: str
    verdict: str
    outcome_class: str
    reason_code: str
    candidate_present: bool
    runtime_verified: bool
    eligible: bool
    safety_certified: bool
    performance_skipped: str | None
    throughput_gain_pct: float | None
    accuracy_regression_pct: float | None
    ttft_p99_regression_pct: float | None
    tpot_p99_regression_pct: float | None
    throughput_ratio: float | None
    ttft_p99_ratio: float | None
    tpot_p99_ratio: float | None
    anchor_measurement_id: str | None
    candidate_measurement_id: str | None
    runtime_component: float
    eligible_base_component: float
    throughput_component: float
    ttft_p99_component: float
    tpot_p99_component: float
    ge2e: float | None
    scalar_reward: float

    def __post_init__(self) -> None:
        if (
            self.scope not in {"attempt", "task_terminal"}
            or
            self.policy_id != "e2e_throughput_qos_v1"
            or not _digest(self.policy_digest)
            or self.verdict not in _VERDICTS
            or not self.reason_code
            or not self.outcome_class
            or not all(
                isinstance(item, bool)
                for item in (
                    self.candidate_present,
                    self.runtime_verified,
                    self.eligible,
                    self.safety_certified,
                )
            )
            or self.performance_skipped not in {None, "quality_gate"}
        ):
            raise ContractError("E2E reward vector is invalid", "invalid_e2e_reward_vector")
        if any(not math.isfinite(item) for item in self._numbers()):
            raise ContractError("E2E reward vector is invalid", "invalid_e2e_reward_vector")
        if not 0.0 <= self.scalar_reward <= 320.0:
            raise ContractError("E2E reward vector is invalid", "invalid_e2e_reward_vector")

    def _numbers(self) -> tuple[float, ...]:
        optional = (
            self.throughput_gain_pct,
            self.accuracy_regression_pct,
            self.ttft_p99_regression_pct,
            self.tpot_p99_regression_pct,
            self.throughput_ratio,
            self.ttft_p99_ratio,
            self.tpot_p99_ratio,
            self.ge2e,
        )
        return (
            self.runtime_component,
            self.eligible_base_component,
            self.throughput_component,
            self.ttft_p99_component,
            self.tpot_p99_component,
            self.scalar_reward,
            *(item for item in optional if item is not None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-reward-vector/v1",
            "task_kind": "e2e_kernel_only",
            "scope": self.scope,
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "verdict": self.verdict,
            "outcome_class": self.outcome_class,
            "reason_code": self.reason_code,
            "candidate_present": self.candidate_present,
            "runtime_verified": self.runtime_verified,
            "eligible": self.eligible,
            "safety_certified": self.safety_certified,
            "performance_skipped": self.performance_skipped,
            "metrics": {
                "throughput_gain_pct": self.throughput_gain_pct,
                "accuracy_regression_pct": self.accuracy_regression_pct,
                "ttft_p99_regression_pct": self.ttft_p99_regression_pct,
                "tpot_p99_regression_pct": self.tpot_p99_regression_pct,
                "anchor_measurement_id": self.anchor_measurement_id,
                "candidate_measurement_id": self.candidate_measurement_id,
            },
            "ratios": {
                "total_token_throughput": self.throughput_ratio,
                "ttft_p99": self.ttft_p99_ratio,
                "tpot_p99": self.tpot_p99_ratio,
            },
            "components": {
                "runtime": self.runtime_component,
                "eligible_base": self.eligible_base_component,
                "total_token_throughput": self.throughput_component,
                "ttft_p99": self.ttft_p99_component,
                "tpot_p99": self.tpot_p99_component,
                "ge2e": self.ge2e,
            },
            "scalar_reward": self.scalar_reward,
        }


def grade_e2e_outcome(
    *,
    verdict: str,
    reason_code: str,
    candidate_present: bool,
    measurement_verdict: E2EPairedVerdict | None = None,
    safety_certified: bool = False,
    performance_skipped: str | None = None,
    scope: str = "attempt",
    policy: E2ERewardPolicy | None = None,
) -> E2ERewardVector:
    """Grade one attributable attempt; missing measurement evidence is rejected."""

    chosen = policy or E2ERewardPolicy()
    _validate_grade_inputs(
        verdict,
        reason_code,
        candidate_present,
        measurement_verdict,
        performance_skipped,
        scope,
    )
    runtime_verified = measurement_verdict is not None or performance_skipped is not None
    eligible = measurement_verdict is not None and reason_code not in _HARD_GATE_REASONS
    ratios = _measurement_ratios(measurement_verdict)
    components = _reward_components(chosen, runtime_verified, eligible, ratios)
    measured = measurement_verdict
    return E2ERewardVector(
        scope=scope,
        policy_id=chosen.policy_id,
        policy_digest=chosen.digest,
        verdict=verdict,
        outcome_class=_outcome_class(verdict, reason_code, candidate_present),
        reason_code=reason_code,
        candidate_present=candidate_present,
        runtime_verified=runtime_verified,
        eligible=eligible,
        safety_certified=safety_certified,
        performance_skipped=performance_skipped,
        throughput_gain_pct=measured.throughput_gain_pct if measured else None,
        accuracy_regression_pct=measured.accuracy_regression_pct if measured else None,
        ttft_p99_regression_pct=measured.ttft_p99_regression_pct if measured else None,
        tpot_p99_regression_pct=measured.tpot_p99_regression_pct if measured else None,
        throughput_ratio=ratios[0],
        ttft_p99_ratio=ratios[1],
        tpot_p99_ratio=ratios[2],
        anchor_measurement_id=measured.anchor_measurement_id if measured else None,
        candidate_measurement_id=measured.candidate_measurement_id if measured else None,
        runtime_component=components[0],
        eligible_base_component=components[1],
        throughput_component=components[2],
        ttft_p99_component=components[3],
        tpot_p99_component=components[4],
        ge2e=components[5],
        scalar_reward=components[6],
    )


def replay_e2e_reward(
    vector: Mapping[str, Any],
    *,
    policy: E2ERewardPolicy | None = None,
) -> float:
    """Recompute a stored E2E reward without trusting its scalar field."""

    chosen = policy or E2ERewardPolicy()
    if (
        vector.get("schema") != "apex.e2e-reward-vector/v1"
        or vector.get("task_kind") != "e2e_kernel_only"
        or vector.get("scope") not in {"attempt", "task_terminal"}
        or vector.get("policy_id") != chosen.policy_id
        or vector.get("policy_digest") != chosen.digest
    ):
        raise ContractError("E2E reward policy differs", "e2e_reward_policy_mismatch")
    _validate_stored_semantics(vector)
    ratios = _stored_ratios(vector)
    runtime_verified = vector.get("runtime_verified") is True
    eligible = vector.get("eligible") is True
    expected = _reward_components(chosen, runtime_verified, eligible, ratios)
    components = _mapping(vector.get("components"), "components")
    if set(components) != {
        "runtime",
        "eligible_base",
        "total_token_throughput",
        "ttft_p99",
        "tpot_p99",
        "ge2e",
    }:
        raise ContractError("E2E reward components differ", "e2e_reward_replay_mismatch")
    observed = (
        _finite(components.get("runtime")),
        _finite(components.get("eligible_base")),
        _finite(components.get("total_token_throughput")),
        _finite(components.get("ttft_p99")),
        _finite(components.get("tpot_p99")),
        _optional_finite(components.get("ge2e")),
    )
    if observed != expected[:6]:
        raise ContractError("E2E reward components differ", "e2e_reward_replay_mismatch")
    if vector.get("outcome_class") != _outcome_class(
        str(vector.get("verdict", "")),
        str(vector.get("reason_code", "")),
        vector.get("candidate_present") is True,
    ):
        raise ContractError("E2E outcome class differs", "e2e_reward_replay_mismatch")
    return expected[6]


def _validate_grade_inputs(
    verdict: str,
    reason_code: str,
    candidate_present: bool,
    measurement_verdict: E2EPairedVerdict | None,
    performance_skipped: str | None,
    scope: str,
) -> None:
    if (
        verdict not in _VERDICTS
        or not reason_code
        or not isinstance(candidate_present, bool)
        or scope not in {"attempt", "task_terminal"}
    ):
        raise ContractError("E2E reward outcome is invalid", "invalid_e2e_reward_outcome")
    quality_skip = performance_skipped == "quality_gate"
    if performance_skipped not in {None, "quality_gate"}:
        raise ContractError(
            "E2E performance skip reason is invalid",
            "invalid_e2e_reward_outcome",
        )
    if quality_skip and (
        verdict != "revert"
        or reason_code != "quality_gate_failed"
        or not candidate_present
        or measurement_verdict is not None
    ):
        raise ContractError(
            "Quality-gate reward semantics are invalid",
            "invalid_e2e_reward_outcome",
        )
    if verdict in {"keep", "revert"} and measurement_verdict is None and not quality_skip:
        raise ContractError(
            "Measured E2E outcomes require an evaluator verdict",
            "missing_e2e_reward_measurement",
        )
    if verdict not in {"keep", "revert"} and measurement_verdict is not None:
        raise ContractError(
            "Rejected or inconclusive outcomes cannot carry a measured verdict",
            "unexpected_e2e_reward_measurement",
        )
    if verdict in {"keep", "revert"} and not candidate_present and scope == "attempt":
        raise ContractError(
            "Measured E2E outcomes require a source candidate",
            "missing_e2e_reward_candidate",
        )
    if measurement_verdict is not None and (
        measurement_verdict.keep != (verdict == "keep")
        or measurement_verdict.reason_code != reason_code
    ):
        raise ContractError(
            "E2E decision and measurement verdict differ",
            "e2e_reward_verdict_mismatch",
        )


def _measurement_ratios(
    measured: E2EPairedVerdict | None,
) -> tuple[float | None, float | None, float | None]:
    if measured is None:
        return None, None, None
    throughput = measured.estimate.throughput_ratio
    ttft = measured.estimate.ttft_p99_ratio
    tpot = measured.estimate.tpot_p99_ratio
    if (
        min(throughput, ttft, tpot) <= 0
        or any(
            not math.isfinite(item)
            for item in (throughput, ttft, tpot)
        )
    ):
        raise ContractError("E2E reward ratios are invalid", "invalid_e2e_reward_measurement")
    return throughput, ttft, tpot


def _reward_components(
    policy: E2ERewardPolicy,
    runtime_verified: bool,
    eligible: bool,
    ratios: tuple[float | None, float | None, float | None],
) -> tuple[float, float, float, float, float, float | None, float]:
    runtime = policy.runtime_reward if runtime_verified else 0.0
    if not eligible:
        return runtime, 0.0, 0.0, 0.0, 0.0, None, runtime
    if any(item is None for item in ratios):
        raise ContractError("Eligible E2E reward lacks ratios", "e2e_reward_replay_mismatch")
    throughput, ttft, tpot = (float(item) for item in ratios)
    gains = tuple(_clip(item - 1.0, policy.clip_lower, policy.clip_upper) for item in (throughput, ttft, tpot))
    weighted = (
        policy.throughput_weight * gains[0],
        policy.ttft_p99_weight * gains[1],
        policy.tpot_p99_weight * gains[2],
    )
    shaped = tuple(policy.gain_scale * item for item in weighted)
    ge2e = sum(weighted)
    scalar = runtime + policy.eligible_reward + sum(shaped)
    return runtime, policy.eligible_reward, *shaped, ge2e, scalar


def _stored_ratios(
    vector: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None]:
    ratios = _mapping(vector.get("ratios"), "ratios")
    if set(ratios) != {"total_token_throughput", "ttft_p99", "tpot_p99"}:
        raise ContractError("E2E reward ratios differ", "e2e_reward_replay_mismatch")
    stored = (
        _optional_finite(ratios.get("total_token_throughput")),
        _optional_finite(ratios.get("ttft_p99")),
        _optional_finite(ratios.get("tpot_p99")),
    )
    metrics = _mapping(vector.get("metrics"), "metrics")
    expected_keys = {
        "throughput_gain_pct",
        "accuracy_regression_pct",
        "ttft_p99_regression_pct",
        "tpot_p99_regression_pct",
        "anchor_measurement_id",
        "candidate_measurement_id",
    }
    if set(metrics) != expected_keys:
        raise ContractError("E2E reward metrics differ", "e2e_reward_replay_mismatch")
    skipped = vector.get("performance_skipped")
    if vector.get("runtime_verified") is True and skipped is None:
        expected = _ratios_from_metrics(metrics)
        if stored != expected:
            raise ContractError("E2E reward ratios differ", "e2e_reward_replay_mismatch")
    elif stored != (None, None, None) or any(value is not None for value in metrics.values()):
        raise ContractError("E2E reward metrics differ", "e2e_reward_replay_mismatch")
    return stored


def _validate_stored_semantics(vector: Mapping[str, Any]) -> None:
    verdict = str(vector.get("verdict", ""))
    reason = str(vector.get("reason_code", ""))
    candidate_present = vector.get("candidate_present")
    runtime = vector.get("runtime_verified")
    eligible = vector.get("eligible")
    safety_certified = vector.get("safety_certified")
    scope = vector.get("scope")
    performance_skipped = vector.get("performance_skipped")
    expected_runtime = verdict in {"keep", "revert"}
    expected_eligible = (
        expected_runtime
        and performance_skipped is None
        and reason not in _HARD_GATE_REASONS
    )
    valid_skip = performance_skipped is None or (
        performance_skipped == "quality_gate"
        and verdict == "revert"
        and reason == "quality_gate_failed"
        and candidate_present is True
    )
    if (
        verdict not in _VERDICTS
        or not reason
        or not isinstance(candidate_present, bool)
        or not isinstance(runtime, bool)
        or not isinstance(eligible, bool)
        or not isinstance(safety_certified, bool)
        or not valid_skip
        or runtime != expected_runtime
        or eligible != expected_eligible
        or (expected_runtime and not candidate_present and scope == "attempt")
    ):
        raise ContractError("E2E reward semantics differ", "e2e_reward_replay_mismatch")


def _ratios_from_metrics(
    metrics: Mapping[str, Any],
) -> tuple[float, float, float]:
    throughput = 1.0 + _finite(metrics.get("throughput_gain_pct")) / 100.0
    ttft_denominator = 1.0 + _finite(metrics.get("ttft_p99_regression_pct")) / 100.0
    tpot_denominator = 1.0 + _finite(metrics.get("tpot_p99_regression_pct")) / 100.0
    if min(throughput, ttft_denominator, tpot_denominator) <= 0:
        raise ContractError("E2E reward ratios differ", "e2e_reward_replay_mismatch")
    return throughput, 1.0 / ttft_denominator, 1.0 / tpot_denominator


def _outcome_class(verdict: str, reason: str, candidate_present: bool) -> str:
    if verdict == "keep":
        return "accepted"
    if verdict == "revert":
        return "hard_gate_regression" if reason in _HARD_GATE_REASONS else "no_gain"
    if verdict == "reject":
        return "no_source" if reason == _NO_SOURCE_REASON and not candidate_present else "candidate_rejected"
    if verdict == "needs_more_measurement":
        return "inconclusive"
    raise ContractError("E2E reward outcome is invalid", "invalid_e2e_reward_outcome")


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"E2E reward {name} is invalid", "e2e_reward_replay_mismatch")
    return value


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError("E2E reward number is invalid", "e2e_reward_replay_mismatch") from error
    if not math.isfinite(result):
        raise ContractError("E2E reward number is invalid", "e2e_reward_replay_mismatch")
    return result


def _optional_finite(value: object) -> float | None:
    return None if value is None else _finite(value)


def _digest(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


__all__ = [
    "E2ERewardPolicy",
    "E2ERewardVector",
    "grade_e2e_outcome",
    "replay_e2e_reward",
]
