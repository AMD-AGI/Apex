"""Evaluator-owned reward for one E2E kernel-candidate attempt."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from apex.core import ContractError, sha256_json

from .e2e import E2EVerdict


_VERDICTS = frozenset({"keep", "revert", "reject", "needs_more_measurement"})
_HARD_GATE_REASONS = frozenset(
    {"accuracy_regression", "ttft_p99_regression", "tpot_p99_regression"}
)
_NO_SOURCE_REASON = "agent_made_no_source_change"
_COMPARISON_SELECTION_POLICY_ID = "conservative_e2e_reward_v1"


@dataclass(frozen=True, slots=True)
class E2ERewardPolicy:
    """Frozen scalarization for evaluator-owned E2E outcomes."""

    policy_id: str = "e2e_kernel_candidate_v1"
    keep_base: float = 100.0
    revert_base: float = -10.0
    hard_gate_base: float = -100.0
    reject_reward: float = -100.0
    no_source_reward: float = -20.0
    inconclusive_reward: float = 0.0
    throughput_scale: float = 10.0
    throughput_clip_pct: float = 10.0
    reward_min: float = -200.0
    reward_max: float = 200.0

    def __post_init__(self) -> None:
        values = (
            self.keep_base,
            self.revert_base,
            self.hard_gate_base,
            self.reject_reward,
            self.no_source_reward,
            self.inconclusive_reward,
            self.throughput_scale,
            self.throughput_clip_pct,
            self.reward_min,
            self.reward_max,
        )
        if self.policy_id != "e2e_kernel_candidate_v1" or any(
            not math.isfinite(item) for item in values
        ):
            raise ContractError("E2E reward policy is invalid", "invalid_e2e_reward_policy")
        if (
            self.throughput_scale <= 0
            or self.throughput_clip_pct <= 0
            or self.reward_min >= self.reward_max
        ):
            raise ContractError("E2E reward policy is invalid", "invalid_e2e_reward_policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-reward-policy/v1",
            "policy_id": self.policy_id,
            "keep_base": self.keep_base,
            "revert_base": self.revert_base,
            "hard_gate_base": self.hard_gate_base,
            "reject_reward": self.reject_reward,
            "no_source_reward": self.no_source_reward,
            "inconclusive_reward": self.inconclusive_reward,
            "throughput_scale": self.throughput_scale,
            "throughput_clip_pct": self.throughput_clip_pct,
            "reward_bounds": [self.reward_min, self.reward_max],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class E2ERewardGrade:
    """Replayable scalar plus the measured or verified facts that produced it."""

    policy_id: str
    policy_digest: str
    verdict: str
    outcome_class: str
    reason_code: str
    candidate_present: bool
    throughput_gain_pct: float | None
    accuracy_regression_pct: float | None
    ttft_p99_regression_pct: float | None
    tpot_p99_regression_pct: float | None
    anchor_measurement_id: str | None
    candidate_measurement_id: str | None
    outcome_base: float
    throughput_component: float
    scalar_reward: float

    def __post_init__(self) -> None:
        if (
            self.policy_id != "e2e_kernel_candidate_v1"
            or len(self.policy_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in self.policy_digest)
            or self.verdict not in _VERDICTS
            or not self.reason_code
            or not self.outcome_class
            or not isinstance(self.candidate_present, bool)
        ):
            raise ContractError("E2E reward grade is invalid", "invalid_e2e_reward_grade")
        numeric = (
            self.outcome_base,
            self.throughput_component,
            self.scalar_reward,
            *(
                item
                for item in (
                    self.throughput_gain_pct,
                    self.accuracy_regression_pct,
                    self.ttft_p99_regression_pct,
                    self.tpot_p99_regression_pct,
                )
                if item is not None
            ),
        )
        if any(not math.isfinite(item) for item in numeric):
            raise ContractError("E2E reward grade is invalid", "invalid_e2e_reward_grade")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-reward-grade/v1",
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "verdict": self.verdict,
            "outcome_class": self.outcome_class,
            "reason_code": self.reason_code,
            "candidate_present": self.candidate_present,
            "metrics": {
                "throughput_gain_pct": self.throughput_gain_pct,
                "accuracy_regression_pct": self.accuracy_regression_pct,
                "ttft_p99_regression_pct": self.ttft_p99_regression_pct,
                "tpot_p99_regression_pct": self.tpot_p99_regression_pct,
                "anchor_measurement_id": self.anchor_measurement_id,
                "candidate_measurement_id": self.candidate_measurement_id,
            },
            "components": {
                "outcome_base": self.outcome_base,
                "throughput": self.throughput_component,
            },
            "scalar_reward": self.scalar_reward,
        }


def grade_e2e_outcome(
    *,
    verdict: str,
    reason_code: str,
    candidate_present: bool,
    measurement_verdict: E2EVerdict | None = None,
    policy: E2ERewardPolicy | None = None,
) -> E2ERewardGrade:
    """Grade one attributable attempt; infrastructure failures are not inputs."""

    chosen = policy or E2ERewardPolicy()
    _validate_grade_inputs(
        verdict,
        reason_code,
        candidate_present,
        measurement_verdict,
    )
    throughput = (
        measurement_verdict.throughput_gain_pct
        if measurement_verdict is not None
        else None
    )
    shaped = 0.0 if throughput is None else chosen.throughput_scale * _clip(
        throughput,
        -chosen.throughput_clip_pct,
        chosen.throughput_clip_pct,
    )
    outcome_class, base, component = _components(
        verdict,
        reason_code,
        candidate_present,
        shaped,
        chosen,
    )
    scalar = _clip(base + component, chosen.reward_min, chosen.reward_max)
    measured = measurement_verdict
    return E2ERewardGrade(
        policy_id=chosen.policy_id,
        policy_digest=chosen.digest,
        verdict=verdict,
        outcome_class=outcome_class,
        reason_code=reason_code,
        candidate_present=candidate_present,
        throughput_gain_pct=throughput,
        accuracy_regression_pct=(
            measured.accuracy_regression_pct if measured is not None else None
        ),
        ttft_p99_regression_pct=(
            measured.ttft_p99_regression_pct if measured is not None else None
        ),
        tpot_p99_regression_pct=(
            measured.tpot_p99_regression_pct if measured is not None else None
        ),
        anchor_measurement_id=(
            measured.anchor_measurement_id if measured is not None else None
        ),
        candidate_measurement_id=(
            measured.candidate_measurement_id if measured is not None else None
        ),
        outcome_base=base,
        throughput_component=component,
        scalar_reward=scalar,
    )


def e2e_comparison_selection_policy(
    policy: E2ERewardPolicy | None = None,
) -> dict[str, Any]:
    """Describe the frozen conservative ordering for matched E2E comparisons."""

    chosen = policy or E2ERewardPolicy()
    return {
        "schema": "apex.e2e-comparison-selection-policy/v1",
        "policy_id": _COMPARISON_SELECTION_POLICY_ID,
        "reward_policy_id": chosen.policy_id,
        "reward_policy_digest": chosen.digest,
        "ordering": [
            "failure_before_keep",
            "scalar_reward_ascending",
            "throughput_gain_pct_ascending",
            "accuracy_regression_pct_descending",
            "ttft_p99_regression_pct_descending",
            "tpot_p99_regression_pct_descending",
            "measurement_ids_ascending",
        ],
    }


def select_conservative_e2e_verdict(
    comparisons: Sequence[E2EVerdict],
    policy: E2ERewardPolicy | None = None,
) -> int:
    """Select one replayable worst comparison without depending on tuple order."""

    if not comparisons or any(not isinstance(item, E2EVerdict) for item in comparisons):
        raise ContractError(
            "E2E comparison set is invalid",
            "invalid_e2e_comparison_set",
        )
    chosen = policy or E2ERewardPolicy()
    return min(
        range(len(comparisons)),
        key=lambda index: _comparison_selection_key(comparisons[index], chosen),
    )


def _comparison_selection_key(
    verdict: E2EVerdict,
    policy: E2ERewardPolicy,
) -> tuple[object, ...]:
    decision = "keep" if verdict.keep else "revert"
    grade = grade_e2e_outcome(
        verdict=decision,
        reason_code=verdict.reason_code,
        candidate_present=True,
        measurement_verdict=verdict,
        policy=policy,
    )
    return (
        1 if verdict.keep else 0,
        grade.scalar_reward,
        verdict.throughput_gain_pct,
        -verdict.accuracy_regression_pct,
        -verdict.ttft_p99_regression_pct,
        -verdict.tpot_p99_regression_pct,
        verdict.anchor_measurement_id,
        verdict.candidate_measurement_id,
    )


def _validate_grade_inputs(
    verdict: str,
    reason_code: str,
    candidate_present: bool,
    measurement_verdict: E2EVerdict | None,
) -> None:
    if verdict not in _VERDICTS or not reason_code:
        raise ContractError("E2E reward outcome is invalid", "invalid_e2e_reward_outcome")
    if not isinstance(candidate_present, bool):
        raise ContractError("E2E reward outcome is invalid", "invalid_e2e_reward_outcome")
    if verdict in {"keep", "revert"} and measurement_verdict is None:
        raise ContractError(
            "Measured E2E outcomes require an evaluator verdict",
            "missing_e2e_reward_measurement",
        )
    if verdict not in {"keep", "revert"} and measurement_verdict is not None:
        raise ContractError(
            "Rejected or inconclusive outcomes cannot carry a measured verdict",
            "unexpected_e2e_reward_measurement",
        )
    if verdict in {"keep", "revert"} and not candidate_present:
        raise ContractError(
            "Measured E2E outcomes require a source candidate",
            "missing_e2e_reward_candidate",
        )
    if measurement_verdict is not None:
        expected_keep = verdict == "keep"
        if measurement_verdict.keep != expected_keep or (
            measurement_verdict.reason_code != reason_code
        ):
            raise ContractError(
                "E2E decision and measurement verdict differ",
                "e2e_reward_verdict_mismatch",
            )


def replay_e2e_reward(
    vector: Mapping[str, Any],
    *,
    policy: E2ERewardPolicy | None = None,
) -> float:
    """Recompute a stored E2E reward without trusting its scalar field."""

    chosen = policy or E2ERewardPolicy()
    if vector.get("policy_id") != chosen.policy_id or vector.get(
        "policy_digest"
    ) != chosen.digest:
        raise ContractError("E2E reward policy differs", "e2e_reward_policy_mismatch")
    metrics = _mapping(vector.get("metrics"), "metrics")
    throughput_value = metrics.get("throughput_gain_pct")
    throughput = None if throughput_value is None else _finite(throughput_value)
    shaped = 0.0 if throughput is None else chosen.throughput_scale * _clip(
        throughput,
        -chosen.throughput_clip_pct,
        chosen.throughput_clip_pct,
    )
    outcome_class, base, component = _components(
        str(vector.get("verdict", "")),
        str(vector.get("reason_code", "")),
        vector.get("candidate_present") is True,
        shaped,
        chosen,
    )
    if vector.get("outcome_class") != outcome_class:
        raise ContractError("E2E outcome class differs", "e2e_reward_replay_mismatch")
    components = _mapping(vector.get("components"), "components")
    if (
        abs(_finite(components.get("outcome_base")) - base) > 1e-9
        or abs(_finite(components.get("throughput")) - component) > 1e-9
    ):
        raise ContractError("E2E reward components differ", "e2e_reward_replay_mismatch")
    return _clip(base + component, chosen.reward_min, chosen.reward_max)


def _components(
    verdict: str,
    reason_code: str,
    candidate_present: bool,
    shaped: float,
    policy: E2ERewardPolicy,
) -> tuple[str, float, float]:
    if verdict == "keep":
        return "accepted", policy.keep_base, max(shaped, 0.0)
    if verdict == "revert":
        if reason_code in _HARD_GATE_REASONS:
            return "hard_gate_regression", policy.hard_gate_base, min(shaped, 0.0)
        return "no_gain", policy.revert_base, shaped
    if verdict == "reject":
        if reason_code == _NO_SOURCE_REASON and not candidate_present:
            return "no_source", policy.no_source_reward, 0.0
        return "candidate_rejected", policy.reject_reward, 0.0
    if verdict == "needs_more_measurement":
        return "inconclusive", policy.inconclusive_reward, 0.0
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


__all__ = [
    "E2ERewardGrade",
    "E2ERewardPolicy",
    "e2e_comparison_selection_policy",
    "grade_e2e_outcome",
    "replay_e2e_reward",
    "select_conservative_e2e_verdict",
]
