"""Pure task-terminal grading for one formal standalone kernel campaign."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ContractError, sha256_json, validate_identifier

from .kernel import GateVerdict, KernelGrade, kernel_reward
from .statistics import MeasurementPolicy


KERNEL_REWARD_POLICY_ID = "kernel_robust_v1"


def kernel_terminal_policy_source() -> dict[str, object]:
    return {
        "schema": "apex.kernel-terminal-reward-policy/v1",
        "policy_id": KERNEL_REWARD_POLICY_ID,
        "formula": (
            "20*Icompile + Icorrect*(100 + "
            "200*clip(min(S50,S99)-1,-0.25,1.00))"
        ),
        "terminal_semantics": {
            "selected_candidate": "selected grade relative to frozen reference",
            "measured_noop": "Srobust=1 and reward=120",
            "compile_failure": "reward=0",
            "correctness_failure": "reward=20",
            "invalid_environment_or_measurement": "reward=null",
            "attempt_rewards_aggregated": False,
        },
    }


def kernel_reward_policy_source(policy: MeasurementPolicy) -> dict[str, object]:
    """Canonical measured-attempt policy bound to raw invocation evidence."""

    return {
        "schema": "apex.kernel-reward-policy/v1",
        "measurement_schema": "apex.kernel-measurement/v1",
        "policy_id": KERNEL_REWARD_POLICY_ID,
        "measurement_policy": policy.to_dict(),
        "timing_protocol": {
            "execution_receipt": "apex.kernel-measurement-execution/v1",
            "writer": "trusted_evaluator_adapter",
            "phase": "measurement",
            "protected_harness_digest_required": True,
            "ordering": "seeded_paired_abba_blocks",
            "inner_repeats": 1,
            "timer_resolution_required": True,
            "measurement_method_sha256_required": True,
            "gpu_health_before_after_each_block_required": True,
        },
        "formula": (
            "20*Icompile + Icorrect*(100 + "
            "200*clip(min(S50,S99)-1,-0.25,1.00))"
        ),
        "promotion": {
            "point_threshold": "Srobust > keep_srobust_threshold",
            "confidence": "Srobust_CI_lower > confidence_srobust_floor",
            "noise": "max_case_cv <= max_cv",
            "worst_case": "worst_case_srobust >= worst_case_srobust_floor",
        },
    }


@dataclass(frozen=True, slots=True)
class KernelTerminalGrade:
    outcome: str
    reason_code: str
    source_attempt_id: str | None
    gates: GateVerdict | None
    s50: float | None
    s99: float | None
    srobust: float | None
    scalar_reward: float | None
    trainability: str
    untrainable_reason: str | None
    policy_id: str = KERNEL_REWARD_POLICY_ID

    def __post_init__(self) -> None:
        if self.policy_id != KERNEL_REWARD_POLICY_ID or self.outcome not in {
            "selected_candidate",
            "measured_noop",
            "compile_failure",
            "correctness_failure",
            "untrainable",
        } or not self.reason_code:
            raise ContractError("Kernel terminal outcome is invalid", "invalid_kernel_terminal_grade")
        if self.trainability not in {"trainable", "untrainable"}:
            raise ContractError("Kernel terminal trainability is invalid", "invalid_kernel_terminal_grade")
        if self.source_attempt_id is not None:
            validate_identifier(self.source_attempt_id, field_name="source_attempt_id")
        points = tuple(item for item in (self.s50, self.s99, self.srobust) if item is not None)
        if any(not math.isfinite(item) or item <= 0 for item in points):
            raise ContractError("Kernel terminal speedups are invalid", "invalid_kernel_terminal_grade")
        if self.scalar_reward is not None and (
            not math.isfinite(self.scalar_reward) or not 0.0 <= self.scalar_reward <= 320.0
        ):
            raise ContractError("Kernel terminal reward is invalid", "invalid_kernel_terminal_grade")
        if (self.trainability == "trainable") != (self.scalar_reward is not None):
            raise ContractError("Kernel terminal reward is incoherent", "invalid_kernel_terminal_grade")
        if (self.trainability == "untrainable") != bool(self.untrainable_reason):
            raise ContractError("Kernel terminal null reason is incoherent", "invalid_kernel_terminal_grade")
        if self.trainability == "untrainable":
            if (
                self.outcome != "untrainable"
                or self.source_attempt_id is not None
                or self.gates is not None
                or points
            ):
                raise ContractError("Untrainable terminal grade fabricates evidence", "invalid_kernel_terminal_grade")
            return
        if self.gates is None or self.source_attempt_id is None or self.untrainable_reason is not None:
            raise ContractError("Kernel terminal grade lacks source evidence", "invalid_kernel_terminal_grade")
        if kernel_reward(self.gates, self.srobust) != self.scalar_reward:
            raise ContractError("Kernel terminal reward differs from policy", "invalid_kernel_terminal_grade")
        self._validate_outcome_semantics()

    def _validate_outcome_semantics(self) -> None:
        assert self.gates is not None
        all_speedups = self.s50 is not None and self.s99 is not None and self.srobust is not None
        no_speedups = self.s50 is None and self.s99 is None and self.srobust is None
        if self.outcome == "selected_candidate" and (
            not all_speedups
            or not self.gates.correctness_gate_passed
            or self.gates.safety_finding
            or self.srobust != min(self.s50, self.s99)  # type: ignore[type-var]
        ):
            raise ContractError("Selected terminal grade is incoherent", "invalid_kernel_terminal_grade")
        if self.outcome == "measured_noop" and (
            (self.s50, self.s99, self.srobust) != (1.0, 1.0, 1.0)
            or not self.gates.correctness_gate_passed
            or self.gates.safety_finding
        ):
            raise ContractError("No-op terminal grade is incoherent", "invalid_kernel_terminal_grade")
        if self.outcome == "compile_failure" and (
            not no_speedups or self.gates.compiled or self.scalar_reward != 0.0
        ):
            raise ContractError("Compile terminal grade is incoherent", "invalid_kernel_terminal_grade")
        if self.outcome == "correctness_failure" and (
            not no_speedups
            or not self.gates.compiled
            or self.gates.correct
            or not self.gates.integrity_passed
            or not self.gates.tampering_passed
            or self.scalar_reward != 20.0
        ):
            raise ContractError("Correctness terminal grade is incoherent", "invalid_kernel_terminal_grade")

    @property
    def policy_digest(self) -> str:
        return sha256_json(kernel_terminal_policy_source())

    def to_dict(self) -> dict[str, Any]:
        gates = self.gates
        return {
            "schema": "apex.kernel-terminal-grade/v1",
            "scope": "task_terminal",
            "policy_id": self.policy_id,
            "policy_digest": self.policy_digest,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "source_attempt_id": self.source_attempt_id,
            "gates": (
                {
                    "compiled": gates.compiled,
                    "correct": gates.correct,
                    "integrity_passed": gates.integrity_passed,
                    "tampering_passed": gates.tampering_passed,
                    "safety_finding": gates.safety_finding,
                }
                if gates is not None
                else None
            ),
            "s50": self.s50,
            "s99": self.s99,
            "srobust": self.srobust,
            "scalar_reward": self.scalar_reward,
            "trainability": self.trainability,
            "untrainable_reason": self.untrainable_reason,
        }


def selected_kernel_terminal_grade(
    attempt_id: str, grade: KernelGrade
) -> KernelTerminalGrade:
    return KernelTerminalGrade(
        "selected_candidate",
        grade.promotion_reason_code,
        attempt_id,
        grade.gates,
        grade.s50,
        grade.s99,
        grade.srobust,
        grade.reward,
        "trainable",
        None,
    )


def no_op_kernel_terminal_grade(attempt_id: str) -> KernelTerminalGrade:
    gates = GateVerdict(True, True, True, True)
    return KernelTerminalGrade(
        "measured_noop",
        "terminal_baseline_noop",
        attempt_id,
        gates,
        1.0,
        1.0,
        1.0,
        kernel_reward(gates, 1.0),
        "trainable",
        None,
    )


def failed_kernel_terminal_grade(
    attempt_id: str, *, correctness_reached: bool
) -> KernelTerminalGrade:
    gates = GateVerdict(
        compiled=correctness_reached,
        correct=False,
        integrity_passed=correctness_reached,
        tampering_passed=correctness_reached,
    )
    return KernelTerminalGrade(
        "correctness_failure" if correctness_reached else "compile_failure",
        "correctness_failed" if correctness_reached else "compile_failed",
        attempt_id,
        gates,
        None,
        None,
        None,
        kernel_reward(gates, None),
        "trainable",
        None,
    )


def untrainable_kernel_terminal_grade(reason: str) -> KernelTerminalGrade:
    return KernelTerminalGrade(
        "untrainable",
        reason,
        None,
        None,
        None,
        None,
        None,
        None,
        "untrainable",
        reason,
    )


def load_kernel_terminal_grade(value: Mapping[str, Any]) -> KernelTerminalGrade:
    """Load the exact terminal vector while rejecting policy or shape drift."""

    expected = {
        "schema",
        "scope",
        "policy_id",
        "policy_digest",
        "outcome",
        "reason_code",
        "source_attempt_id",
        "gates",
        "s50",
        "s99",
        "srobust",
        "scalar_reward",
        "trainability",
        "untrainable_reason",
    }
    if set(value) != expected or value.get("schema") != "apex.kernel-terminal-grade/v1" or value.get("scope") != "task_terminal":
        raise ContractError("Kernel terminal grade fields are invalid", "invalid_kernel_terminal_grade")
    gates_value = value.get("gates")
    gates: GateVerdict | None = None
    if gates_value is not None:
        if not isinstance(gates_value, Mapping) or set(gates_value) != {
            "compiled",
            "correct",
            "integrity_passed",
            "tampering_passed",
            "safety_finding",
        } or any(not isinstance(item, bool) for item in gates_value.values()):
            raise ContractError("Kernel terminal gates are invalid", "invalid_kernel_terminal_grade")
        gates = GateVerdict(
            gates_value["compiled"],
            gates_value["correct"],
            gates_value["integrity_passed"],
            gates_value["tampering_passed"],
            gates_value["safety_finding"],
        )
    source = value.get("source_attempt_id")
    reason = value.get("untrainable_reason")
    grade = KernelTerminalGrade(
        outcome=_text(value.get("outcome"), "outcome"),
        reason_code=_text(value.get("reason_code"), "reason_code"),
        source_attempt_id=None if source is None else _text(source, "source_attempt_id"),
        gates=gates,
        s50=_optional_number(value.get("s50")),
        s99=_optional_number(value.get("s99")),
        srobust=_optional_number(value.get("srobust")),
        scalar_reward=_optional_number(value.get("scalar_reward")),
        trainability=_text(value.get("trainability"), "trainability"),
        untrainable_reason=None if reason is None else _text(reason, "untrainable_reason"),
        policy_id=_text(value.get("policy_id"), "policy_id"),
    )
    if value.get("policy_digest") != grade.policy_digest or value != grade.to_dict():
        raise ContractError("Kernel terminal grade differs from policy", "invalid_kernel_terminal_grade")
    return grade


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"Kernel terminal {field} is invalid", "invalid_kernel_terminal_grade")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("Kernel terminal number is invalid", "invalid_kernel_terminal_grade")
    return float(value)


__all__ = [
    "KERNEL_REWARD_POLICY_ID",
    "KernelTerminalGrade",
    "failed_kernel_terminal_grade",
    "kernel_terminal_policy_source",
    "kernel_reward_policy_source",
    "load_kernel_terminal_grade",
    "no_op_kernel_terminal_grade",
    "selected_kernel_terminal_grade",
    "untrainable_kernel_terminal_grade",
]
