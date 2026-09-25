"""Path-free frozen contract for offline E2E reward and promotion replay."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from apex.core import ContractError, sha256_json, validate_identifier
from apex.intake import RegressionGates

from .e2e import E2EAcceptancePolicy


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class E2ERewardContract:
    """Only metric, gate, estimator, and protocol truth needed for scoring."""

    run_id: str
    measurement_protocol_hash: str
    acceptance_policy: E2EAcceptancePolicy
    primary_metric: str = "throughput"
    direction: str = "maximize"

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field_name="run_id")
        if (
            not _DIGEST.fullmatch(self.measurement_protocol_hash)
            or self.primary_metric != "throughput"
            or self.direction != "maximize"
            or self.acceptance_policy.policy_id != "current_anchor_throughput_v1"
        ):
            raise ContractError(
                "E2E reward contract is invalid", "invalid_e2e_reward_contract"
            )

    @property
    def objective(self) -> dict[str, Any]:
        return {
            "primary": self.primary_metric,
            "direction": self.direction,
            "gates": asdict(self.acceptance_policy.gates),
        }

    @property
    def objective_policy_hash(self) -> str:
        return sha256_json(self.objective)

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-reward-contract/v1",
            "task_kind": "e2e_kernel_only",
            "run_id": self.run_id,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "measurement_protocol_hash": self.measurement_protocol_hash,
            "objective_policy_hash": self.objective_policy_hash,
            "acceptance_policy": self.acceptance_policy.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "E2ERewardContract":
        expected = {
            "schema", "task_kind", "run_id", "primary_metric", "direction",
            "measurement_protocol_hash", "objective_policy_hash",
            "acceptance_policy",
        }
        if (
            set(value) != expected
            or value.get("schema") != "apex.e2e-reward-contract/v1"
            or value.get("task_kind") != "e2e_kernel_only"
            or not isinstance(value.get("acceptance_policy"), Mapping)
        ):
            raise ContractError(
                "E2E reward contract shape is invalid",
                "invalid_e2e_reward_contract",
            )
        policy = _load_policy(value["acceptance_policy"])
        contract = cls(
            run_id=str(value["run_id"]),
            measurement_protocol_hash=str(value["measurement_protocol_hash"]),
            acceptance_policy=policy,
            primary_metric=str(value["primary_metric"]),
            direction=str(value["direction"]),
        )
        if contract.to_dict() != dict(value):
            raise ContractError(
                "E2E reward contract values are invalid",
                "invalid_e2e_reward_contract",
            )
        return contract


def _load_policy(value: Mapping[str, Any]) -> E2EAcceptancePolicy:
    expected = set(E2EAcceptancePolicy().to_dict())
    gates = value.get("gates")
    if set(value) != expected or not isinstance(gates, Mapping):
        raise ContractError(
            "E2E acceptance policy is invalid", "invalid_e2e_reward_contract"
        )
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
    except (ContractError, KeyError, TypeError, ValueError) as error:
        raise ContractError(
            "E2E acceptance policy is invalid", "invalid_e2e_reward_contract"
        ) from error
    if policy.to_dict() != dict(value):
        raise ContractError(
            "E2E acceptance policy is noncanonical", "invalid_e2e_reward_contract"
        )
    return policy


__all__ = ["E2ERewardContract"]
