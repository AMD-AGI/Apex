"""Semantic validation shared by atomic E2E decision/reward commits."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ContractError


def validate_e2e_reward_payload(
    reward: Mapping[str, Any],
    *,
    verdict: str,
    reason: str,
    candidate_present: bool,
) -> None:
    vector = reward.get("reward_vector")
    if not isinstance(vector, Mapping):
        raise ContractError(
            "E2E reward vector is missing",
            "e2e_reward_decision_mismatch",
        )
    pairs = (
        (reward.get("verdict"), verdict),
        (reward.get("reason_code"), reason),
        (vector.get("verdict"), verdict),
        (vector.get("reason_code"), reason),
        (vector.get("candidate_present"), candidate_present),
        (vector.get("policy_id"), reward.get("policy_id")),
        (vector.get("policy_digest"), reward.get("policy_digest")),
        (vector.get("scalar_reward"), reward.get("scalar_reward")),
    )
    if reward.get("evidence_class") != "derived" or any(
        observed != expected for observed, expected in pairs
    ):
        raise ContractError(
            "E2E reward conflicts with its decision or grade vector",
            "e2e_reward_decision_mismatch",
        )


__all__ = ["validate_e2e_reward_payload"]
