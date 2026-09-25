"""Recovery checks for scored and reward-null E2E decisions."""

from __future__ import annotations

from typing import Protocol

from apex.core import IntegrityError
from apex.storage import EventRecord


class OutcomeEventIndex(Protocol):
    def keyed(self, key: str) -> EventRecord | None: ...


def verify_decision_reward_transaction(
    index: OutcomeEventIndex, attempt_id: str
) -> EventRecord:
    decision = index.keyed(f"e2e.attempt.{attempt_id}.decision")
    reward = index.keyed(f"e2e.attempt.{attempt_id}.reward")
    if (
        decision is None
        or reward is None
        or decision.event_type != "e2e.candidate_decided"
        or reward.event_type != "reward_committed"
        or decision.transaction_id != reward.transaction_id
    ):
        raise IntegrityError(
            "Decision and reward are not one transaction",
            "e2e_reward_transaction_mismatch",
        )
    return reward


def verify_untrainable_decision(
    index: OutcomeEventIndex, attempt_id: str, reason: str
) -> None:
    decision = index.keyed(f"e2e.attempt.{attempt_id}.decision")
    reward = index.keyed(f"e2e.attempt.{attempt_id}.reward")
    if (
        decision is None
        or reward is not None
        or decision.event_type != "e2e.candidate_decided"
        or decision.payload.get("verdict") != "needs_more_measurement"
        or decision.payload.get("reason") != reason
        or decision.payload.get("trainability") != "untrainable"
        or decision.payload.get("untrainable_reason") != reason
    ):
        raise IntegrityError(
            "Untrainable decision has fabricated or ambiguous reward lineage",
            "e2e_reward_transaction_mismatch",
        )


__all__ = ["verify_decision_reward_transaction", "verify_untrainable_decision"]
