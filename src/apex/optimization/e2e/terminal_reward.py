"""Unique task-terminal E2E reward publication."""

from __future__ import annotations

from apex.core import ContractError
from apex.evaluation import E2ERewardPolicy, E2ERewardVector
from apex.storage import ArtifactReceipt

from .run_record import E2ERunRecord


def record_terminal_reward(
    record: E2ERunRecord,
    grade: E2ERewardVector,
    *,
    reward_source: ArtifactReceipt,
    paired_measurement: ArtifactReceipt,
    raw_measurement_receipts: tuple[str, ...],
    raw_evidence: tuple[ArtifactReceipt, ...],
) -> ArtifactReceipt:
    """Commit the unique task-terminal reward from final clean replay evidence."""

    if grade.scope != "task_terminal" or not raw_measurement_receipts or not raw_evidence:
        raise ContractError(
            "Terminal E2E reward evidence is incomplete",
            "invalid_terminal_e2e_reward",
        )
    policy = E2ERewardPolicy()
    if grade.policy_digest != policy.digest:
        raise ContractError(
            "Terminal E2E reward policy differs",
            "e2e_reward_policy_mismatch",
        )
    vector = record.put_json(grade.to_dict())
    policy_receipt = record.put_json(policy.to_dict())
    record.controller.record_domain_event(
        "reward_committed",
        {
            "scope": "task_terminal",
            "task_kind": "e2e_kernel_only",
            "policy_id": grade.policy_id,
            "policy_digest": grade.policy_digest,
            "scalar_reward": grade.scalar_reward,
            "reward_vector": grade.to_dict(),
            "reward_source_receipt": reward_source.digest,
            "raw_measurement_receipts": list(raw_measurement_receipts),
            "evidence_class": "derived",
            "artifacts": [
                _binding("terminal_reward_source", reward_source),
                _binding("terminal_paired_measurement", paired_measurement),
                _binding("e2e_reward_vector", vector),
                _binding("reward_policy", policy_receipt),
                *(
                    _binding(f"terminal_raw_{index}", receipt)
                    for index, receipt in enumerate(raw_evidence)
                ),
            ],
        },
        idempotency_key="e2e.task_terminal.reward",
    )
    return vector


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = ["record_terminal_reward"]
