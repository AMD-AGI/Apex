"""Untrusted chat-to-formal draft creation without agent, GPU, or evaluator work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError, canonical_json_bytes
from apex.evaluation import EvaluationContractFreezer, EvaluationContractReceipt
from apex.intake import TaskResolver, TaskSpec
from apex.runtime import WorkspaceGitIdentityResolver

from .contract_recording import record_evaluation_contract
from .run_record import KernelRunRecord


@dataclass(frozen=True, slots=True)
class KernelCampaignDraft:
    run_id: str
    root: Path
    contract: EvaluationContractReceipt
    high_water_mark: int


class KernelCampaignDraftUseCase:
    """Persist a discovered task as an explicitly unverified campaign draft."""

    def start(
        self,
        task_data: Mapping[str, Any],
        *,
        workspace: Path,
        results_dir: Path,
        run_root: Path,
        run_id: str,
    ) -> KernelCampaignDraft:
        task = TaskSpec.from_mapping(
            {
                **dict(task_data),
                "workspace": str(workspace),
                "results_dir": str(results_dir),
            }
        )
        resolved = TaskResolver().resolve(task)
        repository = WorkspaceGitIdentityResolver().inspect(resolved.workspace)
        contract = EvaluationContractFreezer().freeze(resolved, repository)
        record = KernelRunRecord.create(
            run_id=run_id,
            root=run_root,
            initial_anchor_id=f"anchor-{resolved.resolution_hash[:16]}",
            dataset_split=task.dataset_split,
            data_visibility=task.data_visibility,
        )
        task_receipt = record.artifacts.put_bytes(
            canonical_json_bytes(task.to_dict()), media_type="application/json"
        )
        source_bindings = _frozen_file_bindings(
            record,
            resolved.workspace,
            resolved.baseline_file_hashes,
            role="baseline_source",
        )
        harness_bindings = _frozen_file_bindings(
            record,
            resolved.workspace,
            resolved.harness_file_hashes,
            role="protected_harness",
        )
        record.controller.record_domain_event(
            "provenance_observed",
            {
                "kind": "kernel_campaign_draft",
                "task_id": task.task_id,
                "task_digest": contract.draft.task_digest,
                "resolution_hash": resolved.resolution_hash,
                "repository": repository.to_dict(),
                "verified": False,
                "artifacts": [
                    {"role": "task_input", "receipt": task_receipt.to_dict()},
                    *source_bindings,
                    *harness_bindings,
                ],
            },
            idempotency_key="campaign.draft.task",
        )
        record_evaluation_contract(
            artifacts=record.artifacts,
            controller=record.controller,
            contract=contract,
        )
        return KernelCampaignDraft(
            run_id,
            run_root,
            contract,
            record.controller.state.sequence,
        )


def _frozen_file_bindings(
    record: KernelRunRecord,
    workspace: Path,
    expected_hashes: Mapping[str, str],
    *,
    role: str,
) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for relative, expected in sorted(expected_hashes.items()):
        receipt = record.artifacts.put_file(
            workspace.joinpath(*relative.split("/")),
            media_type="application/octet-stream",
        )
        if receipt.digest != expected:
            raise IntegrityError(
                "Formal campaign input changed during draft freeze",
                "campaign_draft_input_changed",
                {"path": relative, "role": role},
            )
        bindings.append(
            {"role": role, "path": relative, "receipt": receipt.to_dict()}
        )
    return bindings


__all__ = ["KernelCampaignDraft", "KernelCampaignDraftUseCase"]
