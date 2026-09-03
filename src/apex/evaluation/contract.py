"""Evaluator-owned freeze of a discovered standalone evaluation contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from apex.core import ContractError, sha256_json, validate_identifier
from apex.intake import ResolvedTaskSpec, TaskSpec
from apex.ports import WorkspaceRepositoryIdentity


_SHA256 = re.compile(r"[0-9a-f]{64}")


class EvaluationAuthorityKind(str, Enum):
    REVIEWED_TEMPLATE = "reviewed_template"
    EXTERNAL_EVALUATOR = "external_evaluator"
    USER_CONFIRMATION = "user_confirmation"


@dataclass(frozen=True, slots=True)
class EvaluationAuthorityIdentity:
    authority_id: str
    kind: EvaluationAuthorityKind
    issuer: str
    policy_sha256: str
    template_sha256: str

    def __post_init__(self) -> None:
        if not self.authority_id or not self.issuer:
            raise ContractError("Evaluation authority is incomplete", "invalid_evaluation_authority")
        validate_identifier(self.authority_id, field_name="evaluation authority ID")
        if not _SHA256.fullmatch(self.policy_sha256) or not _SHA256.fullmatch(
            self.template_sha256
        ):
            raise ContractError("Evaluation authority digest is invalid", "invalid_evaluation_authority")


@dataclass(frozen=True, slots=True)
class EvaluationAuthorityReceipt:
    authority: EvaluationAuthorityIdentity
    draft_digest: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.draft_digest):
            raise ContractError("Authority draft binding is invalid", "invalid_evaluation_authority")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.evaluation-authority-receipt/v1",
            "authority_id": self.authority.authority_id,
            "authority_kind": self.authority.kind.value,
            "issuer": self.authority.issuer,
            "policy_sha256": self.authority.policy_sha256,
            "template_sha256": self.authority.template_sha256,
            "draft_digest": self.draft_digest,
        }


@dataclass(frozen=True, slots=True)
class EvaluationContractDraft:
    task_id: str
    task_digest: str
    resolution_hash: str
    repository: WorkspaceRepositoryIdentity
    baseline_file_hashes: tuple[tuple[str, str], ...]
    harness_file_hashes: tuple[tuple[str, str], ...]
    harness_sha256: str | None
    gpu_arch: str
    source_scope: Mapping[str, Any]
    budget: Mapping[str, Any]
    commands: Mapping[str, Mapping[str, Any]]
    measurement: Mapping[str, Any] | None
    recipe_claim: Mapping[str, Any] | None
    policies: Mapping[str, str]

    @classmethod
    def from_resolved(
        cls,
        resolved: ResolvedTaskSpec,
        repository: WorkspaceRepositoryIdentity,
    ) -> "EvaluationContractDraft":
        task = resolved.task
        return cls(
            task_id=task.task_id,
            task_digest=sha256_json(task.to_dict()),
            resolution_hash=resolved.resolution_hash,
            repository=repository,
            baseline_file_hashes=tuple(sorted(resolved.baseline_file_hashes.items())),
            harness_file_hashes=tuple(sorted(resolved.harness_file_hashes.items())),
            harness_sha256=resolved.harness_sha256,
            gpu_arch=task.gpu_arch,
            source_scope={
                "mode": task.mode,
                "language": task.language,
                "editable_files": list(task.editable_files),
                "target_functions": list(task.target_functions),
                "template_authority": (
                    task.template_authority.to_dict()
                    if task.template_authority is not None else None
                ),
            },
            budget={
                "max_iterations": task.budget.max_iterations,
                "max_turns": task.budget.max_turns,
                "timeout_seconds": task.budget.timeout_seconds,
            },
            commands={name: command.to_dict() for name, command in sorted(task.commands.items())},
            measurement=task.measurement.to_dict() if task.measurement else None,
            recipe_claim=task.recipe.to_dict() if task.recipe else None,
            policies={
                "contract": "apex.evaluation-contract-receipt/v1",
                "task_parser": "apex.task-spec/v1",
                "repository_identity": "apex.workspace-git-identity/v1",
                "grading": "kernel_robust_v1",
                "safety": "external_receipt_or_no_tools_v1",
            },
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.evaluation-contract-draft/v1",
            "task_id": self.task_id,
            "task_digest": self.task_digest,
            "resolution_hash": self.resolution_hash,
            "repository": self.repository.to_dict(),
            "baseline_file_hashes": dict(self.baseline_file_hashes),
            "harness_file_hashes": dict(self.harness_file_hashes),
            "harness_sha256": self.harness_sha256,
            "gpu_arch": self.gpu_arch,
            "source_scope": dict(self.source_scope),
            "budget": dict(self.budget),
            "commands": {key: dict(value) for key, value in self.commands.items()},
            "measurement": dict(self.measurement) if self.measurement else None,
            "recipe_claim": dict(self.recipe_claim) if self.recipe_claim else None,
            "policies": dict(self.policies),
        }


class EvaluationContractAuthorizer(Protocol):
    def authorize(
        self, draft: EvaluationContractDraft
    ) -> EvaluationAuthorityReceipt | None: ...


@dataclass(frozen=True, slots=True)
class EvaluationContractReceipt:
    draft: EvaluationContractDraft
    authority: EvaluationAuthorityReceipt | None
    status: str
    unverified_reason: str | None

    def __post_init__(self) -> None:
        verified = self.status == "verified"
        if self.status not in {"verified", "unverified"}:
            raise ContractError("Evaluation contract status is invalid", "invalid_evaluation_contract")
        if verified != bool(self.authority) or verified == bool(self.unverified_reason):
            raise ContractError("Evaluation contract receipt is incoherent", "invalid_evaluation_contract")
        if self.authority and self.authority.draft_digest != self.draft.digest:
            raise ContractError("Evaluation authority binds another draft", "evaluation_authority_mismatch")

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.evaluation-contract-receipt/v1",
            "status": self.status,
            "unverified_reason": self.unverified_reason,
            "draft": self.draft.to_dict(),
            "draft_digest": self.draft.digest,
            "authority": self.authority.to_dict() if self.authority else None,
        }


class EvaluationContractFreezer:
    def __init__(self, authorizer: EvaluationContractAuthorizer | None = None) -> None:
        self._authorizer = authorizer

    def freeze(
        self,
        resolved: ResolvedTaskSpec,
        repository: WorkspaceRepositoryIdentity,
    ) -> EvaluationContractReceipt:
        draft = EvaluationContractDraft.from_resolved(resolved, repository)
        authority = self._authorizer.authorize(draft) if self._authorizer else None
        if authority is not None and authority.draft_digest != draft.digest:
            raise ContractError("Evaluation authority binds another draft", "evaluation_authority_mismatch")
        if not repository.resolved:
            return EvaluationContractReceipt(
                draft, None, "unverified", repository.unavailable_reason
            )
        if authority is None:
            return EvaluationContractReceipt(
                draft, None, "unverified", "evaluation_authority_missing"
            )
        return EvaluationContractReceipt(draft, authority, "verified", None)


class ExactEvaluationAuthorityRegistry:
    """Authorize only exact reviewed draft digests configured by composition."""

    def __init__(self, entries: Mapping[str, EvaluationAuthorityIdentity]) -> None:
        self._entries = dict(entries)

    def authorize(
        self, draft: EvaluationContractDraft
    ) -> EvaluationAuthorityReceipt | None:
        identity = self._entries.get(draft.digest)
        return EvaluationAuthorityReceipt(identity, draft.digest) if identity else None


class DigestBoundEvaluationAuthorizer:
    """Require one pre-inspected draft digest before issuing authority."""

    def __init__(
        self,
        expected_draft_digest: str,
        identity: EvaluationAuthorityIdentity,
    ) -> None:
        if not _SHA256.fullmatch(expected_draft_digest):
            raise ContractError("Confirmed draft digest is invalid", "invalid_evaluation_authority")
        self._expected = expected_draft_digest
        self._identity = identity

    def authorize(self, draft: EvaluationContractDraft) -> EvaluationAuthorityReceipt:
        if draft.digest != self._expected:
            raise ContractError(
                "Confirmed evaluation contract changed before execution",
                "evaluation_authority_mismatch",
                {"expected": self._expected, "observed": draft.digest},
            )
        return EvaluationAuthorityReceipt(self._identity, draft.digest)


class ReviewedTemplateEvaluationAuthorizer:
    """Authorize one internally materialized task and no user-replayed mapping."""

    def __init__(self, task: TaskSpec) -> None:
        authority = task.template_authority
        if authority is None or task.mode != "template_bound_image_kernel":
            raise ContractError(
                "Reviewed template authority is missing", "template_authority_required"
            )
        self._task_digest = sha256_json(task.to_dict())
        self._identity = EvaluationAuthorityIdentity(
            authority_id=authority.template_id,
            kind=EvaluationAuthorityKind.REVIEWED_TEMPLATE,
            issuer="apex-reviewed-kernel-template-registry",
            policy_sha256=authority.evaluator_recipe_sha256,
            template_sha256=authority.manifest_sha256,
        )

    def authorize(self, draft: EvaluationContractDraft) -> EvaluationAuthorityReceipt:
        if draft.task_digest != self._task_digest:
            raise ContractError(
                "Materialized template task changed before contract freeze",
                "evaluation_authority_mismatch",
            )
        return EvaluationAuthorityReceipt(self._identity, draft.digest)


def user_confirmed_evaluation_authorizer(
    expected_draft_digest: str,
) -> DigestBoundEvaluationAuthorizer:
    """Materialize explicit local-user confirmation of one previewed draft."""

    identity = EvaluationAuthorityIdentity(
        authority_id="local-user-confirmation-v1",
        kind=EvaluationAuthorityKind.USER_CONFIRMATION,
        issuer="apex-cli-explicit-draft-confirmation",
        policy_sha256=sha256_json(
            {
                "policy": "explicit_previewed_draft_confirmation_v1",
                "authority_scope": "local_formal_campaign",
            }
        ),
        template_sha256=sha256_json(
            {
                "template": "evaluation_contract_draft_v1",
                "external_attestation": False,
            }
        ),
    )
    return DigestBoundEvaluationAuthorizer(expected_draft_digest, identity)


def reviewed_template_evaluation_authorizer(
    task: TaskSpec,
) -> ReviewedTemplateEvaluationAuthorizer:
    return ReviewedTemplateEvaluationAuthorizer(task)


__all__ = [
    "EvaluationAuthorityIdentity",
    "EvaluationAuthorityKind",
    "EvaluationAuthorityReceipt",
    "EvaluationContractAuthorizer",
    "EvaluationContractDraft",
    "EvaluationContractFreezer",
    "EvaluationContractReceipt",
    "DigestBoundEvaluationAuthorizer",
    "ReviewedTemplateEvaluationAuthorizer",
    "reviewed_template_evaluation_authorizer",
    "ExactEvaluationAuthorityRegistry",
    "user_confirmed_evaluation_authorizer",
]
