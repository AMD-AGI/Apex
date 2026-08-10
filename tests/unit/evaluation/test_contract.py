from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apex.core import ContractError, sha256_bytes
from apex.evaluation import (
    EvaluationAuthorityIdentity,
    EvaluationAuthorityKind,
    EvaluationAuthorityReceipt,
    EvaluationContractDraft,
    EvaluationContractFreezer,
    ExactEvaluationAuthorityRegistry,
    load_evaluation_contract,
    reviewed_template_evaluation_authorizer,
    user_confirmed_evaluation_authorizer,
)
from apex.intake import (
    CommandSpec,
    TaskRecipe,
    TaskResolver,
    TaskSpec,
    TemplateTaskAuthority,
)
from apex.ports import WorkspaceRepositoryIdentity


def _resolved(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n", encoding="utf-8")
    task = TaskSpec.from_mapping(
        {
            "task_id": "contract-test",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize kernel",
            "language": "triton",
            "editable_files": ["kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                phase: {"argv": ["true"]}
                for phase in ("compile", "correctness", "performance")
            },
            "recipe": {
                "kind": "python_triton",
                "recipe_id": "self-claimed-template",
                "sha256": "a" * 64,
                "provenance": "trusted_registry",
            },
        }
    )
    return TaskResolver().resolve(task)


def _repository(tmp_path: Path) -> WorkspaceRepositoryIdentity:
    return WorkspaceRepositoryIdentity(
        str((tmp_path / "workspace").resolve()),
        "resolved",
        "example.invalid/org/repo",
        "1" * 40,
        "2" * 40,
        (),
    )


def _identity() -> EvaluationAuthorityIdentity:
    return EvaluationAuthorityIdentity(
        "reviewed-python-triton-v1",
        EvaluationAuthorityKind.REVIEWED_TEMPLATE,
        "apex-bootstrap",
        "3" * 64,
        "4" * 64,
    )


def test_descriptor_recipe_claim_cannot_self_authorize(tmp_path: Path) -> None:
    receipt = EvaluationContractFreezer().freeze(
        _resolved(tmp_path), _repository(tmp_path)
    )

    assert receipt.verified is False
    assert receipt.unverified_reason == "evaluation_authority_missing"
    assert receipt.authority is None
    assert receipt.draft.recipe_claim["provenance"] == "trusted_registry"


def test_exact_composition_registry_authorizes_only_one_frozen_draft(
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    repository = _repository(tmp_path)
    draft = EvaluationContractDraft.from_resolved(resolved, repository)
    registry = ExactEvaluationAuthorityRegistry({draft.digest: _identity()})

    receipt = EvaluationContractFreezer(registry).freeze(resolved, repository)

    assert receipt.verified is True
    assert receipt.authority is not None
    assert receipt.authority.draft_digest == receipt.draft.digest
    assert receipt.to_dict()["schema"] == "apex.evaluation-contract-receipt/v1"
    assert len(receipt.digest) == 64
    assert receipt.draft.source_scope["editable_files"] == ["kernel.py"]
    assert receipt.draft.policies["grading"] == "kernel_robust_v1"
    repository_document = receipt.to_dict()["draft"]["repository"]
    assert "root" not in repository_document
    assert repository_document["root_sha256"] == sha256_bytes(
        repository.root.encode("utf-8")
    )


def test_cas_contract_loader_restores_redacted_root_and_rejects_wrong_root(
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    repository = _repository(tmp_path)
    draft = EvaluationContractDraft.from_resolved(resolved, repository)
    receipt = EvaluationContractFreezer(
        ExactEvaluationAuthorityRegistry({draft.digest: _identity()})
    ).freeze(resolved, repository)

    loaded = load_evaluation_contract(
        receipt.to_dict(), repository_root=resolved.workspace
    )

    assert loaded == receipt
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    with pytest.raises(ContractError) as error:
        load_evaluation_contract(receipt.to_dict(), repository_root=wrong)
    assert error.value.reason_code == "invalid_evaluation_contract"


def test_explicit_user_confirmation_binds_one_previewed_draft(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    repository = _repository(tmp_path)
    draft = EvaluationContractDraft.from_resolved(resolved, repository)
    authorizer = user_confirmed_evaluation_authorizer(draft.digest)

    receipt = EvaluationContractFreezer(authorizer).freeze(resolved, repository)

    assert receipt.verified is True
    assert receipt.authority is not None
    assert receipt.authority.authority.kind is EvaluationAuthorityKind.USER_CONFIRMATION
    assert receipt.authority.authority.authority_id == "local-user-confirmation-v1"

    (tmp_path / "workspace" / "kernel.py").write_text(
        "def kernel(x): return x + 1\n", encoding="utf-8"
    )
    changed = TaskResolver().resolve(resolved.task)
    with pytest.raises(ContractError) as mismatch:
        EvaluationContractFreezer(authorizer).freeze(changed, repository)
    assert mismatch.value.reason_code == "evaluation_authority_mismatch"


def test_mismatched_authority_receipt_fails_closed(tmp_path: Path) -> None:
    class WrongAuthority:
        def authorize(self, draft):
            return EvaluationAuthorityReceipt(_identity(), "f" * 64)

    with pytest.raises(ContractError) as error:
        EvaluationContractFreezer(WrongAuthority()).freeze(
            _resolved(tmp_path), _repository(tmp_path)
        )
    assert error.value.reason_code == "evaluation_authority_mismatch"


def test_unresolved_repository_cannot_become_verified(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    repository = WorkspaceRepositoryIdentity(
        str((tmp_path / "workspace").resolve()),
        "unresolved",
        None,
        None,
        None,
        (),
        "repository_identity_unavailable",
    )
    draft = EvaluationContractDraft.from_resolved(resolved, repository)
    registry = ExactEvaluationAuthorityRegistry({draft.digest: _identity()})

    receipt = EvaluationContractFreezer(registry).freeze(resolved, repository)

    assert receipt.verified is False
    assert receipt.authority is None
    assert receipt.unverified_reason == "repository_identity_unavailable"


def test_reviewed_template_authority_binds_materialized_task_digest(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "template-workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n", encoding="utf-8")
    authority = TemplateTaskAuthority(
        "reviewed-template", "reviewed-showcase", "3" * 64,
        "example.invalid/image@sha256:" + "4" * 64, "sha256:" + "5" * 64,
        "6" * 64, "7" * 64, "8" * 64,
    )
    task = TaskSpec(
        schema_version=1,
        task_id="reviewed-template",
        workspace=workspace.resolve(),
        results_dir=(tmp_path / "results").resolve(),
        instructions="Optimize",
        language="triton",
        editable_files=("kernel.py",),
        target_functions=("kernel",),
        commands={phase: CommandSpec(("true",)) for phase in ("compile", "correctness", "performance")},
        mode="template_bound_image_kernel",
        recipe=TaskRecipe("python_triton", "reviewed-recipe", "7" * 64, "trusted_registry"),
        template_authority=authority,
    )
    resolved = TaskResolver().resolve(task)
    authorizer = reviewed_template_evaluation_authorizer(task)

    receipt = EvaluationContractFreezer(authorizer).freeze(
        resolved, _repository(tmp_path)
    )

    assert receipt.verified
    assert receipt.authority is not None
    assert receipt.authority.authority.kind is EvaluationAuthorityKind.REVIEWED_TEMPLATE
    assert receipt.draft.source_scope["template_authority"] == authority.to_dict()
    changed = TaskResolver().resolve(replace(task, instructions="Changed"))
    with pytest.raises(ContractError) as mismatch:
        EvaluationContractFreezer(authorizer).freeze(changed, _repository(tmp_path))
    assert mismatch.value.reason_code == "evaluation_authority_mismatch"
