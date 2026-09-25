"""Strict reconstruction of a CAS-backed evaluation contract receipt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, sha256_bytes
from apex.ports import WorkspaceRepositoryIdentity

from .contract import (
    EvaluationAuthorityIdentity,
    EvaluationAuthorityKind,
    EvaluationAuthorityReceipt,
    EvaluationContractDraft,
    EvaluationContractReceipt,
)


def load_evaluation_contract(
    value: Mapping[str, Any], *, repository_root: Path
) -> EvaluationContractReceipt:
    """Rebuild the typed receipt while independently supplying its redacted root."""

    _exact(
        value,
        {"schema", "status", "unverified_reason", "draft", "draft_digest", "authority"},
    )
    if value["schema"] != "apex.evaluation-contract-receipt/v1":
        _invalid()
    draft = _load_draft(_mapping(value["draft"]), repository_root=repository_root)
    if _string(value["draft_digest"]) != draft.digest:
        _invalid()
    authority = _load_authority(value["authority"])
    return EvaluationContractReceipt(
        draft=draft,
        authority=authority,
        status=_string(value["status"]),
        unverified_reason=_optional_string(value["unverified_reason"]),
    )


def _load_draft(
    value: Mapping[str, Any], *, repository_root: Path
) -> EvaluationContractDraft:
    _exact(
        value,
        {
            "schema", "task_id", "task_digest", "resolution_hash", "repository",
            "baseline_file_hashes", "harness_file_hashes", "harness_sha256",
            "gpu_arch", "source_scope", "budget", "commands", "measurement",
            "recipe_claim", "policies",
        },
    )
    if value["schema"] != "apex.evaluation-contract-draft/v1":
        _invalid()
    root = repository_root.resolve(strict=True)
    repository = _load_repository(_mapping(value["repository"]), root=root)
    draft = EvaluationContractDraft(
        task_id=_string(value["task_id"]),
        task_digest=_string(value["task_digest"]),
        resolution_hash=_string(value["resolution_hash"]),
        repository=repository,
        baseline_file_hashes=tuple(sorted(_digest_mapping(value["baseline_file_hashes"]).items())),
        harness_file_hashes=tuple(sorted(_digest_mapping(value["harness_file_hashes"]).items())),
        harness_sha256=_optional_string(value["harness_sha256"]),
        gpu_arch=_string(value["gpu_arch"]),
        source_scope=_mapping(value["source_scope"]),
        budget=_mapping(value["budget"]),
        commands=_nested_mapping(value["commands"]),
        measurement=_optional_mapping(value["measurement"]),
        recipe_claim=_optional_mapping(value["recipe_claim"]),
        policies=_string_mapping(value["policies"]),
    )
    return draft


def _load_repository(
    value: Mapping[str, Any], *, root: Path
) -> WorkspaceRepositoryIdentity:
    _exact(
        value,
        {"root_sha256", "status", "remote", "commit", "tree", "dirty_paths", "unavailable_reason"},
    )
    if _string(value["root_sha256"]) != sha256_bytes(str(root).encode("utf-8")):
        _invalid()
    dirty = value["dirty_paths"]
    if not isinstance(dirty, list) or any(not isinstance(item, str) for item in dirty):
        _invalid()
    return WorkspaceRepositoryIdentity(
        root=str(root),
        status=_string(value["status"]),
        remote=_optional_string(value["remote"]),
        commit=_optional_string(value["commit"]),
        tree=_optional_string(value["tree"]),
        dirty_paths=tuple(dirty),
        unavailable_reason=_optional_string(value["unavailable_reason"]),
    )


def _load_authority(value: object) -> EvaluationAuthorityReceipt | None:
    if value is None:
        return None
    mapping = _mapping(value)
    _exact(
        mapping,
        {"schema", "authority_id", "authority_kind", "issuer", "policy_sha256", "template_sha256", "draft_digest"},
    )
    if mapping["schema"] != "apex.evaluation-authority-receipt/v1":
        _invalid()
    try:
        kind = EvaluationAuthorityKind(_string(mapping["authority_kind"]))
    except ValueError:
        _invalid()
    identity = EvaluationAuthorityIdentity(
        authority_id=_string(mapping["authority_id"]),
        kind=kind,
        issuer=_string(mapping["issuer"]),
        policy_sha256=_string(mapping["policy_sha256"]),
        template_sha256=_string(mapping["template_sha256"]),
    )
    return EvaluationAuthorityReceipt(identity, _string(mapping["draft_digest"]))


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _invalid()
    return dict(value)


def _optional_mapping(value: object) -> dict[str, Any] | None:
    return None if value is None else _mapping(value)


def _nested_mapping(value: object) -> dict[str, Mapping[str, Any]]:
    return {key: _mapping(item) for key, item in _mapping(value).items()}


def _string_mapping(value: object) -> dict[str, str]:
    mapping = _mapping(value)
    if any(not isinstance(item, str) for item in mapping.values()):
        _invalid()
    return dict(mapping)


def _digest_mapping(value: object) -> dict[str, str]:
    mapping = _string_mapping(value)
    if any(len(item) != 64 or set(item) - set("0123456789abcdef") for item in mapping.values()):
        _invalid()
    return mapping


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        _invalid()
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _exact(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        _invalid()


def _invalid() -> None:
    raise ContractError(
        "Evaluation contract artifact is malformed",
        "invalid_evaluation_contract",
    )


__all__ = ["load_evaluation_contract"]
