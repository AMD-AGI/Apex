"""Strict validation for evaluator-owned safety artifacts."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Mapping, Sequence

from apex.core import ContractError, sha256_file

from .plan import VerificationPlan
from .profile import normalize_relative_path
from .results import EvidenceArtifactReceipt


def validate_artifacts(
    value: object,
    artifact_root: Path,
    plan: VerificationPlan,
) -> tuple[tuple[EvidenceArtifactReceipt, ...], tuple[str, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return (), ("invalid_safety_artifacts",)
    artifacts: list[EvidenceArtifactReceipt] = []
    issues: list[str] = []
    roles: set[str] = set()
    root = artifact_root.resolve(strict=True)
    for raw in value:
        if not isinstance(raw, Mapping):
            issues.append("invalid_safety_artifact")
            continue
        role = raw.get("role")
        relative_value = raw.get("path")
        digest = raw.get("sha256")
        declared_size = raw.get("size")
        if not isinstance(role, str) or not isinstance(relative_value, str):
            issues.append("invalid_safety_artifact")
            continue
        try:
            relative = normalize_relative_path(relative_value, field="artifact_path")
        except ContractError:
            issues.append("safety_artifact_path_escape")
            continue
        path = artifact_root.joinpath(*relative.split("/"))
        try:
            cursor = artifact_root
            has_symlink_component = artifact_root.is_symlink()
            for part in relative.split("/"):
                cursor = cursor / part
                if cursor.is_symlink():
                    has_symlink_component = True
                    break
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError:
            issues.append("missing_safety_artifact")
            continue
        if (
            has_symlink_component
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not resolved.is_relative_to(root)
        ):
            issues.append("safety_artifact_path_escape")
            continue
        observed_digest = sha256_file(path)
        if digest != observed_digest or declared_size != metadata.st_size:
            issues.append("safety_artifact_digest_mismatch")
            continue
        if observed_digest == plan.deployed_digest:
            issues.append("sanitizer_artifact_equals_deployed_artifact")
            continue
        try:
            receipt = EvidenceArtifactReceipt(
                role=role,
                path=f"{artifact_root.name}/{relative}",
                digest=observed_digest,
                size=metadata.st_size,
            )
        except ContractError:
            issues.append("invalid_safety_artifact_receipt")
            continue
        artifacts.append(receipt)
        roles.add(role)
    if "instrumented_artifact" not in roles:
        issues.append("missing_instrumented_safety_artifact")
    artifacts.sort(key=lambda artifact: (artifact.role, artifact.path))
    return tuple(artifacts), tuple(dict.fromkeys(issues))


__all__ = ["validate_artifacts"]
