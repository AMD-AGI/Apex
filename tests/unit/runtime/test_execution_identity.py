from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
)
from apex.runtime import ApexExecutionIdentity, collect_apex_execution_identity


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
    )


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "apex"
    package = root / "src" / "apex"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "apex@example.invalid")
    _git(root, "config", "user.name", "Apex Test")
    _git(root, "remote", "add", "origin", "https://github.com/AMD-AGI/Apex.git")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "baseline")
    return root, package


def test_execution_identity_records_dirty_bytes_without_blocking(tmp_path: Path) -> None:
    root, package = _repository(tmp_path)
    clean = collect_apex_execution_identity(root, package_root=package)
    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

    dirty = collect_apex_execution_identity(root, package_root=package)

    assert clean.receipt_sha256 != dirty.receipt_sha256
    assert dirty.document["repository"]["dirty_paths"] != []
    assert ApexExecutionIdentity.from_dict(dirty.to_dict()) == dirty


def test_execution_identity_rejects_tampered_digest(tmp_path: Path) -> None:
    root, package = _repository(tmp_path)
    identity = collect_apex_execution_identity(root, package_root=package)
    value = identity.to_dict()
    value["receipt_sha256"] = "0" * 64

    with pytest.raises(IntegrityError) as caught:
        ApexExecutionIdentity.from_dict(value)
    assert caught.value.reason_code == "execution_identity_tampered"


def test_execution_identity_rejects_partial_unresolved_repository() -> None:
    value = {
        "schema": "apex.execution-identity/v1",
        "repository": {
            "root_sha256": "a" * 64,
            "status": "unresolved",
            "remote": None,
            "commit": "b" * 40,
            "tree": None,
            "dirty_paths": [],
            "unavailable_reason": "repository_identity_unavailable",
        },
        "package": {
            "distribution": "amd-apex-optimizer",
            "version": None,
            "source_manifest_sha256": "c" * 64,
            "file_count": 1,
        },
        "dependency_lock_sha256": None,
        "receipt_sha256": "d" * 64,
    }
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))

    with pytest.raises(IntegrityError) as caught:
        ApexExecutionIdentity.from_dict(value)
    assert caught.value.reason_code == "execution_identity_tampered"


def test_execution_identity_rejects_package_symlink(tmp_path: Path) -> None:
    root, package = _repository(tmp_path)
    (package / "linked.py").symlink_to(package / "__init__.py")

    with pytest.raises(ContractError) as caught:
        collect_apex_execution_identity(root, package_root=package)
    assert caught.value.reason_code == "execution_identity_unavailable"
