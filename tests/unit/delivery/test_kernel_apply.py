from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import apex.delivery.kernel_apply as kernel_apply
from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes
from apex.delivery import apply_verified_kernel_bundle, build_kernel_bundle
from apex.intake import TaskResolver, TaskSpec


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _workspace(tmp_path: Path, *, source: str = "VALUE = 1\n") -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Apex Test")
    _git(root, "config", "user.email", "apex@example.com")
    (root / "kernel.py").write_text(source, encoding="utf-8")
    _git(root, "add", "kernel.py")
    _git(root, "commit", "-q", "-m", "baseline")
    return root


def _bundle(tmp_path: Path, workspace: Path):
    task = TaskSpec.from_mapping(
        {
            "task_id": "apply-test",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize kernel",
            "language": "triton",
            "editable_files": ["kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                "compile": {"argv": ["true"]},
                "correctness": {"argv": ["true"]},
                "performance": {"argv": ["true"]},
            },
        }
    )
    resolved = TaskResolver().resolve(task)
    candidate = tmp_path / "candidate"
    shutil.copytree(workspace, candidate, ignore=shutil.ignore_patterns(".git"))
    (candidate / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    return build_kernel_bundle(
        resolved,
        candidate_root=candidate,
        bundle_dir=tmp_path / "bundle",
    )


def test_apply_mutates_only_exact_clean_baseline(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _bundle(tmp_path, workspace)

    receipt = apply_verified_kernel_bundle(
        bundle.path,
        workspace,
        expected_digest=bundle.digest,
    )

    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert receipt.applied is True
    assert receipt.changed_files == ("kernel.py",)
    assert set(receipt.applied_file_hashes) == {"kernel.py"}


def test_apply_rejects_dirty_workspace_before_mutation(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _bundle(tmp_path, workspace)
    (workspace / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        apply_verified_kernel_bundle(bundle.path, workspace)

    assert raised.value.reason_code == "dirty_apply_workspace"
    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_rejects_clean_but_different_baseline(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _bundle(tmp_path, workspace)
    (workspace / "kernel.py").write_text("VALUE = 9\n", encoding="utf-8")
    _git(workspace, "add", "kernel.py")
    _git(workspace, "commit", "-q", "-m", "different baseline")

    with pytest.raises(IntegrityError) as raised:
        apply_verified_kernel_bundle(bundle.path, workspace)

    assert raised.value.reason_code == "bundle_baseline_mismatch"
    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 9\n"


def test_apply_rolls_back_if_post_apply_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    bundle = _bundle(tmp_path, workspace)

    def fail_verification(*_args, **_kwargs):
        raise IntegrityError("forced verification failure", "forced_failure")

    monkeypatch.setattr(kernel_apply, "_verify_applied", fail_verification)
    with pytest.raises(IntegrityError) as raised:
        apply_verified_kernel_bundle(bundle.path, workspace)

    assert raised.value.reason_code == "forced_failure"
    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_rejects_self_consistent_delete_patch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _bundle(tmp_path, workspace)
    manifest_path = bundle.path / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch_path = bundle.path / manifest["patches"][0]["path"]
    content = b"--- a/kernel.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-VALUE = 1\n"
    patch_path.write_bytes(content)
    manifest["patches"][0]["sha256"] = sha256_bytes(content)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(IntegrityError) as raised:
        apply_verified_kernel_bundle(bundle.path, workspace)

    assert raised.value.reason_code == "unsupported_bundle_patch"
    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_apply_rejects_hardlinked_patch_payload(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    bundle = _bundle(tmp_path, workspace)
    patch_path = bundle.path / bundle.manifest["patches"][0]["path"]
    external = tmp_path / "external.patch"
    patch_path.replace(external)
    os.link(external, patch_path)

    with pytest.raises(IntegrityError) as raised:
        apply_verified_kernel_bundle(bundle.path, workspace)

    assert raised.value.reason_code == "bundle_hardlink"
    assert (workspace / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"
