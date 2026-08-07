from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError, canonical_json_bytes
from apex.delivery import build_kernel_bundle, detect_bundle_kind, load_and_verify_kernel_bundle
from apex.intake import TaskResolver, TaskSpec


def _resolved_task(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "source").mkdir(parents=True)
    (workspace / "source" / "kernel.py").write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    task = TaskSpec.from_mapping(
        {
            "task_id": "kernel-task",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize kernel",
            "language": "triton",
            "editable_files": ["source/kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                "compile": {"argv": ["python", "compile.py"]},
                "correctness": {"argv": ["python", "correctness.py"]},
                "performance": {"argv": ["python", "performance.py"]},
            },
        }
    )
    return TaskResolver().resolve(task)


def _candidate(resolved, tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    shutil.copytree(resolved.workspace, candidate)
    (candidate / "source" / "kernel.py").write_text(
        "def kernel(x):\n    return x.contiguous()\n", encoding="utf-8"
    )
    return candidate


def test_build_and_verify_bundle_matches_aka_digest_contract(tmp_path: Path) -> None:
    resolved = _resolved_task(tmp_path)
    bundle = build_kernel_bundle(
        resolved,
        candidate_root=_candidate(resolved, tmp_path),
        bundle_dir=tmp_path / "results" / "bundle",
    )

    verified = load_and_verify_kernel_bundle(bundle.path, expected_digest=bundle.digest)
    manual = hashlib.sha256()
    manual.update(canonical_json_bytes(bundle.manifest))
    for entry in bundle.manifest["patches"]:
        manual.update((bundle.path / entry["path"]).read_bytes())

    assert verified.digest == manual.hexdigest()
    assert verified.changed_files == ("source/kernel.py",)
    assert verified.manifest["delivery"] == {"mode": "bundle", "applied": False}
    assert detect_bundle_kind(bundle.path) == "kernel"
    assert "return x.contiguous()" in (bundle.path / bundle.manifest["patches"][0]["path"]).read_text()


def test_no_change_does_not_create_winner_bundle(tmp_path: Path) -> None:
    resolved = _resolved_task(tmp_path)

    with pytest.raises(ContractError) as raised:
        build_kernel_bundle(
            resolved,
            candidate_root=resolved.workspace,
            bundle_dir=tmp_path / "results" / "bundle",
        )

    assert raised.value.reason_code == "no_changed_files"
    assert not (tmp_path / "results" / "bundle").exists()


def test_patch_tampering_is_detected(tmp_path: Path) -> None:
    resolved = _resolved_task(tmp_path)
    bundle = build_kernel_bundle(
        resolved,
        candidate_root=_candidate(resolved, tmp_path),
        bundle_dir=tmp_path / "bundle",
    )
    patch = bundle.path / bundle.manifest["patches"][0]["path"]
    patch.write_text(patch.read_text() + "tamper\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        load_and_verify_kernel_bundle(bundle.path)

    assert raised.value.reason_code == "bundle_patch_digest_mismatch"


def test_undeclared_file_and_symlink_are_rejected(tmp_path: Path) -> None:
    resolved = _resolved_task(tmp_path)
    bundle = build_kernel_bundle(
        resolved,
        candidate_root=_candidate(resolved, tmp_path),
        bundle_dir=tmp_path / "bundle",
    )
    (bundle.path / "unexpected.txt").write_text("not declared", encoding="utf-8")
    with pytest.raises(IntegrityError) as raised:
        load_and_verify_kernel_bundle(bundle.path)
    assert raised.value.reason_code == "bundle_file_set_mismatch"

    (bundle.path / "unexpected.txt").unlink()
    (bundle.path / "linked.patch").symlink_to(bundle.path / bundle.manifest["patches"][0]["path"])
    with pytest.raises(IntegrityError) as raised:
        load_and_verify_kernel_bundle(bundle.path)
    assert raised.value.reason_code == "bundle_symlink"


def test_hardlinked_patch_is_rejected(tmp_path: Path) -> None:
    resolved = _resolved_task(tmp_path)
    bundle = build_kernel_bundle(
        resolved,
        candidate_root=_candidate(resolved, tmp_path),
        bundle_dir=tmp_path / "bundle",
    )
    patch = bundle.path / bundle.manifest["patches"][0]["path"]
    external = tmp_path / "external.patch"
    patch.replace(external)
    os.link(external, patch)

    with pytest.raises(IntegrityError) as raised:
        load_and_verify_kernel_bundle(bundle.path)

    assert raised.value.reason_code == "bundle_hardlink"


def test_manifest_extra_file_injection_changes_expected_set(tmp_path: Path) -> None:
    resolved = _resolved_task(tmp_path)
    bundle = build_kernel_bundle(
        resolved,
        candidate_root=_candidate(resolved, tmp_path),
        bundle_dir=tmp_path / "bundle",
    )
    manifest_path = bundle.path / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["patches"].append({"path": "patches/missing.patch", "sha256": "0" * 64})
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(IntegrityError) as raised:
        load_and_verify_kernel_bundle(bundle.path)

    assert raised.value.reason_code == "bundle_file_set_mismatch"
