from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError
from apex.delivery import build_kernel_bundle
from apex.intake import TaskResolver, TaskSpec
from apex.mcp import (
    BundleVerifyHandler,
    CapabilityRegistry,
    CapabilityScope,
    planned_capability_descriptors,
)
from apex.ports import CapabilityAuthority, CapabilityRequest


def _bundle(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "source").mkdir(parents=True)
    (workspace / "source" / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    task = TaskSpec.from_mapping(
        {
            "task_id": "mcp-kernel",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize the kernel",
            "language": "triton",
            "editable_files": ["source/kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                name: {"argv": ["python3", f"{name}.py"]}
                for name in ("compile", "correctness", "performance")
            },
        }
    )
    resolved = TaskResolver().resolve(task)
    candidate = tmp_path / "candidate"
    shutil.copytree(workspace, candidate)
    (candidate / "source" / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    bundle = build_kernel_bundle(
        resolved,
        candidate_root=candidate,
        bundle_dir=tmp_path / "results" / "winner",
    )
    return workspace, bundle


def _registry(tmp_path: Path, workspace: Path) -> CapabilityRegistry:
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "bundle.verify"
    )
    registry = CapabilityRegistry()
    registry.register(
        descriptor,
        BundleVerifyHandler(CapabilityScope(workspace, tmp_path / "results")),
    )
    return registry


def test_bundle_verify_capability_uses_official_loader_and_authority(tmp_path: Path) -> None:
    workspace, bundle = _bundle(tmp_path)
    registry = _registry(tmp_path, workspace)
    request = CapabilityRequest("bundle.verify", {"bundle_path": "winner"})

    with pytest.raises(ContractError) as missing:
        registry.invoke(request)
    assert missing.value.reason_code == "capability_authority_missing"

    result = registry.invoke(
        CapabilityRequest(
            "bundle.verify",
            {"bundle_path": "winner"},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )
    assert result.content["verification"] == {
        "kind": "kernel",
        "digest": bundle.digest,
        "verified": True,
        "task_id": "mcp-kernel",
        "changed_files": ["source/kernel.py"],
    }
    assert result.reward_eligible is False


def test_bundle_verify_capability_rejects_tampering(tmp_path: Path) -> None:
    workspace, bundle = _bundle(tmp_path)
    patch = next((bundle.path / "patches").iterdir())
    patch.write_bytes(patch.read_bytes() + b"tamper")

    with pytest.raises(IntegrityError):
        _registry(tmp_path, workspace).invoke(
            CapabilityRequest(
                "bundle.verify",
                {"bundle_path": "winner"},
                frozenset({CapabilityAuthority.WORKSPACE_USER}),
            )
        )


def test_bundle_verify_capability_rejects_ambiguous_scoped_path(tmp_path: Path) -> None:
    workspace, bundle = _bundle(tmp_path)
    shutil.copytree(bundle.path, workspace / "winner")

    with pytest.raises(ContractError) as raised:
        _registry(tmp_path, workspace).invoke(
            CapabilityRequest(
                "bundle.verify",
                {"bundle_path": "winner"},
                frozenset({CapabilityAuthority.WORKSPACE_USER}),
            )
        )

    assert raised.value.reason_code == "unsafe_capability_path"
