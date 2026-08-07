from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from apex.core import IntegrityError
from apex.optimization.e2e.candidate import SourceCandidateWorkspace
from apex.optimization.e2e.kernel_lane import KernelOpportunity


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, name: str = "source") -> Path:
    root = tmp_path / name
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "apex@example.test")
    _git(root, "config", "user.name", "Apex Test")
    _git(root, "remote", "add", "origin", str(root))
    return root


def _commit(root: Path, message: str = "anchor") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)


def _opportunity(root: Path) -> KernelOpportunity:
    return KernelOpportunity(
        opportunity_id="kernel-opportunity",
        evidence_id="a" * 64,
        runtime_name="kernel",
        operation_name="attention",
        phase="decode",
        rank=0,
        language="triton",
        origin_library="aiter",
        shape_summary=("[16, 128]",),
        dtypes=("float16",),
        graph_mode="eager",
        match_confidence="active_finder",
        measured_gpu_pct=10.0,
        roi_prior=5.0,
        source_path=root / "kernel.py",
        source_root=root,
        test_file=root / "test_kernel.py",
        test_command="pytest test_kernel.py",
        eligibility="eligible",
        reason_code="eligible",
    )


def test_git_checkout_preserves_safe_symlink_identity(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (root / "shared.py").write_text("shared = True\n", encoding="utf-8")
    os.symlink("shared.py", root / "alias.py")
    _commit(root)

    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    (workspace.root / "kernel.py").write_text("value = 2\n", encoding="utf-8")

    changed, baseline_digest, candidate_digest = workspace.freeze()

    assert changed == ("kernel.py",)
    assert baseline_digest != candidate_digest
    assert (workspace.root / "alias.py").is_symlink()


def test_gitlink_content_is_never_an_editable_side_channel(tmp_path: Path) -> None:
    child = _repository(tmp_path, "child")
    (child / "child.py").write_text("value = 1\n", encoding="utf-8")
    _commit(child)
    child_commit = _git(child, "rev-parse", "HEAD")

    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root, "source")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{child_commit},vendor/sub")
    _git(root, "commit", "-m", "gitlink")
    (root / "vendor" / "sub").mkdir(parents=True)

    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    submodule = workspace.root / "vendor" / "sub"
    submodule.mkdir(parents=True, exist_ok=True)
    (submodule / "agent-created.py").write_text("unsafe = True\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        workspace.freeze()

    assert raised.value.reason_code == "undeclared_agent_edit"
    assert "vendor/sub" in raised.value.details["paths"]


def test_tracked_symlink_may_not_escape_checkout(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    os.symlink("../outside.py", root / "escape.py")
    _commit(root)

    with pytest.raises(IntegrityError) as raised:
        SourceCandidateWorkspace.create(
            _opportunity(root), destination=tmp_path / "candidate"
        )

    assert raised.value.reason_code == "workspace_symlink_escape"


def test_mutable_git_excludes_cannot_hide_agent_files(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    info = workspace.root / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "exclude").write_text("hidden.py\n", encoding="utf-8")
    (workspace.root / "hidden.py").write_text("tamper = True\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        workspace.freeze()

    assert raised.value.reason_code == "undeclared_agent_edit"
    assert "hidden.py" in raised.value.details["paths"]
