from __future__ import annotations

import subprocess
from pathlib import Path

from apex.runtime import WorkspaceGitIdentityResolver


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_workspace_git_identity_is_read_only_and_records_dirty_baseline(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(("git", "init", "-q", str(workspace)), check=True)
    _git(workspace, "config", "user.email", "tests@example.invalid")
    _git(workspace, "config", "user.name", "Tests")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/org/repo.git")
    source = workspace / "kernel.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _git(workspace, "add", "kernel.py")
    _git(workspace, "commit", "-q", "-m", "baseline")
    source.write_text("value = 2\n", encoding="utf-8")

    identity = WorkspaceGitIdentityResolver().inspect(workspace)

    assert identity.resolved
    assert identity.remote == "example.invalid/org/repo"
    assert identity.commit == _git(workspace, "rev-parse", "HEAD")
    assert identity.tree == _git(workspace, "rev-parse", "HEAD^{tree}")
    assert identity.dirty_paths == (" M kernel.py",)


def test_non_git_workspace_remains_explicitly_unresolved(tmp_path: Path) -> None:
    identity = WorkspaceGitIdentityResolver().inspect(tmp_path)

    assert identity.resolved is False
    assert identity.unavailable_reason == "repository_identity_unavailable"
