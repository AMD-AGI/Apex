"""Read-only, bounded Git identity for a formal user workspace."""

from __future__ import annotations

from pathlib import Path

from apex.execution import SubprocessSupervisor, build_subprocess_environment
from apex.ports import WorkspaceRepositoryIdentity

from .repositories import canonical_repository


class _IdentityUnavailable(RuntimeError):
    pass


class WorkspaceGitIdentityResolver:
    """Inspect without fetching, modifying Git config, or accepting prompts."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor()
        self._environment = build_subprocess_environment(
            {},
            fixed={
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )

    def inspect(self, workspace: Path) -> WorkspaceRepositoryIdentity:
        root = workspace.resolve(strict=True)
        try:
            top = Path(self._git(root, "rev-parse", "--show-toplevel")).resolve(
                strict=True
            )
            root.relative_to(top)
            commit = self._git(top, "rev-parse", "HEAD")
            tree = self._git(top, "rev-parse", "HEAD^{tree}")
            remote = canonical_repository(
                self._git(top, "remote", "get-url", "origin")
            )
            dirty = tuple(
                item
                for item in self._git(
                    top,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "-z",
                    allow_empty=True,
                ).split("\0")
                if item
            )
        except (OSError, ValueError, _IdentityUnavailable) as error:
            reason = "repository_identity_unavailable"
            if isinstance(error, _IdentityUnavailable) and str(error):
                reason = str(error)
            return WorkspaceRepositoryIdentity(
                str(root), "unresolved", None, None, None, (), reason
            )
        return WorkspaceRepositoryIdentity(
            str(top), "resolved", remote, commit, tree, dirty, None
        )

    def _git(
        self, cwd: Path, *arguments: str, allow_empty: bool = False
    ) -> str:
        result = self._supervisor.run(
            ("git", "-C", str(cwd), *arguments),
            cwd=cwd,
            environment=self._environment,
            timeout_seconds=10,
        )
        if (
            result.exit_code != 0
            or result.timed_out
            or result.stdout_truncated
            or result.stderr_truncated
            or not result.cleanup_succeeded
        ):
            raise _IdentityUnavailable("repository_identity_unavailable")
        value = result.stdout.rstrip("\n")
        if not value and not allow_empty:
            raise _IdentityUnavailable("repository_identity_incomplete")
        return value


__all__ = ["WorkspaceGitIdentityResolver"]
