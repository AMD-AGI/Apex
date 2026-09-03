"""Pinned Git repository resolution for external Apex dependencies."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from urllib.parse import urlparse


class BootstrapError(RuntimeError):
    """A deterministic dependency bootstrap or verification failure."""


class DependencySpec(Protocol):
    """Repository fields required by :class:`RepositoryResolver`."""

    key: str
    name: str
    repository: str
    commit: str
    sibling: str
    managed_checkout: str
    root_env: str


@dataclass(frozen=True)
class RepositoryState:
    """Observed immutable identity of a Git checkout."""

    root: Path
    commit: str
    tree: str
    remote: str
    dirty_paths: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedRepository:
    """Dependency source selected by the resolver."""

    root: Path
    resolution: str
    state: RepositoryState | None
    clone_source: str | None = None


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run argv without a shell and raise a concise deterministic error."""

    try:
        result = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise BootstrapError(f"cannot execute {argv[0]!r}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        rendered = " ".join(argv)
        raise BootstrapError(
            f"command failed ({result.returncode}): {rendered}"
            + (f"\n{detail}" if detail else "")
        )
    return result


def _git(root: Path, *args: str) -> str:
    return run_command(("git", "-C", str(root), *args)).stdout.strip()


def canonical_repository(value: str) -> str:
    """Normalize HTTPS, SSH and local Git locations for identity comparison."""

    text = value.strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return str(Path(parsed.path).expanduser().resolve()).rstrip("/")
    if parsed.scheme and parsed.hostname:
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"{parsed.hostname.lower()}/{path.lower()}"
    scp = re.fullmatch(r"(?:[^@]+@)?([^:]+):(.+)", text)
    if scp and not Path(text).is_absolute():
        path = scp.group(2).strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"{scp.group(1).lower()}/{path.lower()}"
    return str(Path(text).expanduser().resolve()).rstrip("/")


def inspect_repository(root: Path) -> RepositoryState:
    """Read commit, origin and dirty state from a Git checkout."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise BootstrapError(f"dependency checkout does not exist: {root}")
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise BootstrapError(f"dependency path is not a Git worktree: {root}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return RepositoryState(
        root=root,
        commit=_git(root, "rev-parse", "HEAD"),
        tree=_git(root, "rev-parse", "HEAD^{tree}"),
        remote=_git(root, "remote", "get-url", "origin"),
        dirty_paths=tuple(line for line in status.splitlines() if line),
    )


def repository_errors(
    dependency: DependencySpec, state: RepositoryState
) -> tuple[str, ...]:
    """Return all lock mismatches for an observed repository."""

    errors: list[str] = []
    if state.commit != dependency.commit:
        errors.append(f"commit={state.commit}, expected={dependency.commit}")
    expected_tree = getattr(dependency, "tree", None)
    if expected_tree is not None and state.tree != expected_tree:
        errors.append(f"tree={state.tree}, expected={expected_tree}")
    if canonical_repository(state.remote) != canonical_repository(dependency.repository):
        errors.append(f"origin={state.remote!r}, expected={dependency.repository!r}")
    if state.dirty_paths:
        errors.append(f"dirty worktree ({', '.join(state.dirty_paths[:5])})")
    return tuple(errors)


def repository_contains(root: Path, commit: str) -> bool:
    """Return whether a local repository contains an exact commit object."""

    try:
        run_command(("git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"))
    except BootstrapError:
        return False
    return True


class RepositoryResolver:
    """Resolve exact dependency sources without mutating sibling checkouts."""

    def __init__(
        self,
        *,
        sibling_root: Path,
        checkout_root: Path,
        explicit_roots: Mapping[str, Path],
        offline: bool,
        dry_run: bool,
        materialize_siblings: bool = False,
    ) -> None:
        self.sibling_root = sibling_root.resolve()
        self.checkout_root = checkout_root.resolve()
        self.explicit_roots = explicit_roots
        self.offline = offline
        self.dry_run = dry_run
        self.materialize_siblings = materialize_siblings

    def resolve(self, dependency: DependencySpec) -> ResolvedRepository:
        """Select or materialize one checkout matching the dependency lock."""

        explicit = self.explicit_roots.get(dependency.key)
        if explicit is None and os.environ.get(dependency.root_env):
            explicit = Path(os.environ[dependency.root_env])
        if explicit is not None:
            return self._require_exact(dependency, explicit, "explicit")

        managed = self.checkout_root / dependency.managed_checkout
        if self.materialize_siblings and managed.exists():
            return self._require_exact(dependency, managed, "managed")

        sibling = self.sibling_root / dependency.sibling
        sibling_state: RepositoryState | None = None
        if sibling.exists():
            try:
                sibling_state = inspect_repository(sibling)
            except BootstrapError as exc:
                raise BootstrapError(
                    f"invalid sibling checkout for {dependency.name}: {exc}"
                ) from exc
            if (
                not self.materialize_siblings
                and not repository_errors(dependency, sibling_state)
            ):
                return ResolvedRepository(
                    root=sibling_state.root,
                    resolution="sibling",
                    state=sibling_state,
                )

        if managed.exists():
            state = inspect_repository(managed)
            if not repository_errors(dependency, state):
                return ResolvedRepository(managed.resolve(), "managed", state)
            managed = self.checkout_root / (
                f"{dependency.managed_checkout}-{dependency.commit[:12]}"
            )
            if managed.exists():
                return self._require_exact(dependency, managed, "managed-versioned")

        clone_source, resolution = self._select_clone_source(
            dependency, sibling_state
        )
        if clone_source is None:
            observed = "missing"
            if sibling_state is not None:
                observed = "; ".join(repository_errors(dependency, sibling_state))
            raise BootstrapError(
                f"offline resolution failed for {dependency.name}: sibling is {observed}, "
                "and no exact managed checkout exists"
            )
        if self.dry_run:
            return ResolvedRepository(
                root=managed,
                resolution=f"planned-{resolution}",
                state=None,
                clone_source=clone_source,
            )
        return self._clone_exact(dependency, clone_source, managed, resolution)

    def _select_clone_source(
        self,
        dependency: DependencySpec,
        sibling: RepositoryState | None,
    ) -> tuple[str | None, str]:
        if (
            sibling is not None
            and canonical_repository(sibling.remote)
            == canonical_repository(dependency.repository)
            and repository_contains(sibling.root, dependency.commit)
        ):
            return str(sibling.root), "sibling-clone"
        if not self.offline:
            return dependency.repository, "remote-clone"
        return None, "unavailable"

    def _require_exact(
        self, dependency: DependencySpec, root: Path, resolution: str
    ) -> ResolvedRepository:
        state = inspect_repository(root)
        errors = repository_errors(dependency, state)
        if errors:
            raise BootstrapError(
                f"{dependency.name} {resolution} checkout is not locked: "
                + "; ".join(errors)
            )
        return ResolvedRepository(root=state.root, resolution=resolution, state=state)

    def _clone_exact(
        self,
        dependency: DependencySpec,
        source: str,
        target: Path,
        resolution: str,
    ) -> ResolvedRepository:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{dependency.key}-", dir=target.parent)
        )
        checkout = temporary / dependency.managed_checkout
        try:
            argv = ["git", "clone", "--no-checkout"]
            if Path(source).is_absolute():
                argv.append("--no-hardlinks")
            argv.extend((source, str(checkout)))
            run_command(argv)
            run_command(
                ("git", "-C", str(checkout), "checkout", "--detach", dependency.commit)
            )
            run_command(
                (
                    "git",
                    "-C",
                    str(checkout),
                    "remote",
                    "set-url",
                    "origin",
                    dependency.repository,
                )
            )
            state = inspect_repository(checkout)
            errors = repository_errors(dependency, state)
            if errors:
                raise BootstrapError(
                    f"new {dependency.name} checkout failed lock validation: "
                    + "; ".join(errors)
                )
            if target.exists():
                return self._require_exact(dependency, target, "managed-race")
            checkout.rename(target)
            final_state = inspect_repository(target)
            return ResolvedRepository(
                root=final_state.root,
                resolution=resolution,
                state=final_state,
                clone_source=source,
            )
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "BootstrapError",
    "RepositoryResolver",
    "RepositoryState",
    "ResolvedRepository",
    "canonical_repository",
    "inspect_repository",
    "repository_errors",
    "run_command",
]
