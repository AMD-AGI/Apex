"""Exact source checkout locks for formal E2E delivery."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .repositories import (
    BootstrapError,
    RepositoryResolver,
    ResolvedRepository,
    inspect_repository,
    repository_errors,
)


SOURCE_LOCK_SCHEMA = "apex.e2e-source-locks"
SOURCE_LOCK_VERSION = 1
GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclass(frozen=True, slots=True)
class SourceLockSpec:
    """One exact source repository needed by a formal E2E profile."""

    key: str
    name: str
    repository: str
    commit: str
    tree: str
    sibling: str
    managed_checkout: str
    root_env: str


@dataclass(frozen=True, slots=True)
class SourceLockSet:
    """Strictly parsed source lock plus its checked-in content digest."""

    path: Path
    receipt_schema: str
    sources: tuple[SourceLockSpec, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceCheckoutReceipt:
    """Verified checkout identity selected for one source."""

    name: str
    root: Path
    repository: str
    commit: str
    tree: str
    resolution: str
    observed_remote: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "repository": self.repository,
            "observed_remote": self.observed_remote or self.repository,
            "commit": self.commit,
            "tree": self.tree,
            "resolution": self.resolution,
            "dirty": False,
        }


@dataclass(frozen=True, slots=True)
class SourceLockReceipt:
    """Evaluator-consumable proof of all exact E2E source checkouts."""

    schema: str
    lock_path: Path
    lock_sha256: str
    sources: Mapping[str, SourceCheckoutReceipt]

    @property
    def roots(self) -> Mapping[str, Path]:
        return {key: value.root for key, value in self.sources.items()}

    def root(self, name: str) -> Path:
        try:
            return self.sources[name].root
        except KeyError as error:
            raise BootstrapError(f"source lock is absent from receipt: {name}") from error

    def to_dict(self, *, status: str = "verified") -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": status,
            "lock": str(self.lock_path),
            "lock_sha256": self.lock_sha256,
            "sources": {
                key: value.to_dict() for key, value in self.sources.items()
            },
        }


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_relative(value: Any, field: str) -> str:
    text = _required_string(value, field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise BootstrapError(f"{field} must be a safe relative path: {text!r}")
    return text


def _parse_source(key: str, raw: Any) -> SourceLockSpec:
    if not isinstance(raw, dict):
        raise BootstrapError(f"sources.{key} must be an object")
    commit = _required_string(raw.get("commit"), f"sources.{key}.commit")
    tree = _required_string(raw.get("tree"), f"sources.{key}.tree")
    if not GIT_OBJECT.fullmatch(commit) or not GIT_OBJECT.fullmatch(tree):
        raise BootstrapError(
            f"sources.{key}.commit and tree must be lowercase 40-hex Git objects"
        )
    root_env = _required_string(raw.get("root_env"), f"sources.{key}.root_env")
    if not ENV_NAME.fullmatch(root_env):
        raise BootstrapError(f"invalid source root environment name: {root_env!r}")
    return SourceLockSpec(
        key=key,
        name=_required_string(raw.get("name"), f"sources.{key}.name"),
        repository=_required_string(
            raw.get("repository"), f"sources.{key}.repository"
        ),
        commit=commit,
        tree=tree,
        sibling=_safe_relative(raw.get("sibling"), f"sources.{key}.sibling"),
        managed_checkout=_safe_relative(
            raw.get("managed_checkout"), f"sources.{key}.managed_checkout"
        ),
        root_env=root_env,
    )


def load_source_lock(path: Path) -> SourceLockSet:
    """Read and strictly validate the formal E2E source lock."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise BootstrapError(f"cannot read source lock {path}: {error}") from error
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise BootstrapError(f"invalid source lock JSON {path}: {error}") from error
    if not isinstance(raw, dict):
        raise BootstrapError("source lock root must be an object")
    if raw.get("schema") != SOURCE_LOCK_SCHEMA or raw.get("version") != SOURCE_LOCK_VERSION:
        raise BootstrapError(
            f"unsupported source lock; expected {SOURCE_LOCK_SCHEMA} v{SOURCE_LOCK_VERSION}"
        )
    sources = raw.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise BootstrapError("sources must be a non-empty object")
    parsed = tuple(_parse_source(key, value) for key, value in sources.items())
    return SourceLockSet(
        path=path.expanduser().resolve(),
        receipt_schema=_required_string(raw.get("receipt_schema"), "receipt_schema"),
        sources=parsed,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def default_source_lock_path(apex_root: Path) -> Path:
    return apex_root.resolve() / "scripts" / "e2e_source_locks.json"


def default_source_checkout_root() -> Path:
    override = os.environ.get("APEX_SOURCE_LOCK_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "apex" / "source-locks").resolve()


def default_source_roots(
    lock: SourceLockSet, checkout_root: Path | None = None
) -> Mapping[str, Path]:
    root = (checkout_root or default_source_checkout_root()).expanduser().resolve()
    return {item.key: root / item.managed_checkout for item in lock.sources}


class SourceLockManager:
    """Materialize or verify exact source locks without touching sibling repos."""

    def __init__(
        self,
        lock: SourceLockSet,
        *,
        sibling_root: Path,
        checkout_root: Path,
        explicit_roots: Mapping[str, Path],
        offline: bool,
    ) -> None:
        self.lock = lock
        self.sibling_root = sibling_root.expanduser().resolve()
        self.checkout_root = checkout_root.expanduser().resolve()
        self.explicit_roots = explicit_roots
        self.offline = offline

    def materialize(self) -> SourceLockReceipt:
        """Clone missing managed pins, then independently reverify every source."""

        resolver = self._resolver(dry_run=False)
        resolved = {item.key: resolver.resolve(item) for item in self.lock.sources}
        return self._receipt(resolved)

    def plan(self) -> dict[str, Any]:
        """Describe source selection without creating or changing checkouts."""

        resolver = self._resolver(dry_run=True)
        resolved = {item.key: resolver.resolve(item) for item in self.lock.sources}
        sources: dict[str, Any] = {}
        for item in self.lock.sources:
            selected = resolved[item.key]
            sources[item.key] = {
                "name": item.name,
                "root": str(selected.root),
                "repository": item.repository,
                "commit": item.commit,
                "tree": item.tree,
                "resolution": selected.resolution,
                "action": "verify" if selected.state is not None else "materialize",
            }
        return self._result("planned", sources)

    def verify(self) -> SourceLockReceipt:
        """Verify configured roots without cloning, fetching, or changing them."""

        resolved: dict[str, ResolvedRepository] = {}
        defaults = default_source_roots(self.lock, self.checkout_root)
        for item in self.lock.sources:
            root, resolution = self._selected_root(item, defaults[item.key])
            state = inspect_repository(root)
            errors = repository_errors(item, state)
            if errors:
                raise BootstrapError(
                    f"{item.name} {resolution} source is not locked: "
                    + "; ".join(errors)
                )
            resolved[item.key] = ResolvedRepository(state.root, resolution, state)
        return self._receipt(resolved)

    def _selected_root(self, item: SourceLockSpec, default: Path) -> tuple[Path, str]:
        explicit = self.explicit_roots.get(item.key)
        if explicit is not None:
            return explicit.expanduser().resolve(), "explicit"
        environment = os.environ.get(item.root_env)
        if environment:
            return Path(environment).expanduser().resolve(), "environment"
        return default, "managed"

    def _resolver(self, *, dry_run: bool) -> RepositoryResolver:
        return RepositoryResolver(
            sibling_root=self.sibling_root,
            checkout_root=self.checkout_root,
            explicit_roots=self.explicit_roots,
            offline=self.offline,
            dry_run=dry_run,
            materialize_siblings=True,
        )

    def _receipt(
        self, resolved: Mapping[str, ResolvedRepository]
    ) -> SourceLockReceipt:
        sources: dict[str, SourceCheckoutReceipt] = {}
        for item in self.lock.sources:
            selected = resolved[item.key]
            state = inspect_repository(selected.root)
            errors = repository_errors(item, state)
            if errors:
                raise BootstrapError(
                    f"{item.name} source changed during verification: "
                    + "; ".join(errors)
                )
            sources[item.key] = SourceCheckoutReceipt(
                item.name,
                state.root,
                item.repository,
                state.commit,
                state.tree,
                selected.resolution,
                state.remote,
            )
        return SourceLockReceipt(
            self.lock.receipt_schema,
            self.lock.path,
            self.lock.sha256,
            sources,
        )

    def _result(self, status: str, sources: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": self.lock.receipt_schema,
            "status": status,
            "lock": str(self.lock.path),
            "lock_sha256": self.lock.sha256,
            "sources": dict(sources),
        }


__all__ = [
    "SourceCheckoutReceipt",
    "SourceLockManager",
    "SourceLockReceipt",
    "SourceLockSet",
    "SourceLockSpec",
    "default_source_checkout_root",
    "default_source_lock_path",
    "default_source_roots",
    "load_source_lock",
]
