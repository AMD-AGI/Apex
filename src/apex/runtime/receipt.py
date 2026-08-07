"""Resolve the single verified dependency receipt consumed by all adapters."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import DependencyError

from .dependencies import DependencyBootstrapper, PythonEnvironment, load_lock
from .lm_eval_lock import load_lm_eval_runtime_lock
from .lm_eval_runtime import (
    LmEvalRuntimeReceipt,
    default_lm_eval_runtime_root,
    verify_lm_eval_runtime,
)
from .repositories import BootstrapError, RepositoryResolver
from .source_locks import (
    SourceLockManager,
    SourceLockReceipt,
    default_source_checkout_root,
    default_source_lock_path,
    load_source_lock,
)


@dataclass(frozen=True, slots=True)
class DependencyReceipt:
    schema: str
    lock_sha256: str
    python: Path
    roots: Mapping[str, Path]
    commits: Mapping[str, str]
    raw: Mapping[str, Any]
    lm_eval_runtime: LmEvalRuntimeReceipt | None = None
    source_locks: SourceLockReceipt | None = None

    def root(self, name: str) -> Path:
        try:
            return self.roots[name]
        except KeyError as error:
            raise DependencyError(
                f"Dependency is absent from receipt: {name}", "dependency_missing"
            ) from error

    def source_root(self, name: str) -> Path:
        if self.source_locks is None:
            raise DependencyError(
                "E2E source-lock receipt is absent", "dependency_missing"
            )
        try:
            return self.source_locks.root(name)
        except BootstrapError as error:
            raise DependencyError(str(error), "dependency_missing") from error


def verify_runtime_dependencies(*, apex_root: Path | None = None) -> DependencyReceipt:
    """Verify exact checkouts and imports in the current Python environment."""

    root = (apex_root or Path(__file__).resolve().parents[3]).resolve()
    explicit = {
        key: Path(value)
        for key, value in (
            ("magpie", os.environ.get("MAGPIE_ROOT")),
            ("tracelens", os.environ.get("TRACELENS_REPO_PATH")),
            ("inferencex", os.environ.get("MAGPIE_INFERENCEX_PATH")),
        )
        if value
    }
    try:
        lock = load_lock(root / "scripts" / "dependencies.lock.json")
        resolver = RepositoryResolver(
            sibling_root=root.parent,
            checkout_root=root / ".cache" / "apex-dependencies",
            explicit_roots=explicit,
            offline=True,
            dry_run=False,
        )
        environment = PythonEnvironment(Path(sys.prefix), sys.executable, offline=True)
        raw = DependencyBootstrapper(lock, resolver, environment).verify()
        lm_eval_runtime = _optional_lm_eval_runtime(root)
        source_locks = _verify_e2e_source_locks(root)
    except BootstrapError as error:
        raise DependencyError(
            f"Pinned dependency verification failed: {error}",
            "dependency_receipt_invalid",
        ) from error
    dependencies = raw["dependencies"]
    combined = dict(raw)
    combined["e2e_source_locks"] = source_locks.to_dict()
    return DependencyReceipt(
        schema=str(raw["schema"]),
        lock_sha256=str(raw["lock_sha256"]),
        python=Path(str(raw["python"])),
        roots={key: Path(str(value["root"])) for key, value in dependencies.items()},
        commits={key: str(value["commit"]) for key, value in dependencies.items()},
        raw=combined,
        lm_eval_runtime=lm_eval_runtime,
        source_locks=source_locks,
    )


def _verify_e2e_source_locks(apex_root: Path) -> SourceLockReceipt:
    lock = load_source_lock(default_source_lock_path(apex_root))
    return SourceLockManager(
        lock,
        sibling_root=apex_root.parent,
        checkout_root=default_source_checkout_root(),
        explicit_roots={},
        offline=True,
    ).verify()


def _optional_lm_eval_runtime(apex_root: Path) -> LmEvalRuntimeReceipt | None:
    lock = load_lm_eval_runtime_lock(apex_root / "scripts" / "lm_eval_runtime.lock.json")
    explicit = os.environ.get("APEX_LM_EVAL_RUNTIME")
    runtime_root = (
        Path(explicit).expanduser()
        if explicit
        else default_lm_eval_runtime_root(apex_root, lock)
    )
    if not runtime_root.exists():
        if explicit:
            raise BootstrapError(
                f"explicit lm-eval runtime does not exist: {runtime_root}"
            )
        return None
    return verify_lm_eval_runtime(runtime_root, lock)


__all__ = ["DependencyReceipt", "verify_runtime_dependencies"]
