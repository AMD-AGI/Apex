"""Lock, install, verify, and receipt contracts for Apex dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .repositories import (
    BootstrapError,
    RepositoryResolver,
    ResolvedRepository,
    inspect_repository,
    repository_errors,
    run_command,
)


LOCK_SCHEMA = "apex.dependencies.lock"
LOCK_VERSION = 1
HEX_COMMIT = re.compile(r"[0-9a-f]{40}")
MODULE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
DIST_NAME = re.compile(r"[A-Za-z0-9_.-]+")
ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclass(frozen=True)
class LockedDependency:
    """One validated entry from the dependency lock."""

    key: str
    name: str
    repository: str
    commit: str
    sibling: str
    managed_checkout: str
    root_env: str
    distribution: str
    package_version: str
    version_policy: str
    import_root: str
    extras: tuple[str, ...]
    repository_only: bool = False

    @property
    def pip_spec_suffix(self) -> str:
        return "[" + ",".join(self.extras) + "]" if self.extras else ""


@dataclass(frozen=True)
class DependencyLock:
    """Validated dependency lock and its content digest."""

    path: Path
    receipt_schema: str
    dependencies: tuple[LockedDependency, ...]
    sha256: str


@dataclass(frozen=True)
class PythonProbe:
    """Package identity observed from the target Python interpreter."""

    ok: bool
    distribution_version: str | None
    import_file: Path | None
    direct_url: Mapping[str, Any] | None
    error: str | None


def _required_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"{field} must be a non-empty string")
    return value.strip()


def _safe_relative(value: Any, field: str) -> str:
    text = _required_str(value, field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise BootstrapError(f"{field} must be a safe relative path: {text!r}")
    return text


def _parse_python_contract(
    key: str, raw: Mapping[str, Any], *, repository_only: bool
) -> tuple[str, str, str, str, tuple[str, ...]]:
    python = raw.get("python")
    if repository_only:
        if python is not None:
            raise BootstrapError(
                f"dependencies.{key}.python is forbidden for repository-only entries"
            )
        return "", "", "", "", ()
    if not isinstance(python, dict):
        raise BootstrapError(f"dependencies.{key}.python must be an object")
    import_root = _required_str(
        python.get("import_root"), f"dependencies.{key}.python.import_root"
    )
    distribution = _required_str(
        python.get("distribution"), f"dependencies.{key}.python.distribution"
    )
    version_policy = _required_str(
        python.get("version_policy"), f"dependencies.{key}.python.version_policy"
    )
    package_version = _required_str(
        python.get("version"), f"dependencies.{key}.python.version"
    )
    extras = python.get("extras", [])
    if not MODULE_NAME.fullmatch(import_root):
        raise BootstrapError(f"invalid Python import root: {import_root!r}")
    if not DIST_NAME.fullmatch(distribution):
        raise BootstrapError(f"invalid Python distribution name: {distribution!r}")
    if version_policy not in {"exact", "prefix"}:
        raise BootstrapError(
            f"dependencies.{key}.python.version_policy must be exact or prefix"
        )
    if not isinstance(extras, list) or any(
        not isinstance(extra, str) or not DIST_NAME.fullmatch(extra)
        for extra in extras
    ):
        raise BootstrapError(f"dependencies.{key}.python.extras must be names")
    return (
        distribution,
        package_version,
        version_policy,
        import_root,
        tuple(extras),
    )


def _parse_dependency(key: str, raw: Any) -> LockedDependency:
    if not isinstance(raw, dict):
        raise BootstrapError(f"dependencies.{key} must be an object")
    repository_only = raw.get("repository_only", False)
    if not isinstance(repository_only, bool):
        raise BootstrapError(
            f"dependencies.{key}.repository_only must be a boolean"
        )
    commit = _required_str(raw.get("commit"), f"dependencies.{key}.commit")
    if not HEX_COMMIT.fullmatch(commit):
        raise BootstrapError(
            f"dependencies.{key}.commit must be a lowercase 40-hex Git commit"
        )
    distribution, package_version, version_policy, import_root, extras = (
        _parse_python_contract(key, raw, repository_only=repository_only)
    )
    root_env = _required_str(raw.get("root_env"), f"dependencies.{key}.root_env")
    if not ENV_NAME.fullmatch(root_env):
        raise BootstrapError(f"invalid dependency root environment name: {root_env!r}")

    return LockedDependency(
        key=key,
        name=_required_str(raw.get("name"), f"dependencies.{key}.name"),
        repository=_required_str(
            raw.get("repository"), f"dependencies.{key}.repository"
        ),
        commit=commit,
        sibling=_safe_relative(raw.get("sibling"), f"dependencies.{key}.sibling"),
        managed_checkout=_safe_relative(
            raw.get("managed_checkout"), f"dependencies.{key}.managed_checkout"
        ),
        root_env=root_env,
        distribution=distribution,
        package_version=package_version,
        version_policy=version_policy,
        import_root=import_root,
        extras=extras,
        repository_only=repository_only,
    )


def load_lock(path: Path) -> DependencyLock:
    """Load and strictly validate a dependency lock."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BootstrapError(f"cannot read dependency lock {path}: {exc}") from exc
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BootstrapError(f"invalid dependency lock JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BootstrapError("dependency lock root must be an object")
    if raw.get("schema") != LOCK_SCHEMA or raw.get("version") != LOCK_VERSION:
        raise BootstrapError(
            f"unsupported dependency lock; expected {LOCK_SCHEMA} v{LOCK_VERSION}"
        )
    receipt_schema = _required_str(raw.get("receipt_schema"), "receipt_schema")
    dependencies = raw.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise BootstrapError("dependencies must be a non-empty object")
    parsed = tuple(
        _parse_dependency(key, value) for key, value in dependencies.items()
    )
    return DependencyLock(
        path=path.resolve(),
        receipt_schema=receipt_schema,
        dependencies=parsed,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


PROBE_CODE = r"""
import importlib
import importlib.metadata
import json
import pathlib
import sys

distribution, import_root = sys.argv[1:3]
result = {"ok": False}
try:
    dist = importlib.metadata.distribution(distribution)
    module = importlib.import_module(import_root)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{import_root} has no __file__")
    direct_url = dist.read_text("direct_url.json")
    result = {
        "ok": True,
        "distribution_version": dist.version,
        "import_file": str(pathlib.Path(module_file).resolve()),
        "direct_url": json.loads(direct_url) if direct_url else None,
    }
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
"""


def version_matches(installed: str, locked: str, policy: str) -> bool:
    """Apply an exact or PEP 440 base-prefix package version rule."""

    if policy == "exact":
        return installed == locked
    return (
        installed == locked
        or installed.startswith(locked + ".")
        or installed.startswith(locked + "+")
    )


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def probe_errors(
    dependency: LockedDependency,
    probe: PythonProbe,
    expected_root: Path,
) -> tuple[str, ...]:
    """Return distribution-version and import-root mismatches."""

    if dependency.repository_only:
        return ()

    if not probe.ok:
        return (probe.error or "package import failed",)
    assert probe.distribution_version is not None
    assert probe.import_file is not None
    errors: list[str] = []
    if not version_matches(
        probe.distribution_version,
        dependency.package_version,
        dependency.version_policy,
    ):
        errors.append(
            f"distribution version={probe.distribution_version}, "
            f"expected {dependency.version_policy} {dependency.package_version}"
        )
    if not path_is_within(probe.import_file, expected_root):
        errors.append(
            f"import resolved to {probe.import_file}, expected inside {expected_root}"
        )
    return tuple(errors)


class PythonEnvironment:
    """Create a venv, install local sources and prove their import identity."""

    def __init__(self, venv: Path, base_python: str, offline: bool) -> None:
        self.venv = venv.resolve()
        self.base_python = base_python
        self.offline = offline

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    def ensure(self) -> str:
        if self.python.is_file():
            return "existing"
        if self.venv.exists() and any(self.venv.iterdir()):
            raise BootstrapError(
                f"venv path exists but has no bin/python; refusing to overwrite: {self.venv}"
            )
        self.venv.parent.mkdir(parents=True, exist_ok=True)
        run_command((self.base_python, "-m", "venv", str(self.venv)))
        if not self.python.is_file():
            raise BootstrapError(f"venv creation did not produce {self.python}")
        return "created"

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        return env

    def probe(self, dependency: LockedDependency) -> PythonProbe:
        if not self.python.is_file():
            raise BootstrapError(f"Python environment does not exist: {self.python}")
        result = run_command(
            (
                str(self.python),
                "-c",
                PROBE_CODE,
                dependency.distribution,
                dependency.import_root,
            ),
            env=self._env(),
        )
        try:
            raw = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise BootstrapError(
                f"invalid Python probe output for {dependency.name}: {result.stdout!r}"
            ) from exc
        if not isinstance(raw, dict):
            raise BootstrapError(f"invalid Python probe payload for {dependency.name}")
        import_file = raw.get("import_file")
        distribution_version = raw.get("distribution_version")
        return PythonProbe(
            ok=raw.get("ok") is True,
            distribution_version=(
                distribution_version
                if isinstance(distribution_version, str)
                else None
            ),
            import_file=Path(import_file) if isinstance(import_file, str) else None,
            direct_url=raw.get("direct_url") if isinstance(raw.get("direct_url"), dict) else None,
            error=raw.get("error") if isinstance(raw.get("error"), str) else None,
        )

    def install(self, dependency: LockedDependency, root: Path) -> None:
        argv = [
            str(self.python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--editable",
            str(root) + dependency.pip_spec_suffix,
        ]
        if self.offline:
            argv[5:5] = ["--no-index", "--no-deps"]
        run_command(argv, env=self._env())


class DependencyBootstrapper:
    """Coordinate repository and Python-package verification."""

    def __init__(
        self,
        lock: DependencyLock,
        resolver: RepositoryResolver,
        environment: PythonEnvironment,
    ) -> None:
        self.lock = lock
        self.resolver = resolver
        self.environment = environment

    def install(self, *, dry_run: bool) -> dict[str, Any]:
        repositories = self._resolve_all()
        if dry_run:
            planned_actions = {
                item.key: (
                    "verify-repository"
                    if item.repository_only
                    else "verify-or-install"
                )
                for item in self.lock.dependencies
            }
            return self._result(
                "planned",
                repositories,
                venv_action="existing" if self.environment.python.is_file() else "create",
                package_actions=planned_actions,
            )
        venv_action = self.environment.ensure()
        actions: dict[str, str] = {}
        for dependency in self.lock.dependencies:
            resolved = repositories[dependency.key]
            if dependency.repository_only:
                actions[dependency.key] = "repository-verified"
                continue
            if not probe_errors(
                dependency, self.environment.probe(dependency), resolved.root
            ):
                actions[dependency.key] = "already-installed"
                continue
            self.environment.install(dependency, resolved.root)
            errors = probe_errors(
                dependency, self.environment.probe(dependency), resolved.root
            )
            if errors:
                raise BootstrapError(
                    f"{dependency.name} failed post-install verification: "
                    + "; ".join(errors)
                )
            actions[dependency.key] = "installed"
        return self._verified_result(repositories, venv_action, actions)

    def verify(self) -> dict[str, Any]:
        repositories = self._resolve_all()
        if not self.environment.python.is_file():
            raise BootstrapError(f"Python environment does not exist: {self.environment.python}")
        return self._verified_result(
            repositories,
            "existing",
            {
                item.key: (
                    "repository-verified" if item.repository_only else "verified"
                )
                for item in self.lock.dependencies
            },
        )

    def _resolve_all(self) -> dict[str, ResolvedRepository]:
        return {
            dependency.key: self.resolver.resolve(dependency)
            for dependency in self.lock.dependencies
        }

    def _verified_result(
        self,
        repositories: Mapping[str, ResolvedRepository],
        venv_action: str,
        actions: Mapping[str, str],
    ) -> dict[str, Any]:
        probes: dict[str, PythonProbe] = {}
        for dependency in self.lock.dependencies:
            resolved = repositories[dependency.key]
            if resolved.state is None:
                raise BootstrapError(f"{dependency.name} repository was not materialized")
            state = inspect_repository(resolved.root)
            mismatches = repository_errors(dependency, state)
            if mismatches:
                raise BootstrapError(
                    f"{dependency.name} changed during bootstrap: "
                    + "; ".join(mismatches)
                )
            if not dependency.repository_only:
                probe = self.environment.probe(dependency)
                mismatches = probe_errors(dependency, probe, resolved.root)
                if mismatches:
                    raise BootstrapError(
                        f"{dependency.name} Python verification failed: "
                        + "; ".join(mismatches)
                    )
                probes[dependency.key] = probe
        return self._result(
            "verified",
            repositories,
            venv_action=venv_action,
            package_actions=actions,
            probes=probes,
        )

    def _result(
        self,
        status: str,
        repositories: Mapping[str, ResolvedRepository],
        *,
        venv_action: str,
        package_actions: Mapping[str, str],
        probes: Mapping[str, PythonProbe] | None = None,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for dependency in self.lock.dependencies:
            resolved = repositories[dependency.key]
            probe = probes.get(dependency.key) if probes else None
            results[dependency.key] = {
                "name": dependency.name,
                "root": str(resolved.root),
                "resolution": resolved.resolution,
                "repository": dependency.repository,
                "commit": dependency.commit,
                "tree": resolved.state.tree if resolved.state else None,
                "dirty": bool(resolved.state and resolved.state.dirty_paths),
                "repository_only": dependency.repository_only,
                "distribution": dependency.distribution or None,
                "locked_version": dependency.package_version or None,
                "version_policy": dependency.version_policy or None,
                "installed_version": probe.distribution_version if probe else None,
                "import_root": dependency.import_root or None,
                "import_file": str(probe.import_file) if probe and probe.import_file else None,
                "action": package_actions[dependency.key],
                "root_env": dependency.root_env,
            }
        return {
            "schema": self.lock.receipt_schema,
            "status": status,
            "lock": str(self.lock.path),
            "lock_sha256": self.lock.sha256,
            "python": str(self.environment.python),
            "venv_action": venv_action,
            "dependencies": results,
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate executable concerns to the small dependency CLI module."""

    from .dependency_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BootstrapError",
    "DependencyBootstrapper",
    "DependencyLock",
    "LockedDependency",
    "PythonEnvironment",
    "PythonProbe",
    "load_lock",
    "main",
    "probe_errors",
    "version_matches",
]
