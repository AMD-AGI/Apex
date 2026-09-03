"""Best-effort intake provenance and strict source-lock evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, sha256_json
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)

from .magpie_config import MagpieConfigContract


_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ContainerIdentity:
    requested_image: str
    image_id: str | None
    repo_digests: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]

    @property
    def resolved(self) -> bool:
        return bool(self.image_id and _IMAGE_ID.fullmatch(self.image_id))


@dataclass(frozen=True, slots=True)
class RepositoryLock:
    name: str
    path: str
    url: str
    commit: str
    tree: str
    clean: bool
    build_recipe_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not Path(self.path).is_absolute():
            raise ContractError("Repository lock identity is invalid", "invalid_repository_lock")
        if not _GIT_OBJECT.fullmatch(self.commit) or not _GIT_OBJECT.fullmatch(self.tree):
            raise ContractError("Repository lock requires exact Git objects", "invalid_repository_lock")

    @property
    def exact(self) -> bool:
        return self.clean


@dataclass(frozen=True, slots=True)
class ComponentSourceLockSet:
    """Exact per-run source locks keyed only by active runtime component."""

    required_components: tuple[str, ...]
    locks: tuple[RepositoryLock, ...]

    def __post_init__(self) -> None:
        if (
            not self.required_components
            or any(not item.strip() for item in self.required_components)
            or len(set(self.required_components)) != len(self.required_components)
        ):
            raise ContractError(
                "Active source components are invalid", "invalid_component_source_locks"
            )
        names = tuple(item.name for item in self.locks)
        if len(set(names)) != len(names) or not set(names).issubset(
            self.required_components
        ):
            raise ContractError(
                "Source locks do not uniquely match active components",
                "invalid_component_source_locks",
            )

    @property
    def exact_components(self) -> frozenset[str]:
        return frozenset(item.name for item in self.locks if item.exact)

    @property
    def missing_exact_components(self) -> tuple[str, ...]:
        return tuple(
            item for item in self.required_components if item not in self.exact_components
        )

    @property
    def ready(self) -> bool:
        return not self.missing_exact_components

    def lock_for(self, component: str) -> RepositoryLock | None:
        return next((item for item in self.locks if item.name == component), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.component-source-lock-set/v1",
            "required_components": list(self.required_components),
            "locks": [asdict(item) for item in self.locks],
            "exact_components": sorted(self.exact_components),
            "missing_exact_components": list(self.missing_exact_components),
        }


@dataclass(frozen=True, slots=True)
class RunProvenance:
    schema_version: int
    benchmark_config_path: str
    benchmark_config_sha256: str
    framework: str
    model_id: str
    model_revision: str | None
    gpu_arch: str
    run_mode: str
    container: ContainerIdentity
    component_sources: ComponentSourceLockSet
    status: str
    missing_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.run_mode not in {"docker", "local", "ray"}
            or self.status not in {"resolved", "partial", "unresolved"}
        ):
            raise ContractError("Run provenance schema/status is invalid", "invalid_provenance")

    @property
    def active_components(self) -> tuple[str, ...]:
        return self.component_sources.required_components

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_config_path": self.benchmark_config_path,
            "benchmark_config_sha256": self.benchmark_config_sha256,
            "framework": self.framework,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "gpu_arch": self.gpu_arch,
            "run_mode": self.run_mode,
            "container": asdict(self.container),
            "active_components": list(self.active_components),
            "component_source_locks": self.component_sources.to_dict(),
            "status": self.status,
            "missing_evidence": list(self.missing_evidence),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def source_delivery_ready(self) -> bool:
        return self.status == "resolved" and self.component_sources.ready


class ProvenanceResolver:
    """Observe what can be proven without blocking partial diagnostic runs."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=2 * 1024 * 1024)

    def resolve(
        self,
        resolved: MagpieConfigContract,
        *,
        gpu_arch: str,
        hints: Mapping[str, Any] | None = None,
    ) -> RunProvenance:
        config_path = resolved.config_path
        identity = resolved.plan["identity"]
        source_runtime = resolved.plan["source_runtime"]
        raw_image = source_runtime.get("requested_image")
        image = raw_image.strip() if isinstance(raw_image, str) else ""
        framework = str(identity["framework"])
        model = str(identity["model"])
        run_mode = str(identity["run_mode"])
        chosen_hints = dict(hints or {})
        container = (
            self._inspect_image(image, config_path.parent)
            if run_mode == "docker" and image
            else ContainerIdentity(image, None, (), ())
        )
        locks = tuple(
            self._repository_lock(item, config_path.parent)
            for item in chosen_hints.get("source_repositories", ())
        )
        active = resolved.requested_components
        component_sources = ComponentSourceLockSet(active, locks)
        model_revision = str(chosen_hints["model_revision"]) if chosen_hints.get("model_revision") else None
        missing: list[str] = []
        if run_mode == "docker" and not image:
            missing.append("runtime_image_selection")
        elif run_mode == "docker" and not container.resolved:
            missing.append("image_digest")
        elif run_mode == "local":
            missing.append("local_runtime_identity")
        elif run_mode == "ray":
            missing.append("ray_worker_runtime_identity")
        if model_revision is None:
            missing.append("model_revision")
        missing.extend(
            f"source_lock:{component}"
            for component in component_sources.missing_exact_components
        )
        missing.append("runtime_loaded_bytes")
        status = "resolved" if not missing else "partial"
        if run_mode == "docker" and image and not container.resolved:
            status = "unresolved"
        return RunProvenance(
            schema_version=2,
            benchmark_config_path=str(config_path),
            benchmark_config_sha256=resolved.config_sha256,
            framework=framework,
            model_id=model,
            model_revision=model_revision,
            gpu_arch=gpu_arch,
            run_mode=run_mode,
            container=container,
            component_sources=component_sources,
            status=status,
            missing_evidence=tuple(sorted(set(missing))),
        )

    def _inspect_image(self, image: str, cwd: Path) -> ContainerIdentity:
        result = self._run(("docker", "image", "inspect", image), cwd, 30)
        if result.exit_code != 0 or result.timed_out:
            return ContainerIdentity(image, None, (), ())
        try:
            value = json.loads(result.stdout)
            item = value[0]
            image_id = str(item.get("Id")) if item.get("Id") else None
            repo_digests = tuple(sorted(str(entry) for entry in item.get("RepoDigests", ()) if entry))
            labels = item.get("Config", {}).get("Labels") or {}
            return ContainerIdentity(
                image,
                image_id if image_id and _IMAGE_ID.fullmatch(image_id) else None,
                repo_digests,
                tuple(sorted((str(key), str(value)) for key, value in labels.items())),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, IndexError):
            return ContainerIdentity(image, None, (), ())

    def _repository_lock(self, value: object, cwd: Path) -> RepositoryLock:
        if not isinstance(value, Mapping):
            raise ContractError("source_repositories entries must be objects", "invalid_repository_hint")
        name = str(value.get("name", ""))
        path = Path(str(value.get("path", "")))
        if not path.is_absolute() or not path.is_dir():
            raise ContractError("Repository hint path is invalid", "invalid_repository_hint")
        commit = self._git(path, "rev-parse", "HEAD")
        tree = self._git(path, "rev-parse", "HEAD^{tree}")
        url = self._git(path, "remote", "get-url", "origin")
        clean = not self._git(path, "status", "--porcelain=v1")
        asserted = value.get("commit")
        if asserted and str(asserted) != commit:
            raise IntegrityError("Repository hint commit does not match", "repository_commit_mismatch")
        recipe = value.get("build_recipe")
        recipe_hash = sha256_json(recipe) if recipe is not None else None
        return RepositoryLock(name, str(path), url, commit, tree, clean, recipe_hash)

    def _git(self, path: Path, *args: str) -> str:
        result = self._run(("git", *args), path, 30)
        if result.exit_code != 0 or result.timed_out:
            raise IntegrityError("Cannot inspect source repository", "repository_inspection_failed")
        return result.stdout.strip()

    def _run(self, argv: tuple[str, ...], cwd: Path, timeout: int) -> ProcessResult:
        environment = build_subprocess_environment(
            inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS,
            fixed={
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
        return self._supervisor.run(
            argv, cwd=cwd, environment=environment, timeout_seconds=timeout
        )


__all__ = [
    "ComponentSourceLockSet",
    "ContainerIdentity",
    "ProvenanceResolver",
    "RepositoryLock",
    "RunProvenance",
]
