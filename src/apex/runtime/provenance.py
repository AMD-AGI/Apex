"""Best-effort intake provenance and strict source-lock evidence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.core import ContractError, IntegrityError, sha256_file, sha256_json
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)


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
class RunProvenance:
    schema_version: int
    benchmark_config_path: str
    benchmark_config_sha256: str
    framework: str
    model_id: str
    model_revision: str | None
    gpu_arch: str
    container: ContainerIdentity
    active_components: tuple[str, ...]
    source_locks: tuple[RepositoryLock, ...]
    status: str
    missing_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.status not in {"resolved", "partial", "unresolved"}:
            raise ContractError("Run provenance schema/status is invalid", "invalid_provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_config_path": self.benchmark_config_path,
            "benchmark_config_sha256": self.benchmark_config_sha256,
            "framework": self.framework,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "gpu_arch": self.gpu_arch,
            "container": asdict(self.container),
            "active_components": list(self.active_components),
            "source_locks": [asdict(item) for item in self.source_locks],
            "status": self.status,
            "missing_evidence": list(self.missing_evidence),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def source_delivery_ready(self) -> bool:
        return self.status == "resolved" and bool(self.source_locks)


class ProvenanceResolver:
    """Observe what can be proven without blocking partial diagnostic runs."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=2 * 1024 * 1024)

    def resolve(
        self,
        config_path: Path,
        *,
        gpu_arch: str,
        hints: Mapping[str, Any] | None = None,
    ) -> RunProvenance:
        if not config_path.is_absolute() or not config_path.is_file():
            raise ContractError("Benchmark config must be an absolute file", "invalid_benchmark_config")
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ContractError("Benchmark config is not valid YAML", "invalid_benchmark_config") from error
        if not isinstance(document, Mapping) or not isinstance(document.get("benchmark"), Mapping):
            raise ContractError("Benchmark config lacks benchmark object", "invalid_benchmark_config")
        benchmark = document["benchmark"]
        image = str(benchmark.get("docker_image", ""))
        framework = str(benchmark.get("framework", ""))
        model = str(benchmark.get("model", ""))
        if not image or not framework or not model:
            raise ContractError("Benchmark identity is incomplete", "invalid_benchmark_config")
        chosen_hints = dict(hints or {})
        container = self._inspect_image(image, config_path.parent)
        locks = tuple(
            self._repository_lock(item, config_path.parent)
            for item in chosen_hints.get("source_repositories", ())
        )
        active = _active_components(framework, benchmark.get("envs"))
        model_revision = str(chosen_hints["model_revision"]) if chosen_hints.get("model_revision") else None
        missing: list[str] = []
        if not container.resolved:
            missing.append("image_digest")
        if model_revision is None:
            missing.append("model_revision")
        locked_names = {item.name for item in locks if item.exact}
        for component in active:
            if component not in locked_names:
                missing.append(f"source_lock:{component}")
        missing.append("runtime_loaded_bytes")
        status = "resolved" if not missing else ("partial" if container.resolved else "unresolved")
        return RunProvenance(
            schema_version=1,
            benchmark_config_path=str(config_path),
            benchmark_config_sha256=sha256_file(config_path),
            framework=framework,
            model_id=model,
            model_revision=model_revision,
            gpu_arch=gpu_arch,
            container=container,
            active_components=active,
            source_locks=locks,
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


def _active_components(framework: str, envs: object) -> tuple[str, ...]:
    components = [framework.lower()]
    values = envs if isinstance(envs, Mapping) else {}
    if str(values.get("VLLM_ROCM_USE_AITER", "0")).lower() in {"1", "true", "yes"}:
        components.append("aiter")
    return tuple(dict.fromkeys(components))


__all__ = ["ContainerIdentity", "ProvenanceResolver", "RepositoryLock", "RunProvenance"]
