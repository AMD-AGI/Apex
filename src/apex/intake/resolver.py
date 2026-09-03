"""Deterministic validation and resolution of single-kernel task inputs."""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from apex.core import AgentBackendName, ContractError, sha256_file, sha256_json

from .descriptor_loader import load_mapping_document
from .task_intent import NaturalLanguageRequest
from .task_spec import ResolvedTaskSpec, TaskSpec


_SOURCE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:\./)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:py|hip|cpp|cc|cxx)(?![A-Za-z0-9_.-])"
)
_ROOT_DESCRIPTOR_NAMES = (
    "apex-task.yaml",
    "apex-task.yml",
    "apex-task.json",
    "task_spec.yaml",
    "task_spec.yml",
    "task_spec.json",
)


class TaskResolver:
    """Resolve source paths without invoking an agent or arbitrary commands."""

    def resolve(self, task: TaskSpec) -> ResolvedTaskSpec:
        workspace = task.workspace.resolve(strict=True)
        if not workspace.is_dir():
            raise ContractError("workspace is not a directory", "workspace_not_directory")
        results_dir = task.results_dir.resolve(strict=False)
        try:
            results_dir.relative_to(workspace)
        except ValueError:
            pass
        else:
            raise ContractError(
                "results_dir must be outside the source workspace",
                "results_inside_workspace",
                {"workspace": str(workspace), "results_dir": str(results_dir)},
            )

        editable_paths: list[Path] = []
        baseline_hashes: dict[str, str] = {}
        for relative in task.editable_files:
            path = workspace.joinpath(*relative.split("/"))
            self._validate_source(path, workspace, relative)
            editable_paths.append(path)
            baseline_hashes[relative] = sha256_file(path)

        harness_hashes: dict[str, str] = {}
        harness_sha256: str | None = None
        if task.measurement is not None:
            for relative in task.measurement.harness_files:
                path = workspace.joinpath(*relative.split("/"))
                self._validate_harness(path, workspace, relative)
                harness_hashes[relative] = sha256_file(path)
            harness_sha256 = sha256_json(
                {
                    "schema": "apex.kernel-measurement-harness/v1",
                    "files": dict(sorted(harness_hashes.items())),
                }
            )

        resolution = {
            "task": task.to_dict(),
            "workspace": str(workspace),
            "baseline_file_hashes": baseline_hashes,
            "harness_file_hashes": harness_hashes,
            "harness_sha256": harness_sha256,
        }
        return ResolvedTaskSpec(
            task=task,
            workspace=workspace,
            editable_paths=tuple(editable_paths),
            baseline_file_hashes=baseline_hashes,
            harness_file_hashes=harness_hashes,
            harness_sha256=harness_sha256,
            resolution_hash=sha256_json(resolution),
        )

    @staticmethod
    def _validate_source(path: Path, workspace: Path, relative: str) -> None:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ContractError(f"Editable source does not exist: {relative}", "source_missing") from error
        try:
            resolved.relative_to(workspace)
        except ValueError as error:
            raise ContractError(f"Editable source escapes workspace: {relative}", "source_path_escape") from error
        stat = os.lstat(path)
        if path.is_symlink():
            raise ContractError(f"Editable source may not be a symlink: {relative}", "source_symlink")
        if not resolved.is_file():
            raise ContractError(f"Editable source is not a regular file: {relative}", "source_not_file")
        if stat.st_nlink != 1:
            raise ContractError(f"Editable source may not be hard-linked: {relative}", "source_hardlink")

    @classmethod
    def _validate_harness(cls, path: Path, workspace: Path, relative: str) -> None:
        try:
            cls._validate_source(path, workspace, relative)
        except ContractError as error:
            raise ContractError(
                f"Protected measurement harness is invalid: {relative}",
                "measurement_harness_invalid",
                {"path": relative, "cause": error.reason_code},
            ) from error


class NaturalLanguageTaskResolver:
    """Resolve human intent against evaluator-owned task descriptors.

    Natural language may select and describe a task, but it never supplies or
    mutates the trusted compile/correctness/performance policy.  Those fields
    come only from a checked-in descriptor discovered under the workspace.
    """

    def __init__(self, resolver: TaskResolver | None = None) -> None:
        self._resolver = resolver or TaskResolver()

    def resolve(
        self,
        request: NaturalLanguageRequest,
        *,
        backend: AgentBackendName = AgentBackendName.CODEX,
    ) -> ResolvedTaskSpec:
        workspace = request.workspace.resolve(strict=True)
        if not workspace.is_dir():
            raise ContractError("workspace is not a directory", "workspace_not_directory")
        referenced = self._referenced_sources(request.text, workspace)
        descriptors = self._descriptors(workspace, referenced)
        parsed = tuple(self._load_descriptor(path, request, backend) for path in descriptors)
        candidates = self._select(parsed, request.text, referenced)
        if not candidates:
            reason = "task_descriptor_missing" if not descriptors else "target_not_resolved"
            raise ContractError(
                "A trusted task descriptor with compile, correctness, and performance oracles is required",
                reason,
                details={"referenced_sources": sorted(referenced)},
            )
        if len(candidates) != 1:
            raise ContractError(
                "Natural-language request matches multiple kernel tasks",
                "ambiguous_kernel_target",
                details={"task_ids": sorted(task.task_id for task in candidates)},
            )
        return self._resolver.resolve(candidates[0])

    @staticmethod
    def _referenced_sources(text: str, workspace: Path) -> frozenset[str]:
        result: set[str] = set()
        for match in _SOURCE_REFERENCE.finditer(text):
            value = match.group(0).removeprefix("./")
            candidate = workspace.joinpath(*value.split("/"))
            try:
                resolved = candidate.resolve(strict=True)
                relative = resolved.relative_to(workspace).as_posix()
            except (FileNotFoundError, ValueError):
                continue
            if resolved.is_file() and not candidate.is_symlink():
                result.add(relative)
        return frozenset(result)

    @staticmethod
    def _descriptors(workspace: Path, sources: frozenset[str]) -> tuple[Path, ...]:
        candidates: set[Path] = {workspace / name for name in _ROOT_DESCRIPTOR_NAMES}
        task_dir = workspace / ".apex" / "tasks"
        if task_dir.is_dir() and not task_dir.is_symlink():
            for suffix in ("*.yaml", "*.yml", "*.json"):
                candidates.update(task_dir.glob(suffix))
        for source in sources:
            path = workspace.joinpath(*source.split("/"))
            candidates.update(
                {
                    path.with_name(f"{path.name}.apex.yaml"),
                    path.with_name(f"{path.name}.apex.json"),
                    path.with_suffix(".apex.yaml"),
                    path.with_suffix(".apex.json"),
                }
            )
        safe: list[Path] = []
        for path in sorted(candidates):
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                pass
            safe.append(path)
        return tuple(safe)

    @staticmethod
    def _load_descriptor(
        path: Path,
        request: NaturalLanguageRequest,
        backend: AgentBackendName,
    ) -> TaskSpec:
        value = load_mapping_document(
            path,
            reason_code="invalid_task_descriptor",
            document_name="task descriptor",
        )
        data: dict[str, Any] = dict(value)
        declared_workspace = data.get("workspace")
        if declared_workspace not in {None, ".", str(request.workspace), str(request.workspace.resolve())}:
            raise ContractError(
                "Task descriptor workspace does not match the requested workspace",
                "task_descriptor_workspace_mismatch",
            )
        data["workspace"] = str(request.workspace.resolve())
        data["results_dir"] = str(request.results_dir)
        descriptor_instructions = str(data.get("instructions", "")).strip()
        data["instructions"] = (
            request.text.strip()
            if not descriptor_instructions
            else f"{descriptor_instructions}\n\nUser objective:\n{request.text.strip()}"
        )
        data["agent_backend"] = backend.value
        return TaskSpec.from_mapping(data)

    @staticmethod
    def _select(
        tasks: tuple[TaskSpec, ...], text: str, sources: frozenset[str]
    ) -> tuple[TaskSpec, ...]:
        if sources:
            return tuple(task for task in tasks if sources.intersection(task.editable_files))
        lowered = text.casefold()
        named = tuple(
            task
            for task in tasks
            if any(
                re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(symbol.casefold())}(?![A-Za-z0-9_])",
                    lowered,
                )
                for symbol in task.target_functions
            )
        )
        return named or tasks


__all__ = ["NaturalLanguageTaskResolver", "TaskResolver"]
