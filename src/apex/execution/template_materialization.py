"""Digest-bound container source materialization for reviewed kernel templates."""

from __future__ import annotations

import json
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, IntegrityError, sha256_file, sha256_json
from apex.intake import (
    AgentOptions,
    CommandSpec,
    KernelMeasurementSpec,
    ReviewedKernelTemplate,
    TaskRecipe,
    TaskScope,
    TaskSpec,
    TemplateTaskAuthority,
)
from apex.core import AgentBackendName

from .environment import build_subprocess_environment
from .supervisor import ProcessResult, SubprocessSupervisor


@dataclass(frozen=True, slots=True)
class TemplateImageSourceReceipt:
    immutable_locator: str
    image_id: str
    container_root: str


class TemplateImageSourceRuntime(Protocol):
    def materialize(
        self,
        *,
        immutable_locator: str,
        expected_image_id: str,
        container_root: str,
        destination: Path,
    ) -> TemplateImageSourceReceipt: ...


@dataclass(frozen=True, slots=True)
class TemplateMaterializationReceipt:
    template_id: str
    showcase_id: str
    template_manifest_sha256: str
    immutable_image_locator: str
    image_id: str
    container_root: str
    source_tree_sha256: str
    evaluator_recipe_sha256: str
    workspace: str

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.kernel-template-materialization/v1",
            "materializer_policy": "digest_pinned_container_copy_v1",
            "template_id": self.template_id,
            "showcase_id": self.showcase_id,
            "template_manifest_sha256": self.template_manifest_sha256,
            "immutable_image_locator": self.immutable_image_locator,
            "image_id": self.image_id,
            "container_root": self.container_root,
            "source_tree_sha256": self.source_tree_sha256,
            "evaluator_recipe_sha256": self.evaluator_recipe_sha256,
            "workspace": self.workspace,
        }


@dataclass(frozen=True, slots=True)
class MaterializedKernelTemplate:
    task: TaskSpec
    receipt: TemplateMaterializationReceipt
    receipt_path: Path


class DockerTemplateImageSourceRuntime:
    """Copy exact in-image bytes without starting the image or acquiring a GPU."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor()
        self._environment = build_subprocess_environment({})

    def materialize(
        self,
        *,
        immutable_locator: str,
        expected_image_id: str,
        container_root: str,
        destination: Path,
    ) -> TemplateImageSourceReceipt:
        observed = self._run(("docker", "image", "inspect", "--format", "{{.Id}}", immutable_locator))
        if observed != expected_image_id:
            raise IntegrityError(
                "Template image ID differs from its reviewed manifest",
                "template_image_identity_mismatch",
                {"expected": expected_image_id, "observed": observed},
            )
        container_id = self._run(("docker", "create", immutable_locator, "true"))
        try:
            self._run(("docker", "cp", f"{container_id}:{container_root}/.", str(destination)))
        finally:
            self._run(("docker", "rm", container_id))
        return TemplateImageSourceReceipt(immutable_locator, observed, container_root)

    def _run(self, argv: tuple[str, ...]) -> str:
        result = self._supervisor.run(
            argv,
            cwd=Path.cwd().resolve(),
            environment=self._environment,
            timeout_seconds=300,
        )
        _require_process(result, argv[1])
        return result.stdout.strip()


class KernelTemplateMaterializer:
    """Create an isolated baseline TaskSpec only from a reviewed manifest."""

    def __init__(
        self,
        runtime: TemplateImageSourceRuntime | None = None,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._runtime = runtime or DockerTemplateImageSourceRuntime(supervisor)
        self._supervisor = supervisor or SubprocessSupervisor()
        self._git_environment = build_subprocess_environment(
            {},
            fixed={
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )

    def materialize(
        self,
        template: ReviewedKernelTemplate,
        *,
        results_dir: Path,
        instructions: str,
        backend: AgentBackendName,
        model: str | None = None,
        effort: str | None = None,
    ) -> MaterializedKernelTemplate:
        template.require_materializable()
        evaluator = template.evaluator
        assert evaluator is not None
        results = _results_root(results_dir)
        final_parent = results / "template_workspace"
        receipt_path = results / "template_materialization.json"
        if final_parent.exists() or receipt_path.exists():
            raise ContractError(
                "Template materialization output already exists",
                "template_materialization_exists",
            )
        staging = results / f".template-workspace-{uuid.uuid4().hex}"
        workspace = staging / template.source.workspace_subdir
        workspace.mkdir(parents=True)
        try:
            image = self._runtime.materialize(
                immutable_locator=str(template.runtime.immutable_locator),
                expected_image_id=str(template.runtime.image_id),
                container_root=template.source.container_root,
                destination=workspace,
            )
            source_tree = template_source_tree_sha256(workspace)
            if source_tree != template.source.baseline_tree_sha256:
                raise IntegrityError(
                    "Materialized source tree differs from the reviewed image bytes",
                    "template_source_tree_mismatch",
                    {"expected": template.source.baseline_tree_sha256, "observed": source_tree},
                )
            self._copy_evaluator(template, workspace)
            staging.replace(final_parent)
            final_workspace = final_parent / template.source.workspace_subdir
            self._initialize_repository(final_workspace, template)
            receipt = _receipt(template, image, source_tree, final_workspace)
            task = _task(template, receipt, final_workspace, results, instructions, backend, model, effort)
            _write_receipt(receipt_path, receipt)
            return MaterializedKernelTemplate(task, receipt, receipt_path)
        except Exception:
            receipt_path.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(final_parent, ignore_errors=True)
            raise

    @staticmethod
    def _copy_evaluator(template: ReviewedKernelTemplate, workspace: Path) -> None:
        evaluator = template.evaluator
        assert evaluator is not None
        for relative in evaluator.harness_files:
            source = template.root / "template" / relative
            destination = workspace.joinpath(*relative.split("/"))
            if source.is_symlink() or not source.is_file():
                raise IntegrityError("Template evaluator file is missing", "template_evaluator_mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise IntegrityError("Template evaluator overlaps image source", "template_evaluator_overlap")
            shutil.copyfile(source, destination)
            if sha256_file(source) != sha256_file(destination):
                raise IntegrityError("Template evaluator copy changed", "template_evaluator_mismatch")

    def _initialize_repository(
        self, workspace: Path, template: ReviewedKernelTemplate
    ) -> None:
        if (workspace / ".git").exists() or (workspace / ".git").is_symlink():
            raise IntegrityError(
                "Materialized image source contains repository metadata",
                "unsafe_template_source",
            )
        commands = (
            ("git", "init", "--quiet", "--initial-branch=apex-template-baseline"),
            ("git", "add", "--all"),
            (
                "git", "-c", "user.name=Apex Template Materializer", "-c",
                "user.email=template@apex.invalid", "commit", "--quiet", "--no-gpg-sign",
                "-m", f"Materialize {template.template_id}",
            ),
            (
                "git", "remote", "add", "origin",
                f"https://templates.apex.invalid/{template.template_id}/{template.manifest_sha256}.git",
            ),
        )
        for command in commands:
            result = self._supervisor.run(
                command,
                cwd=workspace,
                environment=self._git_environment,
                timeout_seconds=60,
            )
            _require_process(result, "git")


def template_source_tree_sha256(root: Path) -> str:
    """Hash exact regular source bytes and modes, rejecting filesystem aliases."""

    resolved_root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise IntegrityError("Template source contains a symlink", "unsafe_template_source")
        if path.is_dir():
            continue
        resolved = path.resolve(strict=True)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or not resolved.is_relative_to(resolved_root):
            raise IntegrityError("Template source contains an unsafe file", "unsafe_template_source")
        entries.append({
            "path": relative,
            "sha256": sha256_file(path),
            "size": metadata.st_size,
            "mode": metadata.st_mode & 0o777,
        })
    if not entries:
        raise IntegrityError("Template source tree is empty", "template_source_empty")
    return sha256_json({"schema": "apex.template-source-tree/v1", "files": entries})


def _task(
    template: ReviewedKernelTemplate,
    receipt: TemplateMaterializationReceipt,
    workspace: Path,
    results: Path,
    instructions: str,
    backend: AgentBackendName,
    model: str | None,
    effort: str | None,
) -> TaskSpec:
    evaluator = template.evaluator
    assert evaluator is not None
    authority = TemplateTaskAuthority(
        template.template_id, template.showcase_id, template.manifest_sha256,
        receipt.immutable_image_locator, receipt.image_id, receipt.source_tree_sha256,
        evaluator.recipe_sha256, receipt.digest,
    )
    commands = {
        "compile": CommandSpec(evaluator.compile_argv),
        "correctness": CommandSpec(evaluator.correctness_argv),
        "performance": CommandSpec(evaluator.performance_argv),
    }
    measurement = KernelMeasurementSpec(
        adapter_id=evaluator.adapter_id,
        harness_files=evaluator.harness_files,
        measurement_method_sha256=evaluator.measurement_method_sha256,
        runner=CommandSpec(evaluator.measurement_argv, timeout_seconds=1800),
    )
    return TaskSpec(
        schema_version=1, task_id=template.template_id, workspace=workspace,
        results_dir=results, instructions=instructions, language=template.source.language,
        editable_files=template.source.editable_files,
        target_functions=template.source.target_functions, commands=commands,
        gpu_arch=template.runtime.gpu_arch, mode="template_bound_image_kernel",
        agent_backend=backend, agent_options=AgentOptions(model=model, effort=effort),
        scope=TaskScope(framework=("vllm",)), measurement=measurement,
        recipe=TaskRecipe(
            kind="fixed_hip" if template.source.language == "hip" else "python_triton",
            recipe_id=evaluator.recipe_id, sha256=evaluator.recipe_sha256,
            provenance="trusted_registry",
        ),
        template_authority=authority,
    )


def _receipt(
    template: ReviewedKernelTemplate,
    image: TemplateImageSourceReceipt,
    source_tree: str,
    workspace: Path,
) -> TemplateMaterializationReceipt:
    evaluator = template.evaluator
    assert evaluator is not None
    return TemplateMaterializationReceipt(
        template.template_id, template.showcase_id, template.manifest_sha256,
        image.immutable_locator, image.image_id, image.container_root, source_tree,
        evaluator.recipe_sha256, str(workspace),
    )


def _results_root(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute() or selected.is_symlink():
        raise ContractError("Template results must be an absolute non-symlink path", "invalid_template_results")
    selected.mkdir(parents=True, exist_ok=True)
    return selected.resolve(strict=True)


def _write_receipt(path: Path, receipt: TemplateMaterializationReceipt) -> None:
    content = json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _require_process(result: ProcessResult, operation: str) -> None:
    if (
        result.exit_code != 0 or result.timed_out or result.stdout_truncated
        or result.stderr_truncated or not result.cleanup_succeeded
    ):
        raise ContractError(
            f"Template {operation} process failed",
            "template_materialization_process_failed",
            {"operation": operation, "exit_code": result.exit_code, "timed_out": result.timed_out},
        )


__all__ = [
    "DockerTemplateImageSourceRuntime",
    "KernelTemplateMaterializer",
    "MaterializedKernelTemplate",
    "TemplateImageSourceReceipt",
    "TemplateImageSourceRuntime",
    "TemplateMaterializationReceipt",
    "template_source_tree_sha256",
]
