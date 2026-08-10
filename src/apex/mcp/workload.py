"""Lazy, reward-ineligible inspection of one raw Magpie workload config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from apex.benchmark import BenchmarkConfigViews, build_config_views
from apex.core import ContractError, sha256_file
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    CapabilitySideEffect,
)
from apex.runtime import (
    DependencyReceipt,
    MagpieConfigContract,
    MagpieMainConfigAdapter,
    load_magpie_corpus_manifest,
)

from .scope import CapabilityScope


_SCHEMA = "apex.workload-inspection/v2"
_ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "minLength": 1},
        "scope": {"enum": ["workspace", "results"]},
        "path": {"type": "string", "minLength": 1},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "byte_count": {"type": "integer", "minimum": 0},
    },
    "required": ["role", "scope", "path", "sha256", "byte_count"],
    "additionalProperties": False,
}


def workload_inspect_descriptor() -> CapabilityDescriptor:
    """Describe inspection without touching Magpie, a GPU, or the filesystem."""

    return CapabilityDescriptor(
        capability_id="workload.inspect",
        title="Inspect a Magpie E2E workload",
        summary=(
            "Lazily verify pinned dependencies and freeze phase-specific views of "
            "one workspace Magpie config without launching a benchmark or GPU."
        ),
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                },
                "config": {"type": "string", "minLength": 1},
                "replay_image": {"type": "string", "minLength": 1},
                "source_repository_roots": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
                "model_revision": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{40}$",
                },
                "hf_cache_path": {"type": "string", "minLength": 1},
                "hf_offline": {"type": "boolean"},
            },
            "required": ["run_id", "config"],
            "additionalProperties": False,
        },
        output_schema=_output_schema(),
        side_effects=(
            CapabilitySideEffect.READ_WORKSPACE,
            CapabilitySideEffect.WRITE_RESULTS,
            CapabilitySideEffect.RUN_PROCESS,
        ),
        required_authority=CapabilityAuthority.WORKSPACE_USER,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=180,
        artifact_classes=("magpie_workload_contract",),
        reward_role=CapabilityRewardRole.EVIDENCE_ONLY,
    )


def _output_schema() -> Mapping[str, Any]:
    nullable_digest = {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"}
    return {
        "type": "object",
        "properties": {
            "schema": {"const": _SCHEMA},
            "run_id": {"type": "string"},
            "config": _ARTIFACT_SCHEMA,
            "dependency": {
                "type": "object",
                "properties": {
                    "receipt_schema": {"type": "string"},
                    "lock_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "magpie_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
                },
                "required": ["receipt_schema", "lock_sha256", "magpie_commit"],
                "additionalProperties": False,
            },
            "magpie_config_resolution": _resolution_schema(),
            "corpus": {
                "type": "object",
                "properties": {
                    "benchmark_tree": {"type": ["string", "null"]},
                    "manifest_sha256": nullable_digest,
                    "config_count": {"type": ["integer", "null"], "minimum": 0},
                    "member": {"type": ["boolean", "null"]},
                    "matched_path": {"type": ["string", "null"]},
                },
                "required": [
                    "benchmark_tree",
                    "manifest_sha256",
                    "config_count",
                    "member",
                    "matched_path",
                ],
                "additionalProperties": False,
            },
            "workload": _workload_schema(nullable_digest),
            "view_status": {
                "enum": ["materialized", "capability_upgrade_required"]
            },
            "workload_semantics_sha256": nullable_digest,
            "evaluator_policy_sha256": nullable_digest,
            "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
            "reward_eligible": {"const": False},
        },
        "required": [
            "schema", "run_id", "config", "dependency", "magpie_config_resolution",
            "corpus", "workload",
            "view_status", "workload_semantics_sha256",
            "evaluator_policy_sha256", "artifacts", "reward_eligible",
        ],
        "additionalProperties": False,
    }


def _resolution_schema() -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "plan_schema": {"const": "apex.magpie-main-resolved-plan/v1"},
            "plan_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "capability_schema": {
                "const": "apex.magpie-main-capability-receipt/v1"
            },
            "capability_receipt_sha256": {
                "type": "string", "pattern": "^[0-9a-f]{64}$"
            },
            "raw_config_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "effective_config_sha256": {
                "type": "string", "pattern": "^[0-9a-f]{64}$"
            },
            "scoring_config_sha256": {
                "type": "string", "pattern": "^[0-9a-f]{64}$"
            },
            "phase_views_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "resolution_method_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "status": {
                "enum": ["config_compatible", "capability_upgrade_required"]
            },
        },
        "required": [
            "plan_schema", "plan_sha256", "capability_schema",
            "capability_receipt_sha256", "raw_config_sha256",
            "effective_config_sha256", "scoring_config_sha256",
            "phase_views_sha256", "resolution_method_sha256", "status",
        ],
        "additionalProperties": False,
    }


def _workload_schema(nullable_digest: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "type": "object",
        "properties": {
            "framework": {"type": "string", "minLength": 1},
            "model_identity_sha256": nullable_digest,
            "precision": {"type": "string", "minLength": 1},
            "run_mode": {"type": "string", "minLength": 1},
            "server_lifecycle": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "cleanup": {"type": "boolean"},
                },
                "required": ["enabled", "cleanup"],
                "additionalProperties": False,
            },
            "image_status": {
                "enum": [
                    "immutable", "mutable_locator", "not_applicable",
                    "runtime_selection_required",
                ]
            },
            "measurement_image": {"type": ["string", "null"]},
            "replay_image": {"type": ["string", "null"]},
            "quality_tasks": {"type": "string"},
            "compatibility_status": {
                "enum": ["config_compatible", "capability_upgrade_required"]
            },
            "unavailable_dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "runtime_requirements": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        "required": [
            "framework", "model_identity_sha256", "precision", "run_mode",
            "server_lifecycle", "image_status", "measurement_image",
            "replay_image", "quality_tasks", "compatibility_status",
            "unavailable_dimensions", "runtime_requirements",
        ],
        "additionalProperties": False,
    }


class WorkloadInspectHandler:
    """Invoke exact dependency verification only after the tool is called."""

    def __init__(
        self,
        scope: CapabilityScope,
        dependency_provider: Callable[[], DependencyReceipt],
        config_adapter_factory: Callable[
            [DependencyReceipt], object
        ] = MagpieMainConfigAdapter,
    ) -> None:
        self._scope = scope
        self._dependency_provider = dependency_provider
        self._config_adapter_factory = config_adapter_factory

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        arguments = request.arguments
        run_id = str(arguments["run_id"])
        config = self._regular_file(str(arguments["config"]), "config")
        source_roots = self._directories(arguments.get("source_repository_roots", ()))
        cache = self._optional_directory(arguments.get("hf_cache_path"), "hf_cache_path")
        output = self._scope.claim_output("workload-inspection", run_id)
        receipt = self._dependency_provider()
        adapter = self._config_adapter_factory(receipt)
        if not hasattr(adapter, "resolve"):
            raise ContractError(
                "Magpie main config adapter is invalid",
                "capability_provider_mismatch",
            )
        resolved = adapter.resolve(config)
        if resolved.status != "config_compatible":
            return self._upgrade_result(request, run_id, config, receipt, resolved)
        views = build_config_views(
            config,
            output,
            dependency_receipt=receipt,
            resolved_contract=resolved,
            replay_image=_optional_text(arguments.get("replay_image")),
            source_repository_roots=source_roots,
            model_revision=_optional_text(arguments.get("model_revision")),
            hf_cache_path=cache,
            hf_offline=bool(arguments.get("hf_offline", False)),
        )
        artifacts = self._artifact_receipts(config, views)
        replay = _benchmark(views.replay)
        content = {
            "schema": _SCHEMA,
            "run_id": run_id,
            "config": artifacts[0],
            "dependency": _dependency(receipt),
            "magpie_config_resolution": _resolution_binding(resolved),
            "corpus": _corpus(receipt, views.original_sha256),
            "workload": _workload(
                resolved,
                replay_image=_optional_text(replay.get("docker_image")),
                quality_tasks=views.quality_tasks,
            ),
            "view_status": "materialized",
            "workload_semantics_sha256": views.workload_semantics_sha256,
            "evaluator_policy_sha256": views.evaluator_policy_sha256,
            "artifacts": list(artifacts[1:]),
            "reward_eligible": False,
        }
        return CapabilityResult(
            request.capability_id,
            content,
            artifact_receipts=artifacts,
            reward_eligible=False,
        )

    def _upgrade_result(
        self,
        request: CapabilityRequest,
        run_id: str,
        config: Path,
        receipt: DependencyReceipt,
        resolved: MagpieConfigContract,
    ) -> CapabilityResult:
        artifact = _artifact(self._scope, "magpie_source_config", config)
        content = {
            "schema": _SCHEMA,
            "run_id": run_id,
            "config": artifact,
            "dependency": _dependency(receipt),
            "magpie_config_resolution": _resolution_binding(resolved),
            "corpus": _corpus(receipt, resolved.config_sha256),
            "workload": _workload(
                resolved,
                replay_image=None,
                quality_tasks=_quality_tasks(resolved),
            ),
            "view_status": "capability_upgrade_required",
            "workload_semantics_sha256": None,
            "evaluator_policy_sha256": None,
            "artifacts": [],
            "reward_eligible": False,
        }
        return CapabilityResult(
            request.capability_id,
            content,
            artifact_receipts=(artifact,),
            reward_eligible=False,
        )

    def _regular_file(self, value: str, field: str) -> Path:
        path = self._scope.read_workspace(value)
        if not path.is_file():
            raise ContractError(f"{field} must be a regular file", "invalid_capability_arguments")
        return path

    def _optional_directory(self, value: object, field: str) -> Path | None:
        if value is None:
            return None
        path = self._scope.read_workspace(str(value))
        if not path.is_dir():
            raise ContractError(f"{field} must be a directory", "invalid_capability_arguments")
        return path

    def _directories(self, values: object) -> tuple[Path, ...]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ContractError(
                "source_repository_roots must be a list", "invalid_capability_arguments"
            )
        roots = tuple(
            self._optional_directory(value, "source_repository_roots") for value in values
        )
        if any(root is None for root in roots):
            raise AssertionError("Required source repository root disappeared")
        return tuple(root for root in roots if root is not None)

    def _artifact_receipts(
        self, config: Path, views: BenchmarkConfigViews
    ) -> tuple[Mapping[str, object], ...]:
        paths = (
            ("magpie_source_config", config),
            ("benchmark_original", views.original),
            ("benchmark_measurement_view", views.measurement),
            ("benchmark_diagnostic_view", views.diagnostic),
            ("benchmark_replay_view", views.replay),
        )
        return tuple(_artifact(self._scope, role, path) for role, path in paths)


def _artifact(scope: CapabilityScope, role: str, path: Path) -> Mapping[str, object]:
    label, relative = scope.locator(path)
    return {
        "role": role,
        "scope": label,
        "path": relative,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


def _benchmark(path: Path) -> Mapping[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    benchmark = document.get("benchmark") if isinstance(document, Mapping) else None
    if not isinstance(benchmark, Mapping):
        raise ContractError("Resolved benchmark view is invalid", "capability_result_mismatch")
    return benchmark


def _dependency(receipt: DependencyReceipt) -> Mapping[str, str]:
    return {
        "receipt_schema": receipt.schema,
        "lock_sha256": receipt.lock_sha256,
        "magpie_commit": receipt.commits.get("magpie", ""),
    }


def _resolution_binding(resolved: MagpieConfigContract) -> Mapping[str, str]:
    plan = resolved.plan
    capability = resolved.capability_receipt
    return {
        "plan_schema": str(plan["schema"]),
        "plan_sha256": str(plan["plan_sha256"]),
        "capability_schema": str(capability["schema"]),
        "capability_receipt_sha256": str(capability["receipt_sha256"]),
        "raw_config_sha256": str(plan["raw_config_sha256"]),
        "effective_config_sha256": str(plan["effective_config_sha256"]),
        "scoring_config_sha256": str(plan["scoring_config_sha256"]),
        "phase_views_sha256": str(plan["phase_views_sha256"]),
        "resolution_method_sha256": resolved.resolution_method_sha256,
        "status": resolved.status,
    }


def _corpus(receipt: DependencyReceipt, config_sha256: str) -> Mapping[str, object]:
    raw = receipt.raw.get("magpie_corpus")
    corpus = raw if isinstance(raw, Mapping) else {}
    member, matched = _corpus_member(corpus, config_sha256)
    summary = corpus.get("summary")
    count = summary.get("config_count") if isinstance(summary, Mapping) else None
    return {
        "benchmark_tree": _optional_text(corpus.get("benchmark_tree")),
        "manifest_sha256": _optional_text(corpus.get("manifest_sha256")),
        "config_count": count if isinstance(count, int) and not isinstance(count, bool) else None,
        "member": member,
        "matched_path": matched,
    }


def _corpus_member(
    corpus: Mapping[str, Any], config_sha256: str
) -> tuple[bool | None, str | None]:
    manifest_path = corpus.get("path")
    if not isinstance(manifest_path, str) or not manifest_path:
        return None, None
    manifest = load_magpie_corpus_manifest(Path(manifest_path))
    matches = [item.path for item in manifest.files if item.sha256 == config_sha256]
    return (bool(matches), matches[0] if len(matches) == 1 else None)


def _workload(
    resolved: MagpieConfigContract,
    *,
    replay_image: str | None,
    quality_tasks: str,
) -> Mapping[str, Any]:
    identity = resolved.plan["identity"]
    capability = resolved.capability_receipt
    framework = str(identity["framework"])
    run_mode = str(identity["run_mode"])
    measurement_image = _optional_text(
        resolved.plan["source_runtime"].get("requested_image")
    )
    lifecycle = _lifecycle(str(resolved.plan["lifecycle"]))
    return {
        "framework": framework,
        "model_identity_sha256": identity["model_sha256"],
        "precision": str(identity["precision"]),
        "run_mode": run_mode,
        "server_lifecycle": lifecycle,
        "image_status": _image_status(run_mode, measurement_image),
        "measurement_image": measurement_image,
        "replay_image": replay_image,
        "quality_tasks": quality_tasks,
        "compatibility_status": resolved.status,
        "unavailable_dimensions": list(capability["blockers"]),
        "runtime_requirements": _runtime_requirements(
            run_mode, measurement_image, lifecycle
        ),
    }


def _quality_tasks(resolved: MagpieConfigContract) -> str:
    envs = resolved.scoring_config.get("envs")
    tasks = envs.get("MAGPIE_EVAL_TASKS") if isinstance(envs, Mapping) else None
    return tasks.strip() if isinstance(tasks, str) else ""


def _lifecycle(value: str) -> Mapping[str, bool]:
    return {
        "enabled": value in {"reuse", "cleanup"},
        "cleanup": value == "cleanup",
    }


def _image_status(run_mode: str, image: str | None) -> str:
    if run_mode != "docker":
        return "not_applicable"
    if image is None:
        return "runtime_selection_required"
    if image.startswith("sha256:") or "@sha256:" in image:
        return "immutable"
    return "mutable_locator"


def _runtime_requirements(
    run_mode: str, image: str | None, lifecycle: Mapping[str, bool]
) -> list[str]:
    requirements = ["gpu_topology_receipt"]
    if run_mode == "docker":
        requirements.append(
            "docker_image_resolution" if image is None else "immutable_image_resolution"
        )
    elif run_mode == "ray":
        requirements.append("ray_cluster_runtime_receipt")
    elif run_mode == "local":
        requirements.append("local_runtime_engagement_receipt")
    if lifecycle["enabled"]:
        requirements.append("server_lifecycle_receipt")
    return requirements


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


__all__ = ["WorkloadInspectHandler", "workload_inspect_descriptor"]
