"""Validate Magpie's binding from an input config to the Docker runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_PATCH_VERSION = re.compile(r"v[1-9][0-9]*")
_PATCH_PATH = re.compile(
    r"examples/custom_workflows/inference_analysis/vllm_patches/"
    r"config_vllm_v0\.([1-9][0-9]*)\.0\.patch"
)
_CONTAINER_NAME = re.compile(r"magpie-benchmark-[A-Za-z0-9_.-]+")
_SCHEMA = "magpie.serving-runtime-receipt/v2"
_TRACELENS_SCHEMA = "magpie.tracelens-vllm-runtime/v1"
_KEYS = frozenset(
    {
        "schema",
        "execution_mode",
        "input_config_sha256",
        "input_image",
        "input_image_id",
        "requested_image",
        "resolved_image_id",
        "image_derivation",
        "container_name",
        "docker_argv_sha256",
        "process_succeeded",
        "verified",
        "errors",
    }
)


@dataclass(frozen=True, slots=True)
class ServingRuntimeEvidence:
    """Evaluator-owned interpretation of one Magpie serving runtime receipt."""

    required: bool
    passed: bool
    input_config_sha256: str | None
    requested_image: str | None
    resolved_image_id: str | None
    container_name: str | None
    docker_argv_sha256: str | None
    process_succeeded: bool | None
    error: str | None
    input_image: str | None = None
    input_image_id: str | None = None
    image_derivation: Mapping[str, Any] | None = None


def parse_serving_runtime_evidence(
    report: Mapping[str, Any],
    *,
    expected_config_sha256: str | None,
    expected_requested_image: str | None,
    expected_execution_mode: str | None,
    allow_tracelens_derivation: bool = False,
    expected_tracelens_commit: str | None = None,
    expected_tracelens_tree: str | None = None,
) -> ServingRuntimeEvidence:
    """Fail closed when a Docker benchmark is not bound to its exact input."""

    required = expected_execution_mode == "docker"
    if not required:
        return ServingRuntimeEvidence(
            False, True, None, None, None, None, None, None, None
        )
    receipt = report.get("serving_runtime_receipt")
    error = _receipt_error(
        receipt,
        expected_config_sha256=expected_config_sha256,
        expected_requested_image=expected_requested_image,
        expected_framework=_string(report.get("framework")),
        allow_tracelens_derivation=allow_tracelens_derivation,
        expected_tracelens_commit=expected_tracelens_commit,
        expected_tracelens_tree=expected_tracelens_tree,
    )
    data = receipt if isinstance(receipt, Mapping) else {}
    return ServingRuntimeEvidence(
        True,
        error is None,
        _string(data.get("input_config_sha256")),
        _string(data.get("requested_image")),
        _string(data.get("resolved_image_id")),
        _string(data.get("container_name")),
        _string(data.get("docker_argv_sha256")),
        (
            data.get("process_succeeded")
            if isinstance(data.get("process_succeeded"), bool)
            else None
        ),
        error,
        _string(data.get("input_image")),
        _string(data.get("input_image_id")),
        (
            dict(data["image_derivation"])
            if isinstance(data.get("image_derivation"), Mapping)
            else None
        ),
    )


def _receipt_error(
    value: object,
    *,
    expected_config_sha256: str | None,
    expected_requested_image: str | None,
    expected_framework: str | None,
    allow_tracelens_derivation: bool,
    expected_tracelens_commit: str | None,
    expected_tracelens_tree: str | None,
) -> str | None:
    if not isinstance(value, Mapping):
        return "serving_runtime_receipt_missing"
    if frozenset(value) != _KEYS:
        return "serving_runtime_receipt_key_set_mismatch"
    config = value.get("input_config_sha256")
    input_image = value.get("input_image")
    input_image_id = value.get("input_image_id")
    requested = value.get("requested_image")
    errors = value.get("errors")
    if (
        value.get("schema") != _SCHEMA
        or value.get("execution_mode") != "docker"
        or not isinstance(config, str)
        or not _SHA256.fullmatch(config)
        or not isinstance(input_image, str)
        or not input_image
        or not isinstance(input_image_id, str)
        or not _IMAGE_ID.fullmatch(input_image_id)
        or not isinstance(requested, str)
        or not requested
        or not isinstance(value.get("resolved_image_id"), str)
        or not _IMAGE_ID.fullmatch(value["resolved_image_id"])
        or not isinstance(value.get("container_name"), str)
        or not _CONTAINER_NAME.fullmatch(value["container_name"])
        or not isinstance(value.get("docker_argv_sha256"), str)
        or not _SHA256.fullmatch(value["docker_argv_sha256"])
        or value.get("process_succeeded") is not True
        or value.get("verified") is not True
        or not _valid_errors(errors)
        or errors
    ):
        return "serving_runtime_receipt_invalid"
    if config != expected_config_sha256:
        return "serving_runtime_config_mismatch"
    if input_image != expected_requested_image:
        return "serving_runtime_image_mismatch"
    return _image_binding_error(
        value,
        expected_framework=expected_framework,
        allow_tracelens_derivation=allow_tracelens_derivation,
        expected_tracelens_commit=expected_tracelens_commit,
        expected_tracelens_tree=expected_tracelens_tree,
    )


def _image_binding_error(
    value: Mapping[str, Any],
    *,
    expected_framework: str | None,
    allow_tracelens_derivation: bool,
    expected_tracelens_commit: str | None,
    expected_tracelens_tree: str | None,
) -> str | None:
    input_image = str(value["input_image"])
    input_image_id = str(value["input_image_id"])
    requested = str(value["requested_image"])
    resolved = str(value["resolved_image_id"])
    derivation = value.get("image_derivation")
    if not isinstance(derivation, Mapping):
        return "serving_runtime_image_lineage_mismatch"
    if input_image.startswith("sha256:") and input_image != input_image_id:
        return "serving_runtime_image_id_mismatch"
    if derivation.get("kind") == "direct":
        return _direct_lineage_error(
            derivation,
            input_image=input_image,
            input_image_id=input_image_id,
            requested=requested,
            resolved=resolved,
            expected_framework=expected_framework,
        )
    if not allow_tracelens_derivation:
        return "serving_runtime_image_lineage_mismatch"
    return _tracelens_lineage_error(
        derivation,
        input_image=input_image,
        input_image_id=input_image_id,
        requested=requested,
        resolved=resolved,
        expected_framework=expected_framework,
        expected_commit=expected_tracelens_commit,
        expected_tree=expected_tracelens_tree,
    )


_TRACELENS_KEYS = frozenset(
    {
        "kind",
        "framework",
        "runtime_schema",
        "base_image",
        "base_image_id",
        "base_image_locator",
        "derived_image",
        "derived_image_id",
        "tracelens_source_commit",
        "tracelens_source_tree",
        "patch_version",
        "patch_path",
        "patch_sha256",
        "dependency_wheel_manifest_sha256",
        "validator",
        "verified",
    }
)


def _direct_lineage_error(
    value: Mapping[str, Any],
    *,
    input_image: str,
    input_image_id: str,
    requested: str,
    resolved: str,
    expected_framework: str | None,
) -> str | None:
    null_fields = (
        "runtime_schema",
        "tracelens_source_commit",
        "tracelens_source_tree",
        "patch_version",
        "patch_path",
        "patch_sha256",
        "dependency_wheel_manifest_sha256",
    )
    valid = (
        frozenset(value) == _TRACELENS_KEYS
        and isinstance(expected_framework, str)
        and bool(expected_framework)
        and value.get("framework") == expected_framework
        and value.get("base_image") == input_image
        and value.get("base_image_id") == input_image_id
        and value.get("base_image_locator") == input_image
        and value.get("derived_image") == requested == input_image
        and value.get("derived_image_id") == resolved == input_image_id
        and all(value.get(field) is None for field in null_fields)
        and value.get("validator") == "docker-image-id"
        and value.get("verified") is True
    )
    return None if valid else "serving_runtime_image_lineage_mismatch"


def _tracelens_lineage_error(
    value: Mapping[str, Any],
    *,
    input_image: str,
    input_image_id: str,
    requested: str,
    resolved: str,
    expected_framework: str | None,
    expected_commit: str | None,
    expected_tree: str | None,
) -> str | None:
    base_id = value.get("base_image_id")
    base_locator = value.get("base_image_locator")
    commit = value.get("tracelens_source_commit")
    tree = value.get("tracelens_source_tree")
    patch_version = value.get("patch_version")
    patch_path = value.get("patch_path")
    patch_match = (
        _PATCH_PATH.fullmatch(patch_path) if isinstance(patch_path, str) else None
    )
    valid = (
        frozenset(value) == _TRACELENS_KEYS
        and value.get("kind") == "tracelens-derived"
        and value.get("framework") == expected_framework == "vllm"
        and value.get("base_image") == input_image
        and isinstance(base_id, str)
        and bool(_IMAGE_ID.fullmatch(base_id))
        and base_id == input_image_id
        and isinstance(base_locator, str)
        and bool(
            _IMAGE_ID.fullmatch(base_locator)
            or _REPO_DIGEST.fullmatch(base_locator)
        )
        and (not base_locator.startswith("sha256:") or base_locator == base_id)
        and value.get("derived_image") == requested
        and value.get("derived_image_id") == resolved
        and requested != input_image
        and base_id != resolved
        and value.get("runtime_schema") == _TRACELENS_SCHEMA
        and isinstance(commit, str)
        and bool(_COMMIT.fullmatch(commit))
        and commit == expected_commit
        and isinstance(tree, str)
        and bool(_COMMIT.fullmatch(tree))
        and tree == expected_tree
        and isinstance(patch_version, str)
        and bool(_PATCH_VERSION.fullmatch(patch_version))
        and patch_match is not None
        and patch_version == f"v{patch_match.group(1)}"
        and _digest(value.get("patch_sha256"))
        and _digest(value.get("dependency_wheel_manifest_sha256"))
        and value.get("validator") == "vllm-tracelens-runtime-validation/v1"
        and value.get("verified") is True
    )
    return None if valid else "serving_runtime_image_lineage_mismatch"


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _valid_errors(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) <= 16
        and all(isinstance(item, str) and 0 < len(item) <= 256 for item in value)
    ) or value == []


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["ServingRuntimeEvidence", "parse_serving_runtime_evidence"]
