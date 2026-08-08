"""Validate Magpie's binding from an input config to the Docker runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"magpie-benchmark-[A-Za-z0-9_.-]+")
_SCHEMA = "magpie.serving-runtime-receipt/v1"
_KEYS = frozenset(
    {
        "schema",
        "execution_mode",
        "input_config_sha256",
        "requested_image",
        "resolved_image_id",
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


def parse_serving_runtime_evidence(
    report: Mapping[str, Any],
    *,
    expected_config_sha256: str | None,
    expected_requested_image: str | None,
    expected_execution_mode: str | None,
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
    )


def _receipt_error(
    value: object,
    *,
    expected_config_sha256: str | None,
    expected_requested_image: str | None,
) -> str | None:
    if not isinstance(value, Mapping):
        return "serving_runtime_receipt_missing"
    if frozenset(value) != _KEYS:
        return "serving_runtime_receipt_key_set_mismatch"
    config = value.get("input_config_sha256")
    requested = value.get("requested_image")
    errors = value.get("errors")
    if (
        value.get("schema") != _SCHEMA
        or value.get("execution_mode") != "docker"
        or not isinstance(config, str)
        or not _SHA256.fullmatch(config)
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
    if requested != expected_requested_image:
        return "serving_runtime_image_mismatch"
    if requested.startswith("sha256:") and requested != value["resolved_image_id"]:
        return "serving_runtime_image_id_mismatch"
    return None


def _valid_errors(value: object) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) <= 16
        and all(isinstance(item, str) and 0 < len(item) <= 256 for item in value)
    ) or value == []


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["ServingRuntimeEvidence", "parse_serving_runtime_evidence"]
