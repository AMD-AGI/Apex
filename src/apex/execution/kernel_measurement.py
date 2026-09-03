"""Production parent-captured adapter for standalone kernel raw measurements."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from apex.core import ContractError, IntegrityError, canonical_json_bytes
from apex.ports import (
    KernelMeasurementOutput,
    KernelMeasurementRequest,
)

from .environment import (
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)
from .supervisor import SubprocessSupervisor


STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID = "apex-structured-kernel-v1"
STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256 = (
    "4bb99ecf991a6d28448f46c071bc3c09fbe91aba2cf5f5194e3c1928d96990c1"
)
_MAX_REPORT_BYTES = 64 * 1024 * 1024


class StructuredKernelMeasurementAdapter:
    """Capture one protected runner's JSON and seal it after process teardown."""

    adapter_id = STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID
    measurement_method_sha256 = STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(
            max_output_bytes=_MAX_REPORT_BYTES
        )

    def measure(self, request: KernelMeasurementRequest) -> KernelMeasurementOutput:
        if request.adapter_id != self.adapter_id:
            raise ContractError(
                "Kernel measurement request names another adapter",
                "measurement_adapter_mismatch",
            )
        if (
            request.measurement_method_sha256.removeprefix("sha256:")
            != self.measurement_method_sha256
        ):
            raise IntegrityError(
                "Kernel measurement request method does not match this adapter",
                "measurement_method_mismatch",
            )
        environment = build_subprocess_environment(
            request.runner_env,
            inherit=(*GPU_RUNTIME_ENVIRONMENT_KEYS, *HF_RUNTIME_ENVIRONMENT_KEYS),
        )
        result = self._supervisor.run(
            request.runner_argv,
            cwd=request.runner_cwd,
            environment=environment,
            timeout_seconds=request.runner_timeout_seconds,
            require_pid_namespace=True,
        )
        containment = result.process_containment
        if (
            containment is None
            or not containment.namespace_empty_verified
            or not result.cleanup_succeeded
        ):
            raise IntegrityError(
                "Kernel measurement runner process tree was not contained",
                "measurement_runner_containment_failed",
            )
        if (
            result.timed_out
            or result.exit_code != 0
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise ContractError(
                "Kernel measurement runner did not produce complete output",
                "measurement_runner_failed",
                {
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "stdout_truncated": result.stdout_truncated,
                    "stderr_truncated": result.stderr_truncated,
                },
            )
        document = _strict_document(result.stdout)
        _validate_envelope(document, request)
        _write_private_report(request.report_path, canonical_json_bytes(document))
        return KernelMeasurementOutput(self.adapter_id, request.report_path)


def _strict_document(content: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            content,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ContractError(
            "Kernel measurement runner stdout is not one strict JSON document",
            "invalid_measurement_runner_output",
        ) from error
    if not isinstance(value, Mapping):
        raise ContractError(
            "Kernel measurement runner output must be a JSON object",
            "invalid_measurement_runner_output",
        )
    return value


def _validate_envelope(
    document: Mapping[str, Any], request: KernelMeasurementRequest
) -> None:
    if document.get("schema") != "apex.kernel-measurement/v1":
        raise ContractError(
            "Kernel measurement runner output has the wrong schema",
            "invalid_measurement_runner_output",
        )
    observed = str(document.get("measurement_method_sha256", "")).removeprefix(
        "sha256:"
    )
    if observed != request.measurement_method_sha256.removeprefix("sha256:"):
        raise IntegrityError(
            "Kernel measurement runner output has the wrong method identity",
            "measurement_method_mismatch",
        )


def _write_private_report(path: Path, content: bytes) -> None:
    if not content or len(content) > _MAX_REPORT_BYTES:
        raise ContractError(
            "Kernel measurement report size is invalid",
            "invalid_measurement_runner_output",
        )
    if path.exists() or path.is_symlink():
        raise IntegrityError(
            "Kernel measurement output appeared before the evaluator write",
            "stale_measurement_report",
        )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise IntegrityError(
                "Kernel measurement output appeared before publication",
                "stale_measurement_report",
            ) from error
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


__all__ = [
    "STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID",
    "STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256",
    "StructuredKernelMeasurementAdapter",
]
