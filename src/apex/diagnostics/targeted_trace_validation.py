"""Schema-level validation helpers for targeted trace envelopes and events."""

from __future__ import annotations

import math
from typing import Any, Mapping

from apex.core import IntegrityError, sha256_json

from .targeted_trace_models import (
    ENVELOPE_TYPES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    checked_sha256,
    nonempty_text,
    strict_nonnegative_int,
)


def required_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    item = value.get(field)
    if not isinstance(item, Mapping):
        raise IntegrityError(
            f"Event {field} is malformed", "invalid_targeted_event"
        )
    return item


def validate_envelope(
    envelope: Mapping[str, Any], *, expected_sequence: int, previous_checksum: str
) -> Mapping[str, Any]:
    if (
        envelope.get("schema_name") != SCHEMA_NAME
        or envelope.get("schema_version") != SCHEMA_VERSION
    ):
        raise IntegrityError("Unsupported targeted shard schema", "unsupported_schema")
    if envelope.get("record_type") not in ENVELOPE_TYPES:
        raise IntegrityError(
            "Unknown targeted envelope type", "invalid_targeted_envelope"
        )
    if envelope.get("sequence") != expected_sequence:
        raise IntegrityError(
            "Targeted envelope sequence mismatch", "invalid_targeted_sequence"
        )
    if envelope.get("previous_checksum") != previous_checksum:
        raise IntegrityError(
            "Targeted envelope chain mismatch", "targeted_checksum_mismatch"
        )
    checksum = checked_sha256(envelope.get("checksum"), "envelope checksum")
    body = dict(envelope)
    body.pop("checksum", None)
    try:
        expected_checksum = sha256_json(body)
    except (TypeError, ValueError) as error:
        raise IntegrityError(
            "Envelope is not canonical JSON", "invalid_targeted_envelope"
        ) from error
    if checksum != expected_checksum:
        raise IntegrityError(
            "Targeted envelope checksum mismatch", "targeted_checksum_mismatch"
        )
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise IntegrityError(
            "Targeted envelope payload is not an object", "invalid_targeted_envelope"
        )
    return payload


def validate_event(
    payload: Mapping[str, Any], *, run_id: str, rank: int, pid: int
) -> None:
    nonempty_text(payload.get("kind"), "event kind")
    nonempty_text(payload.get("stable_event_key"), "stable event key")
    identity = required_mapping(payload, "identity")
    context = required_mapping(payload, "context")
    semantics = required_mapping(payload, "semantics")
    runtime = required_mapping(payload, "runtime")
    _validate_identity(identity, run_id=run_id)
    _validate_context(context, rank=rank, pid=pid)
    _validate_semantics(semantics)
    _validate_runtime(runtime)
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or any(
        not isinstance(item, str) for item in warnings
    ):
        raise IntegrityError(
            "Event warnings are malformed", "invalid_targeted_event"
        )
    timestamp_ns = payload.get("timestamp_ns")
    if timestamp_ns is not None and (
        isinstance(timestamp_ns, bool)
        or not isinstance(timestamp_ns, int)
        or timestamp_ns < 0
    ):
        raise IntegrityError(
            "Event timestamp_ns is invalid", "invalid_targeted_event"
        )


def _validate_identity(identity: Mapping[str, Any], *, run_id: str) -> None:
    if identity.get("run_id") != run_id:
        raise IntegrityError(
            "Event run_id differs from manifest", "targeted_event_identity_mismatch"
        )
    nonempty_text(identity.get("target_id"), "target_id")
    nonempty_text(identity.get("variant_id"), "variant_id")
    for field in ("package", "image"):
        if identity.get(field) is not None and not isinstance(identity[field], str):
            raise IntegrityError(
                f"Event {field} is malformed", "invalid_targeted_event"
            )
    for field in ("source_hashes", "provenance_hashes"):
        hashes = identity.get(field, {})
        if not isinstance(hashes, Mapping):
            raise IntegrityError(
                f"Event {field} is malformed", "invalid_targeted_event"
            )
        for name, digest in hashes.items():
            nonempty_text(name, f"{field} key")
            checked_sha256(digest, f"{field} digest")


def _validate_context(context: Mapping[str, Any], *, rank: int, pid: int) -> None:
    nonempty_text(context.get("framework"), "framework")
    for field in ("framework_version", "stage", "execution_mode", "graph_id"):
        if context.get(field) is not None and not isinstance(context[field], str):
            raise IntegrityError(
                f"Event context {field} is malformed", "invalid_targeted_event"
            )
    if (
        strict_nonnegative_int(context.get("rank"), "event rank") != rank
        or strict_nonnegative_int(context.get("pid"), "event pid") != pid
    ):
        raise IntegrityError(
            "Event rank/pid differs from shard", "targeted_event_identity_mismatch"
        )
    world_size = context.get("world_size")
    if world_size is not None:
        value = strict_nonnegative_int(world_size, "world_size")
        if value == 0 or rank >= value:
            raise IntegrityError(
                "Event world_size is invalid", "invalid_targeted_event"
            )


def _validate_semantics(semantics: Mapping[str, Any]) -> None:
    source = semantics.get("source")
    if source is not None:
        _validate_source(source)
    tensors = semantics.get("tensors", [])
    if not isinstance(tensors, list):
        raise IntegrityError("Event tensors are malformed", "invalid_targeted_event")
    for tensor in tensors:
        _validate_tensor(tensor)
    for field in ("named_scalars", "constexpr", "meta"):
        if not isinstance(semantics.get(field, {}), Mapping):
            raise IntegrityError(
                f"Event {field} is malformed", "invalid_targeted_event"
            )


def _validate_source(source: object) -> None:
    if not isinstance(source, Mapping):
        raise IntegrityError(
            "Event source evidence is malformed", "invalid_targeted_event"
        )
    nonempty_text(source.get("path"), "source path")
    if source.get("function") is not None and not isinstance(source["function"], str):
        raise IntegrityError(
            "Event source function is malformed", "invalid_targeted_event"
        )
    if (
        source.get("line") is not None
        and strict_nonnegative_int(source["line"], "source line") == 0
    ):
        raise IntegrityError("Event source line is invalid", "invalid_targeted_event")
    if source.get("sha256") is not None:
        checked_sha256(source["sha256"], "source sha256")


def _validate_tensor(tensor: object) -> None:
    if not isinstance(tensor, Mapping):
        raise IntegrityError("Event tensor is malformed", "invalid_targeted_event")
    nonempty_text(tensor.get("name"), "tensor name")
    nonempty_text(tensor.get("dtype"), "tensor dtype")
    if not isinstance(tensor.get("shape"), list):
        raise IntegrityError("Tensor shape is malformed", "invalid_targeted_event")
    stride = tensor.get("stride")
    if stride is not None and not isinstance(stride, list):
        raise IntegrityError("Tensor stride is malformed", "invalid_targeted_event")
    for field in ("device", "layout"):
        if tensor.get(field) is not None and not isinstance(tensor[field], str):
            raise IntegrityError(
                f"Tensor {field} is malformed", "invalid_targeted_event"
            )
    if tensor.get("requires_grad") is not None and not isinstance(
        tensor["requires_grad"], bool
    ):
        raise IntegrityError(
            "Tensor requires_grad is malformed", "invalid_targeted_event"
        )


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    for field in ("grid", "block"):
        vector = runtime.get(field)
        if vector is not None and (
            not isinstance(vector, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in vector)
        ):
            raise IntegrityError(
                f"Runtime {field} is malformed", "invalid_targeted_event"
            )
    for field in ("cpu_uid", "correlation_id", "gpu_uid", "gpu_symbol", "stream"):
        if runtime.get(field) is not None and not isinstance(runtime[field], str):
            raise IntegrityError(
                f"Runtime {field} is malformed", "invalid_targeted_event"
            )
    for field in ("duration_us", "timestamp_us"):
        value = runtime.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise IntegrityError(
                f"Runtime {field} is invalid", "invalid_targeted_event"
            )


__all__ = ["required_mapping", "validate_envelope", "validate_event"]
