"""Internal grouping, shape, source, and allocation helpers for evidence intake."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError, sha256_file, sha256_json

from .evidence_models import KernelVolume, ShapeEvidence
from .targeted_trace_models import EvidenceArtifactReceipt, ValidatedTargetedEvent


@dataclass(slots=True)
class EventGroup:
    signature: str
    runtime_symbol: str
    payload: Mapping[str, Any]
    payload_hash_chain: str
    count: int
    duration_us: float
    duration_count: int
    warnings: tuple[str, ...]

    @classmethod
    def from_event(cls, event: ValidatedTargetedEvent) -> "EventGroup":
        runtime = mapping(event.payload, "runtime")
        duration = runtime.get("duration_us")
        warnings = event.payload.get("warnings", [])
        return cls(
            signature=event_group_signature(event.payload),
            runtime_symbol=str(runtime.get("gpu_symbol") or ""),
            payload=event.payload,
            payload_hash_chain=sha256_json({"event": event.payload_sha256}),
            count=1,
            duration_us=float(duration) if duration is not None else 0.0,
            duration_count=1 if duration is not None else 0,
            warnings=tuple(sorted(set(str(item) for item in warnings))),
        )

    def add(self, event: ValidatedTargetedEvent) -> None:
        runtime = mapping(event.payload, "runtime")
        if str(runtime.get("gpu_symbol") or "") != self.runtime_symbol:
            raise IntegrityError(
                "Targeted event grouping is inconsistent", "targeted_group_mismatch"
            )
        self.payload_hash_chain = sha256_json(
            {"previous": self.payload_hash_chain, "event": event.payload_sha256}
        )
        self.count += 1
        duration = runtime.get("duration_us")
        if duration is not None:
            self.duration_us += float(duration)
            self.duration_count += 1
        warnings = event.payload.get("warnings", [])
        self.warnings = tuple(
            sorted(set(self.warnings).union(str(item) for item in warnings))
        )


def event_group_signature(payload: Mapping[str, Any]) -> str:
    identity = mapping(payload, "identity")
    context = mapping(payload, "context")
    semantics = mapping(payload, "semantics")
    runtime = mapping(payload, "runtime")
    return sha256_json(
        {
            "identity": {
                "target_id": identity.get("target_id"),
                "variant_id": identity.get("variant_id"),
                "package": identity.get("package"),
                "source_hashes": identity.get("source_hashes", {}),
                "provenance_hashes": identity.get("provenance_hashes", {}),
            },
            "context": {
                "framework": context.get("framework"),
                "framework_version": context.get("framework_version"),
                "rank": context.get("rank"),
                "stage": context.get("stage"),
                "execution_mode": context.get("execution_mode"),
            },
            "semantics": dict(semantics),
            "runtime": {
                "gpu_symbol": runtime.get("gpu_symbol"),
                "grid": runtime.get("grid"),
                "block": runtime.get("block"),
                "stream": runtime.get("stream"),
            },
        }
    )


def shape_from_payload(
    identity: Mapping[str, Any],
    context: Mapping[str, Any],
    semantics: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> ShapeEvidence:
    tensors = semantics.get("tensors", [])
    tensor_items = [item for item in tensors if isinstance(item, Mapping)]
    params = _shape_params(identity, semantics, runtime)
    graph_mode = str(context.get("execution_mode", "unknown")).lower()
    if graph_mode in {"graph", "cuda_graph", "hip_graph"}:
        graph_mode = "cudagraph"
    elif graph_mode != "eager":
        graph_mode = "unknown"
    return ShapeEvidence(
        params=tuple(sorted(params.items())),
        input_dims=tuple(
            tuple(param_value(part) for part in item.get("shape", []))
            for item in tensor_items
        ),
        dtypes=tuple(str(item.get("dtype", "unknown")) for item in tensor_items),
        strides=tuple(
            tuple(param_value(part) for part in (item.get("stride") or ()))
            for item in tensor_items
        ),
        concrete_inputs=tuple(_concrete_tensor(item) for item in tensor_items),
        graph_mode=graph_mode,
    )


def _shape_params(
    identity: Mapping[str, Any],
    semantics: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "target_id": identity.get("target_id"),
        "variant_id": identity.get("variant_id"),
        "source_hashes": param_value(identity.get("source_hashes", {})),
        "provenance_hashes": param_value(identity.get("provenance_hashes", {})),
    }
    source = semantics.get("source")
    if isinstance(source, Mapping):
        for field in ("function", "sha256"):
            if source.get(field) is not None:
                params[f"source.{field}"] = source.get(field)
    for namespace in ("named_scalars", "constexpr", "meta"):
        values = semantics.get(namespace, {})
        if isinstance(values, Mapping):
            for name, value in values.items():
                params[f"{namespace}.{name}"] = param_value(value)
    if semantics.get("python_grid") is not None:
        params["python_grid"] = param_value(semantics.get("python_grid"))
    for field in ("grid", "block", "stream"):
        if runtime.get(field) is not None:
            params[f"runtime.{field}"] = param_value(runtime.get(field))
    return params


def _concrete_tensor(item: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "name": item.get("name"),
            "device": item.get("device"),
            "layout": item.get("layout"),
            "requires_grad": item.get("requires_grad"),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def param_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def allocate_aggregate(
    aggregate: Mapping[str, Any] | None, groups: list[EventGroup]
) -> tuple[KernelVolume, ...]:
    if aggregate is None:
        return tuple(
            KernelVolume(group.count, group.duration_us / 1000.0, 0.0)
            for group in groups
        )
    calls = int(aggregate.get("calls", aggregate.get("Calls", 0)) or 0)
    time_ms = float(aggregate.get("time_ms", 0) or 0)
    percent = float(aggregate.get("percent", aggregate.get("% Total", 0)) or 0)
    if len(groups) == 1:
        return (KernelVolume(calls, time_ms, percent),)
    weights = _allocation_weights(groups)
    raw_calls = [calls * weight / sum(weights) for weight in weights]
    allocated_calls = [math.floor(value) for value in raw_calls]
    remainder = calls - sum(allocated_calls)
    order = sorted(
        range(len(groups)),
        key=lambda index: (
            -(raw_calls[index] - allocated_calls[index]),
            groups[index].signature,
        ),
    )
    for index in order[:remainder]:
        allocated_calls[index] += 1
    return tuple(
        KernelVolume(
            allocated_calls[index],
            time_ms * weights[index] / sum(weights),
            percent * weights[index] / sum(weights),
        )
        for index in range(len(groups))
    )


def _allocation_weights(groups: list[EventGroup]) -> list[float]:
    complete_durations = all(
        group.duration_count == group.count for group in groups
    ) and sum(group.duration_us for group in groups) > 0
    return [
        group.duration_us if complete_durations else float(group.count)
        for group in groups
    ]


def phase(stage: str) -> str:
    value = stage.strip().lower()
    if value in {"prefill", "decode", "mixed"}:
        return value
    if value in {"prefilldecode", "prefill_decode", "prefill+decode"}:
        return "mixed"
    return "unknown"


def resolve_launch_source(launch_path: str | None, gap_path: str | None) -> str | None:
    if launch_path:
        launch = Path(launch_path).expanduser()
        if launch.is_absolute():
            return str(launch.resolve())
        if gap_path:
            gap = Path(gap_path).expanduser()
            normalized = "/".join(
                part for part in launch.parts if part not in {"", "."}
            )
            if gap.as_posix() == normalized or gap.as_posix().endswith(
                f"/{normalized}"
            ):
                return str(gap.resolve())
    return gap_path or launch_path


def evidence_receipt(
    kind: str, path: Path, workspace: Path, media_type: str
) -> EvidenceArtifactReceipt:
    try:
        relative = path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise IntegrityError(
            "Diagnostic artifact escapes benchmark workspace", "invalid_artifact_path"
        ) from error
    return EvidenceArtifactReceipt(
        kind, relative, sha256_file(path), path.stat().st_size, media_type
    )


def language(kind: str, runtime_name: str) -> str:
    value = f"{kind} {runtime_name}".lower()
    if "triton" in value:
        return "triton"
    if "hip" in value or "cpp" in value or "aiter" in value:
        return "hip"
    if "python" in value or "inductor" in value:
        return "python"
    if "asm" in value:
        return "asm"
    return "unknown"


def mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise TypeError(key)
    return item


def optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def expand_repo_path(
    value: str, mappings: Mapping[str, Path]
) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    if value.startswith("$"):
        variable, separator, suffix = value.partition("/")
        root = mappings.get(variable)
        if root is None:
            return None, None
        resolved = root / suffix if separator else root
        return str(resolved.resolve()), str(root.resolve())
    path = Path(value).expanduser()
    return (str(path.resolve()), None) if path.is_absolute() else (None, None)


__all__ = [
    "EventGroup",
    "allocate_aggregate",
    "event_group_signature",
    "evidence_receipt",
    "expand_repo_path",
    "language",
    "mapping",
    "optional_text",
    "phase",
    "resolve_launch_source",
    "shape_from_payload",
]
