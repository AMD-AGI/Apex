"""Immutable normalized trace-evidence contracts and stable identities."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, sha256_json

from .targeted_trace_models import AcquisitionCoverage, EvidenceArtifactReceipt


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PHASES = frozenset({"prefill", "decode", "mixed", "unknown"})
_LANGUAGES = frozenset({"python", "triton", "hip", "cpp", "asm", "unknown"})
_SOURCE_CONFIDENCE = frozenset(
    {"exact_launch", "active_finder", "registry", "llm_review", "unknown"}
)
_GRAPH_MODES = frozenset({"eager", "cudagraph", "unknown"})
_ROOFLINE_BOUNDS = frozenset({"memory", "compute", "unknown"})
_CONFIDENCE = frozenset({"low", "medium", "high"})


def require_digest(value: str, field: str) -> str:
    if not _DIGEST.fullmatch(value):
        raise ContractError(
            f"{field} must be a lowercase SHA-256 digest", "invalid_digest"
        )
    return value


def finite_nonnegative(value: float, field: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ContractError(
            f"{field} must be finite and non-negative", "invalid_trace_metric"
        )
    return value


@dataclass(frozen=True, slots=True)
class OperationEvidence:
    category: str
    name: str
    call_path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KernelEvidence:
    runtime_name: str
    language: str = "unknown"
    origin_library: str = "unknown"
    source_path: str | None = None
    source_line: int | None = None
    source_confidence: str = "unknown"
    patchable: bool = False
    source_root: str | None = None
    test_file: str | None = None
    test_command: str | None = None

    def __post_init__(self) -> None:
        if not self.runtime_name.strip():
            raise ContractError("Kernel runtime name is required", "missing_kernel_name")
        if self.language not in _LANGUAGES:
            raise ContractError("Unknown kernel language", "invalid_kernel_language")
        if self.source_confidence not in _SOURCE_CONFIDENCE:
            raise ContractError(
                "Unknown source confidence", "invalid_source_confidence"
            )
        if self.source_line is not None and self.source_line < 1:
            raise ContractError("source_line must be positive", "invalid_source_line")
        if self.patchable and not self.source_path:
            raise ContractError(
                "Patchable evidence requires a source path", "missing_source_path"
            )


@dataclass(frozen=True, slots=True)
class ShapeEvidence:
    params: tuple[tuple[str, Any], ...] = ()
    input_dims: tuple[tuple[Any, ...], ...] = ()
    dtypes: tuple[str, ...] = ()
    strides: tuple[tuple[Any, ...], ...] = ()
    concrete_inputs: tuple[str, ...] = ()
    graph_mode: str = "unknown"

    def __post_init__(self) -> None:
        if self.graph_mode not in _GRAPH_MODES:
            raise ContractError("Unknown graph mode", "invalid_graph_mode")


@dataclass(frozen=True, slots=True)
class KernelVolume:
    calls: int
    gpu_time_ms: float
    gpu_time_pct: float

    def __post_init__(self) -> None:
        if self.calls < 0:
            raise ContractError(
                "Kernel call count cannot be negative", "invalid_trace_metric"
            )
        finite_nonnegative(self.gpu_time_ms, "gpu_time_ms")
        finite_nonnegative(self.gpu_time_pct, "gpu_time_pct")
        if self.gpu_time_pct > 100.000001:
            raise ContractError(
                "gpu_time_pct exceeds 100", "invalid_trace_metric"
            )


@dataclass(frozen=True, slots=True)
class PerformanceModelEvidence:
    has_model: bool = False
    flops: float | None = None
    bytes: float | None = None
    arithmetic_intensity: float | None = None
    compute_spec: str | None = None
    roofline_bound: str = "unknown"
    pct_roofline: float | None = None
    confidence: str = "low"

    def __post_init__(self) -> None:
        if (
            self.roofline_bound not in _ROOFLINE_BOUNDS
            or self.confidence not in _CONFIDENCE
        ):
            raise ContractError(
                "Invalid performance-model vocabulary", "invalid_perf_model"
            )
        numeric = (
            ("flops", self.flops),
            ("bytes", self.bytes),
            ("arithmetic_intensity", self.arithmetic_intensity),
            ("pct_roofline", self.pct_roofline),
        )
        for field, value in numeric:
            if value is not None:
                finite_nonnegative(value, field)
        if not self.has_model and any(value is not None for _, value in numeric):
            raise ContractError(
                "Missing perf model cannot contain numeric estimates",
                "invalid_perf_model",
            )


@dataclass(frozen=True, slots=True)
class EvidenceArtifacts:
    acquisition_schema: str
    coverage: AcquisitionCoverage
    artifacts: tuple[EvidenceArtifactReceipt, ...]
    trace_row_hash: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.acquisition_schema.strip():
            raise ContractError(
                "Acquisition schema is required", "missing_acquisition_schema"
            )
        require_digest(self.trace_row_hash, "trace_row_hash")
        identities: set[tuple[str, str]] = set()
        for receipt in self.artifacts:
            if not isinstance(receipt, EvidenceArtifactReceipt):
                raise ContractError(
                    "Evidence artifacts must be typed receipts",
                    "invalid_artifact_receipt",
                )
            identity = (receipt.kind, receipt.relative_path)
            if identity in identities:
                raise ContractError(
                    "Artifact receipts must be unique", "invalid_artifact_receipt"
                )
            identities.add(identity)
        if self.acquisition_schema == "TargetedKernelTrace":
            kinds = {receipt.kind for receipt in self.artifacts}
            if not {"targeted_manifest", "targeted_shard"}.issubset(kinds):
                raise ContractError(
                    "Targeted trace requires manifest and shard receipts",
                    "missing_artifact_receipt",
                )


@dataclass(frozen=True, slots=True)
class TraceEvidence:
    """One shape/rank/regime-specific kernel observation."""

    schema_version: int
    candidate_id: str
    provenance_hash: str
    phase: str
    rank: int
    op: OperationEvidence
    kernel: KernelEvidence
    shape: ShapeEvidence
    volume: KernelVolume
    perf_model: PerformanceModelEvidence
    evidence: EvidenceArtifacts
    match_confidence: str = "unknown"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractError("Unsupported TraceEvidence schema", "unsupported_schema")
        require_digest(self.candidate_id, "candidate_id")
        require_digest(self.provenance_hash, "provenance_hash")
        if self.phase not in _PHASES or self.rank < 0:
            raise ContractError("Invalid phase or rank", "invalid_trace_identity")
        if self.match_confidence not in {"exact", "probable", "unknown"}:
            raise ContractError(
                "Invalid match confidence", "invalid_match_confidence"
            )
        expected = derive_candidate_id(
            provenance_hash=self.provenance_hash,
            phase=self.phase,
            rank=self.rank,
            kernel=self.kernel,
            shape=self.shape,
        )
        if self.candidate_id != expected:
            raise IntegrityError(
                "TraceEvidence candidate identity is inconsistent",
                "candidate_id_mismatch",
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"]["params"] = dict(self.shape.params)
        value["evidence"]["coverage"]["dropped_by_reason"] = dict(
            self.evidence.coverage.dropped_by_reason
        )
        value["evidence"]["artifacts"] = [
            receipt.to_dict() for receipt in self.evidence.artifacts
        ]
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TraceEvidence":
        """Validate a materialized normalized record before planner use."""

        try:
            op = _mapping(value, "op")
            shape = _mapping(value, "shape")
            volume = _mapping(value, "volume")
            params = shape.get("params", {})
            if not isinstance(params, Mapping):
                raise TypeError("params")
            return cls(
                schema_version=int(value["schema_version"]),
                candidate_id=str(value["candidate_id"]),
                provenance_hash=str(value["provenance_hash"]),
                phase=str(value["phase"]),
                rank=int(value["rank"]),
                op=OperationEvidence(
                    str(op.get("category", "unknown")),
                    str(op.get("name", "")),
                    tuple(str(item) for item in op.get("call_path", ())),
                ),
                kernel=_kernel_from_mapping(_mapping(value, "kernel")),
                shape=_shape_from_mapping(shape, params),
                volume=KernelVolume(
                    int(volume.get("calls", 0)),
                    float(volume.get("gpu_time_ms", 0)),
                    float(volume.get("gpu_time_pct", 0)),
                ),
                perf_model=_perf_model_from_mapping(_mapping(value, "perf_model")),
                evidence=_artifacts_from_mapping(_mapping(value, "evidence")),
                match_confidence=str(value.get("match_confidence", "unknown")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                "Malformed TraceEvidence record", "invalid_trace_evidence"
            ) from error


def derive_candidate_id(
    *,
    provenance_hash: str,
    phase: str,
    rank: int,
    kernel: KernelEvidence,
    shape: ShapeEvidence,
) -> str:
    """Derive a stable identity without trace ordinals or aggregate timing."""

    require_digest(provenance_hash, "provenance_hash")
    return sha256_json(
        {
            "provenance_hash": provenance_hash,
            "phase": phase,
            "rank": rank,
            "kernel": {
                "runtime_name": kernel.runtime_name,
                "language": kernel.language,
                "origin_library": kernel.origin_library,
                "source_path": kernel.source_path,
                "source_line": kernel.source_line,
            },
            "shape": asdict(shape),
        }
    )


def _kernel_from_mapping(value: Mapping[str, Any]) -> KernelEvidence:
    return KernelEvidence(
        runtime_name=str(value.get("runtime_name", "")),
        language=str(value.get("language", "unknown")),
        origin_library=str(value.get("origin_library", "unknown")),
        source_path=_optional_text(value.get("source_path")),
        source_line=int(value["source_line"])
        if value.get("source_line") is not None
        else None,
        source_confidence=str(value.get("source_confidence", "unknown")),
        patchable=bool(value.get("patchable", False)),
        source_root=_optional_text(value.get("source_root")),
        test_file=_optional_text(value.get("test_file")),
        test_command=_optional_text(value.get("test_command")),
    )


def _shape_from_mapping(
    value: Mapping[str, Any], params: Mapping[str, Any]
) -> ShapeEvidence:
    return ShapeEvidence(
        params=tuple(sorted((str(key), item) for key, item in params.items())),
        input_dims=tuple(
            tuple(part for part in item) for item in value.get("input_dims", ())
        ),
        dtypes=tuple(str(item) for item in value.get("dtypes", ())),
        strides=tuple(
            tuple(part for part in item) for item in value.get("strides", ())
        ),
        concrete_inputs=tuple(
            str(item) for item in value.get("concrete_inputs", ())
        ),
        graph_mode=str(value.get("graph_mode", "unknown")),
    )


def _perf_model_from_mapping(value: Mapping[str, Any]) -> PerformanceModelEvidence:
    return PerformanceModelEvidence(
        has_model=bool(value.get("has_model", False)),
        flops=_optional_float(value.get("flops")),
        bytes=_optional_float(value.get("bytes")),
        arithmetic_intensity=_optional_float(value.get("arithmetic_intensity")),
        compute_spec=_optional_text(value.get("compute_spec")),
        roofline_bound=str(value.get("roofline_bound", "unknown")),
        pct_roofline=_optional_float(value.get("pct_roofline")),
        confidence=str(value.get("confidence", "low")),
    )


def _artifacts_from_mapping(value: Mapping[str, Any]) -> EvidenceArtifacts:
    coverage = _mapping(value, "coverage")
    reasons = coverage.get("dropped_by_reason", {})
    artifacts = value.get("artifacts", [])
    if (
        not isinstance(reasons, Mapping)
        or not isinstance(artifacts, list)
        or any(not isinstance(item, Mapping) for item in artifacts)
    ):
        raise TypeError("artifacts")
    return EvidenceArtifacts(
        acquisition_schema=str(value.get("acquisition_schema", "")),
        coverage=AcquisitionCoverage.from_mapping(coverage),
        artifacts=tuple(
            EvidenceArtifactReceipt.from_mapping(item) for item in artifacts
        ),
        trace_row_hash=str(value.get("trace_row_hash", "")),
        warnings=tuple(str(item) for item in value.get("warnings", ())),
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise TypeError(key)
    return item


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "EvidenceArtifacts",
    "KernelEvidence",
    "KernelVolume",
    "OperationEvidence",
    "PerformanceModelEvidence",
    "ShapeEvidence",
    "TraceEvidence",
    "derive_candidate_id",
    "finite_nonnegative",
    "require_digest",
]
