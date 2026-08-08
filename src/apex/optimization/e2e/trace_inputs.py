"""Bind producer-declared diagnostic files to verified CAS comparison inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from apex.core import IntegrityError
from apex.ports import (
    DiagnosticsResult,
    TraceComparisonArtifact,
    TraceDiagnosticEvidence,
)
from apex.storage import ArtifactReceipt, ArtifactStore


_COMPARISON_ROLES = {
    "diagnostic_benchmark_report",
    "diagnostic_raw_trace",
    "diagnostic_tracelens_report",
}


def build_trace_diagnostic_evidence(
    result: DiagnosticsResult,
    receipts: Sequence[ArtifactReceipt],
    *,
    trace_evidence_sha256: str | None,
    store: ArtifactStore,
) -> TraceDiagnosticEvidence:
    """Rebind a diagnostic manifest to CAS paths, never its disposable workspace."""

    manifest = result.summary.get("raw_artifact_manifest")
    if not manifest:
        return TraceDiagnosticEvidence(trace_evidence_sha256, store.root.resolve(), ())
    if not isinstance(manifest, list) or result.benchmark_workspace is None:
        raise IntegrityError(
            "Diagnostic comparison manifest is invalid",
            "invalid_trace_comparison_manifest",
        )
    stored_paths = tuple(path.resolve() for path in result.artifacts if path.is_file())
    if len(stored_paths) != len(receipts) or len(set(stored_paths)) != len(stored_paths):
        raise IntegrityError(
            "Diagnostic CAS publication is ambiguous",
            "invalid_trace_comparison_manifest",
        )
    by_path = dict(zip(stored_paths, receipts, strict=True))
    workspace = result.benchmark_workspace.resolve()
    bound = tuple(
        _bind_manifest_item(item, workspace, by_path, result, store)
        for item in manifest
        if isinstance(item, Mapping)
    )
    if len(bound) != len(manifest):
        raise IntegrityError(
            "Diagnostic comparison manifest entry is malformed",
            "invalid_trace_comparison_manifest",
        )
    return TraceDiagnosticEvidence(
        trace_evidence_sha256,
        store.root.resolve(),
        tuple(item for item in bound if item.role in _COMPARISON_ROLES),
    )


def _bind_manifest_item(
    item: Mapping[str, object],
    workspace: Path,
    by_path: Mapping[Path, ArtifactReceipt],
    result: DiagnosticsResult,
    store: ArtifactStore,
) -> TraceComparisonArtifact:
    relative = item.get("workspace_relative_path")
    logical = item.get("comparison_logical_path")
    role = item.get("role")
    digest = item.get("sha256")
    size = item.get("byte_count")
    if (
        not isinstance(relative, str)
        or not isinstance(logical, str)
        or not isinstance(role, str)
        or not isinstance(digest, str)
        or not isinstance(size, int)
    ):
        raise IntegrityError(
            "Diagnostic comparison manifest entry is malformed",
            "invalid_trace_comparison_manifest",
        )
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace):
        raise IntegrityError(
            "Diagnostic comparison path escapes its workspace",
            "invalid_trace_comparison_manifest",
        )
    receipt = by_path.get(path)
    observed_role = result.artifact_roles.get(str(path))
    if (
        receipt is None
        or observed_role != role
        or receipt.digest != digest
        or receipt.size != size
    ):
        raise IntegrityError(
            "Diagnostic comparison receipt differs from its manifest",
            "trace_comparison_receipt_mismatch",
        )
    store.verify(receipt)
    return TraceComparisonArtifact(
        role,
        logical,
        receipt.digest,
        receipt.size,
        receipt.media_type,
        receipt.relative_path,
    )


__all__ = ["build_trace_diagnostic_evidence"]
