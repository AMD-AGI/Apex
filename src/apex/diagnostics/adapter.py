"""DiagnosticsPort adapter over artifacts already produced by Magpie."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Mapping

from apex.core import ApexError, IntegrityError, canonical_json_bytes, sha256_file
from apex.ports import DiagnosticsRequest, DiagnosticsResult

from .evidence import TraceEvidenceNormalizer
from .targeted_trace import EvidenceArtifactReceipt
from .ranking import rank_evidence


class MagpieTraceEvidenceAdapter:
    """Normalize a completed diagnostic workspace; never run another tracer."""

    def __init__(self, normalizer: TraceEvidenceNormalizer | None = None) -> None:
        self._normalizer = normalizer or TraceEvidenceNormalizer()

    def analyze(self, request: DiagnosticsRequest) -> DiagnosticsResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        reports = sorted(request.benchmark_workspace.rglob("benchmark_report.json"))
        if len(reports) != 1:
            return _failed_result(
                request,
                f"Expected exactly one benchmark_report.json, found {len(reports)}",
            )
        raw_artifacts: tuple[Path, ...] = ()
        raw_roles: dict[str, str] = {}
        raw_manifest: tuple[Mapping[str, object], ...] = ()
        try:
            if request.preserve_raw_trace:
                raw_artifacts, raw_roles, raw_manifest = _tracelens_raw_artifacts(
                    reports[0], request.benchmark_workspace
                )
            records = self._normalizer.from_benchmark_report(
                reports[0], provenance_hash=request.provenance_hash
            )
            rankings = rank_evidence(records)
            receipts = {
                (receipt.kind, receipt.relative_path): receipt
                for record in records
                for receipt in record.evidence.artifacts
            }
            output = request.output_dir / "trace_evidence.json"
            payload = {
                "schema_version": 1,
                "run_id": request.run_id,
                "artifact_receipts": [
                    receipt.to_dict()
                    for receipt in sorted(
                        receipts.values(),
                        key=lambda item: (item.kind, item.relative_path),
                    )
                ],
                "records": [record.to_dict() for record in records],
                "rankings": {
                    "measured": [asdict(item) for item in rankings.measured],
                    "recoverable": [asdict(item) for item in rankings.recoverable],
                },
            }
            _atomic_write(output, canonical_json_bytes(payload) + b"\n")
            artifacts = _validated_artifact_paths(
                request.benchmark_workspace, tuple(receipts.values())
            )
            roles = {str(path): "diagnostic_artifact" for path in artifacts}
            roles[str(output.resolve())] = "diagnostic_trace_evidence"
            if raw_artifacts:
                artifacts = tuple(sorted(set(artifacts).union(raw_artifacts)))
                roles.update(raw_roles)
            return DiagnosticsResult(
                request.run_id,
                bool(records),
                artifacts + (output,),
                {
                    "record_count": len(records),
                    "artifact_receipt_count": len(receipts),
                    "evidence_path": str(output),
                    "raw_trace_preserved": bool(raw_artifacts),
                    "raw_artifact_manifest": list(raw_manifest),
                },
                None if records else "No kernel evidence was present",
                roles,
                request.benchmark_workspace.resolve(),
            )
        except (ApexError, OSError, ValueError, json.JSONDecodeError) as error:
            return _failed_result(
                request,
                str(error),
                artifacts=raw_artifacts,
                roles=raw_roles,
                manifest=raw_manifest,
            )


def _failed_result(
    request: DiagnosticsRequest,
    error: str,
    *,
    artifacts: tuple[Path, ...] = (),
    roles: Mapping[str, str] | None = None,
    manifest: tuple[Mapping[str, object], ...] = (),
) -> DiagnosticsResult:
    return DiagnosticsResult(
        request.run_id,
        False,
        artifacts,
        {
            "raw_trace_preserved": bool(artifacts),
            "raw_artifact_manifest": list(manifest),
        },
        error,
        roles or {},
        request.benchmark_workspace.resolve(),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validated_artifact_paths(
    workspace: Path,
    receipts: tuple[EvidenceArtifactReceipt, ...],
) -> tuple[Path, ...]:
    """Return only evidence-bound files, never the disposable runtime tree."""

    workspace = workspace.resolve()
    paths: set[Path] = set()
    for receipt in receipts:
        path = workspace.joinpath(*Path(receipt.relative_path).parts)
        resolved = path.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as error:
            raise IntegrityError(
                "Diagnostic artifact escapes its workspace",
                "invalid_artifact_path",
            ) from error
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.stat().st_size != receipt.byte_count
            or sha256_file(path) != receipt.sha256
        ):
            raise IntegrityError(
                "Diagnostic artifact changed after normalization",
                "diagnostic_artifact_drift",
            )
        paths.add(resolved)
    return tuple(sorted(paths))


def _tracelens_raw_artifacts(
    report: Path,
    workspace: Path,
) -> tuple[tuple[Path, ...], dict[str, str], tuple[Mapping[str, object], ...]]:
    """Validate only producer-declared TraceLens outputs, including the raw trace."""

    try:
        document = json.loads(report.read_text(encoding="utf-8"))
        analysis = document["tracelens_analysis"]
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as error:
        raise IntegrityError(
            "TraceLens artifact declaration is missing",
            "tracelens_artifact_declaration_missing",
        ) from error
    if not isinstance(analysis, Mapping) or analysis.get("enabled") is not True:
        raise IntegrityError(
            "TraceLens analysis was not enabled",
            "tracelens_artifact_declaration_missing",
        )
    raw_trace = _declared_file(workspace, analysis.get("rank0_trace"))
    if not raw_trace.name.endswith(".pt.trace.json.gz"):
        raise IntegrityError(
            "TraceLens raw trace has an unexpected type",
            "invalid_tracelens_artifact",
        )
    output_root = _declared_directory(workspace, analysis.get("output_dir"))
    declared_outputs = analysis.get("output_files")
    if (
        not isinstance(declared_outputs, list)
        or not declared_outputs
        or len(declared_outputs) > 512
        or any(not isinstance(value, str) or not value for value in declared_outputs)
    ):
        raise IntegrityError(
            "TraceLens output declaration is invalid",
            "invalid_tracelens_artifact",
        )
    output_paths = tuple(
        _declared_file(workspace, value, required_parent=output_root)
        for value in declared_outputs
    )
    stage_paths = _stage_report_paths(analysis, workspace, output_root, output_paths)
    paths = tuple(sorted({report.resolve(), raw_trace, *output_paths}))
    roles = {
        str(path): (
            "diagnostic_benchmark_report"
            if path == report.resolve()
            else "diagnostic_raw_trace"
            if path == raw_trace
            else "diagnostic_tracelens_report"
        )
        for path in paths
    }
    root = workspace.resolve()
    manifest = tuple(
        {
            "role": roles[str(path)],
            "workspace_relative_path": path.relative_to(root).as_posix(),
            "comparison_logical_path": _comparison_logical_path(
                path,
                report=report.resolve(),
                raw_trace=raw_trace,
                output_root=output_root,
                stage=stage_paths.get(path),
            ),
            "byte_count": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    )
    logical = [str(item["comparison_logical_path"]) for item in manifest]
    if len(set(logical)) != len(logical):
        raise IntegrityError(
            "TraceLens comparison artifact names collide",
            "invalid_tracelens_artifact",
        )
    return paths, roles, manifest


def _stage_report_paths(
    analysis: Mapping[str, object],
    workspace: Path,
    output_root: Path,
    declared: tuple[Path, ...],
) -> dict[Path, str]:
    raw = analysis.get("stage_results")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise IntegrityError(
            "TraceLens stage declaration is invalid", "invalid_tracelens_artifact"
        )
    allowed = set(declared)
    result: dict[Path, str] = {}
    for stage, value in raw.items():
        stage_path = PurePosixPath(str(stage))
        if (
            not isinstance(stage, str)
            or not stage
            or stage_path.is_absolute()
            or len(stage_path.parts) != 1
            or stage in {".", ".."}
            or not isinstance(value, Mapping)
        ):
            raise IntegrityError(
                "TraceLens stage declaration is invalid", "invalid_tracelens_artifact"
            )
        files = value.get("files")
        if not isinstance(files, list):
            raise IntegrityError(
                "TraceLens stage files are invalid", "invalid_tracelens_artifact"
            )
        for supplied in files:
            path = _declared_file(workspace, supplied, required_parent=output_root)
            if path not in allowed or (path in result and result[path] != stage):
                raise IntegrityError(
                    "TraceLens stage files conflict with the output manifest",
                    "invalid_tracelens_artifact",
                )
            result[path] = stage
    return result


def _comparison_logical_path(
    path: Path,
    *,
    report: Path,
    raw_trace: Path,
    output_root: Path,
    stage: str | None,
) -> str:
    if path == report:
        return "metadata/benchmark_report.json"
    if path == raw_trace:
        return f"raw/{path.name}"
    relative = path.relative_to(output_root)
    group = stage or (relative.parent.as_posix() if relative.parent != Path(".") else "root")
    return f"reports/{group}/{path.name}"


def _declared_directory(workspace: Path, value: object) -> Path:
    path = _declared_path(workspace, value)
    if path.is_symlink() or not path.is_dir():
        raise IntegrityError(
            "TraceLens output directory is unsafe",
            "invalid_tracelens_artifact",
        )
    return path


def _declared_file(
    workspace: Path,
    value: object,
    *,
    required_parent: Path | None = None,
) -> Path:
    path = _declared_path(workspace, value)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_nlink != 1
        or (required_parent is not None and not path.is_relative_to(required_parent))
    ):
        raise IntegrityError(
            "TraceLens artifact is unsafe",
            "invalid_tracelens_artifact",
        )
    return path


def _declared_path(workspace: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(
            "TraceLens artifact path is missing",
            "invalid_tracelens_artifact",
        )
    root = workspace.resolve()
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise IntegrityError(
            "TraceLens artifact escapes its workspace",
            "invalid_tracelens_artifact",
        )
    return resolved


__all__ = ["MagpieTraceEvidenceAdapter"]
