"""DiagnosticsPort adapter over artifacts already produced by Magpie."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

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
            return DiagnosticsResult(
                request.run_id,
                False,
                (),
                {},
                f"Expected exactly one benchmark_report.json, found {len(reports)}",
            )
        try:
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
            return DiagnosticsResult(
                request.run_id,
                bool(records),
                artifacts + (output,),
                {
                    "record_count": len(records),
                    "artifact_receipt_count": len(receipts),
                    "evidence_path": str(output),
                },
                None if records else "No kernel evidence was present",
            )
        except (ApexError, OSError, ValueError, json.JSONDecodeError) as error:
            return DiagnosticsResult(request.run_id, False, (), {}, str(error))


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


__all__ = ["MagpieTraceEvidenceAdapter"]
