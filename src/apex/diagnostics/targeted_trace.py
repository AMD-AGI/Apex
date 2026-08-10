"""Independent, fail-closed validation of Magpie targeted trace artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from apex.core import IntegrityError, sha256_file, sha256_json

from .targeted_trace_models import (
    AcquisitionCoverage,
    EvidenceArtifactReceipt,
    MAX_JSONL_LINE_BYTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ShardReceipt,
    ValidatedTargetedEvent,
    ValidatedTargetedTrace,
    ZERO_CHECKSUM,
    checked_sha256,
    nonempty_text,
    strict_nonnegative_int,
)
from .targeted_trace_io import read_object, resolve_trace_path, resolve_workspace_path
from .targeted_trace_quality import (
    SemanticQualityAccumulator,
    validate_semantic_quality,
)
from .targeted_trace_validation import validate_envelope, validate_event


@dataclass(slots=True)
class _ShardState:
    receipt: ShardReceipt
    run_id: str
    on_event: Callable[[ValidatedTargetedEvent], None] | None
    expected_sequence: int = 0
    previous_checksum: str = ZERO_CHECKSUM
    saw_header: bool = False
    saw_end: bool = False
    event_count: int = 0
    final_coverage: AcquisitionCoverage | None = None

    def consume(self, envelope: Mapping[str, Any]) -> None:
        payload = validate_envelope(
            envelope,
            expected_sequence=self.expected_sequence,
            previous_checksum=self.previous_checksum,
        )
        record_type = str(envelope["record_type"])
        if self.expected_sequence == 0 and record_type != "header":
            raise IntegrityError(
                "Targeted shard does not start with a header",
                "invalid_targeted_header",
            )
        if record_type == "header":
            self._consume_header(payload)
        elif record_type == "event":
            self._consume_event(payload)
        else:
            self._consume_end(payload)
        self.previous_checksum = str(envelope["checksum"])
        self.expected_sequence += 1

    def _consume_header(self, payload: Mapping[str, Any]) -> None:
        if self.saw_header or self.expected_sequence != 0:
            raise IntegrityError(
                "Duplicate targeted shard header", "invalid_targeted_header"
            )
        if (
            payload.get("run_id") != self.run_id
            or strict_nonnegative_int(payload.get("rank"), "header rank")
            != self.receipt.rank
            or strict_nonnegative_int(payload.get("pid"), "header pid")
            != self.receipt.pid
        ):
            raise IntegrityError(
                "Targeted shard header disagrees with manifest",
                "invalid_targeted_header",
            )
        self.saw_header = True

    def _consume_event(self, payload: Mapping[str, Any]) -> None:
        if not self.saw_header:
            raise IntegrityError(
                "Targeted event precedes header", "invalid_targeted_envelope"
            )
        validate_event(
            payload,
            run_id=self.run_id,
            rank=self.receipt.rank,
            pid=self.receipt.pid,
        )
        self.event_count += 1
        if self.on_event is not None:
            self.on_event(
                ValidatedTargetedEvent(
                    payload=dict(payload),
                    payload_sha256=sha256_json(payload),
                    shard_relative_path=self.receipt.relative_path,
                    sequence=self.expected_sequence,
                )
            )

    def _consume_end(self, payload: Mapping[str, Any]) -> None:
        if not self.saw_header:
            raise IntegrityError(
                "Targeted end precedes header", "invalid_end_sentinel"
            )
        raw_counters = payload.get("counters")
        if not isinstance(raw_counters, Mapping):
            raise IntegrityError(
                "End sentinel has no counters", "invalid_end_sentinel"
            )
        coverage = AcquisitionCoverage.from_mapping(raw_counters)
        if (
            payload.get("run_id") != self.run_id
            or strict_nonnegative_int(payload.get("rank"), "end rank")
            != self.receipt.rank
            or strict_nonnegative_int(payload.get("pid"), "end pid")
            != self.receipt.pid
            or coverage.written != self.event_count
        ):
            raise IntegrityError(
                "End sentinel disagrees with shard", "invalid_end_sentinel"
            )
        self.final_coverage = coverage
        self.saw_end = True


class TargetedTraceValidator:
    """Validate report pointers, manifest, summary, receipts, and hash chains."""

    def validate(
        self,
        targeted: Mapping[str, Any],
        *,
        workspace: Path,
        on_event: Callable[[ValidatedTargetedEvent], None] | None = None,
    ) -> ValidatedTargetedTrace:
        workspace = Path(workspace).resolve()
        _validate_report_contract(targeted)
        manifest_path = resolve_workspace_path(
            targeted.get("manifest_path"), workspace, expected_name="manifest.json"
        )
        summary_path = resolve_workspace_path(
            targeted.get("summary_path"), workspace, expected_name="summary.json"
        )
        initial_artifacts = (
            _artifact("targeted_manifest", manifest_path, workspace, "application/json"),
            _artifact("targeted_summary", summary_path, workspace, "application/json"),
        )
        manifest = read_object(manifest_path, "TargetedKernelTrace manifest")
        run_id, backend, receipts, coverage, warnings = self._parse_manifest(
            manifest, workspace=workspace, trace_dir=manifest_path.parent
        )
        self._validate_shard_set(receipts, trace_dir=manifest_path.parent)
        semantic_quality = SemanticQualityAccumulator()
        actual = tuple(
            self._validate_shard(
                receipt, run_id=run_id, on_event=semantic_quality.observe
            )
            for receipt in receipts
        )
        if AcquisitionCoverage.aggregate(actual) != coverage:
            raise IntegrityError(
                "Manifest coverage differs from shard sentinels", "coverage_mismatch"
            )
        _validate_report_coverage(targeted, coverage)
        semantic_claimed, semantic_reasons = _validate_summary(
            summary_path, run_id, coverage, receipts, semantic_quality
        )
        if on_event is not None:
            self._replay_events(receipts, run_id=run_id, on_event=on_event)
        _assert_metadata_unchanged(initial_artifacts, manifest_path, summary_path, workspace)
        shard_artifacts = tuple(
            _artifact("targeted_shard", receipt.path, workspace, "application/x-ndjson")
            for receipt in receipts
        )
        return ValidatedTargetedTrace(
            SCHEMA_NAME,
            SCHEMA_VERSION,
            run_id,
            backend,
            coverage,
            initial_artifacts + shard_artifacts,
            tuple(sorted(warnings)),
            semantic_claimed,
            semantic_reasons,
        )

    @staticmethod
    def _parse_manifest(
        manifest: Mapping[str, Any], *, workspace: Path, trace_dir: Path
    ) -> tuple[str, str, tuple[ShardReceipt, ...], AcquisitionCoverage, set[str]]:
        _validate_manifest_header(manifest)
        run_id = nonempty_text(manifest.get("run_id"), "manifest run_id")
        backend = nonempty_text(
            manifest.get("acquisition_backend"), "acquisition backend"
        )
        receipts = _parse_shard_receipts(
            manifest.get("shards"), workspace=workspace, trace_dir=trace_dir
        )
        raw_coverage = manifest.get("coverage")
        if not isinstance(raw_coverage, Mapping):
            raise IntegrityError("Manifest has no coverage", "coverage_mismatch")
        coverage = AcquisitionCoverage.from_mapping(raw_coverage)
        if AcquisitionCoverage.aggregate(
            tuple(item.coverage for item in receipts)
        ) != coverage:
            raise IntegrityError(
                "Manifest coverage differs from receipts", "coverage_mismatch"
            )
        known = {
            "schema_name", "schema_version", "run_id", "created_at", "pass_kind",
            "reward_eligible", "acquisition_backend", "targets", "provenance",
            "coverage", "shards",
        }
        warnings = {
            f"unknown_manifest_field:{key}" for key in manifest if key not in known
        }
        return run_id, backend, receipts, coverage, warnings

    @staticmethod
    def _validate_shard_set(
        receipts: tuple[ShardReceipt, ...], *, trace_dir: Path
    ) -> None:
        discovered = set(
            path.resolve() for path in (trace_dir / "shards").glob("*.jsonl")
        )
        expected = {receipt.path for receipt in receipts}
        if discovered != expected:
            missing = sorted(path.as_posix() for path in expected - discovered)
            extra = sorted(path.as_posix() for path in discovered - expected)
            raise IntegrityError(
                f"Targeted trace shard set mismatch (missing={missing}, extra={extra})",
                "targeted_shard_set_mismatch",
            )

    def _validate_shard(
        self,
        receipt: ShardReceipt,
        *,
        run_id: str,
        on_event: Callable[[ValidatedTargetedEvent], None] | None,
    ) -> AcquisitionCoverage:
        state = _ShardState(receipt, run_id, on_event)
        byte_count = 0
        file_hash = hashlib.sha256()
        try:
            stream = receipt.path.open("rb")
        except OSError as error:
            raise IntegrityError(
                "Cannot open targeted trace shard", "missing_targeted_shard"
            ) from error
        with stream:
            for raw_line in stream:
                byte_count += len(raw_line)
                file_hash.update(raw_line)
                state.consume(_decode_line(raw_line, saw_end=state.saw_end))
        return _finish_shard(state, file_hash.hexdigest(), byte_count)

    def _replay_events(
        self,
        receipts: tuple[ShardReceipt, ...],
        *,
        run_id: str,
        on_event: Callable[[ValidatedTargetedEvent], None],
    ) -> None:
        for receipt in receipts:
            repeated = self._validate_shard(
                receipt, run_id=run_id, on_event=on_event
            )
            if repeated != receipt.coverage:
                raise IntegrityError(
                    "Shard changed during ingestion", "targeted_shard_changed"
                )


def _validate_report_contract(targeted: Mapping[str, Any]) -> None:
    if targeted.get("valid") is not True:
        raise IntegrityError(
            "Magpie marked targeted trace invalid", "invalid_targeted_trace"
        )
    if targeted.get("reward_eligible") is not False:
        raise IntegrityError(
            "Targeted trace must be diagnostic-only",
            "reward_eligible_diagnostic_trace",
        )
    issues = targeted.get("issues", [])
    if not isinstance(issues, list) or issues:
        raise IntegrityError(
            "Targeted trace reports acquisition issues", "invalid_targeted_trace"
        )


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_name") != SCHEMA_NAME
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise IntegrityError(
            "Unsupported TargetedKernelTrace schema", "unsupported_schema"
        )
    if (
        manifest.get("pass_kind") != "diagnostic"
        or manifest.get("reward_eligible") is not False
    ):
        raise IntegrityError(
            "Targeted trace manifest is not diagnostic-only", "invalid_targeted_trace"
        )
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise IntegrityError(
            "Targeted trace manifest has no targets", "invalid_targeted_trace"
        )
    if any(
        not isinstance(target, Mapping)
        or not str(target.get("target_id", "")).strip()
        for target in targets
    ):
        raise IntegrityError(
            "Targeted trace contains an invalid target", "invalid_targeted_trace"
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise IntegrityError(
            "Targeted trace provenance is malformed", "invalid_targeted_trace"
        )
    warnings = provenance.get("adapter_warnings", [])
    if (
        not isinstance(warnings, list)
        or any(not isinstance(item, str) for item in warnings)
        or warnings
    ):
        raise IntegrityError(
            "Targeted trace acquisition was incomplete", "invalid_targeted_trace"
        )


def _parse_shard_receipts(
    raw_receipts: object, *, workspace: Path, trace_dir: Path
) -> tuple[ShardReceipt, ...]:
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise IntegrityError(
            "Targeted trace manifest has no shard receipts", "missing_targeted_shards"
        )
    receipts: list[ShardReceipt] = []
    paths: set[Path] = set()
    for raw in raw_receipts:
        if not isinstance(raw, Mapping):
            raise IntegrityError(
                "Malformed targeted shard receipt", "invalid_shard_receipt"
            )
        path = resolve_trace_path(
            raw.get("path"), workspace=workspace, trace_dir=trace_dir
        )
        if path in paths:
            raise IntegrityError(
                "Duplicate targeted shard receipt", "invalid_shard_receipt"
            )
        paths.add(path)
        receipts.append(_parse_shard_receipt(raw, path, workspace))
    receipts.sort(key=lambda item: (item.rank, item.pid, item.relative_path))
    return tuple(receipts)


def _parse_shard_receipt(
    raw: Mapping[str, Any], path: Path, workspace: Path
) -> ShardReceipt:
    raw_coverage = raw.get("counters")
    if not isinstance(raw_coverage, Mapping):
        raise IntegrityError(
            "Shard receipt has no counters", "invalid_shard_receipt"
        )
    receipt = ShardReceipt(
        path=path,
        relative_path=path.relative_to(workspace).as_posix(),
        rank=strict_nonnegative_int(raw.get("rank"), "receipt rank"),
        pid=strict_nonnegative_int(raw.get("pid"), "receipt pid"),
        sequence_end=strict_nonnegative_int(raw.get("sequence_end"), "sequence_end"),
        chain_checksum=checked_sha256(raw.get("chain_checksum"), "chain checksum"),
        file_sha256=checked_sha256(raw.get("file_sha256"), "file sha256"),
        byte_count=strict_nonnegative_int(raw.get("byte_count"), "byte_count"),
        coverage=AcquisitionCoverage.from_mapping(raw_coverage),
        complete=raw.get("complete") is True,
    )
    if not receipt.complete:
        raise IntegrityError("Manifest contains incomplete shard", "incomplete_targeted_shard")
    return receipt


def _decode_line(raw_line: bytes, *, saw_end: bool) -> Mapping[str, Any]:
    if len(raw_line) > MAX_JSONL_LINE_BYTES:
        raise IntegrityError(
            "Targeted shard line exceeds safety limit", "oversize_targeted_record"
        )
    if not raw_line.endswith(b"\n"):
        raise IntegrityError(
            "Targeted shard has a corrupt tail", "corrupt_targeted_tail"
        )
    if saw_end:
        raise IntegrityError(
            "Targeted shard has data after end sentinel", "invalid_end_sentinel"
        )
    try:
        envelope = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Targeted shard contains corrupt JSON", "corrupt_targeted_tail"
        ) from error
    if not isinstance(envelope, Mapping):
        raise IntegrityError(
            "Targeted shard envelope is not an object", "invalid_targeted_envelope"
        )
    return envelope


def _finish_shard(
    state: _ShardState, file_sha256: str, byte_count: int
) -> AcquisitionCoverage:
    if not state.saw_header or not state.saw_end or state.final_coverage is None:
        raise IntegrityError(
            "Targeted shard is missing its end sentinel", "missing_end_sentinel"
        )
    actual = (
        state.expected_sequence - 1,
        state.previous_checksum,
        file_sha256,
        byte_count,
        state.final_coverage,
    )
    expected = (
        state.receipt.sequence_end,
        state.receipt.chain_checksum,
        state.receipt.file_sha256,
        state.receipt.byte_count,
        state.receipt.coverage,
    )
    if actual != expected:
        raise IntegrityError(
            "Targeted shard does not match its receipt",
            "targeted_shard_receipt_mismatch",
        )
    return state.final_coverage


def _validate_report_coverage(
    targeted: Mapping[str, Any], coverage: AcquisitionCoverage
) -> None:
    raw = targeted.get("coverage")
    if not isinstance(raw, Mapping) or AcquisitionCoverage.from_mapping(raw) != coverage:
        raise IntegrityError(
            "Benchmark report coverage differs from manifest", "coverage_mismatch"
        )


def _validate_summary(
    path: Path,
    run_id: str,
    coverage: AcquisitionCoverage,
    receipts: tuple[ShardReceipt, ...],
    semantic_quality: SemanticQualityAccumulator,
) -> tuple[bool, tuple[str, ...]]:
    summary = read_object(path, "TargetedKernelTrace summary")
    if (
        summary.get("schema_name") != SCHEMA_NAME
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("run_id") != run_id
        or summary.get("valid") is not True
        or summary.get("streaming") is not True
    ):
        raise IntegrityError(
            "Targeted trace summary is invalid", "invalid_targeted_summary"
        )
    issues = summary.get("issues")
    if not isinstance(issues, list) or issues:
        raise IntegrityError(
            "Targeted trace summary reports issues", "invalid_targeted_summary"
        )
    raw = summary.get("coverage")
    if not isinstance(raw, Mapping) or AcquisitionCoverage.from_mapping(raw) != coverage:
        raise IntegrityError("Targeted summary coverage mismatch", "coverage_mismatch")
    _validate_summary_shards(summary.get("shards"), receipts)
    return validate_semantic_quality(
        summary.get("evidence_quality"), coverage, semantic_quality
    )


def _validate_summary_shards(
    raw_shards: object, receipts: tuple[ShardReceipt, ...]
) -> None:
    if not isinstance(raw_shards, list) or len(raw_shards) != len(receipts):
        raise IntegrityError(
            "Targeted summary shard count mismatch", "invalid_targeted_summary"
        )
    by_name = {receipt.path.name: receipt for receipt in receipts}
    for raw in raw_shards:
        if (
            not isinstance(raw, Mapping)
            or raw.get("valid") is not True
            or raw.get("complete") is not True
        ):
            raise IntegrityError(
                "Targeted summary contains an invalid shard", "invalid_targeted_summary"
            )
        receipt = by_name.get(Path(str(raw.get("path", ""))).name)
        counters = raw.get("counters")
        if not _summary_receipt_matches(raw, counters, receipt):
            raise IntegrityError(
                "Targeted summary receipt mismatch", "invalid_targeted_summary"
            )


def _summary_receipt_matches(
    raw: Mapping[str, Any], counters: object, receipt: ShardReceipt | None
) -> bool:
    return bool(
        receipt is not None
        and isinstance(counters, Mapping)
        and AcquisitionCoverage.from_mapping(counters) == receipt.coverage
        and raw.get("file_sha256") == receipt.file_sha256
        and raw.get("chain_checksum") == receipt.chain_checksum
        and raw.get("byte_count") == receipt.byte_count
        and raw.get("sequence_end") == receipt.sequence_end
        and raw.get("rank") == receipt.rank
        and raw.get("pid") == receipt.pid
    )


def _artifact(
    kind: str, path: Path, workspace: Path, media_type: str
) -> EvidenceArtifactReceipt:
    return EvidenceArtifactReceipt(
        kind,
        path.relative_to(workspace).as_posix(),
        sha256_file(path),
        path.stat().st_size,
        media_type,
    )


def _assert_metadata_unchanged(
    expected: tuple[EvidenceArtifactReceipt, EvidenceArtifactReceipt],
    manifest_path: Path,
    summary_path: Path,
    workspace: Path,
) -> None:
    actual = (
        _artifact("targeted_manifest", manifest_path, workspace, "application/json"),
        _artifact("targeted_summary", summary_path, workspace, "application/json"),
    )
    if actual != expected:
        raise IntegrityError(
            "Targeted metadata changed during ingestion", "targeted_metadata_changed"
        )


__all__ = [
    "AcquisitionCoverage",
    "EvidenceArtifactReceipt",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TargetedTraceValidator",
    "ValidatedTargetedEvent",
    "ValidatedTargetedTrace",
]
