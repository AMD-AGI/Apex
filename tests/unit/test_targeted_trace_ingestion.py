from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from apex.core import IntegrityError, canonical_json_bytes, sha256_bytes, sha256_json
from apex.diagnostics import MagpieTraceEvidenceAdapter, TraceEvidence, TraceEvidenceNormalizer
from apex.ports import DiagnosticsRequest


PROVENANCE = "a" * 64
SCHEMA_NAME = "magpie.targeted-kernel-trace"
SCHEMA_VERSION = "1.0.0"


def _envelope(
    record_type: str, sequence: int, previous_checksum: str, payload: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "sequence": sequence,
        "previous_checksum": previous_checksum,
        "payload": payload,
    }
    body["checksum"] = sha256_json(body)
    return body


def _event(
    *,
    symbol: str = "triton_kernel",
    shape: list[int] | None = None,
    dtype: str = "torch.float16",
    duration_us: float = 10.0,
    key: str = "event-1",
) -> dict[str, Any]:
    shape = shape or [16, 128]
    return {
        "kind": "torch_profiler_kernel",
        "stable_event_key": key,
        "identity": {
            "run_id": "run-1",
            "target_id": "workload-kernels",
            "variant_id": "baseline",
            "package": "aiter",
            "image": "example@sha256:deadbeef",
            "source_hashes": {},
            "provenance_hashes": {},
        },
        "context": {
            "framework": "vllm",
            "rank": 1,
            "pid": 22,
            "framework_version": "0.19.1",
            "stage": "decode",
            "execution_mode": "graph",
            "graph_id": "7",
            "world_size": 2,
        },
        "semantics": {
            "source": {
                "path": "aiter/ops/kernel.py",
                "line": 4,
                "function": "launch",
                "sha256": "d" * 64,
            },
            "tensors": [
                {
                    "name": "x",
                    "shape": shape,
                    "dtype": dtype,
                    "stride": [shape[-1], 1],
                    "device": "cuda:0",
                    "layout": "torch.strided",
                    "requires_grad": False,
                }
            ],
            "named_scalars": {"scale": 0.5},
            "constexpr": {"BLOCK": 128},
            "meta": {"num_warps": 4},
            "python_grid": {"items": [32, 1, 1]},
        },
        "runtime": {
            "cpu_uid": "cpu-1",
            "correlation_id": "99",
            "gpu_uid": "gpu-1",
            "gpu_symbol": symbol,
            "grid": [32, 1, 1],
            "block": [256, 1, 1],
            "stream": "7",
            "duration_us": duration_us,
            "timestamp_us": 100.0,
        },
        "timestamp_ns": 100_000,
        "warnings": [],
    }


def _coverage(
    *, seen: int, sampled: int, written: int, dropped: int, reasons: dict[str, int]
) -> dict[str, Any]:
    return {
        "seen": seen,
        "sampled": sampled,
        "written": written,
        "dropped": dropped,
        "dropped_by_reason": reasons,
    }


def _write_fixture(
    workspace: Path,
    *,
    events: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    aggregate_symbol: str = "triton_kernel",
) -> dict[str, Path]:
    events = events or [_event()]
    coverage = coverage or _coverage(
        seen=len(events), sampled=len(events), written=len(events), dropped=0, reasons={}
    )
    source_root = workspace / "repos" / "aiter"
    source = source_root / "ops" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("def launch():\n    pass\n", encoding="utf-8")
    for payload in events:
        semantic_source = payload.get("semantics", {}).get("source")
        if isinstance(semantic_source, dict):
            semantic_source["sha256"] = sha256_bytes(source.read_bytes())

    gap_dir = workspace / "gap_analysis"
    gap_dir.mkdir(parents=True)
    gap_path = gap_dir / "gap_analysis.csv"
    with gap_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# Base directory: {workspace / 'repos'}\n")
        handle.write("# $AITER_DIR=./aiter\n#\n")
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Name",
                "Input Shapes",
                "kind",
                "category",
                "source_repo",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Name": aggregate_symbol,
                "Input Shapes": "[[16, 128]]",
                "kind": "triton_jit",
                "category": "attention",
                "source_repo": "aiter",
                "source_file": "$AITER_DIR/ops/kernel.py",
            }
        )

    trace_dir = workspace / "targeted_trace"
    shard_dir = trace_dir / "shards"
    shard_dir.mkdir(parents=True)
    shard_path = shard_dir / "rank-0001-pid-00000022.jsonl"
    envelopes: list[dict[str, Any]] = []
    previous = "0" * 64
    header = _envelope(
        "header",
        0,
        previous,
        {
            "run_id": "run-1",
            "rank": 1,
            "pid": 22,
            "run_seed": "seed",
            "sample_rate": 1.0,
            "max_records": 100,
            "metadata": {"framework": "vllm"},
        },
    )
    envelopes.append(header)
    previous = header["checksum"]
    for payload in events:
        item = _envelope("event", len(envelopes), previous, payload)
        envelopes.append(item)
        previous = item["checksum"]
    end = _envelope(
        "end",
        len(envelopes),
        previous,
        {
            "run_id": "run-1",
            "rank": 1,
            "pid": 22,
            "counters": coverage,
            "end_reason": "complete",
        },
    )
    envelopes.append(end)
    shard_bytes = b"".join(canonical_json_bytes(item) + b"\n" for item in envelopes)
    shard_path.write_bytes(shard_bytes)
    receipt = {
        "path": str(shard_path),
        "rank": 1,
        "pid": 22,
        "sequence_end": len(envelopes) - 1,
        "chain_checksum": end["checksum"],
        "file_sha256": sha256_bytes(shard_bytes),
        "byte_count": len(shard_bytes),
        "counters": coverage,
        "complete": True,
    }
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
        "created_at": "2026-01-01T00:00:00Z",
        "pass_kind": "diagnostic",
        "reward_eligible": False,
        "acquisition_backend": "torch_profiler",
        "targets": [{"target_id": "workload-kernels", "name_patterns": ["*"]}],
        "provenance": {"adapter_warnings": [], "framework": "vllm"},
        "coverage": coverage,
        "shards": [receipt],
    }
    manifest_path = trace_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    summary_shard = {
        "path": str(shard_path),
        "valid": True,
        "complete": True,
        "event_count": len(events),
        "byte_count": len(shard_bytes),
        "file_sha256": receipt["file_sha256"],
        "sequence_end": receipt["sequence_end"],
        "chain_checksum": receipt["chain_checksum"],
        "counters": coverage,
        "rank": 1,
        "pid": 22,
        "issues": [],
    }
    lossless = coverage["seen"] > 0 and coverage["dropped"] == 0
    unresolved_reasons = [
        f"dropped:{reason}"
        for reason, count in sorted(coverage["dropped_by_reason"].items())
        if count
    ]
    summary = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
        "valid": True,
        "streaming": True,
        "coverage": coverage,
        "evidence_quality": {
            "evidence_class": "diagnostic_only",
            "resolution_status": "resolved" if lossless else "unresolved",
            "semantic_coverage_claimed": lossless,
            "record_coverage_fraction": (
                coverage["written"] / coverage["seen"] if coverage["seen"] else 0.0
            ),
            "lossless_record_coverage": lossless,
            "records_evaluated": coverage["written"],
            "records_with_complete_semantics": coverage["written"],
            "missing_by_field": {
                "phase": 0,
                "source": 0,
                "grid": 0,
                "shape": 0,
                "correlation": 0,
            },
            "cross_event_join": "not_performed",
            "join_eligible_records": coverage["written"],
            "unresolved_reasons": unresolved_reasons,
        },
        "events": {"by_target": {"workload-kernels": len(events)}},
        "integrity_failures_by_reason": {},
        "shards": [summary_shard],
        "issues": [],
    }
    summary_path = trace_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = {
        "success": True,
        "kernel_summary": [
            {
                "name": aggregate_symbol,
                "time_ms": 2.0,
                "percent": 20.0,
                "calls": 100,
            }
        ],
        "gap_analysis": {"csv_path": str(gap_path)},
        "targeted_trace": {
            "valid": True,
            "reward_eligible": False,
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
            "coverage": coverage,
            "events": summary["events"],
            "issues": [],
        },
    }
    report_path = workspace / "benchmark_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "report": report_path,
        "manifest": manifest_path,
        "summary": summary_path,
        "shard": shard_path,
        "source": source,
    }


def _rewrite_json(path: Path, update) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_ingestion_validates_and_joins_distinct_shapes_and_dtypes(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        events=[
            _event(shape=[16, 128], dtype="torch.float16", duration_us=10, key="a"),
            _event(shape=[32, 128], dtype="torch.bfloat16", duration_us=30, key="b"),
        ],
    )
    records = TraceEvidenceNormalizer().from_benchmark_report(
        paths["report"], provenance_hash=PROVENANCE
    )

    assert len(records) == 2
    assert len({record.candidate_id for record in records}) == 2
    assert {record.shape.input_dims for record in records} == {
        ((16, 128),),
        ((32, 128),),
    }
    assert {record.shape.dtypes for record in records} == {
        ("torch.float16",),
        ("torch.bfloat16",),
    }
    assert all(record.match_confidence == "exact" for record in records)
    assert all(record.phase == "decode" and record.rank == 1 for record in records)
    assert sum(record.volume.calls for record in records) == 100
    assert sum(record.volume.gpu_time_ms for record in records) == pytest.approx(2.0)
    assert sum(record.volume.gpu_time_pct for record in records) == pytest.approx(20.0)
    for record in records:
        assert TraceEvidence.from_mapping(record.to_dict()) == record
        assert record.kernel.patchable
        assert record.kernel.source_confidence == "exact_launch"
        assert record.shape.graph_mode == "cudagraph"
        kinds = {receipt.kind for receipt in record.evidence.artifacts}
        assert kinds == {
            "benchmark_report",
            "gap_analysis_csv",
            "targeted_manifest",
            "targeted_summary",
            "targeted_shard",
        }
        for receipt in record.evidence.artifacts:
            artifact = tmp_path / receipt.relative_path
            assert artifact.stat().st_size == receipt.byte_count
            assert sha256_bytes(artifact.read_bytes()) == receipt.sha256


def test_sampling_drop_uses_magpie_coverage_semantics(tmp_path: Path) -> None:
    coverage = _coverage(
        seen=2,
        sampled=1,
        written=1,
        dropped=1,
        reasons={"sampling": 1},
    )
    paths = _write_fixture(tmp_path, coverage=coverage)
    record = TraceEvidenceNormalizer().from_benchmark_report(
        paths["report"], provenance_hash=PROVENANCE
    )[0]
    assert record.evidence.coverage.to_dict() == coverage
    assert "targeted_drop:sampling:1" in record.evidence.warnings
    assert "semantic_coverage_unresolved:dropped:sampling" in record.evidence.warnings


def test_unknown_additive_manifest_field_is_hash_preserved_and_warned(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    _rewrite_json(paths["manifest"], lambda value: value.update(vendor_extension={"x": 1}))
    record = TraceEvidenceNormalizer().from_benchmark_report(
        paths["report"], provenance_hash=PROVENANCE
    )[0]
    assert "unknown_manifest_field:vendor_extension" in record.evidence.warnings
    manifest_receipt = next(
        item for item in record.evidence.artifacts if item.kind == "targeted_manifest"
    )
    assert manifest_receipt.sha256 == sha256_bytes(paths["manifest"].read_bytes())


def test_diagnostics_adapter_materializes_validated_evidence(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_fixture(benchmark)
    output = tmp_path / "normalized"
    result = MagpieTraceEvidenceAdapter().analyze(
        DiagnosticsRequest("run-1", benchmark, output, PROVENANCE)
    )
    assert result.succeeded
    payload = json.loads((output / "trace_evidence.json").read_text(encoding="utf-8"))
    assert payload["records"][0]["evidence"]["acquisition_schema"] == "TargetedKernelTrace"
    assert result.summary["artifact_receipt_count"] == 5
    assert len(payload["artifact_receipts"]) == 5
    assert {
        receipt["kind"] for receipt in payload["records"][0]["evidence"]["artifacts"]
    } >= {"targeted_manifest", "targeted_shard"}


def test_diagnostics_adapter_excludes_disposable_runtime_tree(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_fixture(benchmark)
    runtime_file = benchmark / "inferencex_runtime" / "unrelated.py"
    runtime_file.parent.mkdir()
    runtime_file.write_text("unrelated\n", encoding="utf-8")

    result = MagpieTraceEvidenceAdapter().analyze(
        DiagnosticsRequest("run-1", benchmark, tmp_path / "normalized", PROVENANCE)
    )

    assert result.succeeded
    assert runtime_file not in result.artifacts
    assert all("inferencex_runtime" not in path.parts for path in result.artifacts)


def test_terminal_diagnostics_preserves_declared_raw_trace_and_reports(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    paths = _write_fixture(benchmark)
    raw_trace = benchmark / "torch_trace" / "rank0.pt.trace.json.gz"
    raw_trace.parent.mkdir()
    raw_trace.write_bytes(b"raw-trace")
    report_root = benchmark / "tracelens" / "decode_only"
    report_root.mkdir(parents=True)
    report_csv = report_root / "ops_summary.csv"
    report_csv.write_text("name,time ms\nkernel,1.0\n", encoding="utf-8")
    _rewrite_json(
        paths["report"],
        lambda value: value.update(
            tracelens_analysis={
                "enabled": True,
                "rank0_trace": str(raw_trace),
                "output_dir": str(benchmark / "tracelens"),
                "output_files": [str(report_csv)],
            }
        ),
    )

    result = MagpieTraceEvidenceAdapter().analyze(
        DiagnosticsRequest(
            "run-1",
            benchmark,
            tmp_path / "normalized",
            PROVENANCE,
            preserve_raw_trace=True,
        )
    )

    assert result.succeeded
    assert result.summary["raw_trace_preserved"] is True
    assert raw_trace.resolve() in result.artifacts
    assert report_csv.resolve() in result.artifacts
    assert result.artifact_roles[str(raw_trace.resolve())] == "diagnostic_raw_trace"
    assert result.artifact_roles[str(report_csv.resolve())] == (
        "diagnostic_tracelens_report"
    )
    manifest = result.summary["raw_artifact_manifest"]
    assert {item["role"] for item in manifest} >= {
        "diagnostic_raw_trace",
        "diagnostic_tracelens_report",
        "diagnostic_benchmark_report",
    }
    assert {item["comparison_logical_path"] for item in manifest} == {
        "metadata/benchmark_report.json",
        "raw/rank0.pt.trace.json.gz",
        "reports/decode_only/ops_summary.csv",
    }


def test_exact_symbol_join_does_not_guess_by_substring(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        events=[_event(symbol="triton_kernel_variant")],
        aggregate_symbol="triton_kernel",
    )
    records = TraceEvidenceNormalizer().from_benchmark_report(
        paths["report"], provenance_hash=PROVENANCE
    )
    assert len(records) == 2
    by_name = {record.kernel.runtime_name: record for record in records}
    assert by_name["triton_kernel_variant"].match_confidence == "unknown"
    assert "aggregate_profiler_row_unmatched" in by_name["triton_kernel_variant"].evidence.warnings
    assert by_name["triton_kernel"].match_confidence == "probable"
    assert "targeted_record_unmatched" in by_name["triton_kernel"].evidence.warnings


def test_resolved_launch_source_receipt_is_verified(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    paths["source"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(IntegrityError) as captured:
        TraceEvidenceNormalizer().from_benchmark_report(
            paths["report"], provenance_hash=PROVENANCE
        )
    assert captured.value.reason_code == "targeted_source_digest_mismatch"


@pytest.mark.parametrize(
    "mutation, reason_code",
    [
        ("checksum", "targeted_checksum_mismatch"),
        ("sentinel", "corrupt_targeted_tail"),
        ("receipt", "targeted_shard_receipt_mismatch"),
        ("manifest_coverage", "invalid_trace_coverage"),
        ("summary_coverage", "coverage_mismatch"),
        ("report_coverage", "coverage_mismatch"),
        ("schema", "unsupported_schema"),
        ("extra_shard", "targeted_shard_set_mismatch"),
        ("semantic_quality", "invalid_targeted_summary"),
        ("semantic_forgery", "invalid_targeted_summary"),
    ],
)
def test_integrity_failures_are_fail_closed(
    tmp_path: Path, mutation: str, reason_code: str
) -> None:
    paths = _write_fixture(tmp_path)
    if mutation == "checksum":
        lines = paths["shard"].read_text(encoding="utf-8").splitlines()
        event = json.loads(lines[1])
        event["payload"]["runtime"]["gpu_symbol"] = "tampered"
        lines[1] = json.dumps(event, sort_keys=True, separators=(",", ":"))
        paths["shard"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif mutation == "sentinel":
        paths["shard"].write_bytes(paths["shard"].read_bytes()[:-10])
    elif mutation == "receipt":
        _rewrite_json(paths["manifest"], lambda value: value["shards"][0].update(file_sha256="f" * 64))
    elif mutation == "manifest_coverage":
        _rewrite_json(paths["manifest"], lambda value: value["coverage"].update(sampled=0))
    elif mutation == "summary_coverage":
        _rewrite_json(
            paths["summary"],
            lambda value: value["coverage"].update(seen=0, sampled=0, written=0),
        )
    elif mutation == "report_coverage":
        _rewrite_json(
            paths["report"],
            lambda value: value["targeted_trace"]["coverage"].update(
                seen=0, sampled=0, written=0
            ),
        )
    elif mutation == "schema":
        _rewrite_json(paths["manifest"], lambda value: value.update(schema_version="2.0.0"))
    elif mutation == "extra_shard":
        (paths["shard"].parent / "unexpected.jsonl").write_text("{}\n", encoding="utf-8")
    elif mutation == "semantic_quality":
        _rewrite_json(
            paths["summary"],
            lambda value: value["evidence_quality"].update(
                semantic_coverage_claimed=False
            ),
        )
    elif mutation == "semantic_forgery":
        def forge(value):
            quality = value["evidence_quality"]
            quality["resolution_status"] = "unresolved"
            quality["semantic_coverage_claimed"] = False
            quality["records_with_complete_semantics"] = 0
            quality["missing_by_field"]["source"] = 1
            quality["unresolved_reasons"] = ["missing:source"]

        _rewrite_json(paths["summary"], forge)

    with pytest.raises(IntegrityError) as captured:
        TraceEvidenceNormalizer().from_benchmark_report(
            paths["report"], provenance_hash=PROVENANCE
        )
    assert captured.value.reason_code == reason_code


def test_default_normalizer_requires_targeted_trace(tmp_path: Path) -> None:
    report = tmp_path / "benchmark_report.json"
    report.write_text(json.dumps({"success": True, "kernel_summary": []}), encoding="utf-8")
    with pytest.raises(IntegrityError) as captured:
        TraceEvidenceNormalizer().from_benchmark_report(report, provenance_hash=PROVENANCE)
    assert captured.value.reason_code == "missing_targeted_trace"
