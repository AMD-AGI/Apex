"""Persist one benchmark's exact inputs and independently typed evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apex.benchmark import NormalizedBenchmarkResult
from apex.core import ContractError, IntegrityError, canonical_json_bytes
from apex.storage import ArtifactReceipt, ArtifactStore

from .benchmark_document import benchmark_document


@dataclass(frozen=True, slots=True)
class BenchmarkEvidenceReceipts:
    """CAS receipts used to replay a benchmark without trusting event metrics."""

    normalized: ArtifactReceipt
    quality: ArtifactReceipt
    config: ArtifactReceipt
    bindings: tuple[dict[str, object], ...]
    receipts: tuple[ArtifactReceipt, ...]


def persist_benchmark_evidence(
    store: ArtifactStore,
    result: NormalizedBenchmarkResult,
    config_path: Path,
) -> BenchmarkEvidenceReceipts:
    """Store exact config, raw artifacts, and normalized projections separately."""

    config = _put_exact_config(store, config_path)
    raw = _put_raw_artifacts(store, result)
    normalized = store.put_bytes(
        canonical_json_bytes(benchmark_document(result)),
        media_type="application/json",
    )
    quality = store.put_bytes(
        canonical_json_bytes(_quality_document(result, raw)),
        media_type="application/json",
    )
    bindings = (
        _binding("benchmark_config", config),
        _binding("normalized_benchmark", normalized),
        _binding("quality_evidence", quality),
        *(_binding(role, receipt) for role, _path, receipt in raw),
    )
    receipts = _unique_receipts(
        (config, normalized, quality, *(item[2] for item in raw))
    )
    return BenchmarkEvidenceReceipts(
        normalized=normalized,
        quality=quality,
        config=config,
        bindings=bindings,
        receipts=receipts,
    )


def _put_exact_config(store: ArtifactStore, path: Path) -> ArtifactReceipt:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ContractError(
            "Benchmark config must be an absolute regular file",
            "invalid_benchmark_config",
        )
    return store.put_file(path, media_type=_media_type(path))


def _put_raw_artifacts(
    store: ArtifactStore, result: NormalizedBenchmarkResult
) -> tuple[tuple[str, Path, ArtifactReceipt], ...]:
    roles = _artifact_roles(result)
    stored: list[tuple[str, Path, ArtifactReceipt]] = []
    seen: set[Path] = set()
    for raw_path in result.artifacts:
        if raw_path.is_symlink() or not raw_path.is_file():
            raise IntegrityError(
                "Benchmark evidence must be a regular file",
                "unsafe_benchmark_artifact",
            )
        path = raw_path.resolve()
        if path in seen:
            continue
        seen.add(path)
        receipt = store.put_file(path, media_type=_media_type(path))
        stored.append(
            (roles.get(path, "benchmark_side_evidence"), path, receipt)
        )
    return tuple(stored)


def _artifact_roles(result: NormalizedBenchmarkResult) -> dict[Path, str]:
    roles: dict[Path, str] = {}
    for path in result.quality.raw_artifact_paths:
        roles[path.resolve()] = (
            "quality_sample"
            if path.name.startswith("samples")
            else "quality_raw_artifact"
        )
    for path in result.quality.source_paths:
        roles[path.resolve()] = "quality_result"
    if result.report_path is not None:
        roles[result.report_path.resolve()] = "benchmark_report"
    return roles


def _quality_document(
    result: NormalizedBenchmarkResult,
    raw: tuple[tuple[str, Path, ArtifactReceipt], ...],
) -> dict[str, Any]:
    quality = result.quality
    by_path = {path: receipt for _role, path, receipt in raw}
    return {
        "schema": "apex.e2e-quality-evidence/v1",
        "required": quality.required,
        "kind": quality.kind,
        "passed": quality.passed,
        "metrics": [asdict(item) for item in quality.metrics],
        "primary_metrics": [asdict(item) for item in quality.primary_metrics],
        "error": quality.error,
        "outcome_digest": quality.outcome_digest,
        "sample_set_digest": quality.sample_set_digest,
        "result_receipts": _receipts_for(quality.source_paths, by_path),
        "raw_artifact_receipts": _receipts_for(
            quality.raw_artifact_paths, by_path
        ),
    }


def _receipts_for(
    paths: tuple[Path, ...], by_path: dict[Path, ArtifactReceipt]
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in paths:
        receipt = by_path.get(path.resolve())
        if receipt is None:
            raise IntegrityError(
                "Normalized quality references an unstored raw artifact",
                "quality_artifact_receipt_missing",
            )
        receipts.append(receipt.to_dict())
    return receipts


def _unique_receipts(
    receipts: tuple[ArtifactReceipt, ...],
) -> tuple[ArtifactReceipt, ...]:
    unique: dict[str, ArtifactReceipt] = {}
    for receipt in receipts:
        unique.setdefault(receipt.digest, receipt)
    return tuple(unique.values())


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".csv": "text/csv",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".gz": "application/gzip",
    }.get(path.suffix.lower(), "application/octet-stream")


__all__ = ["BenchmarkEvidenceReceipts", "persist_benchmark_evidence"]
