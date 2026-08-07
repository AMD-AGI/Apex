"""Benchmark and diagnostic actions shared by E2E search and finalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apex.benchmark import NormalizedBenchmarkResult, QualityMetric
from apex.core import ContractError, sha256_file
from apex.diagnostics import TraceEvidence
from apex.evaluation import E2EMeasurement
from apex.ports import BenchmarkPass, BenchmarkRequest, DiagnosticsRequest, DiagnosticsResult
from apex.runtime import RunProvenance
from apex.storage import ArtifactReceipt

from .kernel_lane import KernelOpportunityPlan, build_kernel_opportunity_plan
from .oracles import CorrectnessOracleRegistry
from .run_record import E2ERunRecord


class BenchmarkAdapter(Protocol):
    def run_normalized(self, request: BenchmarkRequest) -> NormalizedBenchmarkResult: ...


class DiagnosticsAdapter(Protocol):
    def analyze(self, request: DiagnosticsRequest) -> DiagnosticsResult: ...


@dataclass(frozen=True, slots=True)
class Diagnosis:
    plan: KernelOpportunityPlan
    evidence_path: Path
    evidence_receipt: ArtifactReceipt


class E2EBenchmarkSession:
    """Execute actions and bind normalized evidence to the run journal/CAS."""

    def __init__(
        self,
        *,
        benchmark: BenchmarkAdapter,
        diagnostics: DiagnosticsAdapter,
        record: E2ERunRecord,
        provenance: RunProvenance,
        protocol_hash: str,
        max_kernels: int,
        correctness_oracles: CorrectnessOracleRegistry | None = None,
    ) -> None:
        self._benchmark = benchmark
        self._diagnostics = diagnostics
        self.record = record
        self.provenance = provenance
        self.protocol_hash = protocol_hash
        self.max_kernels = max_kernels
        self.correctness_oracles = correctness_oracles

    def action(
        self,
        action_id: str,
        config: Path,
        pass_type: BenchmarkPass,
    ) -> tuple[NormalizedBenchmarkResult, ArtifactReceipt]:
        self.record.begin_action(action_id, f"benchmark-{pass_type.value}")
        result = self._benchmark.run_normalized(
            BenchmarkRequest(
                run_id=action_id,
                config_path=config,
                output_dir=self.record.root / "benchmarks",
                pass_type=pass_type,
                timeout_seconds=7200,
            )
        )
        return result, self.record.record_benchmark(action_id, result)

    def measure(
        self,
        action_id: str,
        config: Path,
    ) -> tuple[NormalizedBenchmarkResult, E2EMeasurement | None, ArtifactReceipt]:
        result, receipt = self.action(action_id, config, BenchmarkPass.MEASUREMENT)
        measurement = (
            measurement_from_result(result, self.protocol_hash, receipt.digest)
            if result.succeeded
            else None
        )
        return result, measurement, receipt

    def diagnose(self, action_id: str, config: Path) -> Diagnosis:
        result, _ = self.action(action_id, config, BenchmarkPass.DIAGNOSTIC)
        if not result.succeeded:
            raise ContractError("Diagnostic benchmark failed", "diagnostic_benchmark_failed")
        diagnostic = self._diagnostics.analyze(
            DiagnosticsRequest(
                self.record.run_id,
                result.workspace_path,
                self.record.root / "diagnostics" / action_id,
                self.provenance.digest,
            )
        )
        receipts = self.record.record_diagnostics(diagnostic)
        path = _evidence_path(diagnostic)
        plan = build_kernel_opportunity_plan(
            load_evidence(path),
            max_kernels=self.max_kernels,
            correctness_oracles=self.correctness_oracles,
        )
        digest = sha256_file(path)
        receipt = next((item for item in receipts if item.digest == digest), None)
        if receipt is None:
            raise ContractError(
                "Trace evidence was not stored in CAS", "trace_evidence_not_recorded"
            )
        return Diagnosis(plan, path, receipt)


def measurement_from_result(
    result: NormalizedBenchmarkResult,
    protocol_hash: str,
    receipt: str,
) -> E2EMeasurement:
    throughput = (
        result.throughput.total_tokens_per_second
        if result.throughput.total_tokens_per_second is not None
        else result.throughput.output_tokens_per_second
    )
    ttft = result.latency.ttft.p99_ms
    tpot = result.latency.tpot.p99_ms
    completed = result.throughput.completed_requests
    quality = primary_quality(result.quality.metrics)
    if throughput is None or ttft is None or tpot is None or completed is None or quality is None:
        raise ContractError("Benchmark lacks required E2E metrics", "e2e_metrics_missing")
    return E2EMeasurement(
        float(throughput),
        float(ttft),
        float(tpot),
        quality.value,
        int(completed),
        protocol_hash,
        receipt,
        receipt,
    )


def primary_quality(metrics: tuple[QualityMetric, ...]) -> QualityMetric | None:
    eligible = tuple(item for item in metrics if item.higher_is_better)
    for name in ("exact_match", "acc_norm", "acc"):
        match = next(
            (item for item in eligible if item.name.split(",", 1)[0] == name),
            None,
        )
        if match:
            return match
    return eligible[0] if eligible else None


def measurement_metrics(value: E2EMeasurement) -> dict[str, float]:
    return {
        "throughput": value.throughput,
        "ttft_p99_ms": value.ttft_p99_ms,
        "tpot_p99_ms": value.tpot_p99_ms,
        "accuracy": value.accuracy,
    }


def load_evidence(path: Path) -> tuple[TraceEvidence, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value["records"]
        if not isinstance(records, list):
            raise TypeError("records")
        return tuple(TraceEvidence.from_mapping(item) for item in records)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ContractError(
            "Normalized trace evidence is invalid", "invalid_trace_evidence"
        ) from error


def _evidence_path(result: DiagnosticsResult) -> Path:
    value = result.summary.get("evidence_path")
    if not isinstance(value, str) or not value:
        raise ContractError("Diagnostic evidence is missing", "diagnostic_evidence_missing")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ContractError("Diagnostic evidence path is unsafe", "diagnostic_evidence_missing")
    return path


__all__ = [
    "BenchmarkAdapter",
    "Diagnosis",
    "DiagnosticsAdapter",
    "E2EBenchmarkSession",
    "load_evidence",
    "measurement_from_result",
    "measurement_metrics",
    "primary_quality",
]
