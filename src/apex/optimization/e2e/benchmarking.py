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
from apex.ports import (
    BenchmarkPass,
    BenchmarkRequest,
    DiagnosticsRequest,
    DiagnosticsResult,
    TraceComparisonPort,
    TraceComparisonRequest,
    TraceDiagnosticEvidence,
)
from apex.runtime import RunProvenance
from apex.storage import ArtifactReceipt

from .kernel_lane import KernelOpportunityPlan, build_kernel_opportunity_plan
from .oracles import CorrectnessOracleRegistry
from .recovery import persist_diagnosis, write_action_completion
from .benchmark_artifacts import BenchmarkEvidenceReceipts
from .run_record import E2ERunRecord
from .trace_inputs import build_trace_diagnostic_evidence


class BenchmarkAdapter(Protocol):
    def run_normalized(self, request: BenchmarkRequest) -> NormalizedBenchmarkResult: ...


class DiagnosticsAdapter(Protocol):
    def analyze(self, request: DiagnosticsRequest) -> DiagnosticsResult: ...


@dataclass(frozen=True, slots=True)
class Diagnosis:
    plan: KernelOpportunityPlan
    evidence_path: Path
    evidence_receipt: ArtifactReceipt
    state_receipt: ArtifactReceipt
    trace_diagnostic_evidence: TraceDiagnosticEvidence


@dataclass(frozen=True, slots=True)
class TerminalDiagnosticsOutcome:
    benchmark_receipt: str
    diagnostic_succeeded: bool
    diagnostic_artifact_receipts: tuple[str, ...]
    raw_trace_preserved: bool
    comparison_status: str
    comparison_reason_code: str
    comparison_receipt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_receipt": self.benchmark_receipt,
            "diagnostic_succeeded": self.diagnostic_succeeded,
            "diagnostic_artifact_receipts": list(
                self.diagnostic_artifact_receipts
            ),
            "raw_trace_preserved": self.raw_trace_preserved,
            "comparison": {
                "status": self.comparison_status,
                "reason_code": self.comparison_reason_code,
                "receipt": self.comparison_receipt,
                "reward_eligible": False,
            },
        }


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
        trace_comparison: TraceComparisonPort,
        correctness_oracles: CorrectnessOracleRegistry | None = None,
    ) -> None:
        self._benchmark = benchmark
        self._diagnostics = diagnostics
        self.record = record
        self.provenance = provenance
        self.protocol_hash = protocol_hash
        self.max_kernels = max_kernels
        self.trace_comparison = trace_comparison
        self.correctness_oracles = correctness_oracles

    def action(
        self,
        action_id: str,
        config: Path,
        pass_type: BenchmarkPass,
        *,
        attempt_id: str | None = None,
        candidate_id: str | None = None,
        opportunity_id: str | None = None,
    ) -> tuple[NormalizedBenchmarkResult, BenchmarkEvidenceReceipts]:
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
        evidence = self.record.record_benchmark(
            action_id,
            result,
            config,
            attempt_id=attempt_id,
            candidate_id=candidate_id,
            opportunity_id=opportunity_id,
        )
        write_action_completion(
            self.record,
            action_id=action_id,
            normalized=evidence.normalized,
            succeeded=result.succeeded,
            errors=result.errors,
        )
        return result, evidence

    def measure(
        self,
        action_id: str,
        config: Path,
        *,
        attempt_id: str | None = None,
        candidate_id: str | None = None,
        opportunity_id: str | None = None,
    ) -> tuple[NormalizedBenchmarkResult, E2EMeasurement | None, ArtifactReceipt]:
        result, evidence = self.action(
            action_id,
            config,
            BenchmarkPass.MEASUREMENT,
            attempt_id=attempt_id,
            candidate_id=candidate_id,
            opportunity_id=opportunity_id,
        )
        measurement = None
        if result.succeeded:
            try:
                measurement = measurement_from_result(
                    result,
                    self.protocol_hash,
                    quality_receipt=evidence.quality.digest,
                    measurement_receipt=evidence.normalized.digest,
                )
            except ContractError:
                if attempt_id is None:
                    raise
        return result, measurement, evidence.normalized

    def diagnose(
        self,
        action_id: str,
        config: Path,
        *,
        preserve_raw_trace: bool = False,
    ) -> Diagnosis:
        result, _ = self.action(action_id, config, BenchmarkPass.DIAGNOSTIC)
        if not result.succeeded:
            raise ContractError("Diagnostic benchmark failed", "diagnostic_benchmark_failed")
        diagnostic = self._diagnostics.analyze(
            DiagnosticsRequest(
                self.record.run_id,
                result.workspace_path,
                self.record.root / "diagnostics" / action_id,
                self.provenance.digest,
                preserve_raw_trace,
            )
        )
        receipts = self.record.record_diagnostics(
            diagnostic, action_id=action_id
        )
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
        comparison_evidence = build_trace_diagnostic_evidence(
            diagnostic,
            receipts,
            trace_evidence_sha256=digest,
            store=self.record.artifacts,
        )
        state_receipt = persist_diagnosis(
            self.record,
            evidence=receipt,
            plan=plan,
            trace_diagnostic_evidence=comparison_evidence,
        )
        return Diagnosis(
            plan,
            self.record.artifacts.root / receipt.relative_path,
            receipt,
            state_receipt,
            comparison_evidence,
        )

    def terminal_diagnostics(
        self,
        action_id: str,
        config: Path,
        *,
        baseline: Diagnosis,
    ) -> TerminalDiagnosticsOutcome:
        """Capture a profiler-on terminal observation with zero reward authority."""

        result, benchmark = self.action(
            action_id, config, BenchmarkPass.DIAGNOSTIC
        )
        diagnostic = DiagnosticsResult(
            self.record.run_id,
            False,
            (),
            {"raw_trace_preserved": False},
            "terminal_diagnostic_benchmark_failed",
        )
        if result.succeeded:
            diagnostic = self._diagnostics.analyze(
                DiagnosticsRequest(
                    self.record.run_id,
                    result.workspace_path,
                    self.record.root / "diagnostics" / action_id,
                    self.provenance.digest,
                    True,
                )
            )
        receipts = self.record.record_diagnostics(
            diagnostic,
            action_id=action_id,
            terminal=True,
        )
        evidence_digest = _diagnostic_evidence_digest(diagnostic, receipts)
        terminal_evidence = build_trace_diagnostic_evidence(
            diagnostic,
            receipts,
            trace_evidence_sha256=evidence_digest,
            store=self.record.artifacts,
        )
        comparison = self.trace_comparison.compare(
            TraceComparisonRequest(
                self.record.run_id,
                self.provenance.gpu_arch,
                baseline.trace_diagnostic_evidence,
                terminal_evidence,
                benchmark.normalized.digest,
                (self.record.root / "diagnostics" / action_id / "comparison").resolve(),
            )
        )
        comparison_receipt = self.record.record_trace_comparison(
            comparison, action_id=action_id
        )
        return TerminalDiagnosticsOutcome(
            benchmark.normalized.digest,
            diagnostic.succeeded,
            tuple(item.digest for item in receipts),
            diagnostic.summary.get("raw_trace_preserved") is True,
            comparison.status.value,
            comparison.reason_code,
            comparison_receipt.digest,
        )


def measurement_from_result(
    result: NormalizedBenchmarkResult,
    protocol_hash: str,
    *,
    quality_receipt: str,
    measurement_receipt: str,
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
        quality_receipt,
        measurement_receipt,
    )


def primary_quality(metrics: tuple[QualityMetric, ...]) -> QualityMetric | None:
    eligible = tuple(item for item in metrics if item.higher_is_better)
    for name in (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "exact_match",
        "acc_norm,none",
        "acc,none",
        "acc_norm",
        "acc",
    ):
        match = next(
            (item for item in eligible if item.name == name),
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


def _diagnostic_evidence_digest(
    result: DiagnosticsResult,
    receipts: tuple[ArtifactReceipt, ...],
) -> str | None:
    try:
        path = _evidence_path(result)
    except ContractError:
        return None
    digest = sha256_file(path)
    return digest if any(item.digest == digest for item in receipts) else None


__all__ = [
    "BenchmarkAdapter",
    "Diagnosis",
    "DiagnosticsAdapter",
    "E2EBenchmarkSession",
    "TerminalDiagnosticsOutcome",
    "load_evidence",
    "measurement_from_result",
    "measurement_metrics",
    "primary_quality",
]
