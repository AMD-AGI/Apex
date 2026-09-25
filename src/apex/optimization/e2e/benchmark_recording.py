"""Record one E2E benchmark plus optional local persistent-server lineage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apex.benchmark import NormalizedBenchmarkResult
from apex.runtime import GpuMeasurementBracketReceipt

from .benchmark_artifacts import (
    BenchmarkEvidenceReceipts,
    persist_formal_benchmark_evidence,
)
from .server_lineage import capture_local_server_lineage

def record_benchmark_result(
    record: Any,
    action_id: str,
    result: NormalizedBenchmarkResult,
    config_path: Path,
    *,
    attempt_id: str | None,
    candidate_id: str | None,
    opportunity_id: str | None,
    measurement_bracket: GpuMeasurementBracketReceipt | None,
    server_owner_kind: str,
    server_owner_id: str | None,
) -> BenchmarkEvidenceReceipts:
    """Persist evidence before atomically publishing its canonical observation."""

    evidence = persist_formal_benchmark_evidence(
        record.artifacts,
        result,
        config_path,
        run_id=record.run_id,
        action_id=action_id,
        measurement_bracket=measurement_bracket,
    )
    state = record.controller.state
    owner_id = server_owner_id or state.anchor_id
    server = capture_local_server_lineage(
        store=record.artifacts,
        events=record.iter_events(),
        result=result,
        evidence=evidence,
        run_id=record.run_id,
        action_id=action_id,
        owner_kind=server_owner_kind,
        owner_id=owner_id,
        anchor_id=state.anchor_id,
        anchor_generation=state.anchor_generation,
    )
    receipts = (*evidence.receipts, *((server.receipt,) if server else ()))
    record.controller.mark_artifacts_ready(
        action_id, [item.digest for item in receipts]
    )
    if result.succeeded:
        record.controller.verify_action(action_id, evidence.normalized.digest)
        record.controller.complete_action(action_id)
    else:
        record.controller.fail_action(
            action_id, ";".join(result.errors) or "benchmark_failed"
        )
    lineage = (
        {}
        if server is not None and server.document["lifecycle"] == "cleanup"
        else _attempt_lineage(record, attempt_id, candidate_id, opportunity_id)
    )
    bindings = [dict(item) for item in evidence.bindings]
    if server is not None:
        bindings.append(server.binding)
    record.controller.record_domain_event(
        "measurement_result",
        _measurement_payload(action_id, result, evidence, lineage, bindings, server),
        idempotency_key=f"benchmark.{action_id}.measurement",
    )
    return evidence


def _measurement_payload(
    action_id: str,
    result: NormalizedBenchmarkResult,
    evidence: BenchmarkEvidenceReceipts,
    lineage: dict[str, object],
    bindings: list[dict[str, object]],
    server: Any,
) -> dict[str, object]:
    return {
        **lineage,
        "action_id": action_id,
        "pass_type": result.pass_type.value,
        "succeeded": result.succeeded,
        "metrics": {
            key: value
            for key, value in result.metric_mapping().items()
            if value is not None
        },
        "evidence_class": (
            "diagnostic" if result.profiling_enabled else "measured"
        ),
        "run_kind": result.run_kind,
        "reward_eligible": result.reward_eligible,
        "model_revision_verified": result.model_revision.passed,
        "inferencex_runtime_verified": result.inferencex_runtime.passed,
        "lm_eval_runtime_verified": result.lm_eval_runtime.passed,
        "serving_runtime_verified": result.serving_runtime.passed,
        "resolved_image_id": result.serving_runtime.resolved_image_id,
        "config_sha256": evidence.config.digest,
        "normalized_benchmark_receipt": evidence.normalized.digest,
        "quality_receipt": evidence.quality.digest,
        "gpu_measurement_bracket_digest": (
            evidence.measurement_bracket.digest
            if evidence.measurement_bracket is not None
            else None
        ),
        "server_lineage": server.reference if server else None,
        "artifacts": bindings,
    }


def _attempt_lineage(
    record: Any,
    attempt_id: str | None,
    candidate_id: str | None,
    opportunity_id: str | None,
) -> dict[str, object]:
    if attempt_id is None:
        return {}
    lineage = record._attempt_payload(attempt_id, candidate_id=candidate_id)
    if opportunity_id is not None:
        lineage["opportunity_id"] = opportunity_id
    return lineage


__all__ = ["record_benchmark_result"]
