"""Durable request, action receipt, and safe early-stage E2E recovery."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import BenchmarkConfigViews, QualityMetric
from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_file
from apex.evaluation import E2EMeasurement
from apex.intake import E2EOptimizeSpec
from apex.orchestration import RunController
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, SnapshotStore

from .kernel_lane import KernelOpportunity, KernelOpportunityPlan
from .run_record import E2ERunRecord


RUN_REQUEST_SCHEMA = "apex.e2e-run-request/v1"
ACTION_COMPLETION_SCHEMA = "apex.e2e-action-completion/v1"
DIAGNOSIS_SCHEMA = "apex.e2e-diagnosis/v1"


@dataclass(frozen=True, slots=True)
class RecoveredRunRequest:
    run_id: str
    spec: E2EOptimizeSpec
    dependency_lock_sha256: str
    provenance_digest: str
    views: BenchmarkConfigViews
    correctness_oracle_policy_sha256: str | None
    request_receipt: ArtifactReceipt


def persist_run_request(
    record: E2ERunRecord,
    *,
    spec: E2EOptimizeSpec,
    dependency_lock_sha256: str,
    provenance_digest: str,
    views: BenchmarkConfigViews,
    correctness_oracle_policy_sha256: str | None,
) -> ArtifactReceipt:
    payload = {
        "schema": RUN_REQUEST_SCHEMA,
        "run_id": record.run_id,
        "spec": spec.to_dict(),
        "dependency_lock_sha256": dependency_lock_sha256,
        "provenance_digest": provenance_digest,
        "correctness_oracle_policy_sha256": correctness_oracle_policy_sha256,
        "views": _views_dict(views),
    }
    receipt = record.put_json(payload)
    _write_immutable_json(record.root / "run.request.json", payload)
    record.controller.record_domain_event(
        "dependency_verified",
        {
            "kind": "resolved_e2e_run_request",
            "artifacts": [{"role": "run_request", "receipt": receipt.to_dict()}],
        },
        idempotency_key="run.request.persisted",
    )
    return receipt


def load_run_request(root: Path) -> RecoveredRunRequest:
    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ContractError("Run root is not a directory", "run_request_missing")
    local_path = resolved / "run.request.json"
    local = _load_object(local_path)
    run_id = str(local.get("run_id", ""))
    receipt = _bound_run_request_receipt(resolved, run_id)
    store = ArtifactStore(resolved / "artifacts")
    content = store.read_bytes(receipt)
    try:
        value = _mapping(json.loads(content), "run request")
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise IntegrityError("Run request CAS artifact is invalid", "invalid_run_request") from error
    if local_path.read_bytes() != content + b"\n":
        raise IntegrityError(
            "Run request projection differs from canonical CAS evidence",
            "run_request_projection_mismatch",
        )
    if value.get("schema") != RUN_REQUEST_SCHEMA:
        raise ContractError("Run request schema is invalid", "invalid_run_request")
    if value.get("run_id") != run_id:
        raise IntegrityError("Run request identity drifted", "run_request_id_mismatch")
    spec = E2EOptimizeSpec.from_mapping(_mapping(value.get("spec"), "spec"))
    if spec.results_dir.resolve() != resolved:
        raise IntegrityError("Run request root drifted", "run_request_root_mismatch")
    return RecoveredRunRequest(
        run_id=str(value.get("run_id", "")),
        spec=spec,
        dependency_lock_sha256=str(value.get("dependency_lock_sha256", "")),
        provenance_digest=str(value.get("provenance_digest", "")),
        views=_views_from_mapping(_mapping(value.get("views"), "views"), resolved),
        correctness_oracle_policy_sha256=_optional_text(
            value.get("correctness_oracle_policy_sha256")
        ),
        request_receipt=receipt,
    )


def _bound_run_request_receipt(root: Path, run_id: str) -> ArtifactReceipt:
    journal_path = root / "events" / "run.db"
    if journal_path.is_symlink() or not journal_path.is_file():
        raise ContractError("Run event journal is missing", "run_request_missing")
    event = EventJournal(journal_path).get_by_idempotency_key(
        run_id, "run.request.persisted"
    )
    if event is None or event.event_type != "dependency_verified":
        raise IntegrityError("Run request event is missing", "run_request_event_missing")
    if event.payload.get("kind") != "resolved_e2e_run_request":
        raise IntegrityError("Run request event kind is invalid", "invalid_run_request")
    artifacts = event.payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise IntegrityError("Run request event is incomplete", "invalid_run_request")
    binding = _mapping(artifacts[0], "run request binding")
    if binding.get("role") != "run_request":
        raise IntegrityError("Run request event role is invalid", "invalid_run_request")
    return ArtifactReceipt.from_dict(
        dict(_mapping(binding.get("receipt"), "run request receipt"))
    )


def recover_record(request: RecoveredRunRequest) -> E2ERunRecord:
    root = request.spec.results_dir.resolve(strict=True)
    controller = RunController.recover(
        request.run_id,
        EventJournal(root / "events" / "run.db"),
        SnapshotStore(root / "state.snapshot.json"),
    )
    return E2ERunRecord(
        request.run_id,
        root,
        ArtifactStore(root / "artifacts"),
        controller,
        request.spec.dataset_split,
        request.spec.data_visibility,
    )


def write_action_completion(
    record: E2ERunRecord,
    *,
    action_id: str,
    normalized: ArtifactReceipt,
    succeeded: bool,
    errors: tuple[str, ...],
) -> None:
    _write_immutable_json(
        record.root / "action_receipts" / f"{action_id}.json",
        {
            "schema": ACTION_COMPLETION_SCHEMA,
            "run_id": record.run_id,
            "action_id": action_id,
            "succeeded": succeeded,
            "normalized_result": normalized.to_dict(),
            "errors": list(errors),
        },
    )


def persist_diagnosis(
    record: E2ERunRecord,
    *,
    evidence: ArtifactReceipt,
    plan: KernelOpportunityPlan,
) -> ArtifactReceipt:
    plan_receipt = record.put_json(_plan_dict(plan))
    lineage = record.put_json(
        {
            "schema": DIAGNOSIS_SCHEMA,
            "evidence": evidence.to_dict(),
            "opportunity_plan": plan_receipt.to_dict(),
            "correctness_oracle_policy_sha256": getattr(
                plan, "correctness_oracle_policy_sha256", None
            ),
        }
    )
    record.controller.record_domain_event(
        "tool_result",
        {
            "tool": "kernel_opportunity_planner",
            "succeeded": True,
            "opportunity_count": len(plan.opportunities),
            "eligible_count": len(plan.eligible),
            "correctness_oracle_policy_sha256": getattr(
                plan, "correctness_oracle_policy_sha256", None
            ),
            "artifacts": [
                {"role": "opportunity_plan", "receipt": plan_receipt.to_dict()},
                {"role": "diagnosis_lineage", "receipt": lineage.to_dict()},
            ],
        },
        idempotency_key=(
            "diagnostics.plan."
            f"{record.controller.state.e2e.bottleneck_generation + 1}"
        ),
    )
    return lineage


def recover_diagnosis(
    record: E2ERunRecord,
) -> tuple[KernelOpportunityPlan, Path, ArtifactReceipt, ArtifactReceipt]:
    search = record.controller.state.e2e
    if search is None or search.diagnostic_receipt is None:
        raise ContractError("Diagnosis is not committed", "diagnosis_not_committed")
    lineage_receipt = receipt_for_digest(record, search.diagnostic_receipt)
    return _diagnosis_from_lineage(record, lineage_receipt)


def recover_uncommitted_diagnosis(
    record: E2ERunRecord,
) -> tuple[KernelOpportunityPlan, Path, ArtifactReceipt, ArtifactReceipt] | None:
    search = record.controller.state.e2e
    if search is None:
        return None
    journal = EventJournal(record.root / "events" / "run.db")
    event = journal.get_by_idempotency_key(
        record.run_id, f"diagnostics.plan.{search.bottleneck_generation + 1}"
    )
    if event is None:
        return None
    artifacts = event.payload.get("artifacts", ())
    for item in artifacts if isinstance(artifacts, list) else ():
        if isinstance(item, Mapping) and item.get("role") == "diagnosis_lineage":
            receipt = ArtifactReceipt.from_dict(
                dict(_mapping(item.get("receipt"), "diagnosis lineage"))
            )
            return _diagnosis_from_lineage(record, receipt)
    raise IntegrityError("Diagnosis lineage event is incomplete", "invalid_diagnosis")


def _diagnosis_from_lineage(
    record: E2ERunRecord, lineage_receipt: ArtifactReceipt
) -> tuple[KernelOpportunityPlan, Path, ArtifactReceipt, ArtifactReceipt]:
    record.artifacts.verify(lineage_receipt)
    lineage = _mapping(
        json.loads(record.artifacts.read_bytes(lineage_receipt)), "diagnosis"
    )
    if lineage.get("schema") != DIAGNOSIS_SCHEMA:
        raise IntegrityError("Diagnosis lineage schema is invalid", "invalid_diagnosis")
    evidence = ArtifactReceipt.from_dict(dict(_mapping(lineage.get("evidence"), "evidence")))
    plan_receipt = ArtifactReceipt.from_dict(
        dict(_mapping(lineage.get("opportunity_plan"), "opportunity_plan"))
    )
    record.artifacts.verify(evidence)
    plan = _plan_from_mapping(
        _mapping(json.loads(record.artifacts.read_bytes(plan_receipt)), "plan")
    )
    if getattr(plan, "correctness_oracle_policy_sha256", None) != lineage.get(
        "correctness_oracle_policy_sha256"
    ):
        raise IntegrityError("Oracle policy receipt drifted", "invalid_diagnosis")
    return (
        plan,
        record.artifacts.root / evidence.relative_path,
        evidence,
        lineage_receipt,
    )


def recover_baseline(record: E2ERunRecord) -> E2EMeasurement:
    search = record.controller.state.e2e
    if search is None or search.baseline_receipt is None:
        raise ContractError("Baseline is not committed", "baseline_not_committed")
    receipt = receipt_for_digest(record, search.baseline_receipt)
    value = _mapping(json.loads(record.artifacts.read_bytes(receipt)), "baseline")
    throughput = _mapping(value.get("throughput"), "throughput")
    latency = _mapping(value.get("latency"), "latency")
    quality = _mapping(value.get("quality"), "quality")
    metrics = tuple(
        QualityMetric(
            str(item["task"]),
            str(item["name"]),
            float(item["value"]),
            bool(item["higher_is_better"]),
        )
        for item in quality.get("metrics", ())
        if isinstance(item, Mapping)
    )
    primary = _primary_metric(metrics)
    ttft = _mapping(latency.get("ttft"), "ttft")
    tpot = _mapping(latency.get("tpot"), "tpot")
    total = throughput.get("total_tokens_per_second")
    selected = total if total is not None else throughput.get("output_tokens_per_second")
    return E2EMeasurement(
        float(selected),
        float(ttft["p99_ms"]),
        float(tpot["p99_ms"]),
        primary.value,
        int(throughput["completed_requests"]),
        search.measurement_protocol_hash,
        receipt.digest,
        receipt.digest,
    )


def receipt_for_digest(record: E2ERunRecord, digest: str) -> ArtifactReceipt:
    if len(digest) != 64:
        raise IntegrityError("CAS digest is invalid", "artifact_digest_mismatch")
    path = record.artifacts.root / "sha256" / digest[:2] / digest
    if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
        raise IntegrityError("CAS artifact failed verification", "artifact_digest_mismatch")
    return ArtifactReceipt(
        digest,
        path.stat().st_size,
        "application/json",
        str(path.relative_to(record.artifacts.root)),
    )


def _primary_metric(metrics: tuple[QualityMetric, ...]) -> QualityMetric:
    order = ("exact_match,strict-match", "acc_norm,none", "acc,none")
    for name in order:
        found = next((item for item in metrics if item.name == name), None)
        if found is not None:
            return found
    raise IntegrityError("Baseline primary quality is missing", "invalid_baseline_receipt")


def _views_dict(views: BenchmarkConfigViews) -> dict[str, Any]:
    return {
        "original": str(views.original),
        "measurement": str(views.measurement),
        "diagnostic": str(views.diagnostic),
        "replay": str(views.replay),
        "original_sha256": views.original_sha256,
        "workload_semantics_sha256": views.workload_semantics_sha256,
        "quality_tasks": views.quality_tasks,
        "evaluator_policy_sha256": views.evaluator_policy_sha256,
    }


def _views_from_mapping(value: Mapping[str, Any], root: Path) -> BenchmarkConfigViews:
    paths = {
        name: Path(str(value.get(name, ""))).resolve(strict=True)
        for name in ("original", "measurement", "diagnostic", "replay")
    }
    if any(not path.is_file() or not path.is_relative_to(root) for path in paths.values()):
        raise IntegrityError("Recovered benchmark view is unsafe", "invalid_run_request")
    return BenchmarkConfigViews(
        **paths,
        original_sha256=str(value.get("original_sha256", "")),
        workload_semantics_sha256=str(value.get("workload_semantics_sha256", "")),
        quality_tasks=str(value.get("quality_tasks", "")),
        evaluator_policy_sha256=_optional_text(value.get("evaluator_policy_sha256")),
    )


def _plan_dict(plan: KernelOpportunityPlan) -> dict[str, Any]:
    opportunities = []
    for item in plan.opportunities:
        value = asdict(item)
        for name in ("source_path", "source_root", "test_file"):
            value[name] = str(value[name]) if value[name] else None
        opportunities.append(value)
    return {
        "schema": "apex.kernel-opportunity-plan/v1",
        "opportunities": opportunities,
        "measured_order": list(plan.measured_order),
        "recoverable_order": list(plan.recoverable_order),
        "correctness_oracle_policy_sha256": getattr(
            plan, "correctness_oracle_policy_sha256", None
        ),
    }


def _plan_from_mapping(value: Mapping[str, Any]) -> KernelOpportunityPlan:
    if value.get("schema") != "apex.kernel-opportunity-plan/v1":
        raise IntegrityError("Opportunity plan schema is invalid", "invalid_diagnosis")
    opportunities = []
    for raw in value.get("opportunities", ()):
        item = dict(_mapping(raw, "opportunity"))
        for name in ("source_path", "source_root", "test_file"):
            item[name] = Path(item[name]) if item.get(name) else None
        opportunities.append(KernelOpportunity(**item))
    kwargs: dict[str, Any] = {
        "opportunities": tuple(opportunities),
        "measured_order": tuple(value.get("measured_order", ())),
        "recoverable_order": tuple(value.get("recoverable_order", ())),
    }
    if "correctness_oracle_policy_sha256" in KernelOpportunityPlan.__dataclass_fields__:
        kwargs["correctness_oracle_policy_sha256"] = _optional_text(
            value.get("correctness_oracle_policy_sha256")
        )
    return KernelOpportunityPlan(**kwargs)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    content = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise IntegrityError("Immutable run receipt drifted", "immutable_run_receipt")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("Run request is missing", "run_request_missing")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), "run request")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError("Run request projection is invalid", "invalid_run_request") from error


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object", "invalid_run_request")
    return value


def _optional_text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


__all__ = [
    "RecoveredRunRequest",
    "load_run_request",
    "persist_diagnosis",
    "persist_run_request",
    "receipt_for_digest",
    "recover_baseline",
    "recover_diagnosis",
    "recover_uncommitted_diagnosis",
    "recover_record",
    "write_action_completion",
]
