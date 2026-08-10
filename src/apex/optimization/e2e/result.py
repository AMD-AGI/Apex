"""Machine-readable terminal result for an E2E optimization run."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import BenchmarkConfigViews
from apex.core import (
    IntegrityError,
    TaskStatus,
    ValidationLevel,
    canonical_json_bytes,
    sha256_file,
)
from apex.evaluation import (
    E2EObservation,
    E2ERewardPolicy,
    E2ERewardVector,
    replay_e2e_reward,
)
from apex.orchestration import RunPhase
from apex.runtime import RunProvenance
from apex.storage import ArtifactReceipt, EventJournal

from .benchmarking import measurement_metrics
from .kernel_lane import KernelOpportunityPlan
from .run_record import E2ERunRecord


@dataclass(frozen=True, slots=True)
class E2EOptimizationResult:
    schema_version: int
    run_id: str
    status: TaskStatus
    reason_code: str
    validation_level: ValidationLevel
    intake_provenance_status: str
    intake_missing_evidence: tuple[str, ...]
    formal_delivery_verified: bool
    provenance_hash: str
    task_kind: str
    task_reward: float | None
    reward_policy_id: str
    reward_policy_digest: str
    reward_vector: Mapping[str, Any] | None
    reward_source_receipt: str | None
    raw_measurement_receipts: tuple[str, ...]
    trainability: str
    untrainable_reason: str | None
    baseline_metrics: Mapping[str, float]
    final_metrics: Mapping[str, float]
    accepted_patch_ids: tuple[str, ...]
    opportunity_count: int
    eligible_opportunity_count: int
    event_journal: str
    artifact_store: str
    benchmark_original: str
    benchmark_measurement: str
    benchmark_diagnostic: str
    benchmark_replay: str
    diagnostic_evidence: str | None
    no_regression: bool | None
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        policy = E2ERewardPolicy()
        if (
            self.task_kind != "e2e_kernel_only"
            or self.reward_policy_id != policy.policy_id
            or self.reward_policy_digest != policy.digest
            or self.trainability not in {"trainable", "untrainable"}
        ):
            raise IntegrityError("Terminal E2E reward contract is invalid", "invalid_e2e_result")
        if self.trainability == "trainable":
            if (
                self.task_reward is None
                or self.reward_vector is None
                or not self.reward_source_receipt
                or not self.raw_measurement_receipts
                or self.untrainable_reason is not None
                or self.reward_vector.get("scope") != "task_terminal"
                or replay_e2e_reward(self.reward_vector) != self.task_reward
            ):
                raise IntegrityError("Terminal E2E reward is incomplete", "invalid_e2e_result")
        elif self.task_reward is not None or self.reward_vector is not None or not self.untrainable_reason:
            raise IntegrityError("Untrainable E2E result claims reward", "invalid_e2e_result")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["validation_level"] = self.validation_level.value
        value["intake_missing_evidence"] = list(self.intake_missing_evidence)
        value["accepted_patch_ids"] = list(self.accepted_patch_ids)
        value["raw_measurement_receipts"] = list(self.raw_measurement_receipts)
        return value

    @classmethod
    def _from_mapping(cls, value: Mapping[str, Any]) -> "E2EOptimizationResult":
        try:
            return cls(
                schema_version=1,
                run_id=str(value["run_id"]),
                status=TaskStatus(str(value["status"])),
                reason_code=str(value["reason_code"]),
                validation_level=ValidationLevel(str(value["validation_level"])),
                intake_provenance_status=str(value["intake_provenance_status"]),
                intake_missing_evidence=tuple(value.get("intake_missing_evidence", ())),
                formal_delivery_verified=value.get("formal_delivery_verified") is True,
                provenance_hash=str(value["provenance_hash"]),
                task_kind=str(value["task_kind"]),
                task_reward=(
                    float(value["task_reward"])
                    if value.get("task_reward") is not None
                    else None
                ),
                reward_policy_id=str(value["reward_policy_id"]),
                reward_policy_digest=str(value["reward_policy_digest"]),
                reward_vector=(
                    dict(value["reward_vector"])
                    if isinstance(value.get("reward_vector"), Mapping)
                    else None
                ),
                reward_source_receipt=(
                    str(value["reward_source_receipt"])
                    if value.get("reward_source_receipt")
                    else None
                ),
                raw_measurement_receipts=tuple(value["raw_measurement_receipts"]),
                trainability=str(value["trainability"]),
                untrainable_reason=(
                    str(value["untrainable_reason"])
                    if value.get("untrainable_reason")
                    else None
                ),
                baseline_metrics=dict(value.get("baseline_metrics", {})),
                final_metrics=dict(value.get("final_metrics", {})),
                accepted_patch_ids=tuple(value.get("accepted_patch_ids", ())),
                opportunity_count=int(value.get("opportunity_count", 0)),
                eligible_opportunity_count=int(value.get("eligible_opportunity_count", 0)),
                event_journal=str(value["event_journal"]),
                artifact_store=str(value["artifact_store"]),
                benchmark_original=str(value["benchmark_original"]),
                benchmark_measurement=str(value["benchmark_measurement"]),
                benchmark_diagnostic=str(value["benchmark_diagnostic"]),
                benchmark_replay=str(value["benchmark_replay"]),
                diagnostic_evidence=(
                    str(value["diagnostic_evidence"])
                    if value.get("diagnostic_evidence")
                    else None
                ),
                no_regression=value.get("no_regression"),
                details=dict(value.get("details", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError("Terminal E2E result is invalid", "invalid_e2e_result") from error

    def _verify_resume_binding(
        self,
        expected_run_id: str,
        root: Path,
        expected_provenance_hash: str,
        views: BenchmarkConfigViews,
    ) -> None:
        expected = {
            "event_journal": root / "events" / "run.db",
            "artifact_store": root / "artifacts",
            "benchmark_original": views.original,
            "benchmark_measurement": views.measurement,
            "benchmark_diagnostic": views.diagnostic,
            "benchmark_replay": views.replay,
        }
        if self.run_id != expected_run_id or self.provenance_hash != expected_provenance_hash:
            raise IntegrityError("Terminal result identity drifted", "e2e_result_binding_mismatch")
        for name, expected_path in expected.items():
            observed = Path(str(getattr(self, name)))
            if not observed.is_absolute() or observed.resolve() != expected_path.resolve():
                raise IntegrityError("Terminal result path drifted", "e2e_result_binding_mismatch")
        if self.diagnostic_evidence:
            evidence = Path(self.diagnostic_evidence)
            expected_evidence = (
                root / "artifacts" / "sha256" / evidence.name[:2] / evidence.name
            )
            if (
                not evidence.is_absolute()
                or not evidence.resolve().is_relative_to((root / "artifacts").resolve())
                or evidence.resolve() != expected_evidence.resolve()
                or evidence.is_symlink()
                or not evidence.is_file()
                or len(evidence.name) != 64
                or sha256_file(evidence) != evidence.name
            ):
                raise IntegrityError(
                    "Terminal diagnostic evidence escaped the run",
                    "e2e_result_binding_mismatch",
                )


@dataclass(frozen=True, slots=True)
class BoundTerminalResult:
    result: E2EOptimizationResult
    phase: RunPhase
    stop_reason: str


def bind_terminal_result(
    record: E2ERunRecord,
    result: E2EOptimizationResult,
    *,
    phase: RunPhase,
    stop_reason: str,
) -> ArtifactReceipt:
    """Bind terminal bytes in CAS/journal before the terminal transition."""

    if phase not in {RunPhase.SUCCEEDED, RunPhase.FAILED} or not stop_reason:
        raise IntegrityError("Terminal result phase is invalid", "invalid_e2e_result")
    receipt = record.put_json(result.to_dict())
    record.controller.record_domain_event(
        "delivery_result",
        {
            "kind": "e2e_terminal_result",
            "terminal_phase": phase.value,
            "stop_reason": stop_reason,
            "status": result.status.value,
            "artifacts": [
                {"role": "e2e_terminal_result", "receipt": receipt.to_dict()}
            ],
        },
        idempotency_key="e2e.terminal_result",
    )
    return receipt


def load_bound_terminal_result(
    record: E2ERunRecord,
    *,
    expected_provenance_hash: str,
    expected_views: BenchmarkConfigViews,
) -> BoundTerminalResult | None:
    """Load terminal truth from its journal-bound CAS receipt."""

    event = EventJournal(record.root / "events" / "run.db").get_by_idempotency_key(
        record.run_id, "e2e.terminal_result"
    )
    if event is None:
        return None
    if event.event_type != "delivery_result" or event.payload.get("kind") != "e2e_terminal_result":
        raise IntegrityError("Terminal result event is invalid", "invalid_e2e_result")
    artifacts = event.payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise IntegrityError("Terminal result receipt is missing", "invalid_e2e_result")
    binding = artifacts[0]
    if not isinstance(binding, Mapping) or binding.get("role") != "e2e_terminal_result":
        raise IntegrityError("Terminal result receipt role is invalid", "invalid_e2e_result")
    receipt_value = binding.get("receipt")
    if not isinstance(receipt_value, Mapping):
        raise IntegrityError("Terminal result receipt is invalid", "invalid_e2e_result")
    receipt = ArtifactReceipt.from_dict(dict(receipt_value))
    content = record.artifacts.read_bytes(receipt)
    result = _result_from_bytes(content)
    result._verify_resume_binding(
        record.run_id,
        record.root,
        expected_provenance_hash,
        expected_views,
    )
    if event.payload.get("status") != result.status.value:
        raise IntegrityError("Terminal result status drifted", "e2e_result_binding_mismatch")
    try:
        phase = RunPhase(str(event.payload.get("terminal_phase")))
    except ValueError as error:
        raise IntegrityError("Terminal result phase is invalid", "invalid_e2e_result") from error
    stop_reason = str(event.payload.get("stop_reason", ""))
    if phase not in {RunPhase.SUCCEEDED, RunPhase.FAILED} or not stop_reason:
        raise IntegrityError("Terminal result transition is invalid", "invalid_e2e_result")
    projection = record.root / "result.json"
    expected_bytes = content + b"\n"
    if projection.exists():
        if projection.is_symlink() or projection.read_bytes() != expected_bytes:
            raise IntegrityError(
                "Terminal result projection differs from CAS evidence",
                "e2e_result_projection_mismatch",
            )
    else:
        write_e2e_result(result, projection)
    return BoundTerminalResult(result, phase, stop_reason)


def _result_from_bytes(content: bytes) -> E2EOptimizationResult:
    try:
        value = json.loads(content)
        if not isinstance(value, Mapping) or int(value.get("schema_version", 0)) != 1:
            raise ValueError("schema")
        return E2EOptimizationResult._from_mapping(value)
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise IntegrityError("Terminal E2E result is invalid", "invalid_e2e_result") from error


def build_e2e_result(
    *,
    record: E2ERunRecord,
    views: BenchmarkConfigViews,
    provenance: RunProvenance,
    status: TaskStatus,
    reason: str,
    validation_level: ValidationLevel,
    baseline: E2EObservation | None,
    final: E2EObservation | None,
    plan: KernelOpportunityPlan | None,
    evidence_path: str | None,
    no_regression: bool | None,
    details: Mapping[str, Any],
    terminal_reward: E2ERewardVector | None = None,
    reward_source_receipt: str | None = None,
    raw_measurement_receipts: tuple[str, ...] = (),
    untrainable_reason: str | None = None,
) -> E2EOptimizationResult:
    """Assemble the sole machine-readable terminal result shape."""

    policy = E2ERewardPolicy()
    if terminal_reward is not None and terminal_reward.scope != "task_terminal":
        raise IntegrityError("Terminal reward scope is invalid", "invalid_e2e_result")
    trainable = terminal_reward is not None
    return E2EOptimizationResult(
        schema_version=1,
        run_id=record.run_id,
        status=status,
        reason_code=reason,
        validation_level=validation_level,
        intake_provenance_status=provenance.status,
        intake_missing_evidence=provenance.missing_evidence,
        formal_delivery_verified=(
            status is TaskStatus.SUCCEEDED
            and validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
        ),
        provenance_hash=provenance.digest,
        task_kind="e2e_kernel_only",
        task_reward=terminal_reward.scalar_reward if terminal_reward else None,
        reward_policy_id=policy.policy_id,
        reward_policy_digest=policy.digest,
        reward_vector=terminal_reward.to_dict() if terminal_reward else None,
        reward_source_receipt=reward_source_receipt,
        raw_measurement_receipts=raw_measurement_receipts,
        trainability="trainable" if trainable else "untrainable",
        untrainable_reason=None if trainable else (untrainable_reason or reason),
        baseline_metrics=measurement_metrics(baseline) if baseline else {},
        final_metrics=measurement_metrics(final) if final else {},
        accepted_patch_ids=record.controller.state.accepted_patch_ids,
        opportunity_count=len(plan.opportunities) if plan else 0,
        eligible_opportunity_count=len(plan.eligible) if plan else 0,
        event_journal=str(record.root / "events" / "run.db"),
        artifact_store=str(record.root / "artifacts"),
        benchmark_original=str(views.original),
        benchmark_measurement=str(views.measurement),
        benchmark_diagnostic=str(views.diagnostic),
        benchmark_replay=str(views.replay),
        diagnostic_evidence=evidence_path,
        no_regression=no_regression,
        details=dict(details),
    )


def write_e2e_result(result: E2EOptimizationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(result.to_dict()) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BoundTerminalResult",
    "E2EOptimizationResult",
    "bind_terminal_result",
    "build_e2e_result",
    "load_bound_terminal_result",
    "write_e2e_result",
]
