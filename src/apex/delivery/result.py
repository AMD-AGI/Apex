"""Atomic machine-readable TaskResult returned to external evaluators."""

from __future__ import annotations

import os
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from apex.core import (
    TaskStatus,
    ValidationLevel,
    canonical_json_bytes,
    sha256_json,
    validate_identifier,
)


_SAFETY_STATUSES = {
    "not_run",
    "not_configured",
    "certified",
    "clean_unqualified",
    "advisory_incomplete",
    "rejected_finding",
    "required_incomplete",
    "gate_rejected",
}
_MEASUREMENT_STATUSES = {
    "not_configured",
    "not_run_due_to_gate",
    "not_run_due_to_safety",
    "valid",
    "unsupported",
    "insufficient_samples",
    "invalid",
    "error",
}
_EVALUATION_CONTRACT_STATUSES = {"not_frozen", "verified", "unverified"}


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class TaskResult:
    schema_version: int
    run_id: str
    task_id: str
    status: TaskStatus
    reason_code: str
    applied: bool
    external_verification_required: bool
    bundle_path: str | None
    bundle_digest: str | None
    changed_files: tuple[str, ...]
    baseline_lock: Mapping[str, Any] | None = None
    internal_verdict: str | None = None
    internal_verdict_ref: str | None = None
    verification_summary_refs: tuple[str, ...] = ()
    event_journal_ref: Mapping[str, Any] | None = None
    artifact_store_ref: Mapping[str, Any] | None = None
    gpu_lease: Mapping[str, Any] | None = None
    gpu_lease_receipt_digest: str | None = None
    evaluation_contract_status: str = "not_frozen"
    evaluation_contract_receipt_digest: str | None = None
    evaluation_contract_unverified_reason: str | None = None
    evaluation_authority_id: str | None = None
    evaluation_authority_kind: str | None = None
    error: Mapping[str, Any] | None = None
    validation_level: ValidationLevel = ValidationLevel.NONE
    safety_status: str = "not_run"
    safety_certified: bool = False
    safety_result_fingerprint: str | None = None
    safety_receipt_digest: str | None = None
    measurement_status: str = "not_configured"
    measurement_report_sha256: str | None = None
    grade_policy_id: str | None = None
    s50: float | None = None
    s99: float | None = None
    srobust: float | None = None
    worst_case_srobust: float | None = None
    reward: float | None = None
    max_cv: float | None = None
    s50_ci_lower: float | None = None
    s50_ci_upper: float | None = None
    s99_ci_lower: float | None = None
    s99_ci_upper: float | None = None
    srobust_ci_lower: float | None = None
    srobust_ci_upper: float | None = None
    confidence_level: float | None = None
    threshold_pass: bool | None = None
    confidence_pass: bool | None = None
    noise_pass: bool | None = None
    worst_case_pass: bool | None = None
    promotion_eligible: bool | None = None
    promotion_reason_code: str | None = None
    task_reward: float | None = None
    task_reward_vector: Mapping[str, Any] | None = None
    reward_policy_id: str | None = None
    reward_policy_digest: str | None = None
    reward_source_receipt: str | None = None
    raw_measurement_receipts: tuple[str, ...] = ()
    task_trainability: str = "unscored"
    untrainable_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_files", tuple(self.changed_files))
        object.__setattr__(
            self, "verification_summary_refs", tuple(self.verification_summary_refs)
        )
        object.__setattr__(
            self, "raw_measurement_receipts", tuple(self.raw_measurement_receipts)
        )
        self._validate_lineage()
        self._validate_evaluation_contract()
        if self.safety_status not in _SAFETY_STATUSES:
            raise ValueError(f"unsupported safety_status: {self.safety_status}")
        for field_name, digest in (
            ("safety_result_fingerprint", self.safety_result_fingerprint),
            ("safety_receipt_digest", self.safety_receipt_digest),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.safety_status == "not_run" and (
            self.safety_result_fingerprint is not None
            or self.safety_receipt_digest is not None
        ):
            raise ValueError("not_run safety cannot carry a result receipt")
        if self.safety_status != "not_run" and (
            self.safety_result_fingerprint is None
            or self.safety_receipt_digest is None
        ):
            raise ValueError("evaluated safety status requires exact result receipts")
        if self.safety_certified and self.safety_status != "certified":
            raise ValueError("safety_certified requires safety_status=certified")
        if self.safety_status == "certified" and not self.safety_certified:
            raise ValueError("certified safety status requires safety_certified=true")
        self._validate_measurement()
        self._validate_task_reward()

    def _validate_task_reward(self) -> None:
        if self.task_trainability not in {"unscored", "trainable", "untrainable"}:
            raise ValueError("unsupported task_trainability")
        receipts = self.raw_measurement_receipts
        if any(not _is_digest(item) for item in receipts):
            raise ValueError("raw measurement receipt is invalid")
        identities = (
            self.reward_policy_id,
            self.reward_policy_digest,
            self.reward_source_receipt,
        )
        if self.task_trainability == "unscored":
            if (
                self.task_reward is not None
                or self.task_reward_vector is not None
                or any(identities)
                or receipts
                or self.untrainable_reason is not None
            ):
                raise ValueError("unscored task cannot carry terminal reward evidence")
            return
        if not self.reward_policy_id or not _is_digest(self.reward_policy_digest or ""):
            raise ValueError("terminal task requires reward policy identity")
        if self.task_trainability == "untrainable":
            if (
                self.task_reward is not None
                or self.task_reward_vector is not None
                or self.reward_source_receipt is not None
                or receipts
                or not self.untrainable_reason
            ):
                raise ValueError("untrainable task reward is incoherent")
            return
        if (
            self.task_reward is None
            or not math.isfinite(self.task_reward)
            or self.task_reward_vector is None
            or not _is_digest(self.reward_source_receipt or "")
            or self.untrainable_reason is not None
        ):
            raise ValueError("trainable task requires complete terminal reward evidence")

    def _validate_evaluation_contract(self) -> None:
        status = self.evaluation_contract_status
        digest = self.evaluation_contract_receipt_digest
        reason = self.evaluation_contract_unverified_reason
        authority = (self.evaluation_authority_id, self.evaluation_authority_kind)
        if status not in _EVALUATION_CONTRACT_STATUSES:
            raise ValueError(f"unsupported evaluation_contract_status: {status}")
        if status == "not_frozen":
            if digest is not None or reason is not None or any(authority):
                raise ValueError("unfrozen evaluation contract cannot carry evidence")
            return
        if not isinstance(digest, str) or not _is_digest(digest):
            raise ValueError("frozen evaluation contract requires its receipt digest")
        if status == "verified" and reason is not None:
            raise ValueError("verified evaluation contract cannot carry a failure reason")
        if status == "verified" and any(not item for item in authority):
            raise ValueError("verified evaluation contract requires authority identity")
        if status == "unverified" and not reason:
            raise ValueError("unverified evaluation contract requires a failure reason")
        if status == "unverified" and any(authority):
            raise ValueError("unverified evaluation contract cannot carry authority identity")

    def _validate_lineage(self) -> None:
        validate_identifier(self.run_id, field_name="run_id")
        if self.internal_verdict not in {
            None,
            "keep",
            "revert",
            "reject",
            "needs_more_measurement",
        }:
            raise ValueError("unsupported internal_verdict")
        if self.internal_verdict_ref is not None:
            validate_identifier(self.internal_verdict_ref, field_name="internal_verdict_ref")
        for digest in self.verification_summary_refs:
            if not _is_digest(digest):
                raise ValueError("lineage receipt must be a lowercase SHA-256 identity")
        self._validate_gpu_lease()
        if self.baseline_lock is not None:
            resolution = self.baseline_lock.get("resolution_hash")
            hashes = self.baseline_lock.get("file_hashes")
            if not isinstance(resolution, str) or not _is_digest(resolution):
                raise ValueError("baseline lock requires a resolution hash")
            if not isinstance(hashes, Mapping) or any(
                not isinstance(path, str)
                or not isinstance(digest, str)
                or not _is_digest(digest)
                for path, digest in hashes.items()
            ):
                raise ValueError("baseline lock file hashes are invalid")
        self._validate_storage_refs()

    def _validate_gpu_lease(self) -> None:
        if self.gpu_lease is None:
            if self.gpu_lease_receipt_digest is not None:
                raise ValueError("GPU lease digest requires a lease receipt")
            return
        if not isinstance(self.gpu_lease_receipt_digest, str) or not _is_digest(
            self.gpu_lease_receipt_digest
        ):
            raise ValueError("GPU lease receipt requires its SHA-256 identity")
        lease = dict(self.gpu_lease)
        doctor = lease.get("doctor")
        owner_pid = lease.get("owner_pid")
        acquired = lease.get("acquired_unix_seconds")
        if (
            lease.get("schema_version") != 3
            or lease.get("run_id") != self.run_id
            or not isinstance(lease.get("execution_scope"), str)
            or not lease["execution_scope"]
            or not isinstance(lease.get("physical_scope"), str)
            or not lease["physical_scope"]
            or isinstance(owner_pid, bool)
            or not isinstance(owner_pid, int)
            or owner_pid <= 0
            or isinstance(acquired, bool)
            or not isinstance(acquired, (int, float))
            or not math.isfinite(float(acquired))
            or acquired <= 0
            or not isinstance(lease.get("lock_path"), str)
            or not Path(lease["lock_path"]).is_absolute()
            or not isinstance(doctor, Mapping)
            or doctor.get("status") != "ready"
            or doctor.get("formal_measurement_ready") is not True
            or doctor.get("ownership_receipt_sha256")
            != sha256_json(lease.get("ownership"))
            or doctor.get("rocm_health_status") != "healthy"
        ):
            raise ValueError("GPU lease receipt is invalid")
        if sha256_json(lease) != self.gpu_lease_receipt_digest:
            raise ValueError("GPU lease receipt digest does not match its contents")

    def _validate_storage_refs(self) -> None:
        if self.event_journal_ref is not None:
            path = self.event_journal_ref.get("path")
            head = self.event_journal_ref.get("head_event_id")
            checksum = self.event_journal_ref.get("head_checksum")
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError("event journal reference path must be absolute")
            if not isinstance(head, str):
                raise ValueError("event journal reference requires a head event")
            validate_identifier(head, field_name="event_journal_head")
            if not isinstance(checksum, str) or not _is_digest(checksum):
                raise ValueError("event journal head checksum is invalid")
        if self.artifact_store_ref is not None:
            path = self.artifact_store_ref.get("path")
            receipts = self.artifact_store_ref.get("receipt_digests")
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError("artifact store reference path must be absolute")
            if not isinstance(receipts, (list, tuple)) or any(
                not isinstance(digest, str) or not _is_digest(digest)
                for digest in receipts
            ):
                raise ValueError("artifact store receipt identities are invalid")

    def _validate_measurement(self) -> None:
        if self.measurement_status not in _MEASUREMENT_STATUSES:
            raise ValueError(f"unsupported measurement_status: {self.measurement_status}")
        digest = self.measurement_report_sha256
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("measurement_report_sha256 must be a lowercase SHA-256 digest")
        values = self._measurement_values()
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("kernel grade values must be finite")
        evidence = (
            digest,
            self.grade_policy_id,
            *values,
            self.threshold_pass,
            self.confidence_pass,
            self.noise_pass,
            self.worst_case_pass,
            self.promotion_eligible,
            self.promotion_reason_code,
        )
        if self.measurement_status == "not_configured" and any(
            value is not None for value in evidence
        ):
            raise ValueError("unconfigured measurement cannot carry grade evidence")
        if self.measurement_status == "valid":
            self._validate_valid_measurement(digest)

    def _measurement_values(self) -> tuple[float | None, ...]:
        return (
            self.s50,
            self.s99,
            self.srobust,
            self.worst_case_srobust,
            self.reward,
            self.max_cv,
            self.s50_ci_lower,
            self.s50_ci_upper,
            self.s99_ci_lower,
            self.s99_ci_upper,
            self.srobust_ci_lower,
            self.srobust_ci_upper,
            self.confidence_level,
        )

    def _validate_valid_measurement(self, digest: str | None) -> None:
        if digest is None or self.grade_policy_id != "kernel_robust_v1":
            raise ValueError("valid measurement requires exact report and policy identity")
        point_values = (
            self.s50,
            self.s99,
            self.srobust,
            self.worst_case_srobust,
            self.reward,
            self.max_cv,
            self.confidence_level,
        )
        if any(value is None for value in point_values):
            raise ValueError("valid measurement requires complete robust point evidence")
        flags = (
            self.threshold_pass,
            self.confidence_pass,
            self.noise_pass,
            self.worst_case_pass,
            self.promotion_eligible,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise ValueError("valid measurement requires complete promotion gates")
        if not self.promotion_reason_code:
            raise ValueError("valid measurement requires a promotion reason")
        assert self.s50 is not None and self.s99 is not None and self.srobust is not None
        if self.srobust != min(self.s50, self.s99):
            raise ValueError("Srobust must be min(S50, S99)")
        ci_values = self._measurement_values()[6:12]
        if self.confidence_pass and any(value is None for value in ci_values):
            raise ValueError("passing confidence requires complete bootstrap intervals")
        expected = bool(
            self.threshold_pass
            and self.confidence_pass
            and self.noise_pass
            and self.worst_case_pass
        )
        if self.promotion_eligible != expected:
            raise ValueError("promotion eligibility must equal all canonical gates")
        if self.promotion_eligible and self.promotion_reason_code != "promotion_eligible":
            raise ValueError("eligible measurement requires the canonical promotion reason")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["status"] = self.status.value
        value["validation_level"] = self.validation_level.value
        value["changed_files"] = list(self.changed_files)
        value["verification_summary_refs"] = list(self.verification_summary_refs)
        value["raw_measurement_receipts"] = list(self.raw_measurement_receipts)
        return value


def write_task_result(result: TaskResult, path: Path) -> None:
    """Atomically replace a result file and fsync its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json_bytes(result.to_dict()) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
