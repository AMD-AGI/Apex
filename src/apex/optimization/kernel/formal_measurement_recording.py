"""Canonical raw-capture and failed-measurement event recording."""

from __future__ import annotations

from apex.core import IntegrityError
from apex.storage import ArtifactReceipt

from .measurement import KernelMeasurementCapture
from .run_record import KernelRunRecord


def record_measurement_capture(
    record: KernelRunRecord,
    attempt_id: str,
    *,
    capture: KernelMeasurementCapture,
    harness_receipt: ArtifactReceipt,
) -> tuple[ArtifactReceipt, ArtifactReceipt]:
    """Persist raw evaluator output without grading or committing reward."""

    raw = record.artifacts.put_file(
        capture.artifact.path, media_type="application/json"
    )
    if raw.digest != capture.artifact.sha256:
        raise IntegrityError(
            "Kernel timing report changed after validation",
            "measurement_report_changed",
        )
    execution = record.artifacts.put_bytes(
        capture.execution.canonical_bytes, media_type="application/json"
    )
    record.artifacts.verify(harness_receipt)
    record.controller.record_domain_event(
        "provenance_observed",
        {
            **record.attempt_payload(attempt_id),
            "kind": "kernel_measurement_capture",
            "report_sha256": raw.digest,
            "execution_fingerprint": capture.execution.fingerprint,
            "evidence_class": "measured",
            "reward_eligible": False,
            "artifacts": [
                _binding("raw_measurement", raw),
                _binding("measurement_execution", execution),
                _binding("harness", harness_receipt),
            ],
        },
        idempotency_key=f"attempt.{attempt_id}.measurement.capture",
    )
    return raw, execution


def record_measurement_error(
    record: KernelRunRecord, attempt_id: str, *, reason_code: str
) -> None:
    record.controller.record_domain_event(
        "measurement_result",
        {
            **record.attempt_payload(attempt_id),
            "measurement_status": "error",
            "reason_code": reason_code,
            "reward": None,
            "evidence_class": "measured",
        },
        idempotency_key=f"attempt.{attempt_id}.measurement",
    )


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = ["record_measurement_capture", "record_measurement_error"]
