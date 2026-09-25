"""Cross-check recovered benchmark/deployment event and CAS bindings."""

from __future__ import annotations

from typing import Mapping

from apex.core import IntegrityError
from apex.storage import ArtifactReceipt, EventRecord

from .run_record import E2ERunRecord
from .services import CandidateDeployment


def verify_deployment_config_bindings(
    record: E2ERunRecord,
    events: tuple[EventRecord, ...],
    deployment: CandidateDeployment,
) -> None:
    digests = deployment.config_sha256
    if digests is None:
        raise IntegrityError(
            "Deployment config identity is absent", "deployment_config_drift"
        )
    for role, expected in (
        ("delivery_measurement_config", digests.measurement),
        ("delivery_diagnostic_config", digests.diagnostic),
        ("delivery_replay_config", digests.replay),
    ):
        receipt = unique_role(events, role)
        if receipt is None or receipt.digest != expected:
            raise IntegrityError(
                "Deployment config CAS binding drifted", "deployment_config_drift"
            )
        record.artifacts.verify(receipt)


def verify_benchmark_config(
    record: E2ERunRecord,
    receipt: ArtifactReceipt | None,
    deployment: CandidateDeployment,
) -> None:
    expected = deployment.config_sha256
    if receipt is None or expected is None or receipt.digest != expected.measurement:
        raise IntegrityError(
            "Benchmark used another config", "benchmark_config_drift"
        )
    record.artifacts.verify(receipt)


def verify_benchmark_event(
    events: tuple[EventRecord, ...],
    *,
    normalized: ArtifactReceipt,
    quality: ArtifactReceipt | None,
    config: ArtifactReceipt | None,
) -> None:
    matches = tuple(
        event
        for event in events
        if event.event_type == "measurement_result"
        and _event_has_receipt(
            event, "normalized_benchmark", normalized.digest
        )
    )
    if len(matches) != 1 or quality is None or config is None:
        raise IntegrityError(
            "Benchmark event binding is incomplete", "recovery_lineage_incomplete"
        )
    expected = {
        "normalized_benchmark_receipt": normalized.digest,
        "quality_receipt": quality.digest,
        "config_sha256": config.digest,
    }
    if any(matches[0].payload.get(name) != value for name, value in expected.items()):
        raise IntegrityError(
            "Benchmark event receipt drifted", "recovery_lineage_incomplete"
        )


def unique_role(
    events: tuple[EventRecord, ...], role: str
) -> ArtifactReceipt | None:
    receipts = []
    for event in events:
        artifacts = event.payload.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for item in artifacts:
            if not isinstance(item, Mapping) or item.get("role") != role:
                continue
            raw = item.get("receipt")
            if isinstance(raw, Mapping):
                receipts.append(ArtifactReceipt.from_dict(dict(raw)))
    by_digest = {item.digest: item for item in receipts}
    if len(by_digest) > 1:
        raise IntegrityError(
            f"{role} is ambiguous", "recovery_lineage_incomplete"
        )
    return next(iter(by_digest.values()), None)


def _event_has_receipt(event: EventRecord, role: str, digest: str) -> bool:
    artifacts = event.payload.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("role") == role
        and isinstance(item.get("receipt"), Mapping)
        and item["receipt"].get("digest") == digest
        for item in artifacts
    )


__all__ = [
    "unique_role",
    "verify_benchmark_config",
    "verify_benchmark_event",
    "verify_deployment_config_bindings",
]
