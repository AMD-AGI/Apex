"""Offline validation of standalone kernel measurement execution authority."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from apex.evaluation import KernelMeasurementExecutionReceipt
from apex.core import sha256_json
from apex.storage import ArtifactReceipt, ArtifactStore

from .models import EpisodeArtifact, EpisodeEvent


_MISMATCH = "kernel_measurement_execution_evidence_mismatch"


def kernel_measurement_evidence_reasons(
    events: Sequence[EpisodeEvent],
    artifacts: ArtifactStore,
) -> set[str]:
    """Verify reward and measurement bind the same typed execution receipt."""

    measurement = _single_event(events, "measurement_result")
    reward = _single_event(events, "reward_committed")
    if measurement is None or reward is None:
        return {_MISMATCH}
    measured_roles = _roles(measurement.artifacts)
    reward_roles = _roles(reward.artifacts)
    shared = {"raw_measurement", "measurement_execution", "harness"}
    if not shared <= set(measured_roles) or not (shared | {"reward_policy"}) <= set(
        reward_roles
    ):
        return {_MISMATCH}
    if any(
        len(measured_roles[role]) != 1 or len(reward_roles[role]) != 1
        for role in shared
    ):
        return {_MISMATCH}
    if len(reward_roles["reward_policy"]) != 1:
        return {_MISMATCH}
    if any(
        measured_roles[role][0].digest != reward_roles[role][0].digest
        for role in shared
    ):
        return {_MISMATCH}
    raw = measured_roles["raw_measurement"][0]
    execution_artifact = measured_roles["measurement_execution"][0]
    try:
        document = json.loads(artifacts.read_bytes(execution_artifact))
        if not isinstance(document, Mapping):
            return {_MISMATCH}
        execution = KernelMeasurementExecutionReceipt.from_mapping(document)
        raw_document = _document(artifacts, raw)
        harness_document = _document(artifacts, measured_roles["harness"][0])
        policy_document = _document(
            artifacts, reward_roles["reward_policy"][0]
        )
    except Exception:
        return {_MISMATCH}
    attempt_id = str(measurement.payload.get("attempt_id", ""))
    if (
        not attempt_id
        or attempt_id != str(reward.payload.get("attempt_id", ""))
        or execution.attempt_id != attempt_id
        or execution.report_sha256 != raw.digest
        or execution.report_size != raw.size
        or measurement.payload.get("measurement_execution_sha256")
        != execution.fingerprint
        or measurement.payload.get("measurement_writer_id") != execution.writer_id
        or measurement.payload.get("measurement_harness_sha256")
        != execution.harness_sha256
        or str(raw_document.get("measurement_method_sha256", "")).removeprefix(
            "sha256:"
        )
        != execution.measurement_method_sha256.removeprefix("sha256:")
        or harness_document.get("harness_sha256") != execution.harness_sha256
        or sha256_json(policy_document.get("measurement_policy"))
        != execution.measurement_policy_sha256
    ):
        return {_MISMATCH}
    return set()


def _document(
    artifacts: ArtifactStore, receipt: ArtifactReceipt
) -> Mapping[str, object]:
    value = json.loads(artifacts.read_bytes(receipt))
    if not isinstance(value, Mapping):
        raise ValueError("artifact document is not an object")
    return value


def _single_event(
    events: Sequence[EpisodeEvent], event_type: str
) -> EpisodeEvent | None:
    found = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == event_type
    )
    return found[0] if len(found) == 1 else None


def _roles(
    bindings: Sequence[EpisodeArtifact],
) -> dict[str, tuple[ArtifactReceipt, ...]]:
    result: dict[str, list[ArtifactReceipt]] = {}
    for binding in bindings:
        result.setdefault(binding.role, []).append(binding.receipt)
    return {role: tuple(receipts) for role, receipts in result.items()}


__all__ = ["kernel_measurement_evidence_reasons"]
