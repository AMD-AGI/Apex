"""Typed reload helpers for phased formal-kernel evaluator evidence."""

from __future__ import annotations

import json
from typing import Mapping

from apex.core import IntegrityError, canonical_json_bytes
from apex.evaluation import KernelMeasurementExecutionReceipt
from apex.storage import ArtifactReceipt, EventRecord

from .formal_campaign import FormalKernelCampaign
from .measurement import (
    KernelMeasurementCapture,
    KernelMeasurementEvaluation,
    grade_kernel_measurement,
    load_kernel_measurement_capture,
)


def attempt_event(
    campaign: FormalKernelCampaign,
    attempt_id: str,
    event_type: str,
    *,
    required: bool = True,
) -> EventRecord | None:
    matches = [
        event
        for event in campaign.record.iter_events()
        if event.event_type == event_type
        and event.payload.get("attempt_id") == attempt_id
    ]
    if len(matches) > 1 or (required and not matches):
        raise IntegrityError(
            "Attempt evidence is missing or ambiguous",
            "attempt_evidence_ambiguous",
            {"attempt_id": attempt_id, "event_type": event_type},
        )
    return matches[0] if matches else None


def kind_event(
    campaign: FormalKernelCampaign,
    attempt_id: str,
    kind: str,
    *,
    required: bool = True,
) -> EventRecord | None:
    matches = [
        event
        for event in campaign.record.iter_events()
        if event.payload.get("attempt_id") == attempt_id
        and event.payload.get("kind") == kind
    ]
    if len(matches) > 1 or (required and not matches):
        raise IntegrityError(
            "Attempt evidence is missing or ambiguous",
            "attempt_evidence_ambiguous",
            {"attempt_id": attempt_id, "kind": kind},
        )
    return matches[0] if matches else None


def event_receipt(event: EventRecord, role: str) -> ArtifactReceipt:
    matches = [
        binding.get("receipt")
        for binding in event.payload.get("artifacts", ())
        if isinstance(binding, Mapping) and binding.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise IntegrityError(
            "Attempt artifact role is missing or ambiguous",
            "attempt_evidence_ambiguous",
            {"event_type": event.event_type, "role": role},
        )
    return ArtifactReceipt.from_dict(matches[0])


def require_passed(
    campaign: FormalKernelCampaign, attempt_id: str, event_type: str
) -> EventRecord:
    event = attempt_event(campaign, attempt_id, event_type)
    assert event is not None
    if event.payload.get("passed") is not True:
        raise IntegrityError(
            "A prior formal evaluator gate did not pass",
            "formal_gate_not_passed",
            {"attempt_id": attempt_id, "event_type": event_type},
        )
    return event


def load_capture(
    campaign: FormalKernelCampaign, attempt_id: str
) -> tuple[KernelMeasurementCapture, ArtifactReceipt]:
    event = kind_event(campaign, attempt_id, "kernel_measurement_capture")
    assert event is not None
    raw = event_receipt(event, "raw_measurement")
    execution_receipt = event_receipt(event, "measurement_execution")
    execution_value = _canonical_mapping(campaign, execution_receipt)
    execution = KernelMeasurementExecutionReceipt.from_mapping(execution_value)
    if execution.attempt_id != attempt_id or execution.run_id != campaign.record.run_id:
        raise IntegrityError(
            "Measurement execution lineage differs from the campaign",
            "invalid_measurement_execution_receipt",
        )
    report_path = campaign.record.artifacts.root / raw.relative_path
    capture = load_kernel_measurement_capture(
        campaign.resolved,
        report_path=report_path,
        execution=execution,
    )
    return capture, event_receipt(event, "harness")


def load_grade(
    campaign: FormalKernelCampaign, attempt_id: str
) -> tuple[KernelMeasurementEvaluation, ArtifactReceipt]:
    capture, harness = load_capture(campaign, attempt_id)
    return grade_kernel_measurement(capture), harness


def _canonical_mapping(
    campaign: FormalKernelCampaign, receipt: ArtifactReceipt
) -> dict[str, object]:
    content = campaign.record.artifacts.read_bytes(receipt)
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IntegrityError(
            "Formal evaluator artifact is invalid JSON",
            "attempt_evidence_invalid",
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise IntegrityError(
            "Formal evaluator JSON artifact is not canonical",
            "attempt_evidence_invalid",
        )
    return value


__all__ = [
    "attempt_event",
    "event_receipt",
    "kind_event",
    "load_capture",
    "load_grade",
    "require_passed",
]
