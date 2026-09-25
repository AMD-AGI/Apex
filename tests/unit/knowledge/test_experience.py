from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from apex.core import ContractError, sha256_json
from apex.knowledge import ExperienceIdentity, ExperienceView


@dataclass(frozen=True)
class Event:
    sequence: int
    event_id: str
    event_type: str
    payload: Mapping[str, Any]


def _identity(source: str = "a") -> dict[str, object]:
    return {
        "task_id": "rms-norm",
        "operator": "rms_norm",
        "gpu_arch": "gfx950",
        "framework": "vllm",
        "versions": {"rocm": "7.2"},
        "shape_hash": sha256_json("shape"),
        "source_hash": sha256_json(source),
        "harness_hash": sha256_json("harness"),
        "policy_hash": sha256_json("policy"),
    }


def _measured(sequence: int, *, evidence_class: str = "measured", dry_run: bool = False) -> Event:
    return Event(
        sequence,
        f"event-{sequence}",
        "experience.measured",
        {
            "evidence_class": evidence_class,
            "dry_run": dry_run,
            "candidate_id": f"candidate-{sequence}",
            "identity": _identity(),
            "outcome": "no_gain",
            "strategy_fingerprint": sha256_json(f"strategy-{sequence}"),
            "mechanism": "Increase vector width.",
            "micro_verdict": "correct_no_gain",
            "e2e_verdict": None,
            "evidence_receipts": [sha256_json(f"receipt-{sequence}")],
            "failure_reason": "Tail latency did not improve.",
            "retry_condition": "Retry only after shape or source changes.",
        },
    )


def test_view_is_deterministic_receipt_only_and_exactly_scoped() -> None:
    measured = _measured(2)
    self_reported = _measured(1, evidence_class="self_reported")
    dry_run = _measured(3, dry_run=True)
    link = Event(
        4,
        "event-4",
        "knowledge_outcome_linked",
        {
            "read_id": "read-1",
            "card_id": "card-aaaaaaaaaaaaaaaaaaaaaaaa",
            "outcome": "inconclusive",
            "evidence_receipt": sha256_json("link"),
        },
    )

    first = ExperienceView.from_events((link, dry_run, measured, self_reported))
    second = ExperienceView.from_events((self_reported, measured, link, dry_run))

    assert first == second
    assert first.digest == second.digest
    assert len(first.records) == 1
    assert len(first.knowledge_outcomes) == 1
    assert len(first.compatible(ExperienceIdentity.from_mapping(_identity()))) == 1
    assert first.compatible(ExperienceIdentity.from_mapping(_identity("different"))) == ()


def test_malformed_canonical_measured_event_fails_closed() -> None:
    event = _measured(1)
    payload = dict(event.payload)
    payload["evidence_receipts"] = []

    with pytest.raises(ContractError) as failure:
        ExperienceView.from_events((Event(1, "event-1", event.event_type, payload),))
    assert failure.value.reason_code == "missing_experience_evidence"
