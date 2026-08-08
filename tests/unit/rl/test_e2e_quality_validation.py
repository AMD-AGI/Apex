from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import IntegrityError, canonical_json_bytes
from apex.rl.e2e_quality_validation import validate_quality_evidence
from apex.rl.models import (
    EpisodeArtifact,
    EpisodeEvent,
    EvidenceClass,
    SemanticRole,
)
from apex.storage import ArtifactStore


def test_quality_result_and_raw_receipts_may_be_disjoint(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_bytes(
        canonical_json_bytes(
            {"results": {"gsm8k": {"exact_match,strict-match": 0.81}}}
        ),
        media_type="application/json",
    )
    sample = artifacts.put_bytes(b'{"doc_id":0}\n', media_type="application/jsonl")
    quality = _quality(result.to_dict(), sample.to_dict())
    event = _event(result, sample)

    accuracy = validate_quality_evidence(
        event,
        artifacts,
        {"quality": _normalized_quality()},
        quality,
    )

    assert accuracy == 0.81


def test_quality_raw_receipt_must_be_bound_by_the_event(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    result = artifacts.put_bytes(
        canonical_json_bytes(
            {"results": {"gsm8k": {"exact_match,strict-match": 0.81}}}
        ),
        media_type="application/json",
    )
    sample = artifacts.put_bytes(b'{"doc_id":0}\n', media_type="application/jsonl")
    unbound = artifacts.put_bytes(b'{"doc_id":1}\n', media_type="application/jsonl")
    quality = _quality(result.to_dict(), unbound.to_dict())

    with pytest.raises(IntegrityError) as error:
        validate_quality_evidence(
            _event(result, sample),
            artifacts,
            {"quality": _normalized_quality()},
            quality,
        )

    assert error.value.reason_code == "e2e_measurement_evidence_mismatch"


def _quality(result: dict[str, object], raw: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "apex.e2e-quality-evidence/v1",
        **_normalized_quality(),
        "result_receipts": [result],
        "raw_artifact_receipts": [raw],
    }


def _normalized_quality() -> dict[str, object]:
    metric = {
        "task": "gsm8k",
        "name": "exact_match,strict-match",
        "value": 0.81,
        "higher_is_better": True,
    }
    return {
        "required": True,
        "kind": "lm_eval",
        "passed": True,
        "metrics": [metric],
        "primary_metrics": [metric],
        "error": None,
        "outcome_digest": "a" * 64,
        "sample_set_digest": "b" * 64,
    }


def _event(result, sample) -> EpisodeEvent:
    event_id = "event-quality"
    return EpisodeEvent(
        1,
        event_id,
        "transaction-quality",
        None,
        "measurement_result",
        SemanticRole.OBSERVATION,
        EvidenceClass.MEASURED,
        {},
        (
            EpisodeArtifact("quality_result", result, event_id),
            EpisodeArtifact("quality_sample", sample, event_id),
        ),
    )
