from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.core import TaskStatus, ValidationLevel
from apex.delivery import TaskResult, write_task_result


def _statistical_fields() -> dict[str, object]:
    return {
        "max_cv": 0.02,
        "s50_ci_lower": 1.20,
        "s50_ci_upper": 1.30,
        "s99_ci_lower": 1.08,
        "s99_ci_upper": 1.16,
        "srobust_ci_lower": 1.08,
        "srobust_ci_upper": 1.16,
        "confidence_level": 0.95,
        "threshold_pass": True,
        "confidence_pass": True,
        "noise_pass": True,
        "worst_case_pass": True,
        "promotion_eligible": True,
        "promotion_reason_code": "promotion_eligible",
    }


def test_task_result_is_atomic_machine_contract(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    result = TaskResult(
        schema_version=1,
        run_id="run-test",
        task_id="kernel-task",
        status=TaskStatus.CANDIDATE_READY,
        reason_code="verified_candidate",
        applied=False,
        external_verification_required=True,
        bundle_path="/results/bundle",
        bundle_digest="a" * 64,
        changed_files=("source/kernel.py",),
        validation_level=ValidationLevel.RUNTIME_OVERLAY_VERIFIED,
    )

    write_task_result(result, path)
    data = json.loads(path.read_text())

    assert data["status"] == "candidate_ready"
    assert data["applied"] is False
    assert data["external_verification_required"] is True
    assert data["validation_level"] == "runtime_overlay_verified"
    assert data["safety_status"] == "not_run"
    assert data["safety_certified"] is False
    assert data["safety_result_fingerprint"] is None
    assert data["safety_receipt_digest"] is None
    assert not list(path.parent.glob(".result.json.*"))


def test_task_result_serializes_standalone_lineage_without_changing_bundle() -> None:
    result = TaskResult(
        schema_version=1,
        run_id="run-123",
        task_id="kernel-task",
        status=TaskStatus.CANDIDATE_READY,
        reason_code="verified_candidate",
        applied=False,
        external_verification_required=True,
        bundle_path="/results/bundle",
        bundle_digest="a" * 64,
        changed_files=("source/kernel.py",),
        baseline_lock={
            "resolution_hash": "b" * 64,
            "file_hashes": {"source/kernel.py": "c" * 64},
        },
        internal_verdict="keep",
        internal_verdict_ref="evt-123",
        verification_summary_refs=("d" * 64,),
        event_journal_ref={
            "path": "/results/events/run.db",
            "head_event_id": "evt-456",
            "head_checksum": "e" * 64,
        },
        artifact_store_ref={
            "path": "/results/artifacts",
            "receipt_digests": ["d" * 64],
        },
    )

    value = result.to_dict()

    assert value["run_id"] == "run-123"
    assert value["baseline_lock"]["resolution_hash"] == "b" * 64
    assert value["internal_verdict"] == "keep"
    assert value["verification_summary_refs"] == ["d" * 64]
    assert value["bundle_digest"] == "a" * 64


def test_task_result_validates_frozen_evaluation_contract_receipt() -> None:
    result = TaskResult(
        schema_version=1,
        run_id="run-test",
        task_id="kernel-task",
        status=TaskStatus.CANDIDATE_READY,
        reason_code="verified_candidate",
        applied=False,
        external_verification_required=True,
        bundle_path="/results/bundle",
        bundle_digest="a" * 64,
        changed_files=("source/kernel.py",),
        evaluation_contract_status="verified",
        evaluation_contract_receipt_digest="b" * 64,
        evaluation_authority_id="reviewed-template-v1",
        evaluation_authority_kind="reviewed_template",
    )

    assert result.to_dict()["evaluation_contract_status"] == "verified"

    with pytest.raises(ValueError, match="requires its receipt digest"):
        TaskResult(
            schema_version=1,
            run_id="run-test",
            task_id="kernel-task",
            status=TaskStatus.INVALID_REQUEST,
            reason_code="evaluation_authority_missing",
            applied=False,
            external_verification_required=True,
            bundle_path=None,
            bundle_digest=None,
            changed_files=(),
            evaluation_contract_status="unverified",
            evaluation_contract_unverified_reason="evaluation_authority_missing",
        )


def test_task_result_requires_receipts_for_evaluated_safety() -> None:
    with pytest.raises(ValueError, match="requires exact result receipts"):
        TaskResult(
            schema_version=1,
            run_id="run-test",
            task_id="kernel-task",
            status=TaskStatus.REJECTED,
            reason_code="required_safety_incomplete",
            applied=False,
            external_verification_required=True,
            bundle_path=None,
            bundle_digest=None,
            changed_files=(),
            safety_status="required_incomplete",
        )

    result = TaskResult(
        schema_version=1,
        run_id="run-test",
        task_id="kernel-task",
        status=TaskStatus.CANDIDATE_READY,
        reason_code="verified_candidate",
        applied=False,
        external_verification_required=True,
        bundle_path="/results/bundle",
        bundle_digest="a" * 64,
        changed_files=("source/kernel.py",),
        safety_status="advisory_incomplete",
        safety_certified=False,
        safety_result_fingerprint="b" * 64,
        safety_receipt_digest="c" * 64,
    )

    assert result.to_dict()["safety_status"] == "advisory_incomplete"


def test_valid_measurement_serializes_complete_robust_grade(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    result = TaskResult(
        schema_version=1,
        run_id="run-test",
        task_id="kernel-task",
        status=TaskStatus.CANDIDATE_READY,
        reason_code="robust_improvement",
        applied=False,
        external_verification_required=True,
        bundle_path="/results/bundle",
        bundle_digest="a" * 64,
        changed_files=("source/kernel.py",),
        measurement_status="valid",
        measurement_report_sha256="d" * 64,
        grade_policy_id="kernel_robust_v1",
        s50=1.25,
        s99=1.125,
        srobust=1.125,
        worst_case_srobust=1.05,
        reward=145.0,
        **_statistical_fields(),
    )

    write_task_result(result, path)
    data = json.loads(path.read_text())

    assert data["measurement_status"] == "valid"
    assert data["measurement_report_sha256"] == "d" * 64
    assert data["grade_policy_id"] == "kernel_robust_v1"
    assert data["s50"] == 1.25
    assert data["s99"] == 1.125
    assert data["srobust"] == min(data["s50"], data["s99"])
    assert data["worst_case_srobust"] == 1.05
    assert data["reward"] == 145.0
    assert data["srobust_ci_lower"] == 1.08
    assert data["max_cv"] == 0.02
    assert data["promotion_eligible"] is True


def test_insufficient_measurement_serializes_evidence_without_reward() -> None:
    result = TaskResult(
        schema_version=1,
        run_id="run-test",
        task_id="kernel-task",
        status=TaskStatus.NO_MEASUREMENT,
        reason_code="insufficient_samples",
        applied=False,
        external_verification_required=True,
        bundle_path=None,
        bundle_digest=None,
        changed_files=(),
        measurement_status="insufficient_samples",
        measurement_report_sha256="e" * 64,
        grade_policy_id="kernel_robust_v1",
    )

    data = result.to_dict()

    assert data["measurement_status"] == "insufficient_samples"
    assert data["measurement_report_sha256"] == "e" * 64
    assert data["grade_policy_id"] == "kernel_robust_v1"
    assert data["s50"] is None
    assert data["s99"] is None
    assert data["srobust"] is None
    assert data["worst_case_srobust"] is None
    assert data["reward"] is None


def test_measurement_error_serializes_without_grade_claims() -> None:
    result = TaskResult(
        schema_version=1,
        run_id="run-test",
        task_id="kernel-task",
        status=TaskStatus.NO_MEASUREMENT,
        reason_code="invalid_measurement_report",
        applied=False,
        external_verification_required=True,
        bundle_path=None,
        bundle_digest=None,
        changed_files=(),
        measurement_status="error",
    )

    data = result.to_dict()

    assert data["measurement_status"] == "error"
    assert data["measurement_report_sha256"] is None
    assert data["grade_policy_id"] is None
    assert data["s50"] is None
    assert data["s99"] is None
    assert data["srobust"] is None
    assert data["worst_case_srobust"] is None
    assert data["reward"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"measurement_report_sha256": None}, "exact report and policy identity"),
        ({"grade_policy_id": "legacy_reward"}, "exact report and policy identity"),
        ({"s99": None}, "complete robust point evidence"),
        ({"srobust": 1.2}, "Srobust must be min"),
    ],
)
def test_valid_measurement_rejects_incoherent_grade(
    overrides: dict[str, object],
    message: str,
) -> None:
    fields: dict[str, object] = {
        "measurement_report_sha256": "f" * 64,
        "grade_policy_id": "kernel_robust_v1",
        "s50": 1.25,
        "s99": 1.125,
        "srobust": 1.125,
        "worst_case_srobust": 1.05,
        "reward": 145.0,
        **_statistical_fields(),
    }
    fields.update(overrides)

    with pytest.raises(ValueError, match=message):
        TaskResult(
            schema_version=1,
            run_id="run-test",
            task_id="kernel-task",
            status=TaskStatus.CANDIDATE_READY,
            reason_code="robust_improvement",
            applied=False,
            external_verification_required=True,
            bundle_path="/results/bundle",
            bundle_digest="a" * 64,
            changed_files=("source/kernel.py",),
            measurement_status="valid",
            **fields,  # type: ignore[arg-type]
        )


def test_unconfigured_measurement_rejects_grade_evidence() -> None:
    with pytest.raises(ValueError, match="cannot carry grade evidence"):
        TaskResult(
            schema_version=1,
            run_id="run-test",
            task_id="kernel-task",
            status=TaskStatus.CANDIDATE_READY,
            reason_code="external_measurement_required",
            applied=False,
            external_verification_required=True,
            bundle_path="/results/bundle",
            bundle_digest="a" * 64,
            changed_files=("source/kernel.py",),
            measurement_report_sha256="f" * 64,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_measurement_rejects_non_finite_grade_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        TaskResult(
            schema_version=1,
            run_id="run-test",
            task_id="kernel-task",
            status=TaskStatus.NO_MEASUREMENT,
            reason_code="invalid_measurement",
            applied=False,
            external_verification_required=True,
            bundle_path=None,
            bundle_digest=None,
            changed_files=(),
            measurement_status="invalid",
            s50=value,
        )
