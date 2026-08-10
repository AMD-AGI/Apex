"""Typed live-qualification receipt contracts and adversarial cases."""

from __future__ import annotations

import copy

import pytest

from apex.core import ContractError, sha256_json
from apex.runtime import QualificationEvidence, build_qualification_evidence


_SHA = "a" * 64
_TREE = "b" * 40


def _details(name: str, subject: str = _SHA) -> dict:
    if name.startswith("backend-"):
        return _backend_details(name, subject)
    if name == "crash-resume-recovery":
        return _recovery_details(subject)
    if name == "knowledge-ablation":
        return _ablation_details(subject)
    if name == "aka-v14-matched":
        return _aka_details(subject)
    return _magpie_details(subject)


def _backend_details(name: str, subject: str) -> dict:
    return {
        "schema": "apex.backend-live-qualification/v1",
        "qualification_manifest_sha256": subject,
        "backend": name.split("-", 2)[1],
        "gpu_arch": "gfx950",
        "agent_identity_sha256": _SHA,
        "coding_receipt_sha256": _SHA,
        "kernel_receipt_sha256": _SHA,
        "measurement_policy_sha256": _SHA,
    }


def _recovery_details(subject: str) -> dict:
    return {
        "schema": "apex.crash-resume-qualification/v1",
        "qualification_manifest_sha256": subject,
        "task_kinds": ["e2e_kernel_only", "single_kernel"],
        "fault_boundaries": [
            "agent_invocation", "candidate_freeze", "evaluation",
            "final_build_bundle", "image_build_engagement", "keep_reprofile",
            "paired_observation", "second_replay",
        ],
        "fault_matrix_sha256": _SHA,
        "reference_runs_sha256": _SHA,
        "recovered_runs_sha256": _SHA,
        "no_duplicate_apply": True,
        "no_duplicate_decision": True,
        "no_duplicate_reward": True,
        "no_duplicate_stack_mutation": True,
        "partial_windows_discarded": True,
        "gpu_identity_change_rejected": True,
    }


def _ablation_details(subject: str) -> dict:
    return {
        "schema": "apex.knowledge-ablation-qualification/v1",
        "qualification_manifest_sha256": subject,
        "arms": ["disabled", "static_cards", "static_cards_plus_experience"],
        "task_kinds": ["e2e_kernel_only", "single_kernel"],
        "matched_dimensions": [
            "backend_model", "budget", "cohort", "gpu_identity",
            "measurement_policy", "seed",
        ],
        "report_sha256": _SHA,
        "episode_manifest_sha256": _SHA,
        "measured_outcomes_only": True,
        "evaluator_owned_experience_updates": True,
    }


def _aka_details(subject: str) -> dict:
    return {
        "schema": "apex.aka-matched-qualification/v1",
        "qualification_manifest_sha256": subject,
        "repository": "https://github.com/AMD-AGI/AgentKernelArena",
        "commit": "c" * 40,
        "tree": "d" * 40,
        "validator_sha256": _SHA,
        "cohort_manifest_sha256": _SHA,
        "control_receipts_sha256": _SHA,
        "treatment_receipts_sha256": _SHA,
        "central_regrade_sha256": _SHA,
        "cohort_count": 10,
        "matched_dimensions": [
            "budget", "cloud_policy", "cohort", "commit_tree", "gpu_pool",
            "images", "seed", "time_window",
        ],
        "independent_validator": True,
    }


def _magpie_details(subject: str) -> dict:
    return {
        "schema": "apex.magpie-corpus-live-qualification/v3",
        "resolved_manifest_sha256": subject,
        "workflow_manifest_sha256": _SHA,
        "quality_receipts_sha256": _SHA,
        "reward_receipts_sha256": _SHA,
        "frameworks": ["atom", "sglang", "vllm"],
        "run_modes": ["docker", "local", "ray"],
        "lifecycles": ["cleanup", "one_shot", "reuse"],
        "source_adapters": ["aiter", "vllm"],
        "formal_delivery_representatives": [
            {
                "framework": "atom", "run_mode": "docker",
                "lifecycle": "cleanup", "source_adapter": "aiter",
                "delivery_receipt_sha256": "1" * 64,
            },
            {
                "framework": "sglang", "run_mode": "local",
                "lifecycle": "one_shot", "source_adapter": "vllm",
                "delivery_receipt_sha256": "2" * 64,
            },
            {
                "framework": "vllm", "run_mode": "ray",
                "lifecycle": "reuse", "source_adapter": "vllm",
                "delivery_receipt_sha256": "3" * 64,
            },
        ],
        "ray_config_count": 3,
        "ray_plan_manifest_sha256": _SHA,
        "ray_shared_storage_receipts_sha256": _SHA,
        "ray_runtime_receipts_sha256": _SHA,
        "ray_worker_reports_sha256": _SHA,
        "ray_driver_replay_receipts_sha256": _SHA,
        "ray_quality_sync_only": True,
        "ray_shared_runtime_verified": True,
        "ray_driver_evidence_replayed": True,
    }


def _counts(name: str) -> tuple[int, int]:
    if name == "crash-resume-recovery":
        return 32, 2
    if name == "knowledge-ablation":
        return 6, 0
    if name == "aka-v14-matched":
        return 20, 1
    if name == "magpie-corpus-live":
        return 27, 3
    return 2, 1


@pytest.mark.parametrize("name", [
    "aka-v14-matched",
    "backend-claude-gfx950",
    "backend-codex-gfx950",
    "backend-cursor-gfx950",
    "crash-resume-recovery",
    "knowledge-ablation",
    "magpie-corpus-live",
])
def test_every_release_qualification_round_trips_typed_evidence(name: str) -> None:
    coverage, formal = _counts(name)
    receipt = build_qualification_evidence(
        qualification_id=name,
        apex_tree=_TREE,
        subject_sha256=_SHA,
        status="qualified",
        coverage_count=coverage,
        formal_delivery_count=formal,
        details=_details(name),
    )

    assert QualificationEvidence.from_dict(receipt.to_dict()) == receipt


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    [
        (
            "backend-codex-gfx950",
            lambda value: value.update(backend="claude"),
            "backend qualification identity differs",
        ),
        (
            "crash-resume-recovery",
            lambda value: value.update(no_duplicate_reward=False),
            "truth claim is incomplete",
        ),
        (
            "knowledge-ablation",
            lambda value: value.update(arms=["static_cards"]),
            "knowledge ablation arms differs",
        ),
        (
            "aka-v14-matched",
            lambda value: value.update(independent_validator=False),
            "truth claim is incomplete",
        ),
        (
            "magpie-corpus-live",
            lambda value: value.update(run_modes=["docker", "local"]),
            "run-mode coverage differs",
        ),
        (
            "magpie-corpus-live",
            lambda value: value.update(ray_driver_evidence_replayed=False),
            "truth claim is incomplete",
        ),
        (
            "magpie-corpus-live",
            lambda value: value.update(ray_config_count=0),
            "Ray qualification coverage is incomplete",
        ),
        (
            "magpie-corpus-live",
            lambda value: (
                value["formal_delivery_representatives"][1].update(
                    source_adapter="aiter"
                ),
                value["formal_delivery_representatives"][2].update(
                    source_adapter="aiter"
                ),
            ),
            "formal-delivery source_adapters coverage is incomplete",
        ),
    ],
)
def test_typed_claims_cannot_be_replaced_by_name_and_count(
    name: str,
    mutation,
    message: str,
) -> None:
    details = _details(name)
    mutation(details)
    coverage, formal = _counts(name)

    with pytest.raises(ContractError, match=message):
        build_qualification_evidence(
            qualification_id=name,
            apex_tree=_TREE,
            subject_sha256=_SHA,
            status="qualified",
            coverage_count=coverage,
            formal_delivery_count=formal,
            details=details,
        )


def test_subject_and_self_digest_tampering_fail_closed() -> None:
    coverage, formal = _counts("knowledge-ablation")
    receipt = build_qualification_evidence(
        qualification_id="knowledge-ablation",
        apex_tree=_TREE,
        subject_sha256=_SHA,
        status="qualified",
        coverage_count=coverage,
        formal_delivery_count=formal,
        details=_details("knowledge-ablation"),
    ).to_dict()

    changed = copy.deepcopy(receipt)
    changed["subject_sha256"] = "f" * 64
    payload = {key: value for key, value in changed.items() if key != "receipt_sha256"}
    changed["receipt_sha256"] = sha256_json(payload)
    with pytest.raises(ContractError, match="subject differs"):
        QualificationEvidence.from_dict(changed)

    changed = copy.deepcopy(receipt)
    changed["receipt_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="receipt digest differs"):
        QualificationEvidence.from_dict(changed)


def test_legacy_untyped_qualification_schema_is_not_accepted() -> None:
    with pytest.raises(ContractError, match="fields differ"):
        QualificationEvidence.from_dict({
            "schema": "apex.release-qualification/v1",
            "qualification_id": "knowledge-ablation",
            "apex_tree": _TREE,
            "subject_sha256": _SHA,
            "status": "qualified",
            "coverage_count": 1,
            "formal_delivery_count": 0,
            "receipt_sha256": _SHA,
        })
