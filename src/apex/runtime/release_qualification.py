"""Typed, self-digested live qualification evidence for release gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ContractError, sha256_json


QUALIFICATION_IDS = (
    "aka-v14-matched",
    "backend-claude-gfx950",
    "backend-codex-gfx950",
    "backend-cursor-gfx950",
    "crash-resume-recovery",
    "knowledge-ablation",
    "magpie-corpus-live",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")
_TASK_KINDS = ["e2e_kernel_only", "single_kernel"]
_RECOVERY_BOUNDARIES = [
    "agent_invocation",
    "candidate_freeze",
    "evaluation",
    "final_build_bundle",
    "image_build_engagement",
    "keep_reprofile",
    "paired_observation",
    "second_replay",
]
_ABLATION_ARMS = [
    "disabled",
    "static_cards",
    "static_cards_plus_experience",
]
_ABLATION_MATCHED = [
    "backend_model",
    "budget",
    "cohort",
    "gpu_identity",
    "measurement_policy",
    "seed",
]
_AKA_MATCHED = [
    "budget",
    "cloud_policy",
    "cohort",
    "commit_tree",
    "gpu_pool",
    "images",
    "seed",
    "time_window",
]


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    """One source-bound qualification plus its kind-specific verified claims."""

    qualification_id: str
    apex_tree: str
    subject_sha256: str
    status: str
    coverage_count: int
    formal_delivery_count: int
    details: Mapping[str, Any]
    receipt_sha256: str

    SCHEMA = "apex.release-qualification/v2"

    def __post_init__(self) -> None:
        if self.qualification_id not in QUALIFICATION_IDS:
            raise _invalid("qualification_id is not a release gate")
        _match(self.apex_tree, _GIT, "qualification Apex tree")
        _match(self.subject_sha256, _SHA256, "qualification subject")
        if self.status not in {"pending", "qualified"}:
            raise _invalid("qualification status is invalid")
        _count(self.coverage_count, "coverage_count")
        _count(self.formal_delivery_count, "formal_delivery_count")
        if not isinstance(self.details, Mapping):
            raise _invalid("qualification details are invalid")
        _validate_details(self)
        _match(self.receipt_sha256, _SHA256, "qualification receipt")
        if self.receipt_sha256 != sha256_json(self.payload()):
            raise _invalid("qualification receipt digest differs")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "qualification_id": self.qualification_id,
            "apex_tree": self.apex_tree,
            "subject_sha256": self.subject_sha256,
            "status": self.status,
            "coverage_count": self.coverage_count,
            "formal_delivery_count": self.formal_delivery_count,
            "details": dict(self.details),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: object) -> QualificationEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "qualification")
        if raw["schema"] != cls.SCHEMA:
            raise _invalid("qualification schema differs")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


def build_qualification_evidence(
    *,
    qualification_id: str,
    apex_tree: str,
    subject_sha256: str,
    status: str,
    coverage_count: int,
    formal_delivery_count: int,
    details: Mapping[str, Any],
) -> QualificationEvidence:
    """Build one canonical v2 receipt after kind-specific validation."""

    payload = {
        "schema": QualificationEvidence.SCHEMA,
        "qualification_id": qualification_id,
        "apex_tree": apex_tree,
        "subject_sha256": subject_sha256,
        "status": status,
        "coverage_count": coverage_count,
        "formal_delivery_count": formal_delivery_count,
        "details": dict(details),
    }
    return QualificationEvidence(
        qualification_id,
        apex_tree,
        subject_sha256,
        status,
        coverage_count,
        formal_delivery_count,
        dict(details),
        sha256_json(payload),
    )


def _validate_details(value: QualificationEvidence) -> None:
    if value.qualification_id.startswith("backend-"):
        _validate_backend(value)
    elif value.qualification_id == "crash-resume-recovery":
        _validate_recovery(value)
    elif value.qualification_id == "knowledge-ablation":
        _validate_ablation(value)
    elif value.qualification_id == "aka-v14-matched":
        _validate_aka(value)
    else:
        _validate_magpie(value)


def _validate_backend(value: QualificationEvidence) -> None:
    raw = _details(value, {
        "schema", "qualification_manifest_sha256", "backend", "gpu_arch",
        "agent_identity_sha256", "coding_receipt_sha256", "kernel_receipt_sha256",
        "measurement_policy_sha256",
    }, "apex.backend-live-qualification/v1")
    expected = value.qualification_id.split("-", 2)[1]
    if raw["backend"] != expected or raw["gpu_arch"] != "gfx950":
        raise _invalid("backend qualification identity differs")
    _hash_fields(raw, {
        "qualification_manifest_sha256", "agent_identity_sha256",
        "coding_receipt_sha256", "kernel_receipt_sha256",
        "measurement_policy_sha256",
    })
    _subject(value, raw["qualification_manifest_sha256"])
    if value.status == "qualified" and (
        value.coverage_count < 2 or value.formal_delivery_count < 1
    ):
        raise _invalid("backend qualification coverage is incomplete")


def _validate_recovery(value: QualificationEvidence) -> None:
    raw = _details(value, {
        "schema", "qualification_manifest_sha256", "task_kinds",
        "fault_boundaries", "fault_matrix_sha256", "reference_runs_sha256",
        "recovered_runs_sha256", "no_duplicate_apply", "no_duplicate_decision",
        "no_duplicate_reward", "no_duplicate_stack_mutation",
        "partial_windows_discarded", "gpu_identity_change_rejected",
    }, "apex.crash-resume-qualification/v1")
    _hash_fields(raw, {
        "qualification_manifest_sha256", "fault_matrix_sha256",
        "reference_runs_sha256", "recovered_runs_sha256",
    })
    _exact_list(raw["task_kinds"], _TASK_KINDS, "recovery task kinds")
    _exact_list(raw["fault_boundaries"], _RECOVERY_BOUNDARIES, "fault boundaries")
    _true_fields(raw, {
        "no_duplicate_apply", "no_duplicate_decision", "no_duplicate_reward",
        "no_duplicate_stack_mutation", "partial_windows_discarded",
        "gpu_identity_change_rejected",
    })
    _subject(value, raw["qualification_manifest_sha256"])
    if value.status == "qualified" and (
        value.coverage_count < 32 or value.formal_delivery_count < 2
    ):
        raise _invalid("crash-resume qualification coverage is incomplete")


def _validate_ablation(value: QualificationEvidence) -> None:
    raw = _details(value, {
        "schema", "qualification_manifest_sha256", "arms", "task_kinds",
        "matched_dimensions", "report_sha256", "episode_manifest_sha256",
        "measured_outcomes_only", "evaluator_owned_experience_updates",
    }, "apex.knowledge-ablation-qualification/v1")
    _hash_fields(raw, {
        "qualification_manifest_sha256", "report_sha256",
        "episode_manifest_sha256",
    })
    _exact_list(raw["arms"], _ABLATION_ARMS, "knowledge ablation arms")
    _exact_list(raw["task_kinds"], _TASK_KINDS, "ablation task kinds")
    _exact_list(raw["matched_dimensions"], _ABLATION_MATCHED, "matched dimensions")
    _true_fields(raw, {"measured_outcomes_only", "evaluator_owned_experience_updates"})
    _subject(value, raw["qualification_manifest_sha256"])
    if value.status == "qualified" and value.coverage_count < 6:
        raise _invalid("knowledge ablation coverage is incomplete")


def _validate_aka(value: QualificationEvidence) -> None:
    raw = _details(value, {
        "schema", "qualification_manifest_sha256", "repository", "commit", "tree",
        "validator_sha256", "cohort_manifest_sha256", "control_receipts_sha256",
        "treatment_receipts_sha256", "central_regrade_sha256", "cohort_count",
        "matched_dimensions", "independent_validator",
    }, "apex.aka-matched-qualification/v1")
    if raw["repository"] != "https://github.com/AMD-AGI/AgentKernelArena":
        raise _invalid("AKA qualification repository differs")
    _match(raw["commit"], _GIT, "AKA commit")
    _match(raw["tree"], _GIT, "AKA tree")
    _hash_fields(raw, {
        "qualification_manifest_sha256", "validator_sha256",
        "cohort_manifest_sha256", "control_receipts_sha256",
        "treatment_receipts_sha256", "central_regrade_sha256",
    })
    _exact_list(raw["matched_dimensions"], _AKA_MATCHED, "AKA matched dimensions")
    if type(raw["cohort_count"]) is not int or raw["cohort_count"] < 10:
        raise _invalid("AKA cohort is incomplete")
    _true_fields(raw, {"independent_validator"})
    _subject(value, raw["qualification_manifest_sha256"])
    if value.status == "qualified" and (
        value.coverage_count < 2 * raw["cohort_count"]
        or value.formal_delivery_count < 1
    ):
        raise _invalid("AKA matched qualification coverage is incomplete")


def _validate_magpie(value: QualificationEvidence) -> None:
    raw = _details(value, {
        "schema", "resolved_manifest_sha256", "workflow_manifest_sha256",
        "quality_receipts_sha256", "reward_receipts_sha256", "frameworks",
        "run_modes", "lifecycles", "source_adapters",
        "formal_delivery_representatives", "e2e_v2_scope",
        "e2e_v2_config_count", "e2e_v2_plan_manifest_sha256",
        "e2e_v2_rejection_count", "e2e_v2_rejection_manifest_sha256",
        "early_rejection_receipts_sha256", "rejected_before_provenance",
        "rejected_before_gpu", "rejected_before_agent",
        "rejected_without_result_root",
    }, "apex.magpie-corpus-live-qualification/v4")
    _hash_fields(raw, {
        "resolved_manifest_sha256", "workflow_manifest_sha256",
        "quality_receipts_sha256", "reward_receipts_sha256",
        "e2e_v2_plan_manifest_sha256",
        "e2e_v2_rejection_manifest_sha256",
        "early_rejection_receipts_sha256",
    })
    _exact_list(raw["frameworks"], ["sglang", "vllm"], "framework coverage")
    _exact_list(raw["run_modes"], ["docker"], "run-mode coverage")
    _exact_list(raw["lifecycles"], ["one_shot"], "lifecycle coverage")
    adapters = raw["source_adapters"]
    if not isinstance(adapters, list) or not adapters or adapters != sorted(set(adapters)):
        raise _invalid("source-adapter coverage is invalid")
    _validate_formal_delivery_representatives(value, raw)
    if raw["e2e_v2_scope"] != "docker_one_shot":
        raise _invalid("E2E V2 qualification scope differs")
    _count(raw["e2e_v2_config_count"], "e2e_v2_config_count")
    _count(raw["e2e_v2_rejection_count"], "e2e_v2_rejection_count")
    _true_fields(raw, {
        "rejected_before_provenance", "rejected_before_gpu",
        "rejected_before_agent", "rejected_without_result_root",
    })
    _subject(value, raw["resolved_manifest_sha256"])
    if value.status == "qualified":
        if value.formal_delivery_count < 1:
            raise _invalid("Magpie formal-delivery coverage is incomplete")
        if (
            raw["e2e_v2_config_count"] < 1
            or value.coverage_count != raw["e2e_v2_config_count"]
            or raw["e2e_v2_rejection_count"] < 1
        ):
            raise _invalid("Docker qualification coverage is incomplete")


def _validate_formal_delivery_representatives(
    value: QualificationEvidence, raw: Mapping[str, Any]
) -> None:
    representatives = raw["formal_delivery_representatives"]
    fields = {
        "framework", "run_mode", "lifecycle", "source_adapter",
        "config_path", "config_sha256", "plan_sha256",
        "capability_receipt_sha256", "delivery_receipt_sha256",
    }
    if not isinstance(representatives, list) or len(representatives) != (
        value.formal_delivery_count
    ):
        raise _invalid("formal-delivery representatives are incomplete")
    identities: list[tuple[str, ...]] = []
    for item in representatives:
        entry = _strict(item, fields, "formal-delivery representative")
        _match(
            entry["delivery_receipt_sha256"], _SHA256,
            "formal-delivery receipt",
        )
        _match(entry["config_sha256"], _SHA256, "representative config")
        _match(entry["plan_sha256"], _SHA256, "representative plan")
        _match(
            entry["capability_receipt_sha256"],
            _SHA256,
            "representative capability receipt",
        )
        identity = (
            str(entry["framework"]),
            str(entry["run_mode"]),
            str(entry["lifecycle"]),
            str(entry["source_adapter"]),
            str(entry["config_path"]),
            str(entry["config_sha256"]),
            str(entry["plan_sha256"]),
            str(entry["capability_receipt_sha256"]),
            str(entry["delivery_receipt_sha256"]),
        )
        identities.append(identity)
    if identities != sorted(set(identities)):
        raise _invalid("formal-delivery representatives are not unique/sorted")
    for identity in identities:
        if (
            identity[0] not in raw["frameworks"]
            or identity[1] not in raw["run_modes"]
            or identity[2] not in raw["lifecycles"]
            or identity[3] not in raw["source_adapters"]
        ):
            raise _invalid("formal-delivery representative is outside product scope")


def _details(
    value: QualificationEvidence,
    fields: set[str],
    schema: str,
) -> Mapping[str, Any]:
    raw = _strict(value.details, fields, "qualification details")
    if raw["schema"] != schema:
        raise _invalid("qualification detail schema differs")
    return raw


def _strict(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _invalid(f"{label} fields differ")
    return value


def _hash_fields(value: Mapping[str, Any], fields: set[str]) -> None:
    for field in fields:
        _match(value[field], _SHA256, field)


def _match(value: object, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _invalid(f"{label} is invalid")


def _count(value: object, label: str) -> None:
    if type(value) is not int or value < 0:
        raise _invalid(f"{label} is invalid")


def _exact_list(value: object, expected: list[str], label: str) -> None:
    if value != expected:
        raise _invalid(f"{label} differs")


def _true_fields(value: Mapping[str, Any], fields: set[str]) -> None:
    if any(value[field] is not True for field in fields):
        raise _invalid("qualification truth claim is incomplete")


def _subject(value: QualificationEvidence, expected: object) -> None:
    if value.subject_sha256 != expected:
        raise _invalid("qualification subject differs from its detail receipt")


def _invalid(message: str) -> ContractError:
    return ContractError(message, "invalid_release_evidence")


__all__ = [
    "QUALIFICATION_IDS",
    "QualificationEvidence",
    "build_qualification_evidence",
]
