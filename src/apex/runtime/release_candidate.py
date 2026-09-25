"""Deterministic, fail-closed release-candidate identity and readiness gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes
from apex.ports import QualificationAuthorityPort, QualificationAuthorityReceipt

from .release_commands import (
    CPU_GATE_COMPILEALL_ARGV,
    CPU_GATE_PYTEST_ARGV,
    CPU_GATE_SCAN_ARGV,
)
from .release_baseline import (
    assess_campaign_apex_baseline,
    assess_release_baseline,
    repository_key,
)
from .release_evidence import (
    CpuGateEvidence,
    DependencyVerificationEvidence,
    MagpieConfigResolutionEntryEvidence,
    ReleaseEvidence,
)
from .release_qualification import QUALIFICATION_IDS
from .release_static import collect_release_static_identity


SCHEMA = "apex.release-candidate-receipt/v2"
APEX_REPOSITORY = "github.com/amd-agi/apex"
E2E_SHOWCASE = "e2e-qwen3-next-80b-fp8"
KERNEL_SHOWCASES = (
    "kernel-ck-moe-2stage",
    "kernel-cktile-moe-2stage",
    "kernel-triton-paged-attention-2d",
)
REQUIRED_SHOWCASES = (E2E_SHOWCASE, *KERNEL_SHOWCASES)
REQUIRED_IMAGES = ("lm-eval-parent", *REQUIRED_SHOWCASES)
REQUIRED_QUALIFICATIONS = QUALIFICATION_IDS


@dataclass(frozen=True, slots=True)
class ReleaseCandidateReceipt:
    """Immutable canonical document whose digest excludes only itself."""

    canonical_payload: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        if sha256_bytes(self.canonical_payload) != self.receipt_sha256:
            raise IntegrityError("Release receipt digest differs", "release_receipt_tampered")
        _validate_payload(self.document)

    @property
    def document(self) -> Mapping[str, Any]:
        value = json.loads(self.canonical_payload)
        if not isinstance(value, Mapping):
            raise IntegrityError("Release receipt root differs", "release_receipt_tampered")
        return value

    @property
    def status(self) -> str:
        return str(self.document["status"])

    @property
    def baseline_status(self) -> str:
        return str(self.document["baseline_status"])

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.document["blockers"])

    @property
    def baseline_blockers(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.document["baseline_blockers"])

    def to_dict(self) -> dict[str, Any]:
        return {**self.document, "receipt_sha256": self.receipt_sha256}


def inspect_release_candidate(
    apex_root: Path,
    evidence: ReleaseEvidence | None = None,
    *,
    qualification_authority: QualificationAuthorityPort | None = None,
) -> ReleaseCandidateReceipt:
    """Combine local identity with claims independently verified by authority."""

    root = apex_root.expanduser().resolve(strict=True)
    supplied = ReleaseEvidence.from_dict((evidence or ReleaseEvidence()).to_dict())
    static = collect_release_static_identity(
        root,
        kernel_showcases=KERNEL_SHOWCASES,
        required_showcases=REQUIRED_SHOWCASES,
        required_images=REQUIRED_IMAGES,
        required_qualifications=REQUIRED_QUALIFICATIONS,
    )
    authorities = _qualification_authorities(supplied, qualification_authority)
    baseline_blockers, blockers = _assess(static, supplied, authorities)
    payload = {
        "schema": SCHEMA,
        "baseline_status": "ready" if not baseline_blockers else "blocked",
        "status": "ready" if not blockers else "blocked",
        "static": static,
        "evidence": supplied.to_dict(),
        "qualification_authorities": [item.to_dict() for item in authorities],
        "baseline_blockers": list(baseline_blockers),
        "blockers": list(blockers),
    }
    canonical = canonical_json_bytes(payload)
    return ReleaseCandidateReceipt(canonical, sha256_bytes(canonical))


def verify_release_candidate_receipt(
    value: Mapping[str, Any],
    *,
    apex_root: Path,
    qualification_authority: QualificationAuthorityPort | None = None,
) -> ReleaseCandidateReceipt:
    """Rebuild the whole receipt; a self-consistent edited status is insufficient."""

    expected = {
        "schema", "baseline_status", "status", "static", "evidence",
        "qualification_authorities", "baseline_blockers", "blockers",
        "receipt_sha256",
    }
    if set(value) != expected:
        raise IntegrityError("Release receipt field set differs", "release_receipt_tampered")
    digest = value.get("receipt_sha256")
    payload = {key: value[key] for key in value if key != "receipt_sha256"}
    if not isinstance(digest, str) or sha256_bytes(canonical_json_bytes(payload)) != digest:
        raise IntegrityError("Release receipt digest differs", "release_receipt_tampered")
    evidence = ReleaseEvidence.from_dict(value.get("evidence"))
    rebuilt = inspect_release_candidate(
        apex_root,
        evidence,
        qualification_authority=qualification_authority,
    )
    if rebuilt.to_dict() != dict(value):
        raise IntegrityError("Release receipt no longer matches source/evidence", "release_receipt_tampered")
    return rebuilt


def freeze_release_candidate(
    value: Mapping[str, Any],
    *,
    apex_root: Path,
    qualification_authority: QualificationAuthorityPort | None = None,
) -> ReleaseCandidateReceipt:
    """Return only a fully verified ready receipt; blocked receipts cannot launch live work."""

    receipt = verify_release_candidate_receipt(
        value,
        apex_root=apex_root,
        qualification_authority=qualification_authority,
    )
    if receipt.status != "ready" or receipt.blockers:
        raise ContractError(
            "Release candidate is blocked: " + ", ".join(receipt.blockers),
            "release_candidate_blocked",
        )
    return receipt


def freeze_campaign_baseline(
    value: Mapping[str, Any],
    *,
    apex_root: Path,
    qualification_authority: QualificationAuthorityPort | None = None,
) -> ReleaseCandidateReceipt:
    """Authorize live qualification from a clean baseline, not future results."""

    receipt = verify_release_candidate_receipt(
        value,
        apex_root=apex_root,
        qualification_authority=qualification_authority,
    )
    if receipt.baseline_status != "ready" or receipt.baseline_blockers:
        raise ContractError(
            "Campaign baseline is blocked: " + ", ".join(receipt.baseline_blockers),
            "campaign_baseline_blocked",
        )
    return receipt


def _assess(
    static: Mapping[str, Any],
    evidence: ReleaseEvidence,
    authorities: tuple[QualificationAuthorityReceipt, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    baseline: set[str] = set()
    checkout = static["apex_checkout"]
    if not checkout["clean"]:
        baseline.add("apex_source_dirty")
    if repository_key(checkout["repository"]) != APEX_REPOSITORY:
        baseline.add("apex_repository_unofficial")
    assess_campaign_apex_baseline(evidence.apex_baseline, checkout, baseline)
    _assess_dependencies(static, evidence.dependencies, baseline)
    _assess_magpie_config_resolution(static, evidence, baseline)
    _assess_cpu_gate(static, evidence.cpu_gate, baseline)
    _assess_cli(static, evidence, baseline)
    release = set(baseline)
    assess_release_baseline("apex", evidence.apex_baseline, checkout, release)
    assess_release_baseline(
        "magpie", evidence.magpie_baseline, static["magpie"], release
    )
    _assess_images(static, evidence, release)
    _assess_showcases(static, evidence, release)
    _assess_qualifications(static, evidence, authorities, release)
    return tuple(sorted(baseline)), tuple(sorted(release))


def _assess_dependencies(
    static: Mapping[str, Any],
    value: DependencyVerificationEvidence | None,
    blockers: set[str],
) -> None:
    if value is None:
        blockers.add("dependency_verification_missing")
        return
    checkout = static["apex_checkout"]
    locks = static["locks"]
    expected_locks = (
        value.dependencies_lock_sha256 == locks["dependencies"]
        and value.e2e_source_lock_sha256 == locks["e2e_sources"]
        and value.lm_eval_runtime_lock_sha256 == locks["lm_eval_runtime"]
        and value.evaluator_policy_lock_sha256 == locks["evaluator_policy"]
        and value.agent_templates_lock_sha256 == locks["agent_templates"]
        and value.lm_eval_runtime_sha256 == static["lm_eval"]["runtime_sha256"]
    )
    if value.apex_tree != checkout["tree"] or not expected_locks:
        blockers.add("dependency_verification_identity_mismatch")
    if not value.all_imports_exact:
        blockers.add("dependency_import_identity_unverified")
    expected = {item["name"]: item for item in (*static["dependencies"], *static["sources"])}
    observed = {item.name: item for item in value.components}
    if set(observed) != set(expected):
        blockers.add("dependency_component_inventory_mismatch")
        return
    for name, item in observed.items():
        target = expected[name]
        valid = (
            item.clean
            and repository_key(item.repository) == repository_key(target["repository"])
            and item.commit == target["commit"]
            and ("tree" not in target or item.tree == target["tree"])
        )
        if not valid:
            blockers.add(f"dependency_component_mismatch:{name}")


def _assess_cpu_gate(
    static: Mapping[str, Any],
    value: CpuGateEvidence | None,
    blockers: set[str],
) -> None:
    if value is None:
        blockers.add("cpu_gate_missing")
        return
    locks = static["locks"]
    magpie = static["magpie"]
    bound = (
        value.apex_tree == static["apex_checkout"]["tree"]
        and value.dependencies_lock_sha256 == locks["dependencies"]
        and value.e2e_source_lock_sha256 == locks["e2e_sources"]
        and value.corpus_manifest_sha256 == magpie["corpus_manifest_sha256"]
        and value.compatibility_ledger_sha256 == magpie["compatibility_ledger_sha256"]
    )
    commands = (
        Path(value.pytest_argv[0]).name == "pytest"
        and value.pytest_argv[1:] == CPU_GATE_PYTEST_ARGV[1:]
        and Path(value.compileall_argv[0]).name in {"python", "python3"}
        and value.compileall_argv[1:] == CPU_GATE_COMPILEALL_ARGV[1:]
        and Path(value.forbidden_scan_argv[0]).name == "rg"
        and value.forbidden_scan_argv[1:] == CPU_GATE_SCAN_ARGV[1:]
    )
    passed = (
        value.pytest_exit_code == 0 and value.passed_count > 0
        and value.failed_count == 0 and value.compileall_exit_code == 0
        and value.forbidden_scan_exit_code == 1 and value.forbidden_scan_clean
    )
    if not bound:
        blockers.add("cpu_gate_identity_mismatch")
    if not commands:
        blockers.add("cpu_gate_command_mismatch")
    if not passed:
        blockers.add("cpu_gate_failed")


def _assess_magpie_config_resolution(
    static: Mapping[str, Any],
    evidence: ReleaseEvidence,
    blockers: set[str],
) -> None:
    value = evidence.magpie_config_resolution
    if value is None:
        blockers.add("magpie_config_resolution_evidence_missing")
        return
    magpie = static["magpie"]
    expected = tuple(
        (
            str(item["path"]),
            str(item["config_sha256"]),
            str(item["status"]),
            str(item["run_mode"]),
            str(item["lifecycle"]),
        )
        for item in magpie["config_resolution_scope"]
    )
    observed = tuple(
        (
            item.path,
            item.config_sha256,
            item.status,
            item.run_mode,
            item.lifecycle,
        )
        for item in value.entries
    )
    identity_matches = (
        value.magpie_commit == magpie["commit"]
        and value.corpus_manifest_sha256 == magpie["corpus_manifest_sha256"]
        and observed == expected
        and len(value.entries) == magpie["config_count"]
        and len(value.e2e_v2_entries()) == magpie["e2e_v2_config_count"]
        and len(value.e2e_v2_rejection_entries())
        == magpie["e2e_v2_rejection_count"]
    )
    if not identity_matches:
        blockers.add("magpie_config_resolution_identity_mismatch")
    if any(item.status != "config_compatible" for item in value.entries):
        blockers.add("magpie_capability_upgrade_required")


def _assess_cli(
    static: Mapping[str, Any],
    evidence: ReleaseEvidence,
    blockers: set[str],
) -> None:
    value = evidence.cli_identity
    if value is None:
        blockers.add("cli_identity_missing")
        return
    project = static["project"]
    installed = static["local_cli"]
    valid = (
        value.apex_tree == static["apex_checkout"]["tree"]
        and value.project_version == project["version"]
        and value.entrypoint == project["entrypoint"]
        and value.import_module == "apex"
        and value.import_file_sha256 == project["import_file_sha256"]
        and installed["status"] == "observed"
        and value.executable_sha256 == installed["executable_sha256"]
    )
    if not valid:
        blockers.add("cli_identity_mismatch")


def _assess_images(
    static: Mapping[str, Any],
    evidence: ReleaseEvidence,
    blockers: set[str],
) -> None:
    images = {item.name: item for item in evidence.images}
    tree = static["apex_checkout"]["tree"]
    for name in REQUIRED_IMAGES:
        item = images.get(name)
        if item is None:
            blockers.add(f"image_identity_missing:{name}")
        elif item.apex_tree != tree:
            blockers.add(f"image_identity_stale:{name}")
    parent = images.get("lm-eval-parent")
    if parent and (
        parent.image_id != static["lm_eval"]["base_image_id"]
        or parent.repo_digest != static["lm_eval"]["base_image_repo_digest"]
    ):
        blockers.add("lm_eval_parent_image_mismatch")


def _assess_showcases(
    static: Mapping[str, Any],
    evidence: ReleaseEvidence,
    blockers: set[str],
) -> None:
    values = {item.showcase_id: item for item in evidence.showcases}
    tree = static["apex_checkout"]["tree"]
    for name in REQUIRED_SHOWCASES:
        item = values.get(name)
        if item is None:
            blockers.add(f"showcase_missing:{name}")
            continue
        valid = (
            item.apex_tree == tree and item.status == "published"
            and item.bundle_verified and item.reward_replayed
            and item.reproduction_verified
        )
        if not valid:
            blockers.add(f"showcase_unqualified:{name}")


def _assess_qualifications(
    static: Mapping[str, Any],
    evidence: ReleaseEvidence,
    authorities: tuple[QualificationAuthorityReceipt, ...],
    blockers: set[str],
) -> None:
    values = {item.qualification_id: item for item in evidence.qualifications}
    verified = {item.qualification_id: item for item in authorities}
    tree = static["apex_checkout"]["tree"]
    for name in REQUIRED_QUALIFICATIONS:
        item = values.get(name)
        if item is None:
            blockers.add(f"qualification_missing:{name}")
            continue
        if item.apex_tree != tree or item.status != "qualified" or item.coverage_count < 1:
            blockers.add(f"qualification_unverified:{name}")
        authority = verified.get(name)
        if authority is None or authority.evidence_receipt_sha256 != item.receipt_sha256:
            blockers.add(f"qualification_authority_missing:{name}")
    magpie = values.get("magpie-corpus-live")
    resolved = evidence.magpie_config_resolution
    if magpie and (
        resolved is None
        or magpie.subject_sha256 != resolved.resolved_manifest_sha256
        or magpie.formal_delivery_count < 1
    ):
        blockers.add("magpie_live_coverage_incomplete")
    if magpie and resolved is not None:
        docker_entries = resolved.e2e_v2_entries()
        if (
            magpie.coverage_count != len(docker_entries)
            or magpie.details["e2e_v2_scope"] != "docker_one_shot"
            or magpie.details["e2e_v2_config_count"] != len(docker_entries)
            or magpie.details["e2e_v2_plan_manifest_sha256"]
            != resolved.e2e_v2_manifest_sha256()
            or magpie.details["e2e_v2_rejection_count"]
            != len(resolved.e2e_v2_rejection_entries())
            or magpie.details["e2e_v2_rejection_manifest_sha256"]
            != resolved.e2e_v2_rejection_manifest_sha256()
            or not _representatives_bind_selected_rows(
                static["magpie"], magpie, docker_entries
            )
        ):
            blockers.add("magpie_e2e_v2_scope_incomplete")


def _representatives_bind_selected_rows(
    static: Mapping[str, Any],
    qualification,
    selected: tuple[MagpieConfigResolutionEntryEvidence, ...],
) -> bool:
    frameworks = {
        item["path"]: item["framework"]
        for item in static["config_resolution_scope"]
    }
    expected = {
        (
            item.path,
            item.config_sha256,
            item.plan_sha256,
            item.capability_receipt_sha256,
            frameworks[item.path],
            item.run_mode,
            item.lifecycle,
        )
        for item in selected
    }
    return all(
        (
            item["config_path"],
            item["config_sha256"],
            item["plan_sha256"],
            item["capability_receipt_sha256"],
            item["framework"],
            item["run_mode"],
            item["lifecycle"],
        )
        in expected
        for item in qualification.details["formal_delivery_representatives"]
    )


def _validate_payload(value: Mapping[str, Any]) -> None:
    expected = {
        "schema", "baseline_status", "status", "static", "evidence",
        "qualification_authorities", "baseline_blockers", "blockers",
    }
    if set(value) != expected:
        raise IntegrityError("Release receipt payload fields differ", "release_receipt_tampered")
    blockers = value.get("blockers")
    baseline = value.get("baseline_blockers")
    authorities = value.get("qualification_authorities")
    if (
        value.get("schema") != SCHEMA
        or value.get("status") not in {"ready", "blocked"}
        or value.get("baseline_status") not in {"ready", "blocked"}
        or not isinstance(blockers, list)
        or not isinstance(baseline, list)
        or not isinstance(authorities, list)
        or blockers != sorted(set(blockers))
        or baseline != sorted(set(baseline))
        or not set(baseline).issubset(blockers)
        or (value.get("status") == "ready") != (not blockers)
        or (value.get("baseline_status") == "ready") != (not baseline)
    ):
        raise IntegrityError("Release receipt status/blockers differ", "release_receipt_tampered")
    try:
        parsed = tuple(
            QualificationAuthorityReceipt.from_dict(item) for item in authorities
        )
    except ContractError as error:
        raise IntegrityError(
            "Release qualification authority receipt is invalid",
            "release_receipt_tampered",
        ) from error
    names = tuple(item.qualification_id for item in parsed)
    if names != tuple(sorted(set(names))):
        raise IntegrityError(
            "Release qualification authorities differ", "release_receipt_tampered"
        )


def _qualification_authorities(
    evidence: ReleaseEvidence,
    authority: QualificationAuthorityPort | None,
) -> tuple[QualificationAuthorityReceipt, ...]:
    if authority is None:
        return ()
    results: list[QualificationAuthorityReceipt] = []
    for item in evidence.qualifications:
        if item.status != "qualified":
            continue
        try:
            observed = authority.verify(item.to_dict())
        except ContractError as error:
            if error.reason_code in {
                "qualification_artifacts_unavailable",
                "qualification_artifacts_invalid",
            }:
                continue
            raise
        receipt = QualificationAuthorityReceipt.from_dict(observed.to_dict())
        if (
            receipt.qualification_id != item.qualification_id
            or receipt.evidence_receipt_sha256 != item.receipt_sha256
        ):
            raise ContractError(
                "Qualification authority verified a different evidence record",
                "qualification_authority_binding_mismatch",
            )
        results.append(receipt)
    return tuple(sorted(results, key=lambda item: item.qualification_id))


__all__ = [
    "APEX_REPOSITORY", "E2E_SHOWCASE", "KERNEL_SHOWCASES",
    "REQUIRED_IMAGES", "REQUIRED_QUALIFICATIONS", "REQUIRED_SHOWCASES",
    "ReleaseCandidateReceipt", "freeze_campaign_baseline", "freeze_release_candidate",
    "inspect_release_candidate", "verify_release_candidate_receipt",
]
