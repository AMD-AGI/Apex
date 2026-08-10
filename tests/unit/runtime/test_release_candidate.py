"""CPU-only release readiness and tamper-resistance contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)
from apex.ports import build_qualification_authority_receipt
from apex.runtime import (
    BaselineAuditEvidence,
    CliIdentityEvidence,
    CpuGateEvidence,
    DependencyVerificationEvidence,
    ImageIdentityEvidence,
    MagpieConfigResolutionEntryEvidence,
    MagpieConfigResolutionEvidence,
    ReleaseEvidence,
    ShowcaseEvidence,
    VerifiedComponentEvidence,
    build_magpie_config_resolution_evidence,
    build_qualification_evidence,
    build_showcase_evidence,
    freeze_campaign_baseline,
    freeze_release_candidate,
    inspect_release_candidate,
    verify_release_candidate_receipt,
)
from apex.runtime.release_candidate import (
    REQUIRED_IMAGES,
    REQUIRED_QUALIFICATIONS,
    REQUIRED_SHOWCASES,
)


ROOT = Path(__file__).resolve().parents[3]
_SHA = "a" * 64


class _FixtureQualificationAuthority:
    def verify(self, evidence):
        return build_qualification_authority_receipt(
            qualification_id=str(evidence["qualification_id"]),
            evidence_receipt_sha256=str(evidence["receipt_sha256"]),
            artifact_manifest_sha256=sha256_json({"artifacts": evidence}),
            verifier_identity_sha256=_SHA,
            authority_id="fixture-offline-artifact-verifier",
        )


_AUTHORITY = _FixtureQualificationAuthority()


def _checked_in_static() -> dict:
    return copy.deepcopy(inspect_release_candidate(ROOT).document["static"])


def _ready_static() -> dict:
    value = _checked_in_static()
    value["apex_checkout"]["clean"] = True
    value["apex_checkout"]["dirty_path_count"] = 0
    value["apex_checkout"]["repository"] = "github.com/amd-agi/apex"
    return value


def _baseline(name: str, static: dict) -> BaselineAuditEvidence:
    source = static["apex_checkout"] if name == "apex" else static["magpie"]
    return BaselineAuditEvidence(
        component=name,
        repository=source["repository"],
        branch="origin/main",
        commit=source["commit"],
        tree=source.get("tree", source.get("repository_tree")),
        remote_tip=source["commit"],
        fetched=True,
        ancestry_reviewed=True,
        clean=True,
    )


def _dependency_evidence(static: dict) -> DependencyVerificationEvidence:
    components = []
    for item in (*static["dependencies"], *static["sources"]):
        tree = item.get("tree", "b" * 40)
        if item["name"] == "inferencex":
            tree = static["lm_eval"]["inferencex_tree"]
        components.append(VerifiedComponentEvidence(
            item["name"], item["repository"], item["commit"], tree, True
        ))
    locks = static["locks"]
    return DependencyVerificationEvidence(
        apex_tree=static["apex_checkout"]["tree"],
        dependencies_lock_sha256=locks["dependencies"],
        e2e_source_lock_sha256=locks["e2e_sources"],
        lm_eval_runtime_lock_sha256=locks["lm_eval_runtime"],
        evaluator_policy_lock_sha256=locks["evaluator_policy"],
        agent_templates_lock_sha256=locks["agent_templates"],
        lm_eval_runtime_sha256=static["lm_eval"]["runtime_sha256"],
        all_imports_exact=True,
        components=tuple(sorted(components, key=lambda item: item.name)),
    )


def _cpu_gate(static: dict) -> CpuGateEvidence:
    locks = static["locks"]
    magpie = static["magpie"]
    return CpuGateEvidence(
        apex_tree=static["apex_checkout"]["tree"],
        dependencies_lock_sha256=locks["dependencies"],
        e2e_source_lock_sha256=locks["e2e_sources"],
        corpus_manifest_sha256=magpie["corpus_manifest_sha256"],
        compatibility_ledger_sha256=magpie["compatibility_ledger_sha256"],
        pytest_argv=(
            "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
            "tests/unit", "tests/contract", "tests/integration", "tests/architecture",
            "tests/test_bootstrap_dependencies.py",
        ),
        pytest_exit_code=0,
        passed_count=1042,
        failed_count=0,
        compileall_argv=("python", "-m", "compileall", "-q", "src/apex", "main.py", "scripts"),
        compileall_exit_code=0,
        forbidden_scan_argv=("rg", "-n", r"shell=True|os\.system", "src/apex"),
        forbidden_scan_exit_code=1,
        forbidden_scan_clean=True,
    )


def _resolver_evidence(static: dict):
    magpie = static["magpie"]
    entries = tuple(
        MagpieConfigResolutionEntryEvidence(
            item["path"],
            item["sha256"],
            f"{index + 1:064x}",
            f"{index + 101:064x}",
            "config_compatible",
            "ray" if index < 3 else ("local" if index < 5 else "docker"),
        )
        for index, item in enumerate(magpie["configs"])
    )
    return build_magpie_config_resolution_evidence(
        magpie_commit=magpie["commit"],
        corpus_manifest_sha256=magpie["corpus_manifest_sha256"],
        plan_schema="apex.magpie-main-resolved-plan/v1",
        capability_schema="apex.magpie-main-capability-receipt/v1",
        result_schema="apex.magpie-main-result-contract/v1",
        entries=entries,
    )


def _ready_evidence(static: dict) -> ReleaseEvidence:
    tree = static["apex_checkout"]["tree"]
    resolver = _resolver_evidence(static)
    images = []
    for index, name in enumerate(sorted(REQUIRED_IMAGES), start=1):
        image_id = f"sha256:{index:064x}"
        repo_digest = f"example.invalid/{name}@sha256:{index:064x}"
        if name == "lm-eval-parent":
            image_id = static["lm_eval"]["base_image_id"]
            repo_digest = static["lm_eval"]["base_image_repo_digest"]
        images.append(ImageIdentityEvidence(name, tree, image_id, repo_digest, _SHA))
    showcases = tuple(
        _showcase_evidence(name, tree) for name in sorted(REQUIRED_SHOWCASES)
    )
    qualifications = []
    for name in sorted(REQUIRED_QUALIFICATIONS):
        subject = _SHA
        coverage, formal = 2, 1
        if name == "magpie-corpus-live":
            subject = resolver.resolved_manifest_sha256
            coverage, formal = static["magpie"]["config_count"], 3
        elif name == "crash-resume-recovery":
            coverage, formal = 32, 2
        elif name == "knowledge-ablation":
            coverage, formal = 6, 0
        elif name == "aka-v14-matched":
            coverage, formal = 20, 1
        qualifications.append(build_qualification_evidence(
            qualification_id=name,
            apex_tree=tree,
            subject_sha256=subject,
            status="qualified",
            coverage_count=coverage,
            formal_delivery_count=formal,
            details=_qualification_details(name, subject, resolver),
        ))
    return ReleaseEvidence(
        apex_baseline=_baseline("apex", static),
        magpie_baseline=_baseline("magpie", static),
        dependencies=_dependency_evidence(static),
        magpie_config_resolution=resolver,
        cpu_gate=_cpu_gate(static),
        cli_identity=CliIdentityEvidence(
            tree,
            static["project"]["version"],
            "apex.cli:main",
            "apex",
            static["local_cli"]["executable_sha256"],
            static["project"]["import_file_sha256"],
        ),
        images=tuple(images),
        showcases=showcases,
        qualifications=tuple(qualifications),
    )


def _showcase_evidence(name: str, tree: str) -> ShowcaseEvidence:
    receipt = {
        "schema": "apex.showcase-verification/v2",
        "showcase_id": name,
        "status": "published",
        "file_count": 12,
        "checksums_sha256": _SHA,
        "event_count": 20,
        "artifact_count": 8,
        "reward_replayed": True,
        "bundle_verified": True,
        "reproduction_verified": True,
        "episode_sha256": _SHA,
        "artifact_manifest_sha256": _SHA,
        "reward_sha256": _SHA,
        "result_sha256": _SHA,
        "reproduction_sha256": _SHA,
    }
    receipt["verification_receipt_sha256"] = sha256_json(receipt)
    return build_showcase_evidence(apex_tree=tree, verifier_receipt=receipt)


def _qualification_details(
    name: str,
    subject: str,
    resolver: MagpieConfigResolutionEvidence,
) -> dict:
    if name.startswith("backend-"):
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
    if name == "crash-resume-recovery":
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
    if name == "knowledge-ablation":
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
    if name == "aka-v14-matched":
        return {
            "schema": "apex.aka-matched-qualification/v1",
            "qualification_manifest_sha256": subject,
            "repository": "https://github.com/AMD-AGI/AgentKernelArena",
            "commit": "b" * 40,
            "tree": "c" * 40,
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
        "ray_config_count": sum(
            item.run_mode == "ray" for item in resolver.entries
        ),
        "ray_plan_manifest_sha256": resolver.run_mode_manifest_sha256("ray"),
        "ray_shared_storage_receipts_sha256": _SHA,
        "ray_runtime_receipts_sha256": _SHA,
        "ray_worker_reports_sha256": _SHA,
        "ray_driver_replay_receipts_sha256": _SHA,
        "ray_quality_sync_only": True,
        "ray_shared_runtime_verified": True,
        "ray_driver_evidence_replayed": True,
    }


def _replace_resolver(
    evidence: ReleaseEvidence,
    resolver: MagpieConfigResolutionEvidence,
) -> ReleaseEvidence:
    return ReleaseEvidence(
        apex_baseline=evidence.apex_baseline,
        magpie_baseline=evidence.magpie_baseline,
        dependencies=evidence.dependencies,
        magpie_config_resolution=resolver,
        cpu_gate=evidence.cpu_gate,
        cli_identity=evidence.cli_identity,
        images=evidence.images,
        showcases=evidence.showcases,
        qualifications=evidence.qualifications,
    )


def test_checked_in_ledger_requires_config_resolution_evidence() -> None:
    first = inspect_release_candidate(ROOT)
    second = inspect_release_candidate(ROOT)

    assert first.to_dict() == second.to_dict()
    assert first.status == "blocked"
    assert first.baseline_status == "blocked"
    magpie = first.document["static"]["magpie"]
    assert (magpie["config_count"], magpie["config_compatible_count"]) == (27, 27)
    assert magpie["compatibility_authority"] == (
        "legacy_apex_projection_not_release_evidence"
    )
    assert (magpie["workflow_qualified_count"], magpie["formal_delivery_qualified_count"]) == (0, 0)
    assert "apex_source_dirty" in first.blockers
    assert "magpie_config_resolution_evidence_missing" in first.blockers
    assert "qualification_missing:magpie-corpus-live" in first.blockers
    assert "showcase_missing:e2e-qwen3-next-80b-fp8" in first.blockers
    assert all(entry["blockers"] for entry in first.document["static"]["templates"]["entries"])


def test_fully_bound_evidence_can_freeze_and_round_trip(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    receipt = inspect_release_candidate(
        ROOT,
        _ready_evidence(static),
        qualification_authority=_AUTHORITY,
    )

    assert receipt.status == "ready"
    assert receipt.baseline_status == "ready"
    assert receipt.blockers == ()
    assert verify_release_candidate_receipt(
        receipt.to_dict(), apex_root=ROOT, qualification_authority=_AUTHORITY
    ) == receipt
    assert freeze_release_candidate(
        receipt.to_dict(), apex_root=ROOT, qualification_authority=_AUTHORITY
    ) == receipt
    assert freeze_campaign_baseline(
        receipt.to_dict(), apex_root=ROOT, qualification_authority=_AUTHORITY
    ) == receipt


def test_self_digested_qualified_json_is_blocked_without_authority(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )

    receipt = inspect_release_candidate(ROOT, _ready_evidence(static))

    assert receipt.status == "blocked"
    assert all(
        f"qualification_authority_missing:{name}" in receipt.blockers
        for name in REQUIRED_QUALIFICATIONS
    )
    assert receipt.document["qualification_authorities"] == []


def test_typed_artifact_unavailability_stays_blocked_instead_of_crashing(
    monkeypatch,
) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )

    class UnavailableAuthority:
        def verify(self, evidence):
            raise ContractError(
                "No kind-specific verifier is installed",
                "qualification_artifacts_unavailable",
            )

    receipt = inspect_release_candidate(
        ROOT,
        _ready_evidence(static),
        qualification_authority=UnavailableAuthority(),
    )

    assert receipt.status == "blocked"
    assert receipt.document["qualification_authorities"] == []
    assert all(
        f"qualification_authority_missing:{name}" in receipt.blockers
        for name in REQUIRED_QUALIFICATIONS
    )


def test_authority_must_bind_the_exact_qualification_receipt(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )

    class WrongAuthority:
        def verify(self, evidence):
            return build_qualification_authority_receipt(
                qualification_id=str(evidence["qualification_id"]),
                evidence_receipt_sha256="f" * 64,
                artifact_manifest_sha256=_SHA,
                verifier_identity_sha256=_SHA,
                authority_id="wrong-fixture",
            )

    with pytest.raises(ContractError, match="verified a different evidence"):
        inspect_release_candidate(
            ROOT,
            _ready_evidence(static),
            qualification_authority=WrongAuthority(),
        )


def test_status_edit_fails_even_if_attacker_recomputes_self_digest(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    value = inspect_release_candidate(
        ROOT,
        _ready_evidence(static),
        qualification_authority=_AUTHORITY,
    ).to_dict()
    value["status"] = "blocked"
    value["blockers"] = ["invented"]
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))

    with pytest.raises(IntegrityError, match="no longer matches"):
        verify_release_candidate_receipt(
            value, apex_root=ROOT, qualification_authority=_AUTHORITY
        )


def test_nested_qualification_mutation_is_revalidated_before_assessment(
    monkeypatch,
) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    evidence = _ready_evidence(static)
    recovery = next(
        item
        for item in evidence.qualifications
        if item.qualification_id == "crash-resume-recovery"
    )
    recovery.details["no_duplicate_reward"] = False

    with pytest.raises(ContractError, match="truth claim is incomplete"):
        inspect_release_candidate(ROOT, evidence)


@pytest.mark.parametrize("mutation", ["missing", "config_digest"])
def test_resolver_corpus_identity_drift_blocks_baseline(monkeypatch, mutation) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    evidence = _ready_evidence(static)
    original = evidence.magpie_config_resolution
    assert original is not None
    entries = list(original.entries)
    if mutation == "missing":
        entries.pop()
    else:
        first = entries[0]
        entries[0] = MagpieConfigResolutionEntryEvidence(
            first.path,
            "f" * 64,
            first.plan_sha256,
            first.capability_receipt_sha256,
            first.status,
            first.run_mode,
        )
    resolver = build_magpie_config_resolution_evidence(
        magpie_commit=original.magpie_commit,
        corpus_manifest_sha256=original.corpus_manifest_sha256,
        plan_schema=original.plan_schema,
        capability_schema=original.capability_schema,
        result_schema=original.result_schema,
        entries=entries,
    )

    receipt = inspect_release_candidate(
        ROOT,
        _replace_resolver(evidence, resolver),
    )

    assert "magpie_config_resolution_identity_mismatch" in receipt.baseline_blockers


def test_resolver_upgrade_status_blocks_baseline(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    evidence = _ready_evidence(static)
    original = evidence.magpie_config_resolution
    assert original is not None
    entries = list(original.entries)
    first = entries[0]
    entries[0] = MagpieConfigResolutionEntryEvidence(
        first.path,
        first.config_sha256,
        first.plan_sha256,
        first.capability_receipt_sha256,
        "capability_upgrade_required",
        first.run_mode,
    )
    resolver = build_magpie_config_resolution_evidence(
        magpie_commit=original.magpie_commit,
        corpus_manifest_sha256=original.corpus_manifest_sha256,
        plan_schema=original.plan_schema,
        capability_schema=original.capability_schema,
        result_schema=original.result_schema,
        entries=entries,
    )

    receipt = inspect_release_candidate(ROOT, _replace_resolver(evidence, resolver))

    assert "magpie_capability_upgrade_required" in receipt.baseline_blockers


def test_resolver_manifest_rejects_reordering_and_digest_tamper() -> None:
    value = _resolver_evidence(_ready_static()).to_dict()
    value["entries"] = list(reversed(value["entries"]))
    with pytest.raises(ContractError, match="not unique/sorted"):
        type(_resolver_evidence(_ready_static())).from_dict(value)

    value = _resolver_evidence(_ready_static()).to_dict()
    value["resolved_manifest_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="manifest digest differs"):
        type(_resolver_evidence(_ready_static())).from_dict(value)


def test_stale_cpu_gate_and_incomplete_magpie_live_coverage_block(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    evidence = _ready_evidence(static)
    stale_gate = CpuGateEvidence.from_dict({
        **evidence.cpu_gate.to_dict(),
        "apex_tree": "c" * 40,
    })
    qualifications = tuple(
        build_qualification_evidence(
            qualification_id=item.qualification_id,
            apex_tree=item.apex_tree,
            subject_sha256=item.subject_sha256,
            status=item.status,
            coverage_count=(
                item.coverage_count - 1
                if item.qualification_id == "magpie-corpus-live"
                else item.coverage_count
            ),
            formal_delivery_count=item.formal_delivery_count,
            details=item.details,
        )
        for item in evidence.qualifications
    )
    blocked = inspect_release_candidate(
        ROOT,
        ReleaseEvidence(
            apex_baseline=evidence.apex_baseline,
            magpie_baseline=evidence.magpie_baseline,
            dependencies=evidence.dependencies,
            magpie_config_resolution=evidence.magpie_config_resolution,
            cpu_gate=stale_gate,
            cli_identity=evidence.cli_identity,
            images=evidence.images,
            showcases=evidence.showcases,
            qualifications=qualifications,
        ),
    )

    assert "cpu_gate_identity_mismatch" in blocked.blockers
    assert "cpu_gate_identity_mismatch" in blocked.baseline_blockers
    assert "magpie_live_coverage_incomplete" in blocked.blockers
    with pytest.raises(ContractError, match="Release candidate is blocked"):
        freeze_release_candidate(blocked.to_dict(), apex_root=ROOT)


def test_magpie_live_ray_evidence_must_bind_exact_resolver_slice(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    evidence = _ready_evidence(static)
    original = next(
        item for item in evidence.qualifications
        if item.qualification_id == "magpie-corpus-live"
    )
    details = dict(original.details)
    details["ray_plan_manifest_sha256"] = "f" * 64
    forged = build_qualification_evidence(
        qualification_id=original.qualification_id,
        apex_tree=original.apex_tree,
        subject_sha256=original.subject_sha256,
        status=original.status,
        coverage_count=original.coverage_count,
        formal_delivery_count=original.formal_delivery_count,
        details=details,
    )
    qualifications = tuple(
        forged if item.qualification_id == original.qualification_id else item
        for item in evidence.qualifications
    )

    blocked = inspect_release_candidate(
        ROOT,
        replace(evidence, qualifications=qualifications),
    )

    assert "magpie_ray_evidence_incomplete" in blocked.blockers


def test_campaign_baseline_does_not_wait_for_future_showcases(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    full = _ready_evidence(static)
    baseline_only = ReleaseEvidence(
        apex_baseline=full.apex_baseline,
        magpie_baseline=full.magpie_baseline,
        dependencies=full.dependencies,
        magpie_config_resolution=full.magpie_config_resolution,
        cpu_gate=full.cpu_gate,
        cli_identity=full.cli_identity,
    )
    receipt = inspect_release_candidate(ROOT, baseline_only)

    assert receipt.baseline_status == "ready"
    assert receipt.baseline_blockers == ()
    assert receipt.status == "blocked"
    assert "showcase_missing:e2e-qwen3-next-80b-fp8" in receipt.blockers
    assert freeze_campaign_baseline(receipt.to_dict(), apex_root=ROOT) == receipt
    with pytest.raises(ContractError, match="Release candidate is blocked"):
        freeze_release_candidate(receipt.to_dict(), apex_root=ROOT)


def test_dependency_receipt_tamper_and_nonportable_argv_fail_closed() -> None:
    value = inspect_release_candidate(ROOT).to_dict()
    value["receipt_sha256"] = "0" * 64
    with pytest.raises(IntegrityError, match="digest differs"):
        verify_release_candidate_receipt(value, apex_root=ROOT)

    with pytest.raises(ContractError, match="not portable"):
        CpuGateEvidence.from_dict({
            **_cpu_gate(_ready_static()).to_dict(),
            "pytest_argv": ["/tmp/private/pytest", "-q"],
        })


def test_cli_byte_identity_and_scan_exit_code_are_rechecked(monkeypatch) -> None:
    static = _ready_static()
    monkeypatch.setattr(
        "apex.runtime.release_candidate.collect_release_static_identity",
        lambda *args, **kwargs: copy.deepcopy(static),
    )
    evidence = _ready_evidence(static)
    forged_cli = CliIdentityEvidence.from_dict({
        **evidence.cli_identity.to_dict(),
        "executable_sha256": _SHA,
    })
    bad_scan = CpuGateEvidence.from_dict({
        **evidence.cpu_gate.to_dict(),
        "forbidden_scan_exit_code": 0,
    })
    blocked = inspect_release_candidate(
        ROOT,
        ReleaseEvidence(
            apex_baseline=evidence.apex_baseline,
            magpie_baseline=evidence.magpie_baseline,
            dependencies=evidence.dependencies,
            magpie_config_resolution=evidence.magpie_config_resolution,
            cpu_gate=bad_scan,
            cli_identity=forged_cli,
            images=evidence.images,
            showcases=evidence.showcases,
            qualifications=evidence.qualifications,
        ),
    )

    assert "cli_identity_mismatch" in blocked.blockers
    assert "cpu_gate_failed" in blocked.blockers
