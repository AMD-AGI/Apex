"""Adversarial replay tests for backend live qualification artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from apex.core import (
    AgentBackendName,
    ContractError,
    canonical_json_bytes,
    sha256_json,
)
from apex.evaluation import (
    GateVerdict,
    KernelMeasurementExecutionReceipt,
    MeasurementPolicy,
    grade_kernel,
    kernel_reward_policy_source,
    kernel_terminal_policy_source,
    load_kernel_measurement_report,
    selected_kernel_terminal_grade,
)
from apex.execution import agent_transcript_document
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentExecutionAuthorityReceipt,
    AgentInvocationReceipt,
    AgentProcessContainmentReceipt,
    AgentResult,
    AgentSemanticEvent,
    AgentTerminationKind,
    AgentTranscriptEvent,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)
from apex.rl import (
    CandidateEpisode,
    EpisodeArtifact,
    EpisodeEvent,
    EpisodeGraph,
    EvidenceClass,
    ParentEpisode,
    backend_live_qualification_verifiers,
)
from apex.rl.episode_semantics import semantic_role
from apex.rl.models import episode_id
from apex.runtime import (
    EvaluatorQualificationArtifactAuthority,
    FormalResultsRootValidator,
    build_qualification_evidence,
)
from apex.runtime.qualification_artifacts import INDEX_NAME, INDEX_SCHEMA
from apex.storage import ArtifactReceipt, ArtifactStore


_TREE = "b" * 40
_RUN = "run-backend-live"
_ATTEMPT = "attempt-live"
_TASK = "task-live"
_SOURCE = "c" * 64
_HARNESS = "d" * 64
_METHOD = "e" * 64


@dataclass(frozen=True)
class _Campaign:
    root: Path
    authority: EvaluatorQualificationArtifactAuthority
    raw_measurement: ArtifactReceipt


def _campaign(
    tmp_path: Path,
    *,
    manifest_backend: str = "codex",
    manifest_tree: str = _TREE,
) -> _Campaign:
    source = tmp_path / "Apex"
    source.mkdir()
    root = tmp_path / "formal-results"
    root.mkdir()
    store = ArtifactStore(root / "artifacts")
    contract = _contract(store)
    coding = _coding_receipt(store, contract)
    evidence = _kernel_evidence(store, contract)
    graph = _graph(store, coding, contract, evidence)
    graph_receipt = store.put_bytes(
        graph.canonical_bytes, media_type="application/json"
    )
    manifest = {
        "schema": "apex.backend-live-qualification-artifacts/v1",
        "qualification_id": "backend-codex-gfx950",
        "backend": manifest_backend,
        "gpu_arch": "gfx950",
        "apex_tree": manifest_tree,
        "coding_receipt": coding.to_dict(),
        "episode_receipt": graph_receipt.to_dict(),
        "kernel_terminal_result_receipt": evidence["result"].to_dict(),
    }
    manifest_receipt = store.put_bytes(
        canonical_json_bytes(manifest), media_type="application/json"
    )
    _write_index(root, manifest_tree, manifest_receipt)
    authority = EvaluatorQualificationArtifactAuthority(
        artifact_root=root,
        results_policy=FormalResultsRootValidator((source,)),
        verifiers=backend_live_qualification_verifiers(),
    )
    return _Campaign(root, authority, evidence["raw"])


def _contract(store: ArtifactStore) -> ArtifactReceipt:
    draft = {
        "schema": "apex.evaluation-contract-draft/v1",
        "gpu_arch": "gfx950",
        "repository": {
            "status": "resolved",
            "tree": _TREE,
            "dirty_paths": [],
        },
    }
    authority = {
        "schema": "apex.evaluation-authority-receipt/v1",
        "authority_id": "fixture-reviewed-v1",
        "authority_kind": "reviewed_template",
        "issuer": "fixture",
        "policy_sha256": "1" * 64,
        "template_sha256": "2" * 64,
        "draft_digest": sha256_json(draft),
    }
    document = {
        "schema": "apex.evaluation-contract-receipt/v1",
        "status": "verified",
        "unverified_reason": None,
        "draft": draft,
        "draft_digest": sha256_json(draft),
        "authority": authority,
    }
    return store.put_bytes(canonical_json_bytes(document), media_type="application/json")


def _coding_receipt(
    store: ArtifactStore, contract: ArtifactReceipt
) -> ArtifactReceipt:
    authority = AgentExecutionAuthorityReceipt(
        authority_id="fixture-agent-authority-v1",
        authority_kind="evaluation_contract",
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        backend="codex",
        workspace="/formal/workspace",
        allowed_files=("kernel.py",),
        requested_environment_keys=(),
        parent_receipt_sha256=contract.digest,
        source_anchor_sha256="3" * 64,
    )
    invocation = AgentInvocationReceipt(
        cli_name="codex",
        cli_version="codex-cli fixture",
        executable_path="/usr/bin/codex",
        resolved_executable_path="/usr/bin/codex",
        entrypoint_sha256="4" * 64,
        argv=("/usr/bin/codex", "exec"),
        workspace="/formal/workspace",
        prompt_transport="stdin",
        execution_authority=authority,
        credential_environment_key="OPENAI_API_KEY",
        requested_allowed_files=("kernel.py",),
        allowed_files_enforced_by_cli=False,
        max_turns=4,
        turn_policy=STRUCTURED_TURN_CHECKPOINT_POLICY,
        process_containment_policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        isolation=(("approval_policy", "never"),),
    )
    containment = _containment()
    result = AgentResult(
        backend=AgentBackendName.CODEX,
        model="fixture-model",
        exit_code=0,
        timed_out=False,
        events=(AgentTranscriptEvent("result", metadata={"type": "result"}),),
        stdout="{}\n",
        stderr="",
        duration_seconds=1.0,
        semantic_events=(
            AgentSemanticEvent(0, 0, "result", "agent_message", "assistant", "done"),
        ),
        invocation=invocation,
        termination_kind=AgentTerminationKind.COMPLETED,
        capture_status=AgentCaptureStatus.COMPLETE,
        observed_turns=1,
        process_containment=containment,
    )
    return store.put_bytes(
        canonical_json_bytes(agent_transcript_document(result)),
        media_type="application/json",
    )


def _containment() -> AgentProcessContainmentReceipt:
    return AgentProcessContainmentReceipt(
        policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="5" * 64,
        namespace_init_host_pid=100,
        namespace_init_starttime=200,
        namespace_init_inner_pid=1,
        pid_namespace_inode=300,
        mount_namespace_inode=301,
        ipc_namespace_inode=302,
        user_namespace_inode=303,
        private_procfs_verified=True,
        pidfd_opened=True,
        termination_reason="natural_exit",
        teardown_mode="natural_exit",
        pidfd_sigkill_sent=False,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=True,
        terminal_status_absent_after_sigkill=False,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
    )


def _kernel_evidence(
    store: ArtifactStore, contract: ArtifactReceipt
) -> dict[str, ArtifactReceipt]:
    policy = MeasurementPolicy()
    raw = store.put_bytes(
        canonical_json_bytes(_measurement_report(policy)),
        media_type="application/json",
    )
    artifact = load_kernel_measurement_report(
        store.root / raw.relative_path, measurement_policy=policy
    )
    grade = grade_kernel(
        GateVerdict(True, True, True, True),
        artifact.cases,
        measurement_policy=policy,
    )
    assert grade.promotion_eligible
    measured_grade = store.put_bytes(
        canonical_json_bytes(grade.to_dict()), media_type="application/json"
    )
    harness = store.put_bytes(
        canonical_json_bytes({"harness_sha256": _HARNESS}),
        media_type="application/json",
    )
    attempt_policy = store.put_bytes(
        canonical_json_bytes(kernel_reward_policy_source(policy)),
        media_type="application/json",
    )
    execution = KernelMeasurementExecutionReceipt(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        writer_id="fixture-evaluator-v1",
        candidate_source_sha256=_SOURCE,
        harness_sha256=_HARNESS,
        measurement_method_sha256=_METHOD,
        measurement_policy_sha256=sha256_json(policy.to_dict()),
        report_sha256=raw.digest,
        report_size=raw.size,
        phase_started_monotonic_ns=1,
        adapter_returned_monotonic_ns=2,
        output_observed_monotonic_ns=3,
        phase_completed_monotonic_ns=4,
    )
    execution_receipt = store.put_bytes(
        execution.canonical_bytes, media_type="application/json"
    )
    compile_receipt = _command_receipt(store, "compile")
    correctness_receipt = _command_receipt(store, "correctness")
    terminal = selected_kernel_terminal_grade(_ATTEMPT, grade)
    vector = store.put_bytes(
        canonical_json_bytes(terminal.to_dict()), media_type="application/json"
    )
    terminal_policy = store.put_bytes(
        canonical_json_bytes(kernel_terminal_policy_source()),
        media_type="application/json",
    )
    source = store.put_bytes(
        canonical_json_bytes({
            "schema": "apex.kernel-terminal-reward-source/v1",
            "run_id": _RUN,
            "evaluation_contract_receipt_digest": contract.digest,
            "source_attempt_id": _ATTEMPT,
            "implementation": "candidate",
            "candidate_source_sha256": _SOURCE,
            "measurement_candidate_source_sha256": _SOURCE,
            "outcome": terminal.outcome,
            "reason_code": terminal.reason_code,
            "attempt_evidence_receipts": [
                measured_grade.digest,
                compile_receipt.digest,
                correctness_receipt.digest,
            ],
        }),
        media_type="application/json",
    )
    result = store.put_bytes(
        canonical_json_bytes(_terminal_result(contract, terminal, source, raw)),
        media_type="application/json",
    )
    return {
        "raw": raw,
        "grade": measured_grade,
        "harness": harness,
        "attempt_policy": attempt_policy,
        "execution": execution_receipt,
        "compile": compile_receipt,
        "correctness": correctness_receipt,
        "vector": vector,
        "terminal_policy": terminal_policy,
        "source": source,
        "result": result,
    }


def _measurement_report(policy: MeasurementPolicy) -> dict[str, Any]:
    order = (
        "reference", "optimized", "optimized", "reference",
        "optimized", "reference", "reference", "optimized",
    )
    health = {
        "device": "gfx950-fixture",
        "healthy": True,
        "temperature_c": 50.0,
        "clock_mhz": 1700.0,
    }
    blocks = [
        {
            "block_id": index,
            "order_position": index,
            "implementation": implementation,
            "samples_ms": [10.0 if implementation == "reference" else 8.0] * 75,
            "invalid_sample_counts": {},
            "gpu_health_before": health,
            "gpu_health_after": health,
        }
        for index, implementation in enumerate(order)
    ]
    return {
        "schema": "apex.kernel-measurement/v1",
        "policy_id": policy.policy_id,
        "sample_unit": policy.sample_unit,
        "quantile_method": policy.quantile_method,
        "timer": "hip_event",
        "timer_resolution_ns": 1.0,
        "inner_repeats": 1,
        "measurement_method_sha256": _METHOD,
        "abba_seed": 7,
        "warmup_samples": policy.warmup_samples,
        "cases": [{"case_id": "case-1", "blocks": blocks}],
    }


def _command_receipt(store: ArtifactStore, phase: str) -> ArtifactReceipt:
    return store.put_bytes(
        canonical_json_bytes({
            "phase": phase,
            "passed": True,
            "process_containment": {"namespace_empty_verified": True},
            "executable_identity_reverified": True,
            "executable_identity": {
                "path": "/usr/bin/fixture",
                "size": 10,
                "sha256": "6" * 64,
                "device": 1,
                "inode": 2,
                "mode": 0o755,
                "mtime_ns": 3,
                "ctime_ns": 4,
            },
            "argv": ["/usr/bin/fixture"],
        }),
        media_type="application/json",
    )


def _terminal_result(
    contract: ArtifactReceipt,
    terminal: Any,
    source: ArtifactReceipt,
    raw: ArtifactReceipt,
) -> dict[str, Any]:
    return {
        "schema": "apex.kernel-terminal-result/v1",
        "task_kind": "single_kernel",
        "run_id": _RUN,
        "task_id": _TASK,
        "evaluation_contract_receipt_digest": contract.digest,
        "task_reward": terminal.scalar_reward,
        "reward_vector": terminal.to_dict(),
        "reward_policy_id": terminal.policy_id,
        "reward_policy_digest": terminal.policy_digest,
        "reward_source_receipt": source.digest,
        "raw_measurement_receipts": [raw.digest],
        "trainability": "trainable",
        "untrainable_reason": None,
    }


def _graph(
    store: ArtifactStore,
    coding: ArtifactReceipt,
    contract: ArtifactReceipt,
    evidence: Mapping[str, ArtifactReceipt],
) -> EpisodeGraph:
    baseline = _campaign_baseline(store)
    baseline_event = _event(
        1,
        "dependency_verified",
        {
            "kind": "campaign_baseline",
            "release_candidate_receipt_sha256": baseline["document"]["receipt_sha256"],
            "apex_tree": _TREE,
        },
        (("campaign_baseline", baseline["receipt"]),),
    )
    contract_event = _event(
        2,
        "dependency_verified",
        {
            "kind": "evaluation_contract",
            "status": "verified",
            "contract_digest": contract.digest,
            "authority_receipt_digest": sha256_json(
                _contract_authority(store, contract)
            ),
            "authority_id": "fixture-reviewed-v1",
            "authority_kind": "reviewed_template",
        },
        (("evaluation_contract", contract),),
    )
    coding_document = _read_store_json(store, coding)
    coding_event = _event(
        3,
        "agent_completed",
        {
            "attempt_id": _ATTEMPT,
            "backend": "codex",
            "model": "fixture-model",
            "exit_code": 0,
            "timed_out": False,
            "termination_kind": "completed",
            "capture_status": "complete",
            "candidate_capture_allowed": True,
            "process_containment": coding_document["termination"]["process_containment"],
        },
        (("agent_transcript", coding),),
    )
    terminal = _read_store_json(store, evidence["result"])
    reward_event = _event(
        4,
        "reward_committed",
        {
            "scope": "task_terminal",
            "task_id": _TASK,
            "policy_id": terminal["reward_policy_id"],
            "policy_digest": terminal["reward_policy_digest"],
            "scalar_reward": terminal["task_reward"],
            "reward_vector": terminal["reward_vector"],
            "reward_source_receipt": evidence["source"].digest,
            "raw_measurement_receipts": [evidence["raw"].digest],
            "evidence_class": "derived",
        },
        (
            ("terminal_reward_source", evidence["source"]),
            ("kernel_terminal_grade", evidence["vector"]),
            ("reward_policy", evidence["terminal_policy"]),
            ("raw_measurement", evidence["raw"]),
            ("measurement_execution", evidence["execution"]),
            ("harness", evidence["harness"]),
            ("kernel_grade", evidence["grade"]),
            ("attempt_reward_policy", evidence["attempt_policy"]),
            ("compile_evidence", evidence["compile"]),
            ("correctness_evidence", evidence["correctness"]),
        ),
    )
    delivery = _event(
        5,
        "delivery_result",
        {"kind": "kernel_terminal_result", "task_id": _TASK},
        (("kernel_terminal_result", evidence["result"]),),
    )
    finished = _event(6, "run_finished", {"status": "succeeded"}, ())
    parent_id = episode_id(_RUN, _TASK)
    child_id = episode_id(_RUN, _ATTEMPT)
    child = CandidateEpisode(
        child_id, parent_id, _ATTEMPT, _ATTEMPT, None, _TASK, None,
        None, None, None, None, (coding_event,), "complete", None, None, None,
        (), "train", "private", "complete", (),
    )
    parent = ParentEpisode(
        parent_id, "single_kernel", _RUN, None, _TASK,
        (baseline_event, contract_event, reward_event, delivery, finished), (child_id,),
        "succeeded", terminal["task_reward"], terminal["reward_vector"],
        terminal["reward_policy_id"], terminal["reward_policy_digest"],
        evidence["source"].digest, (evidence["raw"].digest,), "complete", None,
    )
    return EpisodeGraph(
        1, _RUN, 6, "event-6", None, parent, (child,), {},
        (str(terminal["reward_policy_id"]),),
    )


def _campaign_baseline(store: ArtifactStore) -> dict[str, Any]:
    payload = {
        "schema": "apex.release-candidate-receipt/v2",
        "baseline_status": "ready",
        "status": "blocked",
        "static": {"apex_checkout": {"tree": _TREE, "clean": True}},
        "evidence": {},
        "qualification_authorities": [],
        "baseline_blockers": [],
        "blockers": ["live_qualifications_missing"],
    }
    document = {**payload, "receipt_sha256": sha256_json(payload)}
    receipt = store.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    return {"document": document, "receipt": receipt}


def _event(
    sequence: int,
    event_type: str,
    payload: dict[str, Any],
    artifacts: tuple[tuple[str, ArtifactReceipt], ...],
) -> EpisodeEvent:
    event_id = f"event-{sequence}"
    evidence = EvidenceClass(str(payload.get("evidence_class", "unspecified")))
    return EpisodeEvent(
        sequence, event_id, f"transaction-{sequence}",
        None if sequence == 1 else f"event-{sequence - 1}",
        event_type, semantic_role(event_type), evidence, payload,
        tuple(EpisodeArtifact(role, receipt, event_id) for role, receipt in artifacts),
    )


def _read_store_json(
    store: ArtifactStore, receipt: ArtifactReceipt
) -> Mapping[str, Any]:
    import json
    return json.loads(
        (store.root / receipt.relative_path).read_text(encoding="utf-8")
    )


def _contract_authority(
    store: ArtifactStore, contract: ArtifactReceipt
) -> Mapping[str, Any]:
    return _read_store_json(store, contract)["authority"]


def _write_index(root: Path, tree: str, manifest: ArtifactReceipt) -> None:
    payload = {
        "schema": INDEX_SCHEMA,
        "apex_tree": tree,
        "entries": [{
            "qualification_id": "backend-codex-gfx950",
            "manifest_receipt": manifest.to_dict(),
        }],
    }
    (root / INDEX_NAME).write_bytes(canonical_json_bytes({
        **payload, "manifest_sha256": sha256_json(payload),
    }))


def _qualified(authority: EvaluatorQualificationArtifactAuthority) -> Mapping[str, Any]:
    entry = next(
        item for item in authority.collect().entries
        if item.qualification_id == "backend-codex-gfx950"
    )
    assert entry.status == "verified"
    assert entry.evidence is not None
    return entry.evidence


def test_backend_verifier_replays_raw_coding_and_kernel_evidence(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    evidence = _qualified(campaign.authority)

    assert evidence["status"] == "qualified"
    assert evidence["coverage_count"] == 2
    assert evidence["formal_delivery_count"] == 1
    assert evidence["details"]["backend"] == "codex"


def test_production_factory_registers_only_the_three_gfx950_backends() -> None:
    verifiers = backend_live_qualification_verifiers()

    assert tuple(item.qualification_id for item in verifiers) == (
        "backend-claude-gfx950",
        "backend-codex-gfx950",
        "backend-cursor-gfx950",
    )
    assert len({item.verifier_identity_sha256 for item in verifiers}) == 1


def test_forged_qualification_summary_cannot_replace_recomputation(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    evidence = _qualified(campaign.authority)
    forged = build_qualification_evidence(
        qualification_id="backend-codex-gfx950",
        apex_tree=_TREE,
        subject_sha256=evidence["subject_sha256"],
        status="qualified",
        coverage_count=3,
        formal_delivery_count=1,
        details=evidence["details"],
    )

    with pytest.raises(ContractError, match="differs"):
        campaign.authority.verify(forged.to_dict())


def test_missing_raw_receipt_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    raw_path = campaign.root / "artifacts" / campaign.raw_measurement.relative_path
    raw_path.unlink()

    entry = next(
        item for item in campaign.authority.collect().entries
        if item.qualification_id == "backend-codex-gfx950"
    )

    assert entry.status == "invalid"
    assert entry.evidence is None


@pytest.mark.parametrize(
    ("backend", "tree"),
    [("claude", _TREE), ("codex", "a" * 40)],
)
def test_wrong_backend_or_apex_tree_fails_closed(
    tmp_path: Path, backend: str, tree: str
) -> None:
    campaign = _campaign(tmp_path, manifest_backend=backend, manifest_tree=tree)

    entry = next(
        item for item in campaign.authority.collect().entries
        if item.qualification_id == "backend-codex-gfx950"
    )

    assert entry.status == "invalid"


def test_raw_sample_tamper_is_detected_before_reward_replay(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    path = campaign.root / "artifacts" / campaign.raw_measurement.relative_path
    content = path.read_bytes()
    path.write_bytes(content.replace(b"10.0", b"11.0", 1))

    entry = next(
        item for item in campaign.authority.collect().entries
        if item.qualification_id == "backend-codex-gfx950"
    )

    assert entry.status == "invalid"


def test_manifest_only_self_claim_has_no_qualification_authority(tmp_path: Path) -> None:
    source = tmp_path / "Apex"
    source.mkdir()
    root = tmp_path / "formal-results"
    root.mkdir()
    store = ArtifactStore(root / "artifacts")
    manifest = store.put_bytes(
        canonical_json_bytes({
            "schema": "apex.backend-live-qualification-artifacts/v1",
            "qualification_id": "backend-codex-gfx950",
            "backend": "codex",
            "gpu_arch": "gfx950",
            "apex_tree": _TREE,
            "status": "qualified",
            "coverage_count": 2,
            "formal_delivery_count": 1,
        }),
        media_type="application/json",
    )
    _write_index(root, _TREE, manifest)
    authority = EvaluatorQualificationArtifactAuthority(
        artifact_root=root,
        results_policy=FormalResultsRootValidator((source,)),
        verifiers=backend_live_qualification_verifiers(),
    )

    entry = next(
        item for item in authority.collect().entries
        if item.qualification_id == "backend-codex-gfx950"
    )

    assert entry.status == "invalid"
    assert entry.evidence is None
