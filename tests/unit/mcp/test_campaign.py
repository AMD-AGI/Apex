from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.mcp import (
    CampaignCheckpointHandler,
    CampaignResumeHandler,
    CampaignStartHandler,
    CampaignStatusHandler,
    CapabilityRegistry,
    CapabilityScope,
    planned_capability_descriptors,
)
from apex.ports import CapabilityAuthority, CapabilityRequest
from apex.orchestration import RunController, RunPhase
from apex.reporting import resolve_run_source
from apex.storage import ArtifactStore, EventJournal, SnapshotStore
from apex.optimization.kernel import KernelCampaignDraftUseCase
from apex.cli.app import _parser
from apex.mcp.campaign import _formal_continuation
from tests.support.execution_identity import execution_identity


def _git(workspace: Path, *arguments: str) -> None:
    import subprocess

    subprocess.run(
        ("git", *arguments), cwd=workspace, check=True, capture_output=True
    )


def _registry(tmp_path: Path) -> CapabilityRegistry:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.status"
    )
    registry = CapabilityRegistry()
    registry.register(
        descriptor,
        CampaignStatusHandler(CapabilityScope(workspace, tmp_path / "results")),
    )
    return registry


def test_campaign_status_replays_parent_and_all_attempts(tmp_path: Path) -> None:
    run_id = "run-campaign-status"
    destination = tmp_path / "results" / run_id
    destination.mkdir(parents=True)
    controller = RunController.create(
        run_id,
        EventJournal(destination / "events" / "run.db"),
        SnapshotStore(destination / "state.snapshot.json"),
        initial_anchor_id="anchor-campaign-status",
    )
    controller.queue_action("attempt-1", "kernel-candidate")
    controller.start_action("attempt-1")
    controller.abort_pending("no_gain")
    controller.record_domain_event(
        "decision",
        {"attempt_id": "attempt-1", "candidate_id": "candidate-1", "verdict": "revert"},
        idempotency_key="attempt-1.decision",
    )
    controller.queue_action("attempt-2", "kernel-candidate")
    controller.start_action("attempt-2")
    controller.fail_action("attempt-2", "compile_failed")
    controller.record_domain_event(
        "decision",
        {"attempt_id": "attempt-2", "candidate_id": "candidate-2", "verdict": "reject"},
        idempotency_key="attempt-2.decision",
    )
    controller.finish(RunPhase.SUCCEEDED, reason="completed")
    ArtifactStore(destination / "artifacts").put_bytes(b"evidence")
    registry = _registry(tmp_path)

    result = registry.invoke(
        CapabilityRequest(
            "campaign.status",
            {"run_locator": run_id},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    campaign = result.content["campaign"]
    assert campaign["run_id"] == run_id
    assert campaign["task_kind"] == "single_kernel"
    assert campaign["terminal_status"] == "succeeded"
    assert campaign["state"] == {
        "phase": "succeeded",
        "sequence": campaign["high_water_mark"],
        "anchor_id": "anchor-campaign-status",
        "anchor_generation": 0,
        "accepted_patch_ids": [],
        "stop_reason": "completed",
        "pending_action": None,
        "e2e": None,
    }
    assert [item["attempt_id"] for item in campaign["attempts"]] == [
        "attempt-1",
        "attempt-2",
    ]
    assert result.reward_eligible is False


def test_campaign_status_requires_authority_and_scoped_existing_run(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(ContractError) as authority:
        registry.invoke(
            CapabilityRequest("campaign.status", {"run_locator": "run-missing"})
        )
    assert authority.value.reason_code == "capability_authority_missing"

    with pytest.raises(ContractError) as missing:
        registry.invoke(
            CapabilityRequest(
                "campaign.status",
                {"run_locator": "run-missing"},
                frozenset({CapabilityAuthority.WORKSPACE_USER}),
            )
        )
    assert missing.value.reason_code == "unsafe_capability_path"


def test_campaign_status_projects_active_pending_state_without_writing(
    tmp_path: Path,
) -> None:
    run_id = "run-active-status"
    destination = tmp_path / "results" / run_id
    destination.mkdir(parents=True)
    journal = EventJournal(destination / "events" / "run.db")
    snapshot_path = destination / "state.snapshot.json"
    controller = RunController.create(
        run_id,
        journal,
        SnapshotStore(snapshot_path),
        initial_anchor_id="anchor-active",
    )
    controller.queue_action("attempt-active", "kernel-candidate")
    controller.start_action("attempt-active")
    ArtifactStore(destination / "artifacts").put_bytes(b"evidence")
    events_before = tuple(journal.iter_events(run_id, verify=True))
    snapshot_before = snapshot_path.read_bytes()

    result = _registry(tmp_path).invoke(
        CapabilityRequest(
            "campaign.status",
            {"run_locator": run_id},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    campaign = result.content["campaign"]
    assert campaign["terminal_status"] == "incomplete"
    assert campaign["state"]["phase"] == "running"
    assert campaign["state"]["anchor_id"] == "anchor-active"
    assert campaign["state"]["pending_action"]["action_id"] == "attempt-active"
    assert tuple(journal.iter_events(run_id, verify=True)) == events_before
    assert snapshot_path.read_bytes() == snapshot_before


def test_campaign_checkpoint_rebuilds_disposable_snapshot_only(
    tmp_path: Path,
) -> None:
    run_id = "run-checkpoint"
    destination = tmp_path / "results" / run_id
    destination.mkdir(parents=True)
    journal = EventJournal(destination / "events" / "run.db")
    snapshots = SnapshotStore(destination / "state.snapshot.json")
    controller = RunController.create(
        run_id,
        journal,
        snapshots,
        initial_anchor_id="anchor-checkpoint",
    )
    controller.queue_action("attempt-checkpoint", "kernel-candidate")
    ArtifactStore(destination / "artifacts").put_bytes(b"evidence")
    events_before = tuple(journal.iter_events(run_id, verify=True))
    snapshots.delete()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.checkpoint"
    )
    registry = CapabilityRegistry()
    registry.register(
        descriptor,
        CampaignCheckpointHandler(CapabilityScope(workspace, tmp_path / "results")),
    )

    result = registry.invoke(
        CapabilityRequest(
            "campaign.checkpoint",
            {"run_locator": run_id},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    checkpoint = result.content["campaign"]["checkpoint"]
    rebuilt = snapshots.load()
    assert rebuilt is not None
    assert checkpoint == {
        "schema": "apex.campaign-checkpoint/v1",
        "high_water_mark": rebuilt.high_water_mark,
        "payload_sha256": rebuilt.payload_hash,
        "canonical_events_unchanged": True,
    }
    assert tuple(journal.iter_events(run_id, verify=True)) == events_before
    assert result.reward_eligible is False


def test_campaign_start_freezes_unverified_draft_without_agent_or_gpu(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text(
        "def kernel(x): return x\n", encoding="utf-8"
    )
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/draft.git")
    _git(workspace, "add", "kernel.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    results = tmp_path / "results"
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.start"
    )
    registry = CapabilityRegistry()
    registry.register(
        descriptor,
        CampaignStartHandler(
            CapabilityScope(workspace, results),
            KernelCampaignDraftUseCase(),
            execution_identity(),
        ),
    )

    result = registry.invoke(
        CapabilityRequest(
            "campaign.start",
            {
                "task": {
                    "task_id": "chat-draft",
                    "instructions": "Optimize kernel",
                    "language": "triton",
                    "editable_files": ["kernel.py"],
                    "target_functions": ["kernel"],
                    "commands": {
                        phase: {"argv": ["true"]}
                        for phase in ("compile", "correctness", "performance")
                    },
                }
            },
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    campaign = result.content["campaign"]
    assert campaign["status"] == "unverified"
    assert campaign["next_action"] == "explicit_trusted_user_confirmation_required"
    assert campaign["execution_identity_sha256"] == execution_identity().receipt_sha256
    assert campaign["candidate_projection"] is not None
    assert campaign["agent_invoked"] is False
    assert campaign["gpu_acquired"] is False
    assert campaign["reward"] is None
    assert campaign["evaluation_contract"]["unverified_reason"] == (
        "evaluation_authority_missing"
    )
    assert campaign["formal_continuation"] == {
        "schema": "apex.kernel-campaign-continuation/v1",
        "ready": True,
        "blocked_reason": None,
        "requires_user_confirmation": True,
        "run_only_after_chat_exits": True,
        "argv_template": [
            "apex",
            "optimize",
            "kernel",
            "--campaign",
            str((results / campaign["run_locator"]["relative_path"]).resolve()),
            "--workspace",
            str(workspace.resolve()),
            "--results",
            str(results.resolve()),
            "--evaluation-contract-draft-digest",
            campaign["evaluation_contract_draft_digest"],
        ],
    }
    run = results / campaign["run_locator"]["relative_path"]
    source = resolve_run_source(run, run_id=campaign["run_id"])
    events = tuple(source.journal.iter_events(campaign["run_id"], verify=True))
    assert [item.event_type for item in events] == [
        "run.started",
        "provenance_observed",
        "dependency_verified",
        "provenance_observed",
        "tool_called",
        "tool_result",
    ]
    assert all(item.event_type != "reward_committed" for item in events)
    status_descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.status"
    )
    registry.register(
        status_descriptor, CampaignStatusHandler(CapabilityScope(workspace, results))
    )
    status = registry.invoke(
        CapabilityRequest(
            "campaign.status",
            {"run_locator": campaign["run_locator"]["relative_path"]},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    ).content["campaign"]
    assert status["run_id"] == campaign["run_id"]
    assert status["task_kind"] == "single_kernel"
    assert status["terminal_status"] == "incomplete"
    assert status["task_reward"] is None
    assert result.reward_eligible is False


def test_campaign_start_schema_has_no_release_receipt_input() -> None:
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.start"
    )
    assert "release_candidate_receipt" not in descriptor.input_schema["properties"]


def test_ready_continuation_round_trips_absolute_paths_through_cli_parser(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    results = tmp_path / "results with spaces"
    campaign = results / "campaigns" / "campaign-ready"
    campaign.mkdir(parents=True)
    scope = CapabilityScope(workspace, results)

    continuation = _formal_continuation(scope, campaign, "d" * 64)
    parsed = _parser().parse_args(continuation["argv_template"][1:])

    assert continuation["ready"] is True
    assert continuation["blocked_reason"] is None
    assert parsed.campaign == campaign
    assert parsed.workspace == workspace
    assert parsed.results == results
    assert parsed.evaluation_contract_draft_digest == "d" * 64
    assert not hasattr(parsed, "release_candidate_receipt")


def test_campaign_resume_delegates_after_scoped_run_load(
    tmp_path: Path,
) -> None:
    run_id = "e2e-resume-capability"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    results = tmp_path / "results"
    destination = results / run_id
    destination.mkdir(parents=True)
    journal = EventJournal(destination / "events" / "run.db")
    snapshots = SnapshotStore(destination / "state.snapshot.json")
    RunController.create(
        run_id,
        journal,
        snapshots,
        initial_anchor_id="anchor-resume",
    )
    ArtifactStore(destination / "artifacts").put_bytes(b"evidence")
    resumed: list[Path] = []

    class _Result:
        def to_dict(self):
            return {"status": "no_gain", "run_id": run_id}

    def resume(root: Path):
        resumed.append(root)
        RunController.recover(run_id, journal, snapshots).finish(
            RunPhase.SUCCEEDED, reason="no_gain"
        )
        return _Result()

    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.resume"
    )
    registry = CapabilityRegistry()
    registry.register(
        descriptor,
        CampaignResumeHandler(CapabilityScope(workspace, results), resume),
    )

    result = registry.invoke(
        CapabilityRequest(
            "campaign.resume",
            {"run_locator": run_id},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    campaign = result.content["campaign"]
    assert resumed == [destination]
    assert campaign["terminal_status"] == "succeeded"
    assert campaign["resume"]["result"]["status"] == "no_gain"
    assert result.reward_eligible is False
