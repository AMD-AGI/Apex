"""Scoped campaign projections rebuilt from canonical journal and CAS evidence."""

from __future__ import annotations

from apex.core import ApexError, new_identifier
from apex.optimization.kernel import (
    FormalKernelCampaign,
    begin_formal_capability,
    complete_formal_capability,
)
from apex.orchestration import RunController, WorkloadState
from apex.orchestration.replay import replay_workload_state
from apex.ports import CapabilityRequest, CapabilityResult
from apex.reporting import materialize_run_graph, resolve_run_source
from apex.rl import EpisodeGraph
from apex.storage import SnapshotStore

from .scope import CapabilityScope


class CampaignStatusHandler:
    """Replay one run without appending events or reading backend chat state."""

    def __init__(self, scope: CapabilityScope) -> None:
        self._scope = scope

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        root = self._scope.read_results(str(request.arguments["run_locator"]))
        source = resolve_run_source(root)
        state = replay_workload_state(
            source.run_id,
            tuple(source.journal.iter_events(source.run_id, verify=True)),
        )
        graph = materialize_run_graph(source)
        return CapabilityResult(
            request.capability_id,
            {"campaign": _campaign_projection(graph, state)},
            reward_eligible=False,
        )


class CampaignStartHandler:
    """Create one unverified formal draft through the optimization use case."""

    def __init__(self, scope: CapabilityScope, starter, baseline_loader=None) -> None:
        self._scope = scope
        self._starter = starter
        self._baseline_loader = baseline_loader

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        run_id = new_identifier("campaign")
        run_root = self._scope.claim_output("campaigns", run_id)
        draft = self._starter.start(
            request.arguments["task"],
            workspace=self._scope.workspace,
            results_dir=self._scope.results,
            run_root=run_root,
            run_id=run_id,
        )
        root, locator = self._scope.locator(draft.root)
        contract = draft.contract.to_dict()
        content = {
            "campaign": {
                "schema": "apex.campaign-draft/v1",
                "run_id": draft.run_id,
                "run_locator": {"root": root, "relative_path": locator},
                "status": "unverified",
                "high_water_mark": draft.high_water_mark + 3,
                "evaluation_contract": contract,
                "evaluation_contract_digest": draft.contract.digest,
                "evaluation_contract_draft_digest": draft.contract.draft.digest,
                "next_action": "explicit_user_confirmation_required",
                "agent_invoked": False,
                "gpu_acquired": False,
                "reward": None,
            }
        }
        campaign = FormalKernelCampaign.load(
            draft.root,
            workspace=self._scope.workspace,
            results=self._scope.results,
        )
        baseline, baseline_reason = self._load_baseline(request)
        campaign.bind_release_candidate_baseline(
            baseline, reason_code=baseline_reason
        )
        candidate_locator = None
        if baseline is not None and baseline_reason is None:
            candidate = campaign.ensure_candidate_projection()
            candidate_root, candidate_path = self._scope.locator(candidate)
            candidate_locator = {
                "root": candidate_root,
                "relative_path": candidate_path,
            }
        content["campaign"]["release_candidate_baseline_status"] = (
            "verified" if baseline_reason is None else "unverified"
        )
        content["campaign"]["release_candidate_baseline_reason"] = baseline_reason
        content["campaign"]["release_candidate_receipt_sha256"] = (
            getattr(baseline, "receipt_sha256", None)
        )
        content["campaign"]["candidate_projection"] = candidate_locator
        content["campaign"]["next_action"] = (
            "explicit_trusted_user_confirmation_required"
            if baseline_reason is None
            else "valid_release_candidate_baseline_required"
        )
        content["campaign"]["formal_continuation"] = _formal_continuation(
            self._scope,
            draft.root,
            draft.contract.draft.digest,
            request,
            baseline_reason,
        )
        invocation = begin_formal_capability(
            campaign.record, request.capability_id, request.arguments
        )
        complete_formal_capability(campaign.record, invocation, content)
        return CapabilityResult(
            request.capability_id,
            content,
            reward_eligible=False,
        )

    def _load_baseline(self, request: CapabilityRequest):
        value = request.arguments.get("release_candidate_receipt")
        if value is None:
            return None, "campaign_baseline_receipt_required"
        if self._baseline_loader is None:
            return None, "campaign_baseline_verifier_unavailable"
        try:
            path = self._scope.read_results(str(value))
            return self._baseline_loader(path), None
        except ApexError as error:
            return None, error.reason_code


def _formal_continuation(
    scope: CapabilityScope,
    campaign_root,
    draft_digest: str,
    request: CapabilityRequest,
    baseline_reason: str | None,
) -> dict[str, object]:
    receipt = "<valid-release-candidate-receipt-under-results>"
    supplied = request.arguments.get("release_candidate_receipt")
    if supplied is not None:
        try:
            receipt = str(scope.read_results(str(supplied)))
        except ApexError:
            pass
    return {
        "schema": "apex.kernel-campaign-continuation/v1",
        "ready": baseline_reason is None,
        "blocked_reason": baseline_reason,
        "requires_user_confirmation": True,
        "run_only_after_chat_exits": True,
        "argv_template": [
            "apex",
            "optimize",
            "kernel",
            "--campaign",
            str(campaign_root.resolve(strict=True)),
            "--workspace",
            str(scope.workspace),
            "--results",
            str(scope.results),
            "--evaluation-contract-draft-digest",
            draft_digest,
            "--release-candidate-receipt",
            receipt,
        ],
    }


class CampaignStopHandler:
    """Terminalize one standalone formal campaign through its domain service."""

    def __init__(self, scope: CapabilityScope, stopper) -> None:
        self._scope = scope
        self._stopper = stopper

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        run_locator = str(request.arguments["run_locator"])
        campaign = FormalKernelCampaign.load(
            self._scope.read_results(run_locator),
            workspace=self._scope.workspace,
            results=self._scope.results,
        )
        stopped = self._stopper(
            campaign,
            reason=str(request.arguments.get("reason", "user_requested")),
            capability_arguments=request.arguments,
        )
        status = CampaignStatusHandler(self._scope).invoke(
            CapabilityRequest(
                "campaign.status", {"run_locator": run_locator}, request.authorities
            )
        ).content["campaign"]
        projected = dict(status)
        projected["stop"] = stopped.to_dict()
        return CapabilityResult(
            request.capability_id,
            {"campaign": projected},
            reward_eligible=False,
        )


class CampaignCheckpointHandler:
    """Rebuild one disposable state snapshot without mutating canonical evidence."""

    def __init__(self, scope: CapabilityScope) -> None:
        self._scope = scope

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        root = self._scope.read_results(str(request.arguments["run_locator"]))
        source = resolve_run_source(root)
        events_before = tuple(source.journal.iter_events(source.run_id, verify=True))
        snapshots = SnapshotStore(source.root / "state.snapshot.json")
        controller = RunController.recover(source.run_id, source.journal, snapshots)
        events_after = tuple(source.journal.iter_events(source.run_id, verify=True))
        if events_after != events_before:
            raise AssertionError("checkpoint recovery mutated the canonical journal")
        snapshot = snapshots.load()
        if snapshot is None:
            raise AssertionError("checkpoint recovery did not publish a snapshot")
        graph = materialize_run_graph(source)
        campaign = _campaign_projection(graph, controller.state)
        campaign["checkpoint"] = {
            "schema": "apex.campaign-checkpoint/v1",
            "high_water_mark": snapshot.high_water_mark,
            "payload_sha256": snapshot.payload_hash,
            "canonical_events_unchanged": True,
        }
        return CapabilityResult(
            request.capability_id,
            {"campaign": campaign},
            reward_eligible=False,
        )


class CampaignResumeHandler:
    """Delegate E2E recovery to the formal use case after baseline revalidation."""

    def __init__(self, scope: CapabilityScope, baseline_loader, resumer) -> None:
        self._scope = scope
        self._baseline_loader = baseline_loader
        self._resumer = resumer

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        run_locator = str(request.arguments["run_locator"])
        run_root = self._scope.read_results(run_locator)
        baseline_path = self._scope.read_results(
            str(request.arguments["release_candidate_receipt"])
        )
        baseline = self._baseline_loader(baseline_path)
        resumed = self._resumer(run_root, baseline)
        status = CampaignStatusHandler(self._scope).invoke(
            CapabilityRequest(
                "campaign.status",
                {"run_locator": run_locator},
                request.authorities,
            )
        ).content["campaign"]
        campaign = dict(status)
        campaign["resume"] = {
            "schema": "apex.campaign-resume/v1",
            "release_candidate_receipt_sha256": baseline.receipt_sha256,
            "result": resumed.to_dict(),
        }
        return CapabilityResult(
            request.capability_id,
            {"campaign": campaign},
            reward_eligible=False,
        )


def _campaign_projection(
    graph: EpisodeGraph, state: WorkloadState
) -> dict[str, object]:
    return {
        "run_id": graph.run_id,
        "episode_graph_id": graph.graph_id,
        "task_kind": graph.parent.kind,
        "terminal_status": graph.parent.terminal_status,
        "high_water_mark": graph.high_water_mark,
        "task_reward": graph.parent.task_reward,
        "trainability": graph.parent.trainability,
        "policy_ids": list(graph.policy_ids),
        "state": _state_projection(state),
        "attempts": [
            {
                "attempt_id": item.attempt_id,
                "candidate_id": item.candidate_id,
                "status": item.status,
                "verdict": item.verdict,
                "scalar_reward": item.scalar_reward,
                "trainability": item.trainability,
            }
            for item in graph.children
        ],
    }


def _state_projection(state: WorkloadState) -> dict[str, object]:
    search = state.e2e
    return {
        "phase": state.phase.value,
        "sequence": state.sequence,
        "anchor_id": state.anchor_id,
        "anchor_generation": state.anchor_generation,
        "accepted_patch_ids": list(state.accepted_patch_ids),
        "stop_reason": state.stop_reason,
        "pending_action": (
            state.pending_action.to_dict() if state.pending_action is not None else None
        ),
        "e2e": (
            {
                "stage": search.stage.value,
                "state_generation": search.state_generation,
                "cycle": search.cycle,
                "budget": search.budget.to_dict(),
                "active_attempt_id": search.active_attempt_id,
                "active_opportunity_id": search.active_opportunity_id,
                "active_candidate_id": search.active_candidate_id,
                "exit_reason": search.exit_reason,
            }
            if search is not None
            else None
        ),
    }


__all__ = [
    "CampaignCheckpointHandler",
    "CampaignResumeHandler",
    "CampaignStartHandler",
    "CampaignStatusHandler",
    "CampaignStopHandler",
]
