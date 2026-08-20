"""Trusted CLI handoff from a chat-created draft to the formal kernel loop."""

from __future__ import annotations

import json
from pathlib import Path

from apex.core import AgentBackendName, ApexError, ContractError
from apex.evaluation import user_confirmed_evaluation_authorizer
from apex.optimization.kernel import FormalKernelCampaign, KernelOptimizeRequest

from .formal import formal_result_path, formal_results_root, status_exit_code
from .release import require_campaign_baseline


def run_kernel_campaign_handoff(args, build_application) -> int:
    """Revalidate one untrusted draft, then delegate to the sole formal optimizer."""

    _validate_inputs(args)
    workspace, results = _scope(args.workspace, args.results)
    campaign_root = _campaign_root(args.campaign, results)
    confirmed = args.evaluation_contract_draft_digest
    if not confirmed:
        raise ApexError(
            "Chat-created campaigns require exact draft confirmation",
            "evaluation_authority_missing",
        )
    campaign = FormalKernelCampaign.load(
        campaign_root,
        workspace=workspace,
        results=results,
    )
    if campaign.draft_contract.draft.digest != confirmed:
        raise ContractError(
            "Confirmed digest differs from the chat-created draft",
            "evaluation_authority_mismatch",
        )
    baseline = require_campaign_baseline(args.release_candidate_receipt)
    application = build_application(
        kernel_evaluation_authorizer=user_confirmed_evaluation_authorizer(
            confirmed
        )
    )
    optimizer = application.kernel_optimizer
    if optimizer is None:
        raise ApexError(
            "Kernel optimizer composition is unavailable", "kernel_not_composed"
        )
    contract = optimizer.preview_evaluation_contract(campaign.task)
    if contract.draft != campaign.draft_contract.draft:
        raise ContractError(
            "Workspace or evaluation inputs changed after chat discovery",
            "evaluation_contract_drift",
        )
    result_path = (
        formal_result_path(args.result_json, results)
        if args.result_json is not None
        else results / "result.json"
    )
    result = optimizer.run(
        KernelOptimizeRequest(
            task=campaign.task,
            result_json=result_path,
            backend_override=(
                AgentBackendName(args.agent_backend)
                if args.agent_backend
                else None
            ),
            model_override=args.agent_model,
            effort_override=args.agent_effort,
            campaign_baseline=baseline,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return status_exit_code(result.status)


def _validate_inputs(args) -> None:
    if args.request or args.task_spec is not None or args.template is not None:
        raise ApexError(
            "Campaign handoff cannot be combined with request, task spec, or template",
            "kernel_campaign_input_invalid",
        )


def _scope(workspace: Path | None, results: Path | None) -> tuple[Path, Path]:
    if workspace is None or results is None:
        raise ApexError(
            "Campaign handoff requires --workspace and --results",
            "kernel_campaign_paths_required",
        )
    root = workspace.expanduser().resolve(strict=True)
    return root, formal_results_root(results.expanduser(), workspace=root)


def _campaign_root(path: Path, results: Path) -> Path:
    selected = path.expanduser()
    _reject_symlink_components(selected)
    resolved = selected.resolve(strict=True)
    try:
        resolved.relative_to(results)
    except ValueError as error:
        raise ApexError(
            "Campaign root must be inside --results", "campaign_path_outside_results"
        ) from error
    if not resolved.is_dir():
        raise ApexError("Campaign root is not a directory", "campaign_not_found")
    return resolved


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ApexError(
                "Campaign path cannot contain a symlink", "unsafe_campaign_path"
            )


__all__ = ["run_kernel_campaign_handoff"]
