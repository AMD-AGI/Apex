from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.cli import kernel_handoff
from apex.core import ApexError, TaskStatus
from apex.optimization.kernel import KernelCampaignDraftUseCase


def _git(workspace: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments), cwd=workspace, check=True, capture_output=True
    )


def _draft(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text(
        "def kernel(x): return x\n", encoding="utf-8"
    )
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/handoff.git")
    _git(workspace, "add", "kernel.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    results = tmp_path / "results"
    campaign_root = results / "campaigns" / "campaign-handoff"
    draft = KernelCampaignDraftUseCase().start(
        {
            "task_id": "chat-handoff",
            "instructions": "Optimize kernel and prove it is faster",
            "language": "triton",
            "editable_files": ["kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                phase: {"argv": ["true"]}
                for phase in ("compile", "correctness", "performance")
            },
        },
        workspace=workspace,
        results_dir=results,
        run_root=campaign_root,
        run_id="campaign-handoff",
    )
    return workspace, results, campaign_root, draft


def _args(workspace, results, campaign, digest):
    return SimpleNamespace(
        workspace=workspace,
        results=results,
        campaign=campaign,
        evaluation_contract_draft_digest=digest,
        result_json=None,
        agent_backend="claude",
        agent_model=None,
        agent_effort=None,
        request=None,
        task_spec=None,
        template=None,
    )


def test_confirmed_chat_draft_delegates_once_to_formal_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, results, root, draft = _draft(tmp_path)
    calls = []

    class Optimizer:
        def preview_evaluation_contract(self, task):
            calls.append(("preview", task))
            return draft.contract

        def run(self, request):
            calls.append(("run", request))
            return SimpleNamespace(
                status=TaskStatus.NO_GAIN,
                to_dict=lambda: {"status": TaskStatus.NO_GAIN.value},
            )

    built = []

    def build_application(**values):
        built.append(values)
        return SimpleNamespace(kernel_optimizer=Optimizer())

    status = kernel_handoff.run_kernel_campaign_handoff(
        _args(workspace, results, root, draft.contract.draft.digest),
        build_application,
    )

    assert status == 0
    assert len(built) == 1
    assert [name for name, _ in calls] == ["preview", "run"]
    request = calls[-1][1]
    assert request.backend_override.value == "claude"
    assert not hasattr(request, "campaign_baseline")
    assert request.result_json == results / "result.json"


def test_agent_digest_echo_cannot_bypass_exact_host_confirmation(
    tmp_path: Path,
) -> None:
    workspace, results, root, _draft_receipt = _draft(tmp_path)
    args = _args(workspace, results, root, "0" * 64)

    with pytest.raises(ApexError) as caught:
        kernel_handoff.run_kernel_campaign_handoff(
            args,
            lambda **_values: (_ for _ in ()).throw(
                AssertionError("optimizer must not be composed")
            ),
        )

    assert caught.value.reason_code == "evaluation_authority_mismatch"


def test_campaign_handoff_rejects_symlinked_campaign_component(
    tmp_path: Path,
) -> None:
    workspace, results, root, draft = _draft(tmp_path)
    linked = results / "linked-campaign"
    linked.symlink_to(root, target_is_directory=True)

    with pytest.raises(ApexError) as caught:
        kernel_handoff.run_kernel_campaign_handoff(
            _args(workspace, results, linked, draft.contract.draft.digest),
            lambda **_values: (_ for _ in ()).throw(
                AssertionError("optimizer must not be composed")
            ),
        )

    assert caught.value.reason_code == "unsafe_campaign_path"
