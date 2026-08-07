from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from apex.core import AgentBackendName, IntegrityError
from apex.execution import AgentRegistry
from apex.optimization.e2e.candidate import (
    AgentCandidateWorker,
    E2ECandidateRequest,
    SourceCandidateWorkspace,
)
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.ports import (
    AgentCaptureStatus,
    AgentInvocationReceipt,
    AgentRequest,
    AgentResult,
    AgentTerminationKind,
    BOUNDARY_QUIESCENCE_POLICY,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, name: str = "source") -> Path:
    root = tmp_path / name
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "apex@example.test")
    _git(root, "config", "user.name", "Apex Test")
    _git(root, "remote", "add", "origin", str(root))
    return root


def _commit(root: Path, message: str = "anchor") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)


def _opportunity(root: Path) -> KernelOpportunity:
    return KernelOpportunity(
        opportunity_id="kernel-opportunity",
        evidence_id="a" * 64,
        runtime_name="kernel",
        operation_name="attention",
        phase="decode",
        rank=0,
        language="triton",
        origin_library="aiter",
        shape_summary=("[16, 128]",),
        dtypes=("float16",),
        graph_mode="eager",
        match_confidence="active_finder",
        measured_gpu_pct=10.0,
        roi_prior=5.0,
        source_path=root / "kernel.py",
        source_root=root,
        test_file=root / "test_kernel.py",
        test_command="pytest test_kernel.py",
        eligibility="eligible",
        reason_code="eligible",
    )


class _BoundaryAgent:
    name = AgentBackendName.CODEX

    def __init__(
        self,
        kind: AgentTerminationKind,
        *,
        capture: AgentCaptureStatus = AgentCaptureStatus.COMPLETE,
        make_change: bool = True,
        create_ignored_artifact: bool = False,
    ) -> None:
        self.kind = kind
        self.capture = capture
        self.make_change = make_change
        self.create_ignored_artifact = create_ignored_artifact

    def run(self, request: AgentRequest) -> AgentResult:
        if self.make_change:
            (request.workspace / "kernel.py").write_text("value = 2\n", encoding="utf-8")
        if self.create_ignored_artifact:
            cache = request.workspace / "__pycache__"
            cache.mkdir(exist_ok=True)
            (cache / "poison.pyc").write_bytes(b"untrusted-bytecode")
        exact = self.kind is AgentTerminationKind.EXACT_TURN_BOUNDARY
        reason = {
            AgentTerminationKind.EXACT_TURN_BOUNDARY: "max_turns_exact_boundary",
            AgentTerminationKind.TURN_OVERRUN: "max_turns_overrun",
            AgentTerminationKind.TIMEOUT: "agent_process_timeout",
            AgentTerminationKind.INVALID_STREAM: "unparseable_structured_event",
        }.get(self.kind)
        return AgentResult(
            backend=self.name,
            model=request.model,
            exit_code=-15,
            timed_out=self.kind is AgentTerminationKind.TIMEOUT,
            events=(),
            stdout='{"type":"assistant_message","content":"done"}\n',
            stderr="",
            duration_seconds=0.1,
            invocation=_invocation(request),
            termination_kind=self.kind,
            capture_status=self.capture,
            termination_reason=reason,
            observed_turns=(
                request.max_turns
                if exact
                else request.max_turns + 1
                if self.kind is AgentTerminationKind.TURN_OVERRUN
                else 1
            ),
            observer_stop_sent=self.kind in {
                AgentTerminationKind.EXACT_TURN_BOUNDARY,
                AgentTerminationKind.TURN_OVERRUN,
                AgentTerminationKind.INVALID_STREAM,
            },
            observer_suspend_sent=self.kind in {
                AgentTerminationKind.EXACT_TURN_BOUNDARY,
                AgentTerminationKind.TURN_OVERRUN,
                AgentTerminationKind.INVALID_STREAM,
            },
            suspension_verified=self.kind in {
                AgentTerminationKind.EXACT_TURN_BOUNDARY,
                AgentTerminationKind.TURN_OVERRUN,
                AgentTerminationKind.INVALID_STREAM,
            },
        )


def _invocation(request: AgentRequest) -> AgentInvocationReceipt:
    return AgentInvocationReceipt(
        cli_name="codex",
        cli_version="test",
        executable_path="/usr/bin/codex",
        resolved_executable_path="/usr/bin/codex",
        entrypoint_sha256="a" * 64,
        argv=("codex", "exec"),
        workspace=str(request.workspace),
        prompt_transport="stdin",
        requested_allowed_files=request.allowed_files,
        allowed_files_enforced_by_cli=False,
        max_turns=request.max_turns,
        turn_policy=STRUCTURED_TURN_CHECKPOINT_POLICY,
        boundary_quiescence_policy_id=BOUNDARY_QUIESCENCE_POLICY,
        isolation=(("sandbox", "workspace-write"),),
    )


def _candidate_request(root: Path, destination: Path) -> E2ECandidateRequest:
    return E2ECandidateRequest(
        run_id="run-1",
        attempt_id="attempt-1",
        opportunity=_opportunity(root),
        prompt="Optimize kernel.py",
        destination=destination,
        backend=AgentBackendName.CODEX,
        model=None,
        effort=None,
        max_turns=50,
        timeout_seconds=30,
    )


def test_git_checkout_preserves_safe_symlink_identity(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (root / "shared.py").write_text("shared = True\n", encoding="utf-8")
    os.symlink("shared.py", root / "alias.py")
    _commit(root)

    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    (workspace.root / "kernel.py").write_text("value = 2\n", encoding="utf-8")

    changed, baseline_digest, candidate_digest = workspace.freeze()

    assert changed == ("kernel.py",)
    assert baseline_digest != candidate_digest
    assert (workspace.root / "alias.py").is_symlink()


def test_e2e_worker_freezes_source_at_exact_turn_boundary(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    create_ignored_artifact=True,
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert candidate.succeeded
    assert candidate.reason_code == "candidate_frozen"
    assert candidate.changed_files == ("kernel.py",)
    assert candidate.candidate_source_sha256 is not None
    assert candidate.agent_result.candidate_capture_allowed
    assert not (candidate.workspace / "__pycache__").exists()


@pytest.mark.parametrize(
    ("kind", "capture", "reason"),
    (
        (
            AgentTerminationKind.TURN_OVERRUN,
            AgentCaptureStatus.COMPLETE,
            "agent_turn_budget_overrun",
        ),
        (
            AgentTerminationKind.TIMEOUT,
            AgentCaptureStatus.COMPLETE,
            "agent_timeout",
        ),
        (
            AgentTerminationKind.INVALID_STREAM,
            AgentCaptureStatus.COMPLETE,
            "agent_turn_stream_invalid",
        ),
        (
            AgentTerminationKind.EXACT_TURN_BOUNDARY,
            AgentCaptureStatus.OUTPUT_TRUNCATED,
            "agent_output_truncated",
        ),
        (
            AgentTerminationKind.EXACT_TURN_BOUNDARY,
            AgentCaptureStatus.CLEANUP_FAILED,
            "agent_process_cleanup_failed",
        ),
    ),
)
def test_e2e_worker_rejects_untrusted_boundary_capture(
    tmp_path: Path,
    kind: AgentTerminationKind,
    capture: AgentCaptureStatus,
    reason: str,
) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [_BoundaryAgent(kind, capture=capture)],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert not candidate.succeeded
    assert candidate.reason_code == reason
    assert candidate.candidate_id is None


def test_gitlink_content_is_never_an_editable_side_channel(tmp_path: Path) -> None:
    child = _repository(tmp_path, "child")
    (child / "child.py").write_text("value = 1\n", encoding="utf-8")
    _commit(child)
    child_commit = _git(child, "rev-parse", "HEAD")

    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root, "source")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{child_commit},vendor/sub")
    _git(root, "commit", "-m", "gitlink")
    (root / "vendor" / "sub").mkdir(parents=True)

    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    submodule = workspace.root / "vendor" / "sub"
    submodule.mkdir(parents=True, exist_ok=True)
    (submodule / "agent-created.py").write_text("unsafe = True\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        workspace.freeze()

    assert raised.value.reason_code == "undeclared_agent_edit"
    assert "vendor/sub" in raised.value.details["paths"]


def test_tracked_symlink_may_not_escape_checkout(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    os.symlink("../outside.py", root / "escape.py")
    _commit(root)

    with pytest.raises(IntegrityError) as raised:
        SourceCandidateWorkspace.create(
            _opportunity(root), destination=tmp_path / "candidate"
        )

    assert raised.value.reason_code == "workspace_symlink_escape"


def test_mutable_git_excludes_cannot_hide_agent_files(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    info = workspace.root / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "exclude").write_text("hidden.py\n", encoding="utf-8")
    (workspace.root / "hidden.py").write_text("tamper = True\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        workspace.freeze()

    assert raised.value.reason_code == "undeclared_agent_edit"
    assert "hidden.py" in raised.value.details["paths"]
