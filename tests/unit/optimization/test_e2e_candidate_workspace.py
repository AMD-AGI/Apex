from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from apex.core import AgentBackendName, IntegrityError
from apex.execution import AgentRegistry
from apex.optimization.e2e import candidate as candidate_module
from apex.optimization.e2e import candidate_fingerprint
from apex.optimization.e2e.candidate import (
    AgentCandidateWorker,
    E2ECandidateRequest,
    SourceCandidateWorkspace,
    materialize_frozen_sources,
)
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.optimization.e2e.run_record import E2ERunRecord
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentInvocationReceipt,
    AgentProcessContainmentReceipt,
    AgentRequest,
    AgentResult,
    AgentTerminationKind,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)


def _containment() -> AgentProcessContainmentReceipt:
    return AgentProcessContainmentReceipt(
        policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="b" * 64,
        namespace_init_host_pid=100,
        namespace_init_starttime=200,
        namespace_init_inner_pid=1,
        pid_namespace_inode=300,
        mount_namespace_inode=301,
        ipc_namespace_inode=302,
        user_namespace_inode=303,
        private_procfs_verified=True,
        pidfd_opened=True,
        termination_reason="stdout_budget_boundary",
        teardown_mode="pidfd_sigkill",
        pidfd_sigkill_sent=True,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=True,
        terminal_status_absent_after_sigkill=False,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
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
        extra_edit: str | None = None,
        replace_source_with: Path | None = None,
        oversize_source: bool = False,
        oversize_untracked: bool = False,
        fsmonitor_path: Path | None = None,
        containment_verified: bool = True,
    ) -> None:
        self.kind = kind
        self.capture = capture
        self.make_change = make_change
        self.create_ignored_artifact = create_ignored_artifact
        self.extra_edit = extra_edit
        self.replace_source_with = replace_source_with
        self.oversize_source = oversize_source
        self.oversize_untracked = oversize_untracked
        self.fsmonitor_path = fsmonitor_path
        self.containment_verified = containment_verified

    def run(self, request: AgentRequest) -> AgentResult:
        if self.make_change:
            (request.workspace / "kernel.py").write_text("value = 2\n", encoding="utf-8")
        if self.create_ignored_artifact:
            cache = request.workspace / "__pycache__"
            cache.mkdir(exist_ok=True)
            (cache / "poison.pyc").write_bytes(b"untrusted-bytecode")
        if self.extra_edit is not None:
            (request.workspace / self.extra_edit).write_text(
                "tampered = True\n", encoding="utf-8"
            )
        if self.replace_source_with is not None:
            source = request.workspace / "kernel.py"
            source.unlink()
            os.symlink(self.replace_source_with, source)
        if self.oversize_source:
            with (request.workspace / "kernel.py").open("r+b") as source:
                source.truncate(17 * 1024 * 1024)
        if self.oversize_untracked:
            with (request.workspace / "agent-created.py").open("wb") as source:
                source.truncate(17 * 1024 * 1024)
        if self.fsmonitor_path is not None:
            with (request.workspace / ".git" / "config").open("a", encoding="utf-8") as config:
                config.write(f"\n[core]\n\tfsmonitor = {self.fsmonitor_path}\n")
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
            exit_code=137,
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
            process_containment=_containment() if self.containment_verified else None,
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
        process_containment_policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
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
    assert (candidate.workspace / "__pycache__" / "poison.pyc").exists()
    assert tuple(source.relative_path for source in candidate.frozen_sources) == (
        "kernel.py",
    )


def test_e2e_worker_returns_freeze_time_agent_tampering_as_candidate_failure(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    extra_edit="harness.py",
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert not candidate.succeeded
    assert candidate.candidate_id is None
    assert candidate.reason_code == "undeclared_agent_edit"
    assert candidate.changed_files == ("harness.py",)
    assert candidate.baseline_source_sha256
    assert candidate.candidate_source_sha256 is None
    assert candidate.agent_result.candidate_capture_allowed


def test_failed_symlink_candidate_never_captures_external_bytes(tmp_path: Path) -> None:
    secret = tmp_path / "outside-secret.py"
    secret.write_bytes(b"OUTSIDE-CONTENT")
    root = _repository(tmp_path, "source")
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    replace_source_with=secret,
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))
    assert not candidate.succeeded
    assert candidate.frozen_sources == ()
    assert (candidate.workspace / "kernel.py").is_symlink()

    record = E2ERunRecord.create(
        run_id="symlink-capture-run",
        root=tmp_path / "run",
        initial_anchor_id="anchor-symlink",
        dataset_split="validation",
        data_visibility="public",
    )
    manifest_receipt = record.record_candidate(candidate)
    manifest = json.loads(record.artifacts.read_bytes(manifest_receipt))
    assert manifest["source_receipts"] == []
    assert manifest["frozen_sources"] == []
    stored = tuple(
        path.read_bytes()
        for path in (tmp_path / "run" / "artifacts" / "sha256").rglob("*")
        if path.is_file()
    )
    assert secret.read_bytes() not in stored


def test_oversized_source_is_a_bounded_candidate_rejection(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    oversize_source=True,
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert not candidate.succeeded
    assert candidate.candidate_id is None
    assert candidate.reason_code == "candidate_source_too_large"
    assert candidate.frozen_sources == ()


def test_non_git_oversized_untracked_file_is_rejected_before_hashing(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    oversize_untracked=True,
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert not candidate.succeeded
    assert candidate.reason_code == "candidate_source_too_large"
    assert candidate.changed_files == ()
    assert candidate.frozen_sources == ()


def test_workspace_entry_budget_stops_before_fingerprinting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    for index in range(4):
        (workspace.root / f"empty-{index}.txt").touch()
    monkeypatch.setattr(candidate_fingerprint, "MAX_WORKSPACE_ENTRIES", 3)

    def forbidden_fingerprint(_root: Path):
        raise AssertionError("fingerprint ran before the entry preflight")

    monkeypatch.setattr(candidate_module, "_fingerprint_tree", forbidden_fingerprint)
    with pytest.raises(IntegrityError) as raised:
        workspace.freeze()

    assert raised.value.reason_code == "candidate_workspace_too_large"
    assert raised.value.details["dimension"] == "entries"


def test_workspace_depth_budget_stops_before_fingerprinting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    (workspace.root / "one" / "two" / "three").mkdir(parents=True)
    monkeypatch.setattr(candidate_fingerprint, "MAX_WORKSPACE_DEPTH", 2)

    def forbidden_fingerprint(_root: Path):
        raise AssertionError("fingerprint ran before the depth preflight")

    monkeypatch.setattr(candidate_module, "_fingerprint_tree", forbidden_fingerprint)
    with pytest.raises(IntegrityError) as raised:
        workspace.freeze()

    assert raised.value.reason_code == "candidate_workspace_too_large"
    assert raised.value.details["dimension"] == "depth"


def test_ignored_cache_is_pruned_without_enumerating_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    cache = workspace.root / "__pycache__"
    cache.mkdir()
    for index in range(20):
        (cache / f"ignored-{index}.pyc").touch()
    (workspace.root / "kernel.py").write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(candidate_fingerprint, "MAX_WORKSPACE_ENTRIES", 3)

    changed, _baseline, candidate_digest = workspace.freeze()

    assert changed == ("kernel.py",)
    assert candidate_digest is not None
    assert len(tuple(cache.iterdir())) == 20


def test_git_unexpected_scan_prunes_ignored_cache_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    tracked_cache = root / "__pycache__" / "tracked.pyc"
    tracked_cache.parent.mkdir()
    tracked_cache.write_bytes(b"baseline-cache")
    _git(root, "add", "-f", "__pycache__/tracked.pyc")
    _commit(root)
    workspace = SourceCandidateWorkspace.create(
        _opportunity(root), destination=tmp_path / "candidate"
    )
    cache = workspace.root / "__pycache__"
    assert cache.is_dir()
    for index in range(20):
        (cache / f"ignored-{index}.pyc").touch()
    with (cache / "tracked.pyc").open("r+b") as ignored:
        ignored.truncate(17 * 1024 * 1024)
    (workspace.root / "kernel.py").write_text("value = 2\n", encoding="utf-8")
    # .git, the two tracked files, and the cache root are the only observed
    # entries. Enumerating even one cache child would exceed this budget.
    monkeypatch.setattr(candidate_fingerprint, "MAX_WORKSPACE_ENTRIES", 4)

    changed, _baseline, candidate_digest = workspace.freeze()

    assert changed == ("kernel.py",)
    assert candidate_digest is not None
    assert len(tuple(cache.iterdir())) == 21


def test_post_agent_freeze_never_executes_mutable_git_fsmonitor(tmp_path: Path) -> None:
    marker = tmp_path / "fsmonitor-ran"
    monitor = tmp_path / "malicious-fsmonitor.sh"
    monitor.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n", encoding="utf-8")
    monitor.chmod(0o700)
    root = _repository(tmp_path, "source")
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    fsmonitor_path=monitor,
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert candidate.succeeded
    assert candidate.frozen_sources
    assert not marker.exists()


def test_frozen_snapshot_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    root = _repository(tmp_path, "source")
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [_BoundaryAgent(AgentTerminationKind.EXACT_TURN_BOUNDARY)],
            default=AgentBackendName.CODEX,
        )
    )
    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))
    real_parent = tmp_path / "evaluator-owned"
    real_parent.mkdir()
    symlinked_parent = tmp_path / "parent-link"
    symlinked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(IntegrityError) as raised:
        materialize_frozen_sources(candidate, symlinked_parent / "snapshot")

    assert raised.value.reason_code == "candidate_snapshot_destination_unsafe"
    assert not (real_parent / "snapshot").exists()


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


@pytest.mark.parametrize(
    ("kind", "capture"),
    (
        (AgentTerminationKind.TIMEOUT, AgentCaptureStatus.COMPLETE),
        (AgentTerminationKind.EXACT_TURN_BOUNDARY, AgentCaptureStatus.OUTPUT_TRUNCATED),
    ),
)
def test_untrusted_agent_result_never_reads_post_agent_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: AgentTerminationKind,
    capture: AgentCaptureStatus,
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

    def forbidden_freeze(_workspace: SourceCandidateWorkspace):
        raise AssertionError("untrusted post-agent workspace was traversed")

    monkeypatch.setattr(SourceCandidateWorkspace, "freeze", forbidden_freeze)
    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert not candidate.succeeded
    assert candidate.changed_files == ()
    assert candidate.candidate_source_sha256 is None
    assert candidate.frozen_sources == ()


@pytest.mark.parametrize(
    ("capture", "containment_verified", "reason"),
    (
        (
            AgentCaptureStatus.CLEANUP_FAILED,
            True,
            "agent_process_cleanup_failed",
        ),
        (
            AgentCaptureStatus.COMPLETE,
            False,
            "agent_process_containment_unverified",
        ),
    ),
)
def test_unverified_agent_teardown_returns_source_empty_result_without_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture: AgentCaptureStatus,
    containment_verified: bool,
    reason: str,
) -> None:
    root = _repository(tmp_path)
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    _commit(root)
    worker = AgentCandidateWorker(
        AgentRegistry(
            [
                _BoundaryAgent(
                    AgentTerminationKind.EXACT_TURN_BOUNDARY,
                    capture=capture,
                    containment_verified=containment_verified,
                )
            ],
            default=AgentBackendName.CODEX,
        )
    )

    def forbidden_freeze(_workspace: SourceCandidateWorkspace):
        raise AssertionError("unverified process workspace was traversed")

    monkeypatch.setattr(SourceCandidateWorkspace, "freeze", forbidden_freeze)
    candidate = worker.generate(_candidate_request(root, tmp_path / "candidate"))

    assert not candidate.succeeded
    assert candidate.reason_code == reason
    assert candidate.changed_files == ()
    assert candidate.candidate_source_sha256 is None
    assert candidate.frozen_sources == ()


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
