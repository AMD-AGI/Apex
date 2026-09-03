from __future__ import annotations

import hashlib
import os
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from apex.core import AgentBackendName, ContractError, DependencyError
from apex.execution import SubprocessSupervisor, agent_transcript_document
from apex.execution.transcript import parse_agent_output
from apex.ports import AgentResult, AgentTerminationKind


def _pid_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def test_supervisor_drains_both_pipes_and_bounds_output(tmp_path: Path) -> None:
    script = (
        "import sys; "
        "sys.stderr.write('e' * 1000000); sys.stderr.flush(); "
        "sys.stdout.write('o' * 1000000); sys.stdout.flush()"
    )

    result = SubprocessSupervisor(max_output_bytes=4096).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
    )

    assert result.exit_code == 0
    assert result.stdout_truncated and result.stderr_truncated
    assert len(result.stdout.encode()) < 5000
    assert len(result.stderr.encode()) < 5000


def test_stream_budget_requires_pid_namespace(tmp_path: Path) -> None:
    with pytest.raises(ContractError) as raised:
        SubprocessSupervisor().run(
            [sys.executable, "-c", "print('never')"],
            cwd=tmp_path,
            environment=os.environ,
            timeout_seconds=1,
            stdout_budget=lambda _: True,
        )

    assert raised.value.reason_code == "agent_process_containment_required"


def test_missing_bubblewrap_fails_before_agent_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "must-not-run"
    monkeypatch.setattr("apex.execution.containment.shutil.which", lambda _: None)

    with pytest.raises(DependencyError) as raised:
        SubprocessSupervisor().run(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ],
            cwd=tmp_path,
            environment=os.environ,
            timeout_seconds=1,
            require_pid_namespace=True,
        )

    assert raised.value.reason_code == "agent_process_containment_unavailable"
    assert not marker.exists()


def test_timeout_kills_process_group_descendants(tmp_path: Path) -> None:
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )

    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=1,
    )

    child_pid = int(result.stdout.splitlines()[0])
    for _ in range(20):
        if not _pid_exists(child_pid):
            break
        time.sleep(0.05)
    assert result.timed_out
    assert not _pid_exists(child_pid)


def test_normal_parent_exit_does_not_leave_pipe_holding_child(tmp_path: Path) -> None:
    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid, flush=True)"
    )

    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
    )

    child_pid = int(result.stdout.splitlines()[0])
    for _ in range(20):
        if not _pid_exists(child_pid):
            break
        time.sleep(0.05)
    assert result.exit_code == 0
    assert result.cleanup_succeeded
    assert not _pid_exists(child_pid)


def test_normal_parent_exit_reaps_same_group_child_with_closed_pipes(tmp_path: Path) -> None:
    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True)"
    )

    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
    )

    child_pid = int(result.stdout.splitlines()[0])
    for _ in range(20):
        if not _pid_exists(child_pid):
            break
        time.sleep(0.05)
    assert result.exit_code == 0
    assert result.cleanup_succeeded
    assert not _pid_exists(child_pid)


def test_stream_budget_destroys_pid_namespace_before_timeout(tmp_path: Path) -> None:
    script = (
        "import json,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(json.dumps({'kind':'child','pid':child.pid}),flush=True); "
        "print(json.dumps({'kind':'stop'}),flush=True); time.sleep(60)"
    )

    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
        stdout_budget=lambda line: '"stop"' in line,
        require_pid_namespace=True,
    )

    assert int(json.loads(result.stdout.splitlines()[0])["pid"]) > 0
    assert result.observer_stopped
    assert result.cleanup_succeeded
    assert result.process_containment is not None
    assert result.process_containment.namespace_empty_verified
    assert result.process_containment.pidfd_sigkill_sent
    assert result.process_containment.termination_reason == "stdout_budget_boundary"
    assert result.process_containment.teardown_mode == "pidfd_sigkill"
    assert not result.timed_out
    assert result.duration_seconds < 5


def test_wait_contained_preserves_observer_boundary_after_wrapper_exit() -> None:
    class ExitedProcess:
        @staticmethod
        def poll() -> int:
            return -9

    observer_stop = threading.Event()
    observer_stop.set()

    assert SubprocessSupervisor._wait_contained(
        ExitedProcess(),  # type: ignore[arg-type]
        observer_stop=observer_stop,
        timeout_seconds=1,
    ) == (False, True)


def test_natural_exit_waits_for_slow_stdout_boundary_decision(tmp_path: Path) -> None:
    def slow_boundary(_line: str) -> bool:
        time.sleep(0.15)
        return True

    result = SubprocessSupervisor(kill_grace_seconds=0.2).run(
        [sys.executable, "-c", "print('{}', flush=True)"],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=5,
        stdout_budget=slow_boundary,
        require_pid_namespace=True,
    )

    assert result.observer_stopped
    assert result.observer_termination_started
    assert result.cleanup_succeeded
    assert result.process_containment is not None
    assert result.process_containment.termination_reason == "stdout_budget_boundary"
    assert result.process_containment.namespace_empty_verified


def test_stream_boundary_excludes_and_digests_prebuffered_tail(tmp_path: Path) -> None:
    lines = (
        '{"type":"assistant_message","content":"turn-50"}\n',
        '{"type":"assistant_message","content":"turn-51"}\n',
        '{"type":"result","num_turns":51}\n',
    )
    payload = "".join(lines)
    script = (
        "import os,time; "
        f"os.write(1,{payload.encode()!r}); "
        "time.sleep(60)"
    )

    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
        stdout_budget=lambda line: '"turn-50"' in line,
        require_pid_namespace=True,
    )

    parsed = parse_agent_output(result.stdout)
    discarded = "".join(lines[1:]).encode()
    assert result.stdout == lines[0]
    assert result.process_containment is not None
    assert result.process_containment.namespace_empty_verified
    assert [event.text for event in parsed.semantic_events] == ["turn-50"]
    assert all("turn-51" not in event.text for event in parsed.semantic_events)
    assert result.discarded_stdout_lines == 2
    assert result.discarded_stdout_bytes == len(discarded)
    assert result.discarded_stdout_sha256 == hashlib.sha256(discarded).hexdigest()
    formal = agent_transcript_document(
        AgentResult(
            backend=AgentBackendName.CODEX,
            model=None,
            exit_code=result.exit_code,
            timed_out=False,
            events=parsed.events,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            semantic_events=parsed.semantic_events,
            termination_kind=AgentTerminationKind.PROCESS_FAILED,
            termination_reason="fixture_boundary_stop",
            discarded_stdout_lines=result.discarded_stdout_lines,
            discarded_stdout_bytes=result.discarded_stdout_bytes,
            discarded_stdout_sha256=result.discarded_stdout_sha256,
        )
    )
    assert len(formal["events"]) == 1
    assert formal["termination"]["discarded_stdout_tail"]["lines"] == 2


def test_verified_namespace_teardown_prevents_late_workspace_write(tmp_path: Path) -> None:
    marker = tmp_path / "late-edit.py"
    script = (
        "import json,time; from pathlib import Path; "
        "print(json.dumps({'type':'assistant_message','content':'turn-50'}),flush=True); "
        "time.sleep(0.2); "
        f"Path({str(marker)!r}).write_text('late = True')"
    )

    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
        stdout_budget=lambda line: '"turn-50"' in line,
        require_pid_namespace=True,
    )

    assert result.process_containment is not None
    assert result.process_containment.namespace_empty_verified
    assert result.observer_termination_started
    assert result.cleanup_succeeded
    assert not marker.exists()


def _detached_late_writer_script(marker: Path, *, emit_boundary: bool) -> str:
    return (
        "import os,signal,time; from pathlib import Path; "
        "read_fd,write_fd=os.pipe(); first=os.fork(); "
        "\nif first == 0:\n"
        " os.close(read_fd); os.setsid(); second=os.fork();\n"
        " if second > 0: os._exit(0)\n"
        " os.environ.clear(); signal.signal(signal.SIGTERM,signal.SIG_IGN);\n"
        " [os.close(fd) for fd in (0,1,2) if fd != write_fd];\n"
        " os.write(write_fd,b'1'); os.close(write_fd); time.sleep(0.35);\n"
        f" Path({str(marker)!r}).write_text('escaped'); os._exit(0)\n"
        "os.close(write_fd); os.read(read_fd,1); os.close(read_fd); "
        + (
            "print('{\"type\":\"assistant_message\",\"content\":\"turn-50\"}',flush=True); time.sleep(60)"
            if emit_boundary
            else "raise SystemExit(0)"
        )
    )


def test_pid_namespace_blocks_sets_id_double_fork_clearenv_late_write(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "escaped-after-boundary"
    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [
            sys.executable,
            "-c",
            _detached_late_writer_script(marker, emit_boundary=True),
        ],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
        stdout_budget=lambda line: '"turn-50"' in line,
        require_pid_namespace=True,
    )

    time.sleep(0.5)
    assert result.observer_stopped
    assert result.cleanup_succeeded
    assert result.process_containment is not None
    assert result.process_containment.pidfd_sigkill_sent
    assert result.process_containment.namespace_empty_verified
    assert result.process_containment.live_namespace_members_after == ()
    assert not marker.exists()


def test_natural_exit_pid_namespace_blocks_detached_late_write(tmp_path: Path) -> None:
    marker = tmp_path / "escaped-after-natural-exit"
    result = SubprocessSupervisor(kill_grace_seconds=0.1).run(
        [
            sys.executable,
            "-c",
            _detached_late_writer_script(marker, emit_boundary=False),
        ],
        cwd=tmp_path,
        environment=os.environ,
        timeout_seconds=10,
        require_pid_namespace=True,
    )

    time.sleep(0.5)
    assert result.exit_code == 0
    assert result.cleanup_succeeded
    assert result.process_containment is not None
    assert result.process_containment.teardown_mode == "natural_exit"
    assert result.process_containment.namespace_empty_verified
    assert not marker.exists()
