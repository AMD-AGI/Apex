from __future__ import annotations

import hashlib
import os
import json
import sys
import time
from pathlib import Path

from apex.core import AgentBackendName
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


def test_stream_budget_kills_process_group_before_timeout(tmp_path: Path) -> None:
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
    )

    child_pid = int(json.loads(result.stdout.splitlines()[0])["pid"])
    for _ in range(20):
        if not _pid_exists(child_pid):
            break
        time.sleep(0.05)
    assert result.observer_stopped
    assert result.observer_suspend_sent
    assert result.suspension_verified
    assert result.cleanup_succeeded
    assert not result.timed_out
    assert result.duration_seconds < 5
    assert not _pid_exists(child_pid)


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
    )

    parsed = parse_agent_output(result.stdout)
    discarded = "".join(lines[1:]).encode()
    assert result.stdout == lines[0]
    assert result.observer_suspend_sent
    assert result.suspension_verified
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


def test_verified_boundary_suspension_prevents_late_workspace_write(tmp_path: Path) -> None:
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
    )

    assert result.observer_suspend_sent
    assert result.suspension_verified
    assert result.observer_termination_started
    assert result.cleanup_succeeded
    assert not marker.exists()
