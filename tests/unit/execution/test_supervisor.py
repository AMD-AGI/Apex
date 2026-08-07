from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

from apex.execution import SubprocessSupervisor


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
    assert result.budget_exceeded
    assert not result.timed_out
    assert result.duration_seconds < 5
    assert not _pid_exists(child_pid)
