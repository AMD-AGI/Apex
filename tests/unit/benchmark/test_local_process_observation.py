from __future__ import annotations

import os
from pathlib import Path

import pytest

from apex.benchmark.local_process_observation import (
    ProcfsLocalProcessObservationClient,
    descendant_closure,
    matching_processes,
    same_process,
)


def _proc_process(
    root: Path,
    pid: int,
    *,
    ppid: int,
    start: int,
    argv: tuple[str, ...],
    cwd: Path,
    cgroup: str = "0::/apex.slice\n",
) -> None:
    process = root / str(pid)
    process.mkdir(parents=True)
    tail = ["S", str(ppid), str(pid), str(pid)] + ["0"] * 15 + [str(start)]
    (process / "stat").write_text(
        f"{pid} (python worker) {' '.join(tail)}\n", encoding="utf-8"
    )
    (process / "cmdline").write_bytes(
        b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    )
    (process / "cgroup").write_text(cgroup, encoding="utf-8")
    (process / "cwd").symlink_to(cwd, target_is_directory=True)


def test_procfs_identity_binds_exact_argv_cwd_and_descendants(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    cwd = tmp_path / "magpie"
    cwd.mkdir()
    argv = ("/usr/bin/python3", "-m", "Magpie", "benchmark")
    _proc_process(proc, 101, ppid=1, start=500, argv=argv, cwd=cwd)
    _proc_process(proc, 102, ppid=101, start=501, argv=("worker",), cwd=cwd)

    client = ProcfsLocalProcessObservationClient(proc_root=proc)
    processes = client.snapshot()
    matches = matching_processes(processes, argv=argv, cwd=cwd)

    assert tuple(item.pid for item in matches) == (101,)
    assert tuple(item.pid for item in descendant_closure(processes, matches)) == (
        101,
        102,
    )
    assert matches[0].start_time_ticks == 500
    assert matches[0].cgroup_lines == ("0::/apex.slice",)


def test_pid_reuse_is_not_the_same_process(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    cwd = tmp_path / "magpie"
    cwd.mkdir()
    _proc_process(proc, 101, ppid=1, start=500, argv=("python",), cwd=cwd)
    client = ProcfsLocalProcessObservationClient(proc_root=proc)
    original = client.process(101)
    assert original is not None

    (proc / "101" / "stat").write_text(
        "101 (python worker) "
        + " ".join(["S", "1", "101", "101"] + ["0"] * 15 + ["900"])
        + "\n",
        encoding="utf-8",
    )
    replacement = client.process(101)

    assert replacement is not None
    assert same_process(original, replacement) is False


def test_process_disappearance_is_reported_as_absent(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    client = ProcfsLocalProcessObservationClient(proc_root=proc)

    assert client.process(os.getpid()) is None


def test_procfs_identity_keeps_container_facts_when_cwd_is_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    cwd = tmp_path / "container"
    cwd.mkdir()
    _proc_process(
        proc, 101, ppid=1, start=500, argv=("python", "server.py"), cwd=cwd
    )
    original = os.readlink

    def hidden(path: os.PathLike[str] | str) -> str:
        if Path(path) == proc / "101/cwd":
            raise PermissionError("procfs cwd is hidden")
        return original(path)

    monkeypatch.setattr(os, "readlink", hidden)
    client = ProcfsLocalProcessObservationClient(proc_root=proc)

    identity = client.process(101)

    assert identity is not None
    assert identity.cwd is None
    assert identity.cgroup_lines == ("0::/apex.slice",)
    assert matching_processes(
        (identity,), argv=("python", "server.py"), cwd=cwd
    ) == ()
