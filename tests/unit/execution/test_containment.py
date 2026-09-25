from __future__ import annotations

import json
import os
import threading
import time

import pytest

from apex.core import ContractError
from apex.execution import containment
from apex.execution import procfs


def _status_payload() -> bytes:
    return (
        json.dumps(
            {
                "child-pid": 41,
                "ipc-namespace": 42,
                "mnt-namespace": 43,
                "pid-namespace": 44,
            }
        ).encode()
        + b"\n"
    )


def test_launch_status_reads_fragmented_pipe_payload() -> None:
    read_fd, write_fd = os.pipe()
    payload = _status_payload()

    def writer() -> None:
        try:
            os.write(write_fd, payload[:37])
            time.sleep(0.02)
            os.write(write_fd, payload[37:])
        finally:
            os.close(write_fd)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert containment._read_launch_status(read_fd, 1) == json.loads(payload)
    finally:
        os.close(read_fd)
        thread.join(timeout=1)


def test_launch_status_rejects_bytes_after_newline() -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, _status_payload() + b"{}")
        with pytest.raises(ContractError) as raised:
            containment._read_launch_status(read_fd, 1)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert raised.value.reason_code == "agent_process_containment_failed"


def _launch_identity(*, starttime: int = 101) -> containment._LaunchIdentity:
    return containment._LaunchIdentity(
        starttime=starttime,
        parent_pid=41,
        inner_pid=1,
        pid_namespace_inode=201,
        mount_namespace_inode=202,
        ipc_namespace_inode=203,
        user_namespace_inode=204,
    )


def test_private_procfs_uses_visible_topmost_overmount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = "2362 2315 0:24 / /proc rw master:13 - proc proc rw"
    private = "2416 2362 0:136 / /proc rw - proc proc rw"
    monkeypatch.setattr(
        procfs.Path,
        "read_text",
        lambda self, **_: f"{inherited}\n{private}\n",
    )
    monkeypatch.setattr(procfs, "_private_proc_pid_one_verified", lambda _: True)
    host = procfs.ProcfsMountIdentity(26, 0, 24)

    assert procfs._private_procfs_identity_verified(
        99,
        procfs.ProcfsMountIdentity(2416, 0, 136),
        host,
    )
    assert not procfs._private_procfs_identity_verified(
        99,
        procfs.ProcfsMountIdentity(2362, 0, 24),
        host,
    )
    assert not procfs._private_procfs_identity_verified(
        99,
        procfs.ProcfsMountIdentity(26, 0, 136),
        host,
    )


def test_private_procfs_rejects_propagating_visible_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        procfs.Path,
        "read_text",
        lambda self, **_: "2416 2362 0:136 / /proc rw shared:9 - proc proc rw\n",
    )
    monkeypatch.setattr(procfs, "_private_proc_pid_one_verified", lambda _: True)

    assert not procfs._private_procfs_identity_verified(
        99,
        procfs.ProcfsMountIdentity(2416, 0, 136),
        procfs.ProcfsMountIdentity(26, 0, 24),
    )


def test_launch_readiness_rechecks_identity_until_private_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _launch_identity()
    observations = iter((False, False, True))
    calls = 0

    def observe(_: int) -> containment._LaunchIdentity:
        nonlocal calls
        calls += 1
        return identity

    monkeypatch.setattr(containment, "_launch_identity", observe)
    monkeypatch.setattr(containment, "_pidfd_ready", lambda *_: False)
    monkeypatch.setattr(
        containment,
        "private_procfs_verified",
        lambda *_: next(observations),
    )
    monkeypatch.setattr(containment, "_namespace_inode", lambda *_: 999)
    monkeypatch.setattr(containment, "_READINESS_POLL_SECONDS", 0.0)

    result = containment._await_verified_launch_identity(
        55,
        wrapper_pid=41,
        pidfd=7,
        status={
            "child-pid": 55,
            "pid-namespace": 201,
            "mnt-namespace": 202,
            "ipc-namespace": 203,
        },
        supervisor_procfs=procfs.ProcfsMountIdentity(26, 0, 24),
        deadline=time.monotonic() + 1,
    )

    assert result == identity
    assert calls == 6


def test_establish_releases_gate_only_after_readiness_and_proc_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _launch_identity()
    order: list[str] = []
    prepared = containment.PreparedPidNamespace(
        argv=("bwrap",),
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="0" * 64,
        status_read_fd=-1,
        status_write_fd=-1,
        gate_read_fd=-1,
        gate_write_fd=-1,
    )

    class Wrapper:
        pid = 41

    monkeypatch.setattr(
        containment,
        "_read_launch_status",
        lambda *_: {
            "child-pid": 55,
            "pid-namespace": 201,
            "mnt-namespace": 202,
            "ipc-namespace": 203,
        },
    )
    monkeypatch.setattr(
        containment.os,
        "pidfd_open",
        lambda *_: os.open("/dev/null", os.O_RDONLY),
    )
    monkeypatch.setattr(
        containment,
        "procfs_mount_identity",
        lambda *_: procfs.ProcfsMountIdentity(26, 0, 24),
    )
    monkeypatch.setattr(
        containment,
        "_await_verified_launch_identity",
        lambda *_args, **_kwargs: order.append("ready") or identity,
    )
    monkeypatch.setattr(
        containment,
        "open_verified_private_procfs",
        lambda *_: order.append("retained") or os.open("/dev/null", os.O_RDONLY),
    )
    monkeypatch.setattr(containment, "_launch_identity", lambda _: identity)
    monkeypatch.setattr(containment, "_validate_launch_identity", lambda *_: None)
    monkeypatch.setattr(containment, "_pidfd_ready", lambda *_: False)
    monkeypatch.setattr(
        containment.os,
        "write",
        lambda *_: order.append("released") or 1,
    )

    boundary = containment.establish_pid_namespace(
        prepared,
        Wrapper(),  # type: ignore[arg-type]
    )
    try:
        assert order == ["ready", "retained", "released"]
    finally:
        boundary.close()


def test_launch_readiness_rejects_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter((_launch_identity(), _launch_identity(starttime=102)))
    monkeypatch.setattr(containment, "_launch_identity", lambda _: next(observations))
    monkeypatch.setattr(containment, "_pidfd_ready", lambda *_: False)
    monkeypatch.setattr(containment, "private_procfs_verified", lambda *_: True)
    monkeypatch.setattr(containment, "_namespace_inode", lambda *_: 999)

    with pytest.raises(ContractError) as raised:
        containment._await_verified_launch_identity(
            55,
            wrapper_pid=41,
            pidfd=7,
            status={
                "child-pid": 55,
                "pid-namespace": 201,
                "mnt-namespace": 202,
                "ipc-namespace": 203,
            },
            supervisor_procfs=procfs.ProcfsMountIdentity(26, 0, 24),
            deadline=time.monotonic() + 1,
        )

    assert raised.value.reason_code == "agent_process_containment_failed"


def test_private_proc_membership_permission_error_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(procfs.os, "listdir", lambda _: ["2"])
    monkeypatch.setattr(
        procfs.os,
        "open",
        lambda *_, **__: (_ for _ in ()).throw(PermissionError()),
    )

    assert procfs.live_namespace_members(8) is None


def test_private_proc_membership_list_error_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        procfs.os,
        "listdir",
        lambda _: (_ for _ in ()).throw(OSError()),
    )

    assert procfs.live_namespace_members(8) is None
