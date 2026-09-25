from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.benchmark.local_gpu_observation import (
    LocalGpuLeaseAuthority,
    RocmLocalGpuEngagementObserver,
    validate_active_local_gpu_lease,
)
from apex.benchmark.local_process_observation import LocalProcessIdentity
from apex.core import ContractError, sha256_json
from apex.runtime import GpuProcessIdentity


_GPU_ID = "GPU-0000000000000001"


class FakeProcesses:
    def __init__(self, *processes: LocalProcessIdentity) -> None:
        self.current = {item.pid: item for item in processes}

    def snapshot(self):
        return tuple(self.current[pid] for pid in sorted(self.current))

    def process(self, pid):
        return self.current.get(pid)


class FakeInspector:
    def __init__(self, receipt) -> None:
        self.receipt = receipt

    def inspect(self, selector_scope, *, allowed_pids=()):
        assert selector_scope == "0"
        assert allowed_pids == ()
        return self.receipt


def _process(pid: int, ppid: int, start: int, *, cgroup: str = "7" * 64):
    return LocalProcessIdentity(
        pid,
        os.getuid(),
        ppid,
        pid,
        pid,
        start,
        f"{pid % 10}" * 64,
        (f"process-{pid}",),
        Path("/tmp").resolve(),
        cgroup,
        ("0::/apex.slice",),
    )


def _authority(owner: LocalProcessIdentity) -> LocalGpuLeaseAuthority:
    return LocalGpuLeaseAuthority(
        "run", "1" * 64, "0", ((0, _GPU_ID),), owner, "2" * 64, time.time() + 30
    )


def _ownership(process: LocalProcessIdentity | None):
    owners = (
        (
            GpuProcessIdentity(
                process.pid,
                process.uid,
                process.start_time_ticks,
                process.cmdline_sha256,
                (0,),
            ),
        )
        if process is not None
        else ()
    )
    return SimpleNamespace(
        selected_devices=(SimpleNamespace(rsmi_index=0, unique_id=_GPU_ID),),
        allowed_owners=(),
        foreign_owners=owners,
        digest="3" * 64,
    )


def test_gpu_owner_must_be_exact_descendant_in_lease_owner_cgroup() -> None:
    owner = _process(os.getpid(), 1, 100)
    root = _process(2001, owner.pid, 200)
    worker = _process(2002, root.pid, 201)
    observer = RocmLocalGpuEngagementObserver(
        FakeProcesses(owner, root, worker), FakeInspector(_ownership(worker))
    )

    evidence = observer.observe((root,), _authority(owner))

    assert evidence is not None
    assert evidence["processes"][0]["root_pid"] == root.pid
    assert evidence["processes"][0]["start_time_ticks"] == worker.start_time_ticks


def test_gpu_owner_in_other_cgroup_fails_closed() -> None:
    owner = _process(os.getpid(), 1, 100)
    root = _process(2001, owner.pid, 200)
    worker = _process(2002, root.pid, 201, cgroup="8" * 64)
    observer = RocmLocalGpuEngagementObserver(
        FakeProcesses(owner, root, worker), FakeInspector(_ownership(worker))
    )

    with pytest.raises(ContractError) as caught:
        observer.observe((root,), _authority(owner))

    assert caught.value.reason_code == "magpie_local_gpu_process_escape"


def test_cleanup_rejects_any_remaining_kfd_owner() -> None:
    owner = _process(os.getpid(), 1, 100)
    worker = _process(2002, owner.pid, 201)
    observer = RocmLocalGpuEngagementObserver(
        FakeProcesses(owner, worker), FakeInspector(_ownership(worker))
    )

    with pytest.raises(ContractError) as caught:
        observer.require_quiescent(_authority(owner))

    assert caught.value.reason_code == "magpie_local_gpu_residual_process"


def test_active_lease_requires_current_flock_and_exact_heartbeat(tmp_path: Path) -> None:
    owner = _process(os.getpid(), 1, 100)
    lock = tmp_path / "GPU-0000000000000001.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = {
        "schema_version": 3,
        "run_id": "run",
        "owner_pid": os.getpid(),
        "lock_path": str(lock),
        "lock_paths": [str(lock)],
        "ownership": {
            "selector_scope": "0",
            "selected_devices": [{"rsmi_index": 0, "unique_id": _GPU_ID}],
            "foreign_owners": [],
        },
    }
    heartbeat = {
        "schema_version": 1,
        "run_id": "run",
        "lease_digest": sha256_json(receipt),
        "owner": {
            "pid": owner.pid,
            "uid": owner.uid,
            "start_time_ticks": owner.start_time_ticks,
            "cmdline_sha256": owner.cmdline_sha256,
        },
        "valid_until_unix_seconds": time.time() + 30,
    }
    os.write(descriptor, json.dumps({**receipt, "heartbeat": heartbeat}).encode("utf-8"))
    os.fsync(descriptor)
    try:
        authority = validate_active_local_gpu_lease(
            receipt, run_id="run", processes=FakeProcesses(owner)
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert authority.owner == owner
    assert authority.devices == ((0, _GPU_ID),)
