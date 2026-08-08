from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.runtime import (
    GpuDeviceIdentity,
    GpuOwnershipReceipt,
    GpuProcessIdentity,
    LocalGpuLeaseManager,
    collect_gpu_ownership,
)


class _FakeApi:
    def __init__(
        self,
        snapshots: list[dict[int, tuple[int, ...]]] | None = None,
        *,
        init_status: int = 0,
        shutdown_status: int = 0,
        device_status: int = 0,
    ) -> None:
        self.snapshots = snapshots or [{}, {}]
        self.init_status = init_status
        self.shutdown_status = shutdown_status
        self.device_status = device_status
        self._snapshot: dict[int, tuple[int, ...]] = {}
        self._next = 0
        self.shutdown_calls = 0

    def init(self) -> int:
        return self.init_status

    def shutdown(self) -> int:
        self.shutdown_calls += 1
        return self.shutdown_status

    def device_count(self) -> tuple[int, int]:
        return self.device_status, 2

    def device_identity(self, index: int) -> tuple[int, int, int]:
        return 0, index + 1, 128 + index

    def process_pids(self) -> tuple[int, tuple[int, ...]]:
        self._snapshot = self.snapshots[min(self._next, len(self.snapshots) - 1)]
        self._next += 1
        return 0, tuple(self._snapshot)

    def process_devices(self, pid: int) -> tuple[int, tuple[int, ...]]:
        return 0, self._snapshot[pid]


def _proc_process(root: Path, pid: int, *, start: int = 777) -> None:
    process = root / str(pid)
    process.mkdir(parents=True)
    fields = ["S", *("0" for _ in range(18)), str(start)]
    (process / "stat").write_text(
        f"{pid} (gpu worker) " + " ".join(fields) + "\n",
        encoding="utf-8",
    )
    (process / "cmdline").write_bytes(b"worker\x00--token=redacted-by-digest\x00")


def _collect(
    api: _FakeApi,
    proc: Path,
    *,
    selector: str = "amd-gpu-set=0",
    allowed: tuple[int, ...] = (),
) -> GpuOwnershipReceipt:
    return collect_gpu_ownership(
        api,
        selector_scope=selector,
        allowed_pids=allowed,
        library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
        library_sha256="a" * 64,
        proc_root=proc,
        observed_unix_ns=123,
    )


def test_receipt_binds_selected_physical_identity_and_empty_inventory(
    tmp_path: Path,
) -> None:
    receipt = _collect(_FakeApi(), tmp_path / "proc")

    assert receipt.selected_devices == (
        GpuDeviceIdentity(0, "0x0000000000000001", "/dev/dri/renderD128"),
    )
    assert receipt.foreign_owners == ()
    assert receipt.physical_scope == "amd-gpu-unique-id-set=0x0000000000000001"
    assert len(receipt.digest) == 64


def test_kfd_owner_is_bound_to_exact_process_and_selected_gpu(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_process(proc, 4242)
    receipt = _collect(_FakeApi([{4242: (0,)}, {4242: (0,)}]), proc)

    assert len(receipt.foreign_owners) == 1
    owner = receipt.foreign_owners[0]
    assert owner.pid == 4242
    assert owner.start_time_ticks == 777
    assert owner.device_indices == (0,)
    assert "token" not in str(owner)


def test_current_runner_may_be_recorded_as_an_allowed_owner(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_process(proc, 4242)
    receipt = _collect(
        _FakeApi([{4242: (0,)}, {4242: (0,)}]),
        proc,
        allowed=(4242,),
    )

    assert receipt.foreign_owners == ()
    assert receipt.allowed_owners[0].pid == 4242


def test_ownership_change_during_query_fails_closed(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_process(proc, 4242)
    with pytest.raises(ContractError) as raised:
        _collect(_FakeApi([{}, {4242: (0,)}]), proc)

    assert raised.value.reason_code == "gpu_ownership_race"


@pytest.mark.parametrize(
    ("api", "reason"),
    [
        (_FakeApi(init_status=4), "gpu_ownership_query_failed"),
        (_FakeApi(device_status=4), "gpu_physical_mapping_unresolved"),
        (_FakeApi(shutdown_status=4), "gpu_ownership_query_failed"),
    ],
)
def test_every_api_failure_is_fail_closed(
    tmp_path: Path, api: _FakeApi, reason: str
) -> None:
    with pytest.raises(ContractError) as raised:
        _collect(api, tmp_path / "proc")

    assert raised.value.reason_code == reason


class _ForeignInspector:
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        owner = GpuProcessIdentity(4242, 1000, 777, "b" * 64, (0,))
        return GpuOwnershipReceipt(
            1,
            "rocm_smi_process_gpu_map_v1",
            selector_scope,
            123,
            "/opt/rocm/lib/librocm_smi64.so.7",
            "a" * 64,
            (GpuDeviceIdentity(0, "0x0000000000000001", "/dev/dri/renderD128"),),
            (),
            (owner,),
        )


def test_lease_refuses_foreign_owner_without_terminating_it(tmp_path: Path) -> None:
    manager = LocalGpuLeaseManager(
        lock_root=tmp_path / "leases", ownership_inspector=_ForeignInspector()
    )

    with pytest.raises(ContractError) as raised:
        manager.acquire("run", requested_devices="0")

    assert raised.value.reason_code == "gpu_foreign_owner"
    assert raised.value.details["ownership_receipt"]["foreign_owners"][0]["pid"] == 4242
    assert not (tmp_path / "leases").exists()
