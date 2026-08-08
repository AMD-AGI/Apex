from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError, sha256_json
from apex.runtime import (
    GpuDeviceIdentity,
    GpuOwnershipReceipt,
    LocalGpuLeaseManager,
    resolve_gpu_device_scope,
)


class _FakeOwnershipInspector:
    def __init__(self, *, reverse_devices: bool = False) -> None:
        self.reverse_devices = reverse_devices

    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        devices = (
            GpuDeviceIdentity(0, "0x0000000000000001", "/dev/dri/renderD128"),
            GpuDeviceIdentity(1, "0x0000000000000002", "/dev/dri/renderD129"),
        )
        if selector_scope == "all-visible-amd-gpus":
            selected = devices
        else:
            selectors = selector_scope.removeprefix("amd-gpu-set=").split(",")
            selected = tuple(
                device
                for selector in selectors
                for device in devices
                if selector == str(device.index)
                or selector.lower().removeprefix("gpu-").removeprefix("0x")
                == device.unique_id.removeprefix("0x")
            )
        if self.reverse_devices:
            selected = tuple(reversed(selected))
        return GpuOwnershipReceipt(
            1,
            "rocm_smi_process_gpu_map_v1",
            selector_scope,
            123,
            "/opt/rocm/lib/librocm_smi64.so.7",
            "a" * 64,
            selected,
            (),
            (),
        )


def _manager(
    tmp_path: Path, inspector: _FakeOwnershipInspector | None = None
) -> LocalGpuLeaseManager:
    return LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        ownership_inspector=inspector or _FakeOwnershipInspector(),
    )


def test_gpu_lease_fails_fast_on_contention_and_releases(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    first = manager.acquire("run-one")

    with first:
        assert first.receipt.acquired_unix_seconds > 0
        with pytest.raises(ContractError) as raised:
            manager.acquire("run-two").__enter__()
        assert raised.value.reason_code == "gpu_lease_busy"
        assert raised.value.details["owner"]["run_id"] == "run-one"

    with manager.acquire("run-three") as successor:
        assert successor.receipt.run_id == "run-three"


def test_explicit_gpu_scope_is_bound_to_ambient_visibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    manager = _manager(tmp_path)

    with manager.acquire("run", requested_devices="0,1") as lease:
        assert lease.receipt.device_scope == (
            "amd-gpu-unique-id-set=0x0000000000000001,0x0000000000000002"
        )
        assert lease.receipt.ownership.selector_scope == "amd-gpu-set=0,1"


def test_overlapping_physical_gpu_sets_contend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    manager = _manager(tmp_path)

    with manager.acquire("wide", requested_devices="0,1"):
        with pytest.raises(ContractError) as raised:
            manager.acquire("overlap", requested_devices="1").__enter__()

    assert raised.value.reason_code == "gpu_lease_busy"
    assert raised.value.details["physical_unique_id"] == "0x0000000000000002"
    assert raised.value.details["owner"]["run_id"] == "wide"


def test_all_visible_scope_contends_with_explicit_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    manager = _manager(tmp_path)

    with manager.acquire("all-visible"):
        with pytest.raises(ContractError) as raised:
            manager.acquire("explicit", requested_devices="0").__enter__()

    assert raised.value.reason_code == "gpu_lease_busy"
    assert raised.value.details["owner"]["run_id"] == "all-visible"


def test_partial_multi_gpu_acquisition_releases_earlier_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    manager = _manager(tmp_path)

    with manager.acquire("gpu-one", requested_devices="1"):
        with pytest.raises(ContractError) as raised:
            manager.acquire("wide", requested_devices="0,1").__enter__()
        with manager.acquire("gpu-zero", requested_devices="0") as successor:
            assert successor.receipt.run_id == "gpu-zero"

    assert raised.value.reason_code == "gpu_lease_busy"


def test_physical_locks_are_acquired_in_sorted_unique_id_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    manager = _manager(tmp_path, _FakeOwnershipInspector(reverse_devices=True))
    unique_ids = ("0x0000000000000001", "0x0000000000000002")

    with manager.acquire("ordered", requested_devices="0,1") as lease:
        expected = tuple(
            str(
                (
                    tmp_path
                    / "leases"
                    / f"gpu-{sha256_json({'physical_unique_id': unique_id})[:24]}.lock"
                ).resolve()
            )
            for unique_id in unique_ids
        )
        assert lease.receipt.lock_paths == expected
        assert lease.receipt.lock_path == expected[0]


@pytest.mark.parametrize(
    ("rocr", "hip", "requested"),
    [
        ("0", "1", "0"),
        ("0", "0", "1"),
        ("0,0", "0", "0"),
    ],
)
def test_gpu_scope_rejects_split_brain_or_invalid_visibility(
    monkeypatch: pytest.MonkeyPatch,
    rocr: str,
    hip: str,
    requested: str,
) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", rocr)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", hip)

    with pytest.raises(ContractError) as raised:
        resolve_gpu_device_scope(requested)

    assert raised.value.reason_code in {
        "gpu_visibility_mismatch",
        "invalid_gpu_device_scope",
    }
