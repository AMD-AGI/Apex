from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError, sha256_json
from apex.runtime import (
    GpuDeviceIdentity,
    GpuLeaseOwnerIdentity,
    GpuOwnershipReceipt,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    LocalGpuLeaseManager,
    RsmiDeviceIdentity,
    resolve_gpu_device_scope,
    require_gpu_measurement_guard,
    require_gpu_lease_heartbeat,
)
from apex.runtime.gpu_topology import capture_selector_request, resolve_gpu_selection
from tests.support.gpu_evidence import StaticGpuDoctorInspector


_DEVICES = (
    GpuDeviceIdentity(0, 2, 1, "GPU-0000000000000001", "/dev/dri/renderD128"),
    GpuDeviceIdentity(1, 3, 0, "GPU-0000000000000002", "/dev/dri/renderD129"),
)
_HSA = HsaInventoryEvidence(
    1,
    "clean_unfiltered_hsa_gpu_inventory_v1",
    "/trusted/helper.py",
    "b" * 64,
    "/opt/rocm/lib/libhsa-runtime64.so.1",
    "c" * 64,
    (
        HsaGpuIdentity(0, 2, 2, 100, 0, "GPU-0000000000000001"),
        HsaGpuIdentity(1, 3, 3, 200, 0, "GPU-0000000000000002"),
    ),
)
_RSMI = (
    RsmiDeviceIdentity(0, 3, 200, "GPU-0000000000000002", 129),
    RsmiDeviceIdentity(1, 2, 100, "GPU-0000000000000001", 128),
)


class _FakeOwnershipInspector:
    def __init__(self, *, reverse_devices: bool = False) -> None:
        self.reverse_devices = reverse_devices

    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        request = capture_selector_request(selector_scope)
        selected = resolve_gpu_selection(_DEVICES, request)
        if self.reverse_devices:
            selected = tuple(reversed(selected))
            request = capture_selector_request(
                "amd-gpu-set=" + ",".join(item.unique_id for item in selected),
                environment={},
            )
        return GpuOwnershipReceipt(
            schema_version=2,
            policy_id="clean_hsa_kfd_rsmi_process_gpu_map_v2",
            selector_inputs=request,
            observed_unix_ns=123,
            library_path="/opt/rocm/lib/librocm_smi64.so.7",
            library_sha256="a" * 64,
            topology_root="/sys/class/kfd/kfd/topology/nodes",
            hsa_inventory=_HSA,
            rsmi_monitor_inventory=_RSMI,
            device_inventory=_DEVICES,
            selected_devices=selected,
            allowed_owners=(),
            foreign_owners=(),
        )


def _manager(
    tmp_path: Path, inspector: _FakeOwnershipInspector | None = None
) -> LocalGpuLeaseManager:
    return LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        doctor_inspector=StaticGpuDoctorInspector(
            inspector or _FakeOwnershipInspector()
        ),
    )


def _clear_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_gpu_lease_fails_fast_on_contention_and_releases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
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


def test_execution_order_and_physical_lock_set_are_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    manager = _manager(tmp_path)

    with manager.acquire("run", requested_devices="1,0") as lease:
        assert lease.receipt.execution_scope == (
            "amd-gpu-set=GPU-0000000000000002,GPU-0000000000000001"
        )
        assert lease.receipt.physical_scope == (
            "amd-gpu-unique-id-set=GPU-0000000000000001,"
            "GPU-0000000000000002"
        )


def test_overlapping_physical_gpu_sets_contend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    manager = _manager(tmp_path)

    with manager.acquire("wide", requested_devices="0,1"):
        with pytest.raises(ContractError) as raised:
            manager.acquire("overlap", requested_devices="1").__enter__()

    assert raised.value.reason_code == "gpu_lease_busy"
    assert raised.value.details["physical_unique_id"] == "GPU-0000000000000002"


def test_all_visible_scope_contends_with_explicit_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    manager = _manager(tmp_path)

    with manager.acquire("all-visible"):
        with pytest.raises(ContractError) as raised:
            manager.acquire("explicit", requested_devices="0").__enter__()

    assert raised.value.reason_code == "gpu_lease_busy"


def test_partial_multi_gpu_acquisition_releases_earlier_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
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
    _clear_visibility(monkeypatch)
    manager = _manager(tmp_path)
    unique_ids = ("GPU-0000000000000001", "GPU-0000000000000002")

    with manager.acquire("ordered", requested_devices="1,0") as lease:
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
    tmp_path: Path,
    rocr: str,
    hip: str,
    requested: str,
) -> None:
    _clear_visibility(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", rocr)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", hip)

    with pytest.raises(ContractError) as raised:
        _manager(tmp_path).acquire("run", requested_devices=requested)

    assert raised.value.reason_code in {
        "gpu_visibility_mismatch",
        "invalid_gpu_device_scope",
        "gpu_physical_mapping_unresolved",
    }


def test_selector_validation_preserves_order() -> None:
    assert resolve_gpu_device_scope("1,0") == "amd-gpu-set=1,0"


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _owner(start: int = 77) -> GpuLeaseOwnerIdentity:
    return GpuLeaseOwnerIdentity(123, 1000, start, "d" * 64)


def test_gpu_lease_heartbeat_and_measurement_bracket_are_identity_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    clock = _Clock()
    manager = LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        doctor_inspector=StaticGpuDoctorInspector(_FakeOwnershipInspector()),
        ttl_seconds=30.0,
        clock=clock,
        owner_identity_provider=_owner,
    )

    with manager.acquire("run") as lease:
        acquired = lease.heartbeat("manual")
        clock.value = 110.0
        with lease.measurement("formal-measurement") as guard:
            clock.value = 120.0
        bracket = guard.receipt

    assert acquired.sequence == 2
    assert bracket.pre.sequence == 3
    assert bracket.post.sequence == 4
    assert bracket.action_id == "formal-measurement"
    assert bracket.lease_digest == lease.receipt.digest
    assert bracket.pre.owner == bracket.post.owner == _owner()
    assert bracket.finished_unix_seconds <= bracket.pre.valid_until_unix_seconds


def test_gpu_lease_expiry_cannot_be_renewed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    clock = _Clock()
    manager = LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        doctor_inspector=StaticGpuDoctorInspector(_FakeOwnershipInspector()),
        ttl_seconds=5.0,
        clock=clock,
        owner_identity_provider=_owner,
    )

    with manager.acquire("run") as lease:
        clock.value = 106.0
        with pytest.raises(ContractError) as raised:
            lease.heartbeat()

    assert raised.value.reason_code == "gpu_lease_expired"


def test_gpu_lease_owner_pid_reuse_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    current = [_owner()]
    manager = LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        doctor_inspector=StaticGpuDoctorInspector(_FakeOwnershipInspector()),
        owner_identity_provider=lambda: current[0],
    )

    with manager.acquire("run") as lease:
        current[0] = _owner(start=78)
        with pytest.raises(ContractError) as raised:
            lease.heartbeat()

    assert raised.value.reason_code == "gpu_lease_owner_changed"


def test_gpu_lease_device_drift_fails_before_post_measurement_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_visibility(monkeypatch)
    inspector = _FakeOwnershipInspector()
    manager = LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        doctor_inspector=StaticGpuDoctorInspector(inspector),
        owner_identity_provider=_owner,
    )

    with manager.acquire("run") as lease:
        guard = lease.measurement("formal-measurement")
        guard.__enter__()
        inspector.reverse_devices = True
        with pytest.raises(ContractError) as raised:
            guard.__exit__(None, None, None)

    assert raised.value.reason_code in {
        "gpu_physical_mapping_changed",
        "gpu_lease_device_identity_changed",
    }
    assert not hasattr(guard, "receipt")


def test_formal_measurement_rejects_legacy_lease_without_lifecycle() -> None:
    with pytest.raises(ContractError) as raised:
        require_gpu_measurement_guard(object(), "measurement")

    assert raised.value.reason_code == "gpu_lease_lifecycle_unavailable"

    with pytest.raises(ContractError) as heartbeat:
        require_gpu_lease_heartbeat(object())

    assert heartbeat.value.reason_code == "gpu_lease_lifecycle_unavailable"
