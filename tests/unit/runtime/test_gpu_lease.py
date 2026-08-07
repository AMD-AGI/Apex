from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.runtime import LocalGpuLeaseManager, resolve_gpu_device_scope


def test_gpu_lease_fails_fast_on_contention_and_releases(tmp_path: Path) -> None:
    manager = LocalGpuLeaseManager(lock_root=tmp_path / "leases")
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
    manager = LocalGpuLeaseManager(lock_root=tmp_path / "leases")

    with manager.acquire("run", requested_devices="0,1") as lease:
        assert lease.receipt.device_scope == "amd-gpu-set=0,1"


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
