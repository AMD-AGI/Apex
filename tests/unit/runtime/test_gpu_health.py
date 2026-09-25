from __future__ import annotations

from pathlib import Path

from apex.runtime import (
    GpuDeviceIdentity,
    GpuOwnershipReceipt,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    RsmiDeviceIdentity,
    collect_rocm_health,
)
from apex.runtime.gpu_topology import capture_selector_request


class _Api:
    def init(self) -> int:
        return 0

    def shutdown(self) -> int:
        return 0

    def health(self, index: int) -> tuple[int, int, int, int, int, int]:
        assert index == 0
        return 0, 42_500, 1_700_000_000, 17, 1024, 4096


def _ownership() -> GpuOwnershipReceipt:
    unique_id = "GPU-0000000000000001"
    hsa = HsaInventoryEvidence(
        1,
        "clean_unfiltered_hsa_gpu_inventory_v1",
        "/trusted/helper.py",
        "b" * 64,
        "/opt/rocm/lib/libhsa-runtime64.so.1",
        "c" * 64,
        (HsaGpuIdentity(0, 2, 2, 100, 0, unique_id),),
    )
    monitor = RsmiDeviceIdentity(0, 2, 100, unique_id, 128)
    device = GpuDeviceIdentity(0, 2, 0, unique_id, "/dev/dri/renderD128")
    return GpuOwnershipReceipt(
        2,
        "clean_hsa_kfd_rsmi_process_gpu_map_v2",
        capture_selector_request("amd-gpu-set=0", environment={}),
        123,
        "/opt/rocm/lib/librocm_smi64.so.7",
        "a" * 64,
        "/sys/class/kfd/kfd/topology/nodes",
        hsa,
        (monitor,),
        (device,),
        (device,),
        (),
        (),
    )


def test_rocm_health_receipt_binds_selected_uuid_and_ownership() -> None:
    ownership = _ownership()

    receipt = collect_rocm_health(
        _Api(),
        ownership=ownership,
        library=Path(ownership.library_path),
        library_sha256=ownership.library_sha256,
        observed_unix_ns=456,
    )

    device = receipt.devices[0]
    assert device.unique_id == "GPU-0000000000000001"
    assert device.temperature_c == 42.5
    assert device.clock_mhz == 1700.0
    assert device.busy_percent == 17
    assert receipt.ownership_receipt_sha256 == ownership.digest
    assert len(receipt.digest) == 64
