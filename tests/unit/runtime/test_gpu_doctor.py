from __future__ import annotations

import os
import json
from pathlib import Path

from apex.runtime import (
    GpuDeviceIdentity,
    GpuOwnershipReceipt,
    GpuProcessIdentity,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    LinuxGpuDoctorInspector,
    RocmHealthDevice,
    RocmHealthReceipt,
    RsmiDeviceIdentity,
    load_gpu_doctor_receipt,
)
from apex.runtime.gpu_topology import capture_selector_request


_DEVICE = GpuDeviceIdentity(
    0, 2, 0, "GPU-0000000000000001", "/dev/dri/renderD128"
)
_HSA = HsaInventoryEvidence(
    1,
    "clean_unfiltered_hsa_gpu_inventory_v1",
    "/trusted/helper.py",
    "b" * 64,
    "/opt/rocm/lib/libhsa-runtime64.so.1",
    "c" * 64,
    (HsaGpuIdentity(0, 2, 2, 100, 0, _DEVICE.unique_id),),
)
_RSMI = (RsmiDeviceIdentity(0, 2, 100, _DEVICE.unique_id, 128),)


class _Ownership:
    def __init__(self, owner: GpuProcessIdentity | None) -> None:
        self._owner = owner

    def inspect(self, selector_scope: str, *, allowed_pids=()) -> GpuOwnershipReceipt:
        del allowed_pids
        return GpuOwnershipReceipt(
            schema_version=2,
            policy_id="clean_hsa_kfd_rsmi_process_gpu_map_v2",
            selector_inputs=capture_selector_request(selector_scope, environment={}),
            observed_unix_ns=123,
            library_path="/opt/rocm/lib/librocm_smi64.so.7",
            library_sha256="a" * 64,
            topology_root="/sys/class/kfd/kfd/topology/nodes",
            hsa_inventory=_HSA,
            rsmi_monitor_inventory=_RSMI,
            device_inventory=(_DEVICE,),
            selected_devices=(_DEVICE,),
            allowed_owners=(),
            foreign_owners=((self._owner,) if self._owner is not None else ()),
        )


class _Health:
    def inspect(self, ownership: GpuOwnershipReceipt) -> RocmHealthReceipt:
        return RocmHealthReceipt(
            observed_unix_ns=456,
            library_path=ownership.library_path,
            library_sha256=ownership.library_sha256,
            ownership_receipt_sha256=ownership.digest,
            devices=(
                RocmHealthDevice(
                    _DEVICE.unique_id,
                    _DEVICE.rsmi_index,
                    45.0,
                    1700.0,
                    0,
                    1,
                    1024,
                ),
            ),
        )


def _proc(
    root: Path,
    pid: int,
    *,
    start: int,
    comm: str,
    cgroup: str,
) -> None:
    process = root / str(pid)
    namespaces = process / "ns"
    namespaces.mkdir(parents=True)
    tail = ["S", *("0" for _ in range(18)), str(start), "0"]
    (process / "stat").write_text(
        f"{pid} ({comm}) " + " ".join(tail) + "\n", encoding="utf-8"
    )
    (process / "comm").write_text(comm + "\n", encoding="utf-8")
    (process / "cgroup").write_text(f"0::{cgroup}\n", encoding="utf-8")
    for name in ("pid", "mnt", "user"):
        (namespaces / name).write_text(name, encoding="utf-8")


def test_gpu_doctor_binds_owner_cgroup_scheduler_and_health_activity(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    container_id = "d" * 64
    owner_pid = 4242
    _proc(
        proc_root,
        owner_pid,
        start=777,
        comm="python",
        cgroup=f"/docker/{container_id}/slurm/job_77/step_batch",
    )
    _proc(
        proc_root,
        os.getpid(),
        start=888,
        comm="pytest",
        cgroup="/slurm/job_77/step_batch",
    )
    _proc(
        proc_root,
        5151,
        start=999,
        comm="rocminfo",
        cgroup="/health",
    )
    owner = GpuProcessIdentity(
        owner_pid, os.getuid(), 777, "e" * 64, (0,)
    )
    doctor = LinuxGpuDoctorInspector(
        _Ownership(owner),
        proc_root=proc_root,
        environment={"SLURM_JOB_ID": "77", "SLURM_STEP_ID": "batch"},
        health=_Health(),
    )

    receipt = doctor.inspect("amd-gpu-set=0")

    context = receipt.process_contexts[0]
    assert receipt.status == "blocked"
    assert receipt.formal_measurement_ready is False
    assert context.container_id == container_id
    assert context.slurm_job_id == "77"
    assert context.slurm_step_id == "batch"
    assert receipt.scheduler_identity_consistent is True
    assert [item.comm for item in receipt.health_check_processes] == ["rocminfo"]
    assert receipt.to_dict()["rocm_health_status"] == "healthy"
    assert len(receipt.digest) == 64


def test_gpu_doctor_is_ready_only_with_clean_ownership_and_rocm_health(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _proc(
        proc_root,
        os.getpid(),
        start=888,
        comm="pytest",
        cgroup="/",
    )
    doctor = LinuxGpuDoctorInspector(
        _Ownership(None), proc_root=proc_root, environment={}, health=_Health()
    )

    receipt = doctor.inspect("amd-gpu-set=0")

    assert receipt.status == "ready"
    assert receipt.formal_measurement_ready is True
    assert receipt.process_contexts == ()
    assert receipt.health_check_processes == ()
    document = json.loads(json.dumps(receipt.to_dict()))
    assert load_gpu_doctor_receipt(
        document, ownership=receipt.ownership
    ) == receipt
