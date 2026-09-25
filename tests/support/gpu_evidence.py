"""Typed synthetic GPU doctor evidence for CPU-only tests."""

from __future__ import annotations

from apex.runtime import (
    GpuDoctorReceipt,
    GpuDeviceIdentity,
    GpuLeaseOwnerIdentity,
    GpuLeaseReceipt,
    GpuLeaseHeartbeatReceipt,
    GpuMeasurementBracketReceipt,
    GpuOwnershipReceipt,
    GpuSelectorRequest,
    GpuProcessContext,
    RocmHealthDevice,
    RocmHealthReceipt,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    RsmiDeviceIdentity,
)


def clean_gpu_doctor(ownership: GpuOwnershipReceipt) -> GpuDoctorReceipt:
    contexts = tuple(_context(item.pid, item.uid, item.start_time_ticks) for item in (
        *ownership.allowed_owners,
        *ownership.foreign_owners,
    ))
    health = RocmHealthReceipt(
        observed_unix_ns=ownership.observed_unix_ns,
        library_path=ownership.library_path,
        library_sha256=ownership.library_sha256,
        ownership_receipt_sha256=ownership.digest,
        devices=tuple(
            RocmHealthDevice(
                item.unique_id,
                item.rsmi_index,
                42.0,
                1700.0,
                0,
                1,
                1024,
            )
            for item in ownership.selected_devices
        ),
    )
    return GpuDoctorReceipt(
        ownership=ownership,
        process_contexts=contexts,
        supervisor_context=_context(1, 0, 1),
        scheduler_environment=(),
        scheduler_identity_consistent=True,
        health_check_processes=(),
        process_scan_complete=True,
        rocm_health=health,
        rocm_health_error=None,
    )


class StaticGpuDoctorInspector:
    def __init__(self, ownership_inspector) -> None:
        self._ownership = ownership_inspector

    def inspect(self, selector_scope: str, *, allowed_pids=()) -> GpuDoctorReceipt:
        ownership = self._ownership.inspect(
            selector_scope, allowed_pids=allowed_pids
        )
        return clean_gpu_doctor(ownership)


class SyntheticGpuMeasurementGuard:
    """CPU-only typed bracket for an already synthetic lease receipt."""

    def __init__(self, lease: GpuLeaseReceipt, action_id: str) -> None:
        self._lease = lease
        self._action_id = action_id
        self.receipt: GpuMeasurementBracketReceipt

    def __enter__(self) -> "SyntheticGpuMeasurementGuard":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        owner = GpuLeaseOwnerIdentity(
            self._lease.owner_pid, 0, 1, "d" * 64
        )
        doctor = self._lease.doctor
        pre = GpuLeaseHeartbeatReceipt(
            1,
            self._lease.run_id,
            self._lease.digest,
            1,
            "measurement_pre",
            10.0,
            110.0,
            100.0,
            owner,
            self._lease.ownership,
            doctor,
        )
        post = GpuLeaseHeartbeatReceipt(
            1,
            self._lease.run_id,
            self._lease.digest,
            2,
            "measurement_post",
            12.0,
            112.0,
            100.0,
            owner,
            self._lease.ownership,
            doctor,
        )
        self.receipt = GpuMeasurementBracketReceipt(
            1,
            self._lease.run_id,
            self._action_id,
            self._lease.digest,
            11.0,
            11.5,
            pre,
            post,
        )


def synthetic_gpu_lease(run_id: str) -> GpuLeaseReceipt:
    unique_id = "GPU-0123456789abcdef"
    hsa = HsaInventoryEvidence(
        1,
        "clean_unfiltered_hsa_gpu_inventory_v1",
        "/trusted/hsa-helper.py",
        "b" * 64,
        "/opt/rocm/lib/libhsa-runtime64.so.1",
        "c" * 64,
        (HsaGpuIdentity(0, 2, 2, 100, 0, unique_id),),
    )
    device = GpuDeviceIdentity(
        0, 2, 0, unique_id, "/dev/dri/renderD128"
    )
    ownership = GpuOwnershipReceipt(
        2,
        "clean_hsa_kfd_rsmi_process_gpu_map_v2",
        GpuSelectorRequest(requested=(unique_id,)),
        123,
        "/opt/rocm/lib/librocm_smi64.so.7",
        "a" * 64,
        "/sys/class/kfd/kfd/topology/nodes",
        hsa,
        (RsmiDeviceIdentity(0, 2, 100, unique_id, 128),),
        (device,),
        (device,),
        (),
        (),
    )
    return GpuLeaseReceipt(
        3,
        run_id,
        ownership.execution_scope,
        ownership.physical_scope,
        1234,
        1.0,
        "/tmp/apex-gpu-leases/test.lock",
        ownership,
        clean_gpu_doctor(ownership),
    )


def synthetic_gpu_heartbeat(
    lease: GpuLeaseReceipt,
    *,
    reason: str = "manual",
    sequence: int = 1,
) -> GpuLeaseHeartbeatReceipt:
    observed = float(sequence)
    return GpuLeaseHeartbeatReceipt(
        1,
        lease.run_id,
        lease.digest,
        sequence,
        reason,
        observed,
        observed + 100.0,
        100.0,
        GpuLeaseOwnerIdentity(lease.owner_pid, 0, 1, "d" * 64),
        lease.ownership,
        lease.doctor,
    )


def _context(pid: int, uid: int, start: int) -> GpuProcessContext:
    return GpuProcessContext(
        pid=pid,
        uid=uid,
        start_time_ticks=start,
        comm="test",
        cgroup_sha256="f" * 64,
        cgroup_paths=("/",),
        pid_namespace_inode=1,
        mount_namespace_inode=2,
        user_namespace_inode=3,
        container_id=None,
        slurm_job_id=None,
        slurm_step_id=None,
    )


__all__ = [
    "StaticGpuDoctorInspector",
    "SyntheticGpuMeasurementGuard",
    "clean_gpu_doctor",
    "synthetic_gpu_lease",
    "synthetic_gpu_heartbeat",
]
