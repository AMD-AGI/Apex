"""Strict reconstruction of a serialized GPU doctor receipt."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from apex.core import ContractError

from .gpu_doctor import GpuDoctorReceipt, GpuProcessContext
from .gpu_health import RocmHealthDevice, RocmHealthReceipt
from .gpu_ownership import GpuOwnershipReceipt


def load_gpu_doctor_receipt(
    value: Mapping[str, Any], *, ownership: GpuOwnershipReceipt
) -> GpuDoctorReceipt:
    """Rebuild derived fields and require byte-for-byte semantic identity."""

    health_value = value.get("rocm_health")
    health = None if health_value is None else _health(_mapping(health_value))
    receipt = GpuDoctorReceipt(
        ownership=ownership,
        process_contexts=_objects(value.get("process_contexts"), _context),
        supervisor_context=_context(_mapping(value.get("supervisor_context"))),
        scheduler_environment=tuple(
            sorted(_string_mapping(value.get("scheduler_environment")).items())
        ),
        scheduler_identity_consistent=_boolean(
            value.get("scheduler_identity_consistent")
        ),
        health_check_processes=_objects(
            value.get("health_check_processes"), _context
        ),
        process_scan_complete=_boolean(value.get("process_scan_complete")),
        rocm_health=health,
        rocm_health_error=_optional_string(value.get("rocm_health_error")),
        schema=_string(value.get("schema")),
    )
    if receipt.to_dict() != dict(value):
        _invalid()
    return receipt


def _context(value: Mapping[str, Any]) -> GpuProcessContext:
    try:
        return GpuProcessContext(
            pid=_integer(value["pid"], positive=True),
            uid=_integer(value["uid"]),
            start_time_ticks=_integer(value["start_time_ticks"], positive=True),
            comm=_string(value["comm"]),
            cgroup_sha256=_string(value["cgroup_sha256"]),
            cgroup_paths=_text_tuple(value["cgroup_paths"]),
            pid_namespace_inode=_integer(value["pid_namespace_inode"], positive=True),
            mount_namespace_inode=_integer(value["mount_namespace_inode"], positive=True),
            user_namespace_inode=_integer(value["user_namespace_inode"], positive=True),
            container_id=_optional_string(value["container_id"]),
            slurm_job_id=_optional_string(value["slurm_job_id"]),
            slurm_step_id=_optional_string(value["slurm_step_id"]),
        )
    except KeyError:
        _invalid()


def _health(value: Mapping[str, Any]) -> RocmHealthReceipt:
    try:
        return RocmHealthReceipt(
            observed_unix_ns=_integer(value["observed_unix_ns"], positive=True),
            library_path=_string(value["library_path"]),
            library_sha256=_string(value["library_sha256"]),
            ownership_receipt_sha256=_string(value["ownership_receipt_sha256"]),
            devices=_objects(value["devices"], _health_device),
            schema=_string(value["schema"]),
            policy_id=_string(value["policy_id"]),
        )
    except KeyError:
        _invalid()


def _health_device(value: Mapping[str, Any]) -> RocmHealthDevice:
    try:
        return RocmHealthDevice(
            unique_id=_string(value["unique_id"]),
            rsmi_index=_integer(value["rsmi_index"]),
            temperature_c=_number(value["temperature_c"]),
            clock_mhz=_number(value["clock_mhz"]),
            busy_percent=_integer(value["busy_percent"]),
            vram_used_bytes=_integer(value["vram_used_bytes"]),
            vram_total_bytes=_integer(value["vram_total_bytes"], positive=True),
        )
    except KeyError:
        _invalid()


def _objects(value: object, loader: Callable[[Mapping[str, Any]], Any]) -> tuple[Any, ...]:
    if not isinstance(value, list):
        _invalid()
    return tuple(loader(_mapping(item)) for item in value)


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _invalid()
    return dict(value)


def _string_mapping(value: object) -> dict[str, str]:
    mapping = _mapping(value)
    if any(not isinstance(item, str) or not item for item in mapping.values()):
        _invalid()
    return dict(mapping)


def _text_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _invalid()
    return tuple(value)


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        _invalid()
    return value


def _optional_string(value: object) -> str | None:
    return None if value is None else _string(value)


def _integer(value: object, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _invalid()
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid()
    return float(value)


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        _invalid()
    return value


def _invalid() -> None:
    raise ContractError("GPU doctor receipt is malformed", "invalid_gpu_doctor_receipt")


__all__ = ["load_gpu_doctor_receipt"]
