"""Authoritative AMD GPU process ownership observed before a run starts."""

from __future__ import annotations

import ctypes
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, sha256_bytes, sha256_file, sha256_json


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_DEVICES = 1024
_MAX_PROCESSES = 262_144
_PROCESS_HEADROOM = 64


@dataclass(frozen=True, slots=True)
class GpuDeviceIdentity:
    index: int
    unique_id: str
    render_node: str

    def __post_init__(self) -> None:
        if (
            self.index < 0
            or not re.fullmatch(r"0x[0-9a-f]{16}", self.unique_id)
            or not re.fullmatch(r"/dev/dri/renderD[0-9]+", self.render_node)
        ):
            raise ContractError(
                "ROCm SMI returned an invalid GPU identity",
                "invalid_gpu_physical_identity",
            )


@dataclass(frozen=True, slots=True)
class GpuProcessIdentity:
    pid: int
    uid: int
    start_time_ticks: int
    cmdline_sha256: str
    device_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.pid <= 0
            or self.uid < 0
            or self.start_time_ticks <= 0
            or not _DIGEST.fullmatch(self.cmdline_sha256)
            or not self.device_indices
            or tuple(sorted(set(self.device_indices))) != self.device_indices
        ):
            raise ContractError(
                "ROCm SMI returned an invalid GPU process identity",
                "invalid_gpu_process_identity",
            )


@dataclass(frozen=True, slots=True)
class GpuOwnershipReceipt:
    schema_version: int
    policy_id: str
    selector_scope: str
    observed_unix_ns: int
    library_path: str
    library_sha256: str
    selected_devices: tuple[GpuDeviceIdentity, ...]
    allowed_owners: tuple[GpuProcessIdentity, ...]
    foreign_owners: tuple[GpuProcessIdentity, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.policy_id != "rocm_smi_process_gpu_map_v1"
            or self.observed_unix_ns <= 0
            or not Path(self.library_path).is_absolute()
            or not _DIGEST.fullmatch(self.library_sha256)
            or not self.selected_devices
        ):
            raise ContractError(
                "GPU ownership receipt is incomplete",
                "invalid_gpu_ownership_receipt",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "selector_scope": self.selector_scope,
            "observed_unix_ns": self.observed_unix_ns,
            "library_path": self.library_path,
            "library_sha256": self.library_sha256,
            "selected_devices": [asdict(item) for item in self.selected_devices],
            "allowed_owners": [asdict(item) for item in self.allowed_owners],
            "foreign_owners": [asdict(item) for item in self.foreign_owners],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def physical_scope(self) -> str:
        values = sorted(device.unique_id for device in self.selected_devices)
        return "amd-gpu-unique-id-set=" + ",".join(values)


class GpuOwnershipInspector(Protocol):
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt: ...


class OwnershipApi(Protocol):
    def init(self) -> int: ...

    def shutdown(self) -> int: ...

    def device_count(self) -> tuple[int, int]: ...

    def device_identity(self, index: int) -> tuple[int, int, int]: ...

    def process_pids(self) -> tuple[int, tuple[int, ...]]: ...

    def process_devices(self, pid: int) -> tuple[int, tuple[int, ...]]: ...


class RocmSmiGpuOwnershipInspector:
    """Resolve physical devices and map every KFD PID through librocm_smi."""

    def __init__(self, *, library_path: Path | None = None) -> None:
        self._library_path = library_path

    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        library = _resolve_library(self._library_path)
        return collect_gpu_ownership(
            _CtypesOwnershipApi(library),
            selector_scope=selector_scope,
            allowed_pids=allowed_pids,
            library_path=library,
            library_sha256=sha256_file(library),
        )


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("process_id", ctypes.c_uint32),
        ("pasid", ctypes.c_uint32),
        ("vram_usage", ctypes.c_uint64),
        ("sdma_usage", ctypes.c_uint64),
        ("cu_occupancy", ctypes.c_uint32),
    ]


class _CtypesOwnershipApi:
    def __init__(self, library_path: Path) -> None:
        try:
            self._library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
            self._configure()
        except (OSError, AttributeError) as error:
            raise ContractError(
                "The ROCm SMI process API is unavailable",
                "gpu_ownership_api_unavailable",
            ) from error

    def _configure(self) -> None:
        _signature(self._library.rsmi_init, [ctypes.c_uint64])
        _signature(self._library.rsmi_shut_down, [])
        _signature(self._library.rsmi_num_monitor_devices, [ctypes.POINTER(ctypes.c_uint32)])
        _signature(
            self._library.rsmi_dev_unique_id_get,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)],
        )
        _signature(
            self._library.rsmi_dev_drm_render_minor_get,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)],
        )
        _signature(
            self._library.rsmi_compute_process_info_get,
            [ctypes.POINTER(_ProcessInfo), ctypes.POINTER(ctypes.c_uint32)],
        )
        _signature(
            self._library.rsmi_compute_process_gpus_get,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)],
        )

    def init(self) -> int:
        return int(self._library.rsmi_init(0))

    def shutdown(self) -> int:
        return int(self._library.rsmi_shut_down())

    def device_count(self) -> tuple[int, int]:
        count = ctypes.c_uint32(0)
        status = self._library.rsmi_num_monitor_devices(ctypes.byref(count))
        return int(status), int(count.value)

    def device_identity(self, index: int) -> tuple[int, int, int]:
        unique_id = ctypes.c_uint64(0)
        minor = ctypes.c_uint32(0)
        first = int(self._library.rsmi_dev_unique_id_get(index, ctypes.byref(unique_id)))
        second = int(
            self._library.rsmi_dev_drm_render_minor_get(index, ctypes.byref(minor))
        )
        return first or second, int(unique_id.value), int(minor.value)

    def process_pids(self) -> tuple[int, tuple[int, ...]]:
        count = ctypes.c_uint32(0)
        status = int(
            self._library.rsmi_compute_process_info_get(None, ctypes.byref(count))
        )
        if status != 0 or count.value > _MAX_PROCESSES:
            return status or -1, ()
        capacity = max(int(count.value) + _PROCESS_HEADROOM, _PROCESS_HEADROOM)
        records = (_ProcessInfo * capacity)()
        fetched = ctypes.c_uint32(capacity)
        status = int(
            self._library.rsmi_compute_process_info_get(records, ctypes.byref(fetched))
        )
        if fetched.value > capacity:
            return -1, ()
        return status, tuple(int(records[index].process_id) for index in range(fetched.value))

    def process_devices(self, pid: int) -> tuple[int, tuple[int, ...]]:
        count = ctypes.c_uint32(0)
        status = int(
            self._library.rsmi_compute_process_gpus_get(pid, None, ctypes.byref(count))
        )
        if status != 0 or count.value > _MAX_DEVICES:
            return status or -1, ()
        if count.value == 0:
            return 0, ()
        indices = (ctypes.c_uint32 * int(count.value))()
        fetched = ctypes.c_uint32(count.value)
        status = int(
            self._library.rsmi_compute_process_gpus_get(
                pid, indices, ctypes.byref(fetched)
            )
        )
        if fetched.value > count.value:
            return -1, ()
        return status, tuple(int(indices[index]) for index in range(fetched.value))


def collect_gpu_ownership(
    api: OwnershipApi,
    *,
    selector_scope: str,
    allowed_pids: tuple[int, ...],
    library_path: Path,
    library_sha256: str,
    proc_root: Path = Path("/proc"),
    observed_unix_ns: int | None = None,
) -> GpuOwnershipReceipt:
    """Collect a race-checked physical-device/process map for one selector."""

    if not library_path.is_absolute() or not _DIGEST.fullmatch(library_sha256):
        raise ContractError("ROCm SMI identity is invalid", "gpu_ownership_api_unavailable")
    status = api.init()
    if status != 0:
        raise ContractError(
            "ROCm SMI initialization failed",
            "gpu_ownership_query_failed",
            {"operation": "init", "status": status},
        )
    failure: BaseException | None = None
    receipt: GpuOwnershipReceipt | None = None
    try:
        receipt = _collect_initialized(
            api,
            selector_scope=selector_scope,
            allowed_pids=allowed_pids,
            library_path=library_path,
            library_sha256=library_sha256,
            proc_root=proc_root,
            observed_unix_ns=observed_unix_ns,
        )
    except BaseException as error:
        failure = error
    shutdown = api.shutdown()
    if failure is not None:
        if shutdown != 0:
            raise ContractError(
                "ROCm SMI ownership query and shutdown both failed",
                "gpu_ownership_query_failed",
                {"shutdown_status": shutdown},
            ) from failure
        raise failure
    if shutdown != 0 or receipt is None:
        raise ContractError(
            "ROCm SMI shutdown failed",
            "gpu_ownership_query_failed",
            {"operation": "shutdown", "status": shutdown},
        )
    return receipt


def _collect_initialized(
    api: OwnershipApi,
    *,
    selector_scope: str,
    allowed_pids: tuple[int, ...],
    library_path: Path,
    library_sha256: str,
    proc_root: Path,
    observed_unix_ns: int | None,
) -> GpuOwnershipReceipt:
    devices = _query_devices(api)
    selected = _select_devices(devices, selector_scope)
    selected_indices = {device.index for device in selected}
    first = _query_process_map(api)
    first_owners = _selected_process_identities(
        first, selected_indices=selected_indices, proc_root=proc_root
    )
    second = _query_process_map(api)
    second_owners = _selected_process_identities(
        second, selected_indices=selected_indices, proc_root=proc_root
    )
    if first != second or first_owners != second_owners:
        raise ContractError(
            "GPU ownership changed during preflight",
            "gpu_ownership_race",
        )
    allowed = set(allowed_pids)
    return GpuOwnershipReceipt(
        1,
        "rocm_smi_process_gpu_map_v1",
        selector_scope,
        observed_unix_ns if observed_unix_ns is not None else time.time_ns(),
        str(library_path),
        library_sha256,
        selected,
        tuple(owner for owner in second_owners if owner.pid in allowed),
        tuple(owner for owner in second_owners if owner.pid not in allowed),
    )


def _selected_process_identities(
    process_map: tuple[tuple[int, tuple[int, ...]], ...],
    *,
    selected_indices: set[int],
    proc_root: Path,
) -> tuple[GpuProcessIdentity, ...]:
    return tuple(
        _process_identity(pid, indices, proc_root=proc_root)
        for pid, indices in process_map
        if selected_indices.intersection(indices)
    )


def _query_devices(api: OwnershipApi) -> tuple[GpuDeviceIdentity, ...]:
    status, count = api.device_count()
    if status != 0 or count < 1 or count > _MAX_DEVICES:
        raise ContractError(
            "ROCm SMI device inventory failed",
            "gpu_physical_mapping_unresolved",
            {"status": status, "count": count},
        )
    devices: list[GpuDeviceIdentity] = []
    for index in range(count):
        identity_status, unique_id, render_minor = api.device_identity(index)
        if identity_status != 0:
            raise ContractError(
                "ROCm SMI device identity query failed",
                "gpu_physical_mapping_unresolved",
                {"index": index, "status": identity_status},
            )
        devices.append(
            GpuDeviceIdentity(index, f"0x{unique_id:016x}", f"/dev/dri/renderD{render_minor}")
        )
    if len({device.unique_id for device in devices}) != len(devices):
        raise ContractError(
            "ROCm SMI returned duplicate physical GPU IDs",
            "gpu_physical_mapping_unresolved",
        )
    return tuple(devices)


def _query_process_map(api: OwnershipApi) -> tuple[tuple[int, tuple[int, ...]], ...]:
    status, pids = api.process_pids()
    if status != 0 or len(pids) > _MAX_PROCESSES or any(pid <= 0 for pid in pids):
        raise ContractError(
            "ROCm SMI KFD process inventory failed",
            "gpu_ownership_query_failed",
            {"status": status},
        )
    if len(set(pids)) != len(pids):
        raise ContractError(
            "ROCm SMI returned duplicate KFD process IDs",
            "gpu_ownership_query_failed",
        )
    result: list[tuple[int, tuple[int, ...]]] = []
    for pid in sorted(pids):
        device_status, indices = api.process_devices(pid)
        canonical = tuple(sorted(set(indices)))
        if device_status != 0 or not canonical or canonical != tuple(sorted(indices)):
            raise ContractError(
                "ROCm SMI process-to-GPU query failed",
                "gpu_ownership_query_failed",
                {"pid": pid, "status": device_status},
            )
        result.append((pid, canonical))
    return tuple(result)


def _select_devices(
    devices: tuple[GpuDeviceIdentity, ...], selector_scope: str
) -> tuple[GpuDeviceIdentity, ...]:
    if selector_scope == "all-visible-amd-gpus":
        return devices
    prefix = "amd-gpu-set="
    if not selector_scope.startswith(prefix):
        raise ContractError("GPU selector is invalid", "gpu_physical_mapping_unresolved")
    selectors = selector_scope[len(prefix) :].split(",")
    selected: list[GpuDeviceIdentity] = []
    for selector in selectors:
        matches = [device for device in devices if _selector_matches(device, selector)]
        if len(matches) != 1:
            raise ContractError(
                "GPU selector does not resolve to exactly one physical device",
                "gpu_physical_mapping_unresolved",
                {"selector": selector},
            )
        selected.append(matches[0])
    if len({device.index for device in selected}) != len(selected):
        raise ContractError(
            "GPU selectors resolve to duplicate physical devices",
            "gpu_physical_mapping_unresolved",
        )
    return tuple(sorted(selected, key=lambda device: device.index))


def _selector_matches(device: GpuDeviceIdentity, selector: str) -> bool:
    if selector.isdecimal():
        return device.index == int(selector)
    normalized = selector.lower().removeprefix("gpu-")
    normalized = normalized.removeprefix("0x")
    return normalized == device.unique_id.removeprefix("0x")


def _process_identity(
    pid: int, device_indices: tuple[int, ...], *, proc_root: Path
) -> GpuProcessIdentity:
    root = proc_root / str(pid)
    try:
        metadata = root.stat()
        raw_stat = (root / "stat").read_text(encoding="utf-8")
        cmdline = (root / "cmdline").read_bytes()
        tail = raw_stat[raw_stat.rindex(")") + 2 :].split()
        start_time_ticks = int(tail[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ContractError(
            "A KFD process identity could not be frozen",
            "gpu_process_identity_unavailable",
            {"pid": pid},
        ) from error
    return GpuProcessIdentity(
        pid,
        metadata.st_uid,
        start_time_ticks,
        sha256_bytes(cmdline),
        device_indices,
    )


def _resolve_library(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else [
        Path("/opt/rocm/lib/librocm_smi64.so.7"),
        Path("/opt/rocm/lib/librocm_smi64.so"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    raise ContractError(
        "A concrete ROCm SMI library could not be resolved",
        "gpu_ownership_api_unavailable",
    )


def _signature(function: object, argument_types: list[object]) -> None:
    function.argtypes = argument_types  # type: ignore[attr-defined]
    function.restype = ctypes.c_int  # type: ignore[attr-defined]


__all__ = [
    "GpuDeviceIdentity",
    "GpuOwnershipInspector",
    "GpuOwnershipReceipt",
    "GpuProcessIdentity",
    "OwnershipApi",
    "RocmSmiGpuOwnershipInspector",
    "collect_gpu_ownership",
]
