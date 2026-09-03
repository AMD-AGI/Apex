"""Bounded ctypes adapter for the ROCm SMI identity and process APIs."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Protocol

from apex.core import ContractError


MAX_RSMI_DEVICES = 1024
MAX_RSMI_PROCESSES = 262_144
_PROCESS_HEADROOM = 64


class OwnershipApi(Protocol):
    def init(self) -> int: ...

    def shutdown(self) -> int: ...

    def device_count(self) -> tuple[int, int]: ...

    def device_identity(self, index: int) -> tuple[int, int, int, int, int]: ...

    def process_pids(self) -> tuple[int, tuple[int, ...]]: ...

    def process_devices(self, pid: int) -> tuple[int, tuple[int, ...]]: ...


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("process_id", ctypes.c_uint32),
        ("pasid", ctypes.c_uint32),
        ("vram_usage", ctypes.c_uint64),
        ("sdma_usage", ctypes.c_uint64),
        ("cu_occupancy", ctypes.c_uint32),
    ]


class CtypesOwnershipApi:
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
            self._library.rsmi_dev_node_id_get,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)],
        )
        _signature(
            self._library.rsmi_dev_pci_id_get,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)],
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

    def device_identity(self, index: int) -> tuple[int, int, int, int, int]:
        unique_id = ctypes.c_uint64(0)
        node_id = ctypes.c_uint32(0)
        pci_id = ctypes.c_uint64(0)
        minor = ctypes.c_uint32(0)
        first = int(self._library.rsmi_dev_unique_id_get(index, ctypes.byref(unique_id)))
        second = int(
            self._library.rsmi_dev_drm_render_minor_get(index, ctypes.byref(minor))
        )
        third = int(self._library.rsmi_dev_node_id_get(index, ctypes.byref(node_id)))
        fourth = int(self._library.rsmi_dev_pci_id_get(index, ctypes.byref(pci_id)))
        return (
            first or second or third or fourth,
            int(unique_id.value),
            int(node_id.value),
            int(pci_id.value),
            int(minor.value),
        )

    def process_pids(self) -> tuple[int, tuple[int, ...]]:
        count = ctypes.c_uint32(0)
        status = int(
            self._library.rsmi_compute_process_info_get(None, ctypes.byref(count))
        )
        if status != 0 or count.value > MAX_RSMI_PROCESSES:
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
        if status != 0 or count.value > MAX_RSMI_DEVICES:
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


def resolve_rsmi_library(explicit: Path | None) -> Path:
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
    "CtypesOwnershipApi",
    "MAX_RSMI_DEVICES",
    "MAX_RSMI_PROCESSES",
    "OwnershipApi",
    "resolve_rsmi_library",
]
