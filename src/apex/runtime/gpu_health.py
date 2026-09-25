"""Bounded ROCm SMI health snapshot for selected physical GPUs."""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, sha256_file, sha256_json

from .gpu_ownership import GpuOwnershipReceipt


_MAX_FREQUENCIES = 33
_RSMI_TEMP_JUNCTION = 1
_RSMI_TEMP_CURRENT = 0
_RSMI_CLK_SYS = 0
_RSMI_MEM_VRAM = 0


@dataclass(frozen=True, slots=True)
class RocmHealthDevice:
    unique_id: str
    rsmi_index: int
    temperature_c: float
    clock_mhz: float
    busy_percent: int
    vram_used_bytes: int
    vram_total_bytes: int

    def __post_init__(self) -> None:
        if (
            not self.unique_id.startswith("GPU-")
            or self.rsmi_index < 0
            or not math.isfinite(self.temperature_c)
            or not math.isfinite(self.clock_mhz)
            or self.temperature_c <= 0
            or self.clock_mhz <= 0
            or not 0 <= self.busy_percent <= 100
            or self.vram_used_bytes < 0
            or self.vram_total_bytes <= 0
            or self.vram_used_bytes > self.vram_total_bytes
        ):
            raise ContractError(
                "ROCm health device snapshot is invalid",
                "gpu_health_query_failed",
            )


@dataclass(frozen=True, slots=True)
class RocmHealthReceipt:
    observed_unix_ns: int
    library_path: str
    library_sha256: str
    ownership_receipt_sha256: str
    devices: tuple[RocmHealthDevice, ...]
    schema: str = "apex.rocm-health-receipt/v1"
    policy_id: str = "rsmi_selected_device_health_v1"

    def __post_init__(self) -> None:
        if (
            self.schema != "apex.rocm-health-receipt/v1"
            or self.policy_id != "rsmi_selected_device_health_v1"
            or self.observed_unix_ns <= 0
            or not Path(self.library_path).is_absolute()
            or len(self.library_sha256) != 64
            or len(self.ownership_receipt_sha256) != 64
            or not self.devices
            or len({item.unique_id for item in self.devices}) != len(self.devices)
        ):
            raise ContractError(
                "ROCm health receipt is incomplete",
                "invalid_gpu_health_receipt",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "observed_unix_ns": self.observed_unix_ns,
            "library_path": self.library_path,
            "library_sha256": self.library_sha256,
            "ownership_receipt_sha256": self.ownership_receipt_sha256,
            "devices": [asdict(item) for item in self.devices],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class RocmHealthInspector(Protocol):
    def inspect(self, ownership: GpuOwnershipReceipt) -> RocmHealthReceipt: ...


class RocmHealthApi(Protocol):
    def init(self) -> int: ...
    def shutdown(self) -> int: ...
    def health(self, index: int) -> tuple[int, int, int, int, int, int]: ...


class CtypesRocmHealthInspector:
    """Query only fixed RSMI health APIs from the ownership-bound library."""

    def inspect(self, ownership: GpuOwnershipReceipt) -> RocmHealthReceipt:
        library = Path(ownership.library_path).resolve(strict=True)
        digest = sha256_file(library)
        if digest != ownership.library_sha256:
            raise ContractError(
                "ROCm SMI library differs from ownership preflight",
                "gpu_health_library_mismatch",
            )
        receipt = collect_rocm_health(
            _CtypesRocmHealthApi(library),
            ownership=ownership,
            library=library,
            library_sha256=digest,
        )
        if sha256_file(library) != digest:
            raise ContractError(
                "ROCm SMI library changed during health inspection",
                "gpu_health_library_mismatch",
            )
        return receipt


def collect_rocm_health(
    api: RocmHealthApi,
    *,
    ownership: GpuOwnershipReceipt,
    library: Path,
    library_sha256: str,
    observed_unix_ns: int | None = None,
) -> RocmHealthReceipt:
    if api.init() != 0:
        raise ContractError("ROCm health initialization failed", "gpu_health_query_failed")
    failure: BaseException | None = None
    devices: tuple[RocmHealthDevice, ...] = ()
    try:
        devices = tuple(_device_health(api, item) for item in ownership.selected_devices)
    except BaseException as error:
        failure = error
    shutdown = api.shutdown()
    if failure is not None:
        if shutdown != 0:
            raise ContractError(
                "ROCm health query and shutdown both failed",
                "gpu_health_query_failed",
            ) from failure
        raise failure
    if shutdown != 0:
        raise ContractError("ROCm health shutdown failed", "gpu_health_query_failed")
    return RocmHealthReceipt(
        observed_unix_ns=observed_unix_ns or time.time_ns(),
        library_path=str(library),
        library_sha256=library_sha256,
        ownership_receipt_sha256=ownership.digest,
        devices=devices,
    )


def _device_health(api: RocmHealthApi, device) -> RocmHealthDevice:
    status, temperature, clock_hz, busy, used, total = api.health(device.rsmi_index)
    if status != 0:
        raise ContractError(
            "ROCm health query failed",
            "gpu_health_query_failed",
            {"rsmi_index": device.rsmi_index, "status": status},
        )
    return RocmHealthDevice(
        unique_id=device.unique_id,
        rsmi_index=device.rsmi_index,
        temperature_c=temperature / 1000.0,
        clock_mhz=clock_hz / 1_000_000.0,
        busy_percent=busy,
        vram_used_bytes=used,
        vram_total_bytes=total,
    )


class _Frequencies(ctypes.Structure):
    _fields_ = [
        ("has_deep_sleep", ctypes.c_bool),
        ("num_supported", ctypes.c_uint32),
        ("current", ctypes.c_uint32),
        ("frequency", ctypes.c_uint64 * _MAX_FREQUENCIES),
    ]


class _CtypesRocmHealthApi:
    def __init__(self, library: Path) -> None:
        try:
            self._library = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
            self._configure()
        except (OSError, AttributeError) as error:
            raise ContractError(
                "Required ROCm health APIs are unavailable",
                "gpu_health_api_unavailable",
            ) from error

    def _configure(self) -> None:
        _signature(self._library.rsmi_init, [ctypes.c_uint64])
        _signature(self._library.rsmi_shut_down, [])
        _signature(
            self._library.rsmi_dev_temp_metric_get,
            [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int64)],
        )
        _signature(
            self._library.rsmi_dev_gpu_clk_freq_get,
            [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(_Frequencies)],
        )
        _signature(
            self._library.rsmi_dev_busy_percent_get,
            [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)],
        )
        for name in ("rsmi_dev_memory_usage_get", "rsmi_dev_memory_total_get"):
            _signature(
                getattr(self._library, name),
                [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)],
            )

    def init(self) -> int:
        return int(self._library.rsmi_init(0))

    def shutdown(self) -> int:
        return int(self._library.rsmi_shut_down())

    def health(self, index: int) -> tuple[int, int, int, int, int, int]:
        temperature = ctypes.c_int64(0)
        frequencies = _Frequencies()
        busy = ctypes.c_uint32(0)
        used = ctypes.c_uint64(0)
        total = ctypes.c_uint64(0)
        statuses = (
            self._library.rsmi_dev_temp_metric_get(index, _RSMI_TEMP_JUNCTION, _RSMI_TEMP_CURRENT, ctypes.byref(temperature)),
            self._library.rsmi_dev_gpu_clk_freq_get(index, _RSMI_CLK_SYS, ctypes.byref(frequencies)),
            self._library.rsmi_dev_busy_percent_get(index, ctypes.byref(busy)),
            self._library.rsmi_dev_memory_usage_get(index, _RSMI_MEM_VRAM, ctypes.byref(used)),
            self._library.rsmi_dev_memory_total_get(index, _RSMI_MEM_VRAM, ctypes.byref(total)),
        )
        current = int(frequencies.current)
        if any(int(item) != 0 for item in statuses) or current >= _MAX_FREQUENCIES:
            return next((int(item) for item in statuses if int(item) != 0), -1), 0, 0, 0, 0, 0
        return (
            0,
            int(temperature.value),
            int(frequencies.frequency[current]),
            int(busy.value),
            int(used.value),
            int(total.value),
        )


def _signature(function: object, argument_types: list[object]) -> None:
    function.argtypes = argument_types  # type: ignore[attr-defined]
    function.restype = ctypes.c_int  # type: ignore[attr-defined]


__all__ = [
    "CtypesRocmHealthInspector",
    "RocmHealthApi",
    "RocmHealthDevice",
    "RocmHealthInspector",
    "RocmHealthReceipt",
    "collect_rocm_health",
]
