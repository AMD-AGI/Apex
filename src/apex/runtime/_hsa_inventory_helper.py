"""Emit an unfiltered HSA GPU-agent inventory for the parent Apex process."""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_HSA_STATUS_SUCCESS = 0
_HSA_STATUS_ERROR = 0x1000
_HSA_DEVICE_TYPE_GPU = 1
_HSA_AGENT_INFO_NODE = 16
_HSA_AGENT_INFO_DEVICE = 17
_HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 0xA004
_HSA_AMD_AGENT_INFO_BDFID = 0xA006
_HSA_AMD_AGENT_INFO_DOMAIN = 0xA00F
_HSA_AMD_AGENT_INFO_UUID = 0xA011
_HSA_UUID_CAPACITY = 21
_MAX_GPU_AGENTS = 1024
_GPU_UUID = re.compile(r"^GPU-[0-9a-f]{16}$")
_VISIBILITY_VARIABLES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


class _HsaAgent(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


class _InventoryError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _GpuAgent:
    hsa_gpu_index: int
    node_id: int
    generic_node_id: int
    bdf_id: int
    domain: int
    unique_id: str


class _HsaApi:
    def __init__(self, library_path: Path) -> None:
        try:
            mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            self._library = ctypes.CDLL(str(library_path), mode=mode)
            self.callback_type = ctypes.CFUNCTYPE(
                ctypes.c_uint32, _HsaAgent, ctypes.c_void_p
            )
            self._configure()
        except (OSError, AttributeError) as error:
            raise _InventoryError("HSA library or required symbols are unavailable") from error

    def _configure(self) -> None:
        self._library.hsa_init.argtypes = []
        self._library.hsa_init.restype = ctypes.c_uint32
        self._library.hsa_shut_down.argtypes = []
        self._library.hsa_shut_down.restype = ctypes.c_uint32
        self._library.hsa_iterate_agents.argtypes = [
            self.callback_type,
            ctypes.c_void_p,
        ]
        self._library.hsa_iterate_agents.restype = ctypes.c_uint32
        self._library.hsa_agent_get_info.argtypes = [
            _HsaAgent,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        self._library.hsa_agent_get_info.restype = ctypes.c_uint32

    def init(self) -> int:
        return int(self._library.hsa_init())

    def shutdown(self) -> int:
        return int(self._library.hsa_shut_down())

    def iterate(self, callback: object) -> int:
        return int(self._library.hsa_iterate_agents(callback, None))

    def uint32_info(self, agent: _HsaAgent, attribute: int, name: str) -> int:
        value = ctypes.c_uint32(0)
        status = int(
            self._library.hsa_agent_get_info(agent, attribute, ctypes.byref(value))
        )
        if status != _HSA_STATUS_SUCCESS:
            raise _InventoryError(f"HSA {name} query failed with status {status}")
        return int(value.value)

    def uuid(self, agent: _HsaAgent) -> str:
        value = ctypes.create_string_buffer(_HSA_UUID_CAPACITY)
        status = int(
            self._library.hsa_agent_get_info(
                agent, _HSA_AMD_AGENT_INFO_UUID, ctypes.byref(value)
            )
        )
        if status != _HSA_STATUS_SUCCESS:
            raise _InventoryError(f"HSA UUID query failed with status {status}")
        try:
            observed = value.value.decode("ascii")
        except UnicodeDecodeError as error:
            raise _InventoryError("HSA returned a non-ASCII GPU UUID") from error
        canonical = "GPU-" + observed.removeprefix("GPU-").lower()
        if not _GPU_UUID.fullmatch(canonical):
            raise _InventoryError("HSA returned an invalid or unsupported GPU UUID")
        return canonical


def _inventory(api: _HsaApi) -> tuple[_GpuAgent, ...]:
    devices: list[_GpuAgent] = []
    handles: set[int] = set()
    callback_failure: list[str] = []

    @api.callback_type
    def visit(agent: _HsaAgent, _data: ctypes.c_void_p) -> int:
        try:
            if agent.handle in handles:
                raise _InventoryError("HSA enumerated a duplicate agent handle")
            handles.add(agent.handle)
            device_type = api.uint32_info(agent, _HSA_AGENT_INFO_DEVICE, "device type")
            if device_type != _HSA_DEVICE_TYPE_GPU:
                return _HSA_STATUS_SUCCESS
            if len(devices) >= _MAX_GPU_AGENTS:
                raise _InventoryError("HSA GPU inventory exceeds the supported bound")
            devices.append(_read_gpu(api, agent, len(devices)))
            return _HSA_STATUS_SUCCESS
        except BaseException as error:
            callback_failure.append(str(error) or type(error).__name__)
            return _HSA_STATUS_ERROR

    status = api.iterate(visit)
    if callback_failure:
        raise _InventoryError(callback_failure[0])
    if status != _HSA_STATUS_SUCCESS:
        raise _InventoryError(f"HSA agent iteration failed with status {status}")
    result = tuple(devices)
    _validate_inventory(result)
    return result


def _read_gpu(api: _HsaApi, agent: _HsaAgent, ordinal: int) -> _GpuAgent:
    return _GpuAgent(
        hsa_gpu_index=ordinal,
        node_id=api.uint32_info(
            agent, _HSA_AMD_AGENT_INFO_DRIVER_NODE_ID, "driver node ID"
        ),
        generic_node_id=api.uint32_info(
            agent, _HSA_AGENT_INFO_NODE, "generic node ID"
        ),
        bdf_id=api.uint32_info(agent, _HSA_AMD_AGENT_INFO_BDFID, "BDF ID"),
        domain=api.uint32_info(agent, _HSA_AMD_AGENT_INFO_DOMAIN, "PCI domain"),
        unique_id=api.uuid(agent),
    )


def _validate_inventory(devices: tuple[_GpuAgent, ...]) -> None:
    if not devices:
        raise _InventoryError("HSA returned no GPU agents")
    if tuple(device.hsa_gpu_index for device in devices) != tuple(range(len(devices))):
        raise _InventoryError("HSA GPU ordinals are not contiguous")
    for attribute in ("node_id", "generic_node_id", "unique_id"):
        if len({getattr(device, attribute) for device in devices}) != len(devices):
            raise _InventoryError(f"HSA returned duplicate {attribute} values")
    locations = {(device.domain, device.bdf_id) for device in devices}
    if len(locations) != len(devices):
        raise _InventoryError("HSA returned duplicate PCI locations")


def _library_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise _InventoryError("HSA library path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise _InventoryError("HSA library path does not resolve") from error
    if not resolved.is_file():
        raise _InventoryError("HSA library path is not a regular file")
    return resolved


def _clear_visibility_filters() -> None:
    for name in _VISIBILITY_VARIABLES:
        os.environ.pop(name, None)


def _run(library_path: Path) -> dict[str, object]:
    _clear_visibility_filters()
    api = _HsaApi(library_path)
    init_status = api.init()
    if init_status != _HSA_STATUS_SUCCESS:
        raise _InventoryError(f"HSA initialization failed with status {init_status}")
    failure: BaseException | None = None
    devices: tuple[_GpuAgent, ...] | None = None
    try:
        devices = _inventory(api)
    except BaseException as error:
        failure = error
    shutdown_status = api.shutdown()
    if failure is not None:
        if shutdown_status != _HSA_STATUS_SUCCESS:
            raise _InventoryError(
                f"{failure}; HSA shutdown also failed with status {shutdown_status}"
            ) from failure
        raise failure
    if shutdown_status != _HSA_STATUS_SUCCESS or devices is None:
        raise _InventoryError(f"HSA shutdown failed with status {shutdown_status}")
    return {"schema_version": 1, "devices": [asdict(device) for device in devices]}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write("usage: _hsa_inventory_helper.py /absolute/path/to/libhsa\n")
        return 2
    try:
        payload = _run(_library_path(arguments[0]))
    except BaseException as error:
        sys.stderr.write(f"HSA inventory failed: {error}\n")
        return 1
    sys.stdout.write(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
