"""Fail-closed HSA/KFD selector resolution to physical RSMI devices."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.core import ContractError

from .hsa_inventory import HsaInventoryEvidence


_GPU_UUID = re.compile(r"^GPU-[0-9a-f]{16}$")
_SELECTOR_ENV_NAMES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


@dataclass(frozen=True, slots=True)
class RsmiDeviceIdentity:
    """One device in ROCm SMI's monitoring-index namespace."""

    rsmi_index: int
    node_id: int
    pci_id: int
    unique_id: str
    render_minor: int

    def __post_init__(self) -> None:
        if (
            self.rsmi_index < 0
            or self.node_id < 0
            or self.pci_id < 0
            or self.pci_id >= 2**64
            or not _GPU_UUID.fullmatch(self.unique_id)
            or self.render_minor < 128
        ):
            raise ContractError(
                "ROCm SMI returned an invalid GPU identity",
                "invalid_gpu_physical_identity",
            )


@dataclass(frozen=True, slots=True)
class GpuDeviceIdentity:
    """One GPU joined across HSA, KFD, DRM, and RSMI namespaces."""

    hsa_gpu_index: int
    kfd_node_id: int
    rsmi_index: int
    unique_id: str
    render_node: str

    def __post_init__(self) -> None:
        if (
            self.hsa_gpu_index < 0
            or self.kfd_node_id < 0
            or self.rsmi_index < 0
            or not _GPU_UUID.fullmatch(self.unique_id)
            or not re.fullmatch(r"/dev/dri/renderD[0-9]+", self.render_node)
        ):
            raise ContractError(
                "The HSA/KFD/RSMI GPU identity is invalid",
                "invalid_gpu_physical_identity",
            )

    @property
    def render_minor(self) -> int:
        return int(self.render_node.removeprefix("/dev/dri/renderD"))


@dataclass(frozen=True, slots=True)
class GpuSelectorRequest:
    """Requested selector plus every ambient selector affecting HIP visibility."""

    requested: tuple[str, ...] | None = None
    rocr_visible_devices: tuple[str, ...] | None = None
    hip_visible_devices: tuple[str, ...] | None = None
    cuda_visible_devices: tuple[str, ...] | None = None
    gpu_device_ordinal: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for source, selectors in self.sources:
            if selectors is not None:
                _validate_selectors(selectors, source=source)

    @property
    def sources(self) -> tuple[tuple[str, tuple[str, ...] | None], ...]:
        return (
            ("requested", self.requested),
            ("ROCR_VISIBLE_DEVICES", self.rocr_visible_devices),
            ("HIP_VISIBLE_DEVICES", self.hip_visible_devices),
            ("CUDA_VISIBLE_DEVICES", self.cuda_visible_devices),
            ("GPU_DEVICE_ORDINAL", self.gpu_device_ordinal),
        )

    @property
    def selector_scope(self) -> str:
        if self.requested is None:
            return "all-visible-amd-gpus"
        return "amd-gpu-set=" + ",".join(self.requested)

    def to_dict(self) -> dict[str, list[str] | None]:
        return {
            "requested": _optional_list(self.requested),
            "ROCR_VISIBLE_DEVICES": _optional_list(self.rocr_visible_devices),
            "HIP_VISIBLE_DEVICES": _optional_list(self.hip_visible_devices),
            "CUDA_VISIBLE_DEVICES": _optional_list(self.cuda_visible_devices),
            "GPU_DEVICE_ORDINAL": _optional_list(self.gpu_device_ordinal),
        }


def selector_scope(requested_devices: str | None) -> str:
    """Validate a caller selector while deferring physical resolution."""

    if requested_devices is None:
        return "all-visible-amd-gpus"
    selectors = parse_selector_list(requested_devices, source="requested GPU devices")
    return "amd-gpu-set=" + ",".join(selectors)


def capture_selector_request(
    scope: str, *, environment: Mapping[str, str] | None = None
) -> GpuSelectorRequest:
    """Capture requested and ambient selectors without consulting GPU state."""

    requested = _requested_from_scope(scope)
    values = os.environ if environment is None else environment
    ambient: dict[str, tuple[str, ...] | None] = {}
    for name in _SELECTOR_ENV_NAMES:
        raw = values.get(name)
        ambient[name] = (
            parse_selector_list(raw, source=name) if raw is not None else None
        )
    return GpuSelectorRequest(
        requested=requested,
        rocr_visible_devices=ambient["ROCR_VISIBLE_DEVICES"],
        hip_visible_devices=ambient["HIP_VISIBLE_DEVICES"],
        cuda_visible_devices=ambient["CUDA_VISIBLE_DEVICES"],
        gpu_device_ordinal=ambient["GPU_DEVICE_ORDINAL"],
    )


def parse_selector_list(raw: str, *, source: str) -> tuple[str, ...]:
    selectors = tuple(part.strip() for part in raw.split(","))
    _validate_selectors(selectors, source=source)
    return selectors


def resolve_gpu_inventory(
    rsmi_devices: tuple[RsmiDeviceIdentity, ...],
    *,
    hsa_inventory: HsaInventoryEvidence,
    topology_root: Path,
) -> tuple[GpuDeviceIdentity, ...]:
    """Join clean HSA order to KFD/DRM/RSMI by physical identities."""

    _validate_rsmi_inventory(rsmi_devices)
    hsa_node_ids = frozenset(device.node_id for device in hsa_inventory.devices)
    kfd_devices = _read_kfd_gpu_devices(
        topology_root, required_node_ids=hsa_node_ids
    )
    by_node = {device.node_id: device for device in kfd_devices}
    if set(by_node) != hsa_node_ids:
        _mapping_error("Clean HSA and KFD GPU node inventories disagree")
    by_key = {
        (device.unique_id, device.node_id, device.pci_id, device.render_minor): device
        for device in rsmi_devices
    }
    if len(by_key) != len(rsmi_devices):
        _mapping_error("ROCm SMI returned duplicate GPU identity keys")
    result: list[GpuDeviceIdentity] = []
    for hsa in hsa_inventory.devices:
        kfd = by_node[hsa.node_id]
        if (
            hsa.unique_id != kfd.unique_id
            or hsa.bdf_id != kfd.bdf_id
            or hsa.domain != kfd.domain
        ):
            _mapping_error(
                "Clean HSA and KFD identity fields disagree",
                hsa_gpu_index=hsa.hsa_gpu_index,
                node_id=hsa.node_id,
            )
        match = by_key.get(
            (hsa.unique_id, hsa.node_id, hsa.pci_id, kfd.render_minor)
        )
        if match is None:
            _mapping_error(
                "A KFD GPU agent has no matching ROCm SMI physical device",
                node_id=kfd.node_id,
                unique_id=kfd.unique_id,
                render_minor=kfd.render_minor,
            )
        result.append(
            GpuDeviceIdentity(
                hsa_gpu_index=hsa.hsa_gpu_index,
                kfd_node_id=kfd.node_id,
                rsmi_index=match.rsmi_index,
                unique_id=kfd.unique_id,
                render_node=f"/dev/dri/renderD{kfd.render_minor}",
            )
        )
    if len({device.rsmi_index for device in result}) != len(result):
        _mapping_error(
            "Visible HSA GPU agents do not map injectively onto ROCm SMI devices"
        )
    return tuple(result)


def resolve_gpu_selection(
    inventory: tuple[GpuDeviceIdentity, ...], request: GpuSelectorRequest
) -> tuple[GpuDeviceIdentity, ...]:
    """Compose ROCR and HIP visibility, then verify the caller's physical request."""

    _validate_inventory(inventory)
    base = inventory
    rocr = (
        _resolve_hsa_selectors(base, request.rocr_visible_devices, "ROCR_VISIBLE_DEVICES")
        if request.rocr_visible_devices is not None
        else base
    )
    hip_results: list[tuple[str, tuple[GpuDeviceIdentity, ...]]] = []
    for name, selectors in request.sources[2:]:
        if selectors is not None:
            hip_results.append((name, _resolve_hip_ordinals(rocr, selectors, name)))
    if hip_results and any(result != hip_results[0][1] for _, result in hip_results[1:]):
        raise ContractError(
            "HIP visibility aliases resolve to different physical GPU orderings",
            "gpu_visibility_mismatch",
            {name: [device.unique_id for device in result] for name, result in hip_results},
        )
    ambient_present = request.rocr_visible_devices is not None or bool(hip_results)
    ambient = hip_results[0][1] if hip_results else rocr
    requested = (
        _resolve_hsa_selectors(base, request.requested, "requested GPU devices")
        if request.requested is not None
        else None
    )
    if requested is not None and ambient_present and requested != ambient:
        raise ContractError(
            "Requested and ambient selectors resolve to different physical GPUs",
            "gpu_visibility_mismatch",
            {
                "requested": [device.unique_id for device in requested],
                "ambient": [device.unique_id for device in ambient],
            },
        )
    selected = requested if requested is not None else ambient
    if not selected:
        _mapping_error("GPU selectors resolve to an empty physical device set")
    return selected


@dataclass(frozen=True, slots=True)
class _KfdGpuIdentity:
    node_id: int
    unique_id: str
    render_minor: int
    bdf_id: int
    domain: int


def _read_kfd_gpu_devices(
    topology_root: Path, *, required_node_ids: frozenset[int]
) -> tuple[_KfdGpuIdentity, ...]:
    if not topology_root.is_absolute() or not required_node_ids:
        _mapping_error("The KFD topology root is not absolute")
    try:
        if not topology_root.is_dir():
            raise OSError("not a directory")
    except OSError as error:
        raise ContractError(
            "The KFD topology inventory is unavailable",
            "gpu_physical_mapping_unresolved",
            {"topology_root": str(topology_root)},
        ) from error
    node_paths = tuple(topology_root / str(node_id) for node_id in sorted(required_node_ids))
    devices: list[_KfdGpuIdentity] = []
    for node_path in node_paths:
        properties = _read_properties(node_path / "properties")
        if not properties:
            continue
        if properties.get("simd_count", 0) <= 0:
            continue
        required = (
            "cpu_cores_count",
            "drm_render_minor",
            "unique_id",
            "location_id",
            "domain",
        )
        if any(name not in properties for name in required):
            _mapping_error("A KFD GPU node is missing identity properties", node_id=int(node_path.name))
        if properties["cpu_cores_count"] != 0:
            _mapping_error("A KFD node has both CPU cores and GPU SIMD units", node_id=int(node_path.name))
        unique_id = properties["unique_id"]
        render_minor = properties["drm_render_minor"]
        bdf_id = properties["location_id"]
        domain = properties["domain"]
        if (
            unique_id <= 0
            or unique_id >= 2**64
            or render_minor < 128
            or bdf_id < 0
            or bdf_id >= 2**32
            or domain < 0
            or domain >= 2**32
        ):
            _mapping_error("A KFD GPU node has an invalid physical identity", node_id=int(node_path.name))
        devices.append(
            _KfdGpuIdentity(
                int(node_path.name),
                f"GPU-{unique_id:016x}",
                render_minor,
                bdf_id,
                domain,
            )
        )
    if not devices:
        _mapping_error("The KFD topology contains no visible GPU agents")
    keys = {
        (
            device.unique_id,
            device.node_id,
            (device.domain << 32) | device.bdf_id,
            device.render_minor,
        )
        for device in devices
    }
    if len(keys) != len(devices):
        _mapping_error("KFD returned duplicate GPU identity keys")
    return tuple(devices)


def _read_properties(path: Path) -> dict[str, int]:
    try:
        content = path.read_text(encoding="ascii")
    except OSError as error:
        raise ContractError(
            "A KFD topology node is unreadable",
            "gpu_physical_mapping_unresolved",
            {"path": str(path)},
        ) from error
    if not content.strip():
        return {}
    result: dict[str, int] = {}
    try:
        for line in content.splitlines():
            key, raw = line.split()
            if key in result:
                raise ValueError(key)
            result[key] = int(raw, 10)
    except (ValueError, TypeError) as error:
        raise ContractError(
            "A KFD topology properties file is malformed",
            "gpu_physical_mapping_unresolved",
            {"path": str(path)},
        ) from error
    return result


def _resolve_hsa_selectors(
    devices: tuple[GpuDeviceIdentity, ...], selectors: tuple[str, ...], source: str
) -> tuple[GpuDeviceIdentity, ...]:
    selected: list[GpuDeviceIdentity] = []
    for selector in selectors:
        matches = [device for device in devices if _hsa_selector_matches(device, selector)]
        if len(matches) != 1:
            _mapping_error(
                "A GPU selector does not resolve to exactly one HSA agent",
                source=source,
                selector=selector,
            )
        selected.append(matches[0])
    if len({device.unique_id for device in selected}) != len(selected):
        _mapping_error("GPU selectors resolve to duplicate physical devices", source=source)
    return tuple(selected)


def _resolve_hip_ordinals(
    exposed: tuple[GpuDeviceIdentity, ...], selectors: tuple[str, ...], source: str
) -> tuple[GpuDeviceIdentity, ...]:
    if any(not selector.isdecimal() for selector in selectors):
        raise ContractError(
            f"{source} must contain only HIP logical ordinals",
            "invalid_gpu_device_scope",
        )
    selected: list[GpuDeviceIdentity] = []
    for selector in selectors:
        index = int(selector)
        if index >= len(exposed):
            _mapping_error("A HIP logical ordinal is outside the exposed GPU set", source=source, selector=selector)
        selected.append(exposed[index])
    if len({device.unique_id for device in selected}) != len(selected):
        _mapping_error("HIP logical ordinals resolve to duplicate physical devices", source=source)
    return tuple(selected)


def _hsa_selector_matches(device: GpuDeviceIdentity, selector: str) -> bool:
    if selector.isdecimal():
        return device.hsa_gpu_index == int(selector)
    normalized = selector.lower()
    if normalized.startswith("gpu-"):
        normalized = normalized.removeprefix("gpu-")
    elif normalized.startswith("0x"):
        normalized = normalized.removeprefix("0x")
    return len(normalized) == 16 and f"GPU-{normalized}" == device.unique_id


def _requested_from_scope(scope: str) -> tuple[str, ...] | None:
    if scope == "all-visible-amd-gpus":
        return None
    prefix = "amd-gpu-set="
    if not scope.startswith(prefix):
        raise ContractError("GPU selector scope is invalid", "invalid_gpu_device_scope")
    return parse_selector_list(scope[len(prefix) :], source="requested GPU devices")


def _validate_selectors(selectors: tuple[str, ...], *, source: str) -> None:
    if (
        not selectors
        or any(not item or len(item) > 128 or "\x00" in item for item in selectors)
        or len(selectors) != len(set(selectors))
    ):
        raise ContractError(
            f"{source} must be a non-empty comma-separated list without duplicates",
            "invalid_gpu_device_scope",
        )


def _validate_rsmi_inventory(devices: tuple[RsmiDeviceIdentity, ...]) -> None:
    if not devices or tuple(device.rsmi_index for device in devices) != tuple(range(len(devices))):
        _mapping_error("ROCm SMI inventory indices are not contiguous")
    if len({device.unique_id for device in devices}) != len(devices):
        _mapping_error("ROCm SMI returned duplicate physical GPU IDs")
    if len({device.render_minor for device in devices}) != len(devices):
        _mapping_error("ROCm SMI returned duplicate DRM render nodes")


def _validate_inventory(devices: tuple[GpuDeviceIdentity, ...]) -> None:
    if not devices or tuple(device.hsa_gpu_index for device in devices) != tuple(range(len(devices))):
        _mapping_error("HSA GPU ordinals are not contiguous")
    for attribute in ("kfd_node_id", "rsmi_index", "unique_id", "render_node"):
        if len({getattr(device, attribute) for device in devices}) != len(devices):
            _mapping_error(f"GPU inventory has duplicate {attribute} values")


def _optional_list(value: tuple[str, ...] | None) -> list[str] | None:
    return list(value) if value is not None else None


def _mapping_error(message: str, **details: object) -> None:
    raise ContractError(message, "gpu_physical_mapping_unresolved", details or None)


__all__ = [
    "GpuDeviceIdentity",
    "GpuSelectorRequest",
    "RsmiDeviceIdentity",
    "capture_selector_request",
    "parse_selector_list",
    "resolve_gpu_inventory",
    "resolve_gpu_selection",
    "selector_scope",
]
