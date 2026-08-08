"""Authoritative AMD GPU process ownership observed before a run starts."""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, sha256_bytes, sha256_file, sha256_json

from .gpu_topology import (
    GpuDeviceIdentity,
    GpuSelectorRequest,
    RsmiDeviceIdentity,
    capture_selector_request,
    resolve_gpu_inventory,
    resolve_gpu_selection,
)
from .hsa_inventory import (
    CleanHsaInventoryProvider,
    HsaInventoryEvidence,
    HsaInventoryProvider,
)
from .gpu_rsmi import (
    CtypesOwnershipApi,
    MAX_RSMI_DEVICES,
    MAX_RSMI_PROCESSES,
    OwnershipApi,
    resolve_rsmi_library,
)


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GpuProcessIdentity:
    pid: int
    uid: int
    start_time_ticks: int
    cmdline_sha256: str
    rsmi_device_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.pid <= 0
            or self.uid < 0
            or self.start_time_ticks <= 0
            or not _DIGEST.fullmatch(self.cmdline_sha256)
            or not self.rsmi_device_indices
            or tuple(sorted(set(self.rsmi_device_indices)))
            != self.rsmi_device_indices
        ):
            raise ContractError(
                "ROCm SMI returned an invalid GPU process identity",
                "invalid_gpu_process_identity",
            )


@dataclass(frozen=True, slots=True)
class GpuOwnershipReceipt:
    schema_version: int
    policy_id: str
    selector_inputs: GpuSelectorRequest
    observed_unix_ns: int
    library_path: str
    library_sha256: str
    topology_root: str
    hsa_inventory: HsaInventoryEvidence
    rsmi_monitor_inventory: tuple[RsmiDeviceIdentity, ...]
    device_inventory: tuple[GpuDeviceIdentity, ...]
    selected_devices: tuple[GpuDeviceIdentity, ...]
    allowed_owners: tuple[GpuProcessIdentity, ...]
    foreign_owners: tuple[GpuProcessIdentity, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.policy_id != "clean_hsa_kfd_rsmi_process_gpu_map_v2"
            or self.observed_unix_ns <= 0
            or not Path(self.library_path).is_absolute()
            or not Path(self.topology_root).is_absolute()
            or not _DIGEST.fullmatch(self.library_sha256)
            or not self.rsmi_monitor_inventory
            or not self.device_inventory
            or not self.selected_devices
        ):
            raise ContractError(
                "GPU ownership receipt is incomplete",
                "invalid_gpu_ownership_receipt",
            )
        _validate_receipt_inventory(self)
        expected = resolve_gpu_selection(self.device_inventory, self.selector_inputs)
        if expected != self.selected_devices:
            raise ContractError(
                "GPU ownership receipt selector resolution is inconsistent",
                "invalid_gpu_ownership_receipt",
            )

    @property
    def selector_scope(self) -> str:
        return self.selector_inputs.selector_scope

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "selector_scope": self.selector_scope,
            "selector_inputs": self.selector_inputs.to_dict(),
            "observed_unix_ns": self.observed_unix_ns,
            "library_path": self.library_path,
            "library_sha256": self.library_sha256,
            "topology_root": self.topology_root,
            "hsa_inventory": self.hsa_inventory.to_dict(),
            "rsmi_monitor_inventory": [
                asdict(item) for item in self.rsmi_monitor_inventory
            ],
            "device_inventory": [asdict(item) for item in self.device_inventory],
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

    @property
    def execution_scope(self) -> str:
        values = (device.unique_id for device in self.selected_devices)
        return "amd-gpu-set=" + ",".join(values)


class GpuOwnershipInspector(Protocol):
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt: ...


class RocmSmiGpuOwnershipInspector:
    """Resolve physical devices and map every KFD PID through librocm_smi."""

    def __init__(
        self,
        *,
        library_path: Path | None = None,
        topology_root: Path = Path("/sys/class/kfd/kfd/topology/nodes"),
        hsa_inventory_provider: HsaInventoryProvider | None = None,
    ) -> None:
        self._library_path = library_path
        self._topology_root = topology_root
        self._hsa_inventory_provider = (
            hsa_inventory_provider or CleanHsaInventoryProvider()
        )

    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        library = resolve_rsmi_library(self._library_path)
        selector_inputs = capture_selector_request(selector_scope)
        library_sha256 = sha256_file(library)
        receipt = collect_gpu_ownership(
            CtypesOwnershipApi(library),
            selector_scope=selector_scope,
            selector_inputs=selector_inputs,
            allowed_pids=allowed_pids,
            library_path=library,
            library_sha256=library_sha256,
            topology_root=self._topology_root,
            hsa_inventory=self._hsa_inventory_provider.collect(),
        )
        if sha256_file(library) != library_sha256:
            raise ContractError(
                "The ROCm SMI library changed during ownership inspection",
                "gpu_ownership_api_changed",
            )
        return receipt


def collect_gpu_ownership(
    api: OwnershipApi,
    *,
    selector_scope: str,
    selector_inputs: GpuSelectorRequest | None = None,
    allowed_pids: tuple[int, ...],
    library_path: Path,
    library_sha256: str,
    topology_root: Path,
    hsa_inventory: HsaInventoryEvidence,
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
            selector_inputs=(
                selector_inputs
                if selector_inputs is not None
                else capture_selector_request(selector_scope, environment={})
            ),
            allowed_pids=allowed_pids,
            library_path=library_path,
            library_sha256=library_sha256,
            topology_root=topology_root,
            hsa_inventory=hsa_inventory,
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
    selector_inputs: GpuSelectorRequest,
    allowed_pids: tuple[int, ...],
    library_path: Path,
    library_sha256: str,
    topology_root: Path,
    hsa_inventory: HsaInventoryEvidence,
    proc_root: Path,
    observed_unix_ns: int | None,
) -> GpuOwnershipReceipt:
    if selector_inputs.selector_scope != selector_scope:
        raise ContractError(
            "GPU selector scope and captured inputs disagree",
            "invalid_gpu_device_scope",
        )
    rsmi_devices = _query_devices(api)
    devices = resolve_gpu_inventory(
        rsmi_devices,
        hsa_inventory=hsa_inventory,
        topology_root=topology_root,
    )
    selected = resolve_gpu_selection(devices, selector_inputs)
    selected_indices = {device.rsmi_index for device in selected}
    valid_rsmi_indices = set(range(len(rsmi_devices)))
    first = _query_process_map(api, valid_indices=valid_rsmi_indices)
    first_owners = _selected_process_identities(
        first, selected_indices=selected_indices, proc_root=proc_root
    )
    second = _query_process_map(api, valid_indices=valid_rsmi_indices)
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
        schema_version=2,
        policy_id="clean_hsa_kfd_rsmi_process_gpu_map_v2",
        selector_inputs=selector_inputs,
        observed_unix_ns=(
            observed_unix_ns if observed_unix_ns is not None else time.time_ns()
        ),
        library_path=str(library_path),
        library_sha256=library_sha256,
        topology_root=str(topology_root),
        hsa_inventory=hsa_inventory,
        rsmi_monitor_inventory=rsmi_devices,
        device_inventory=devices,
        selected_devices=selected,
        allowed_owners=tuple(owner for owner in second_owners if owner.pid in allowed),
        foreign_owners=tuple(owner for owner in second_owners if owner.pid not in allowed),
    )


def _validate_receipt_inventory(receipt: GpuOwnershipReceipt) -> None:
    monitors = receipt.rsmi_monitor_inventory
    if tuple(item.rsmi_index for item in monitors) != tuple(range(len(monitors))):
        raise ContractError(
            "GPU ownership monitor inventory is not contiguous",
            "invalid_gpu_ownership_receipt",
        )
    for field in ("unique_id", "node_id", "pci_id", "render_minor"):
        if len({getattr(item, field) for item in monitors}) != len(monitors):
            raise ContractError(
                f"GPU ownership monitor inventory duplicates {field}",
                "invalid_gpu_ownership_receipt",
            )
    if len(receipt.device_inventory) != len(receipt.hsa_inventory.devices):
        raise ContractError(
            "GPU ownership HSA and joined inventory sizes differ",
            "invalid_gpu_ownership_receipt",
        )
    if len({item.rsmi_index for item in receipt.device_inventory}) != len(
        receipt.device_inventory
    ):
        raise ContractError(
            "GPU ownership joined inventory duplicates an RSMI index",
            "invalid_gpu_ownership_receipt",
        )
    valid_rsmi_indices = set(range(len(monitors)))
    owners = (*receipt.allowed_owners, *receipt.foreign_owners)
    if len({owner.pid for owner in owners}) != len(owners) or any(
        not set(owner.rsmi_device_indices).issubset(valid_rsmi_indices)
        for owner in owners
    ):
        raise ContractError(
            "GPU ownership process identities are inconsistent with RSMI",
            "invalid_gpu_ownership_receipt",
        )
    by_rsmi_index = {item.rsmi_index: item for item in monitors}
    for joined, hsa in zip(
        receipt.device_inventory, receipt.hsa_inventory.devices, strict=True
    ):
        monitor = by_rsmi_index.get(joined.rsmi_index)
        if (
            monitor is None
            or joined.hsa_gpu_index != hsa.hsa_gpu_index
            or joined.kfd_node_id != hsa.node_id
            or joined.unique_id != hsa.unique_id
            or monitor.node_id != hsa.node_id
            or monitor.pci_id != hsa.pci_id
            or monitor.unique_id != hsa.unique_id
            or monitor.render_minor != joined.render_minor
        ):
            raise ContractError(
                "GPU ownership namespace inventories are inconsistent",
                "invalid_gpu_ownership_receipt",
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


def _query_devices(api: OwnershipApi) -> tuple[RsmiDeviceIdentity, ...]:
    status, count = api.device_count()
    if status != 0 or count < 1 or count > MAX_RSMI_DEVICES:
        raise ContractError(
            "ROCm SMI device inventory failed",
            "gpu_physical_mapping_unresolved",
            {"status": status, "count": count},
        )
    devices: list[RsmiDeviceIdentity] = []
    for index in range(count):
        identity_status, unique_id, node_id, pci_id, render_minor = api.device_identity(
            index
        )
        if identity_status != 0:
            raise ContractError(
                "ROCm SMI device identity query failed",
                "gpu_physical_mapping_unresolved",
                {"index": index, "status": identity_status},
            )
        devices.append(
            RsmiDeviceIdentity(
                rsmi_index=index,
                node_id=node_id,
                pci_id=pci_id,
                unique_id=f"GPU-{unique_id:016x}",
                render_minor=render_minor,
            )
        )
    if len({device.unique_id for device in devices}) != len(devices):
        raise ContractError(
            "ROCm SMI returned duplicate physical GPU IDs",
            "gpu_physical_mapping_unresolved",
        )
    return tuple(devices)


def _query_process_map(
    api: OwnershipApi, *, valid_indices: set[int]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    status, pids = api.process_pids()
    if status != 0 or len(pids) > MAX_RSMI_PROCESSES or any(pid <= 0 for pid in pids):
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
        if (
            device_status != 0
            or not canonical
            or canonical != tuple(sorted(indices))
            or not set(canonical).issubset(valid_indices)
        ):
            raise ContractError(
                "ROCm SMI process-to-GPU query failed",
                "gpu_ownership_query_failed",
                {"pid": pid, "status": device_status},
            )
        result.append((pid, canonical))
    return tuple(result)


def _process_identity(
    pid: int, rsmi_device_indices: tuple[int, ...], *, proc_root: Path
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
        rsmi_device_indices,
    )


__all__ = [
    "GpuDeviceIdentity",
    "GpuOwnershipInspector",
    "GpuOwnershipReceipt",
    "GpuProcessIdentity",
    "OwnershipApi",
    "RocmSmiGpuOwnershipInspector",
    "collect_gpu_ownership",
]
