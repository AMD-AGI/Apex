"""Bind KFD/RSMI process evidence to one observed Docker container and GPU lease."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError
from apex.runtime import GpuOwnershipInspector, RocmSmiGpuOwnershipInspector

from .docker_observation import DockerContainerObservation


_UNIQUE_ID = re.compile(r"GPU-[0-9a-f]{16}")


class ContainerGpuObserver(Protocol):
    def observe(
        self,
        container: DockerContainerObservation,
        gpu_lease: Mapping[str, object],
    ) -> Mapping[str, object] | None: ...


class RocmContainerGpuObserver:
    """Use Apex's race-checked RSMI map and procfs cgroups as authority."""

    def __init__(
        self,
        inspector: GpuOwnershipInspector | None = None,
        *,
        proc_root: Path = Path("/proc"),
    ) -> None:
        self._inspector = inspector or RocmSmiGpuOwnershipInspector()
        self._proc_root = proc_root

    def observe(
        self,
        container: DockerContainerObservation,
        gpu_lease: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        selector, expected = _lease_devices(gpu_lease)
        receipt = self._inspector.inspect(selector)
        actual = {
            device.rsmi_index: device.unique_id for device in receipt.selected_devices
        }
        if actual != expected:
            raise ContractError(
                "GPU inventory differs from the active lease", "magpie_gpu_lease_drift"
            )
        owners = (*receipt.allowed_owners, *receipt.foreign_owners)
        matched = tuple(
            owner for owner in owners
            if _container_process(
                owner.pid, container.container_id, proc_root=self._proc_root
            )
        )
        if not matched:
            return None
        if len(matched) != len(owners):
            raise ContractError(
                "A leased GPU has a process outside the Magpie container",
                "magpie_gpu_process_escape",
            )
        engaged = {index for owner in matched for index in owner.rsmi_device_indices}
        if engaged != set(expected):
            return None
        return {
            "devices": [
                {"rsmi_index": index, "unique_id": expected[index]}
                for index in sorted(expected)
            ],
            "processes": [
                {
                    "pid": owner.pid,
                    "uid": owner.uid,
                    "start_time_ticks": owner.start_time_ticks,
                    "cmdline_sha256": owner.cmdline_sha256,
                    "rsmi_device_indices": list(owner.rsmi_device_indices),
                    "container_id": container.container_id,
                }
                for owner in matched
            ],
            "ownership_receipt_sha256": receipt.digest,
        }


def _lease_devices(
    value: Mapping[str, object],
) -> tuple[str, dict[int, str]]:
    if value.get("owner_pid") != os.getpid():
        raise ContractError("GPU lease owner differs from Apex", "magpie_gpu_lease_mismatch")
    ownership = value.get("ownership")
    if not isinstance(ownership, Mapping):
        raise ContractError("GPU lease ownership is absent", "magpie_gpu_lease_mismatch")
    selector = ownership.get("selector_scope")
    selected = ownership.get("selected_devices")
    if not isinstance(selector, str) or not selector or not isinstance(selected, list):
        raise ContractError("GPU lease selection is invalid", "magpie_gpu_lease_mismatch")
    devices: dict[int, str] = {}
    for item in selected:
        if not isinstance(item, Mapping):
            raise ContractError("GPU lease device is invalid", "magpie_gpu_lease_mismatch")
        index, unique_id = item.get("rsmi_index"), item.get("unique_id")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(unique_id, str)
            or not _UNIQUE_ID.fullmatch(unique_id)
            or index in devices
        ):
            raise ContractError("GPU lease device is invalid", "magpie_gpu_lease_mismatch")
        devices[index] = unique_id
    if not devices:
        raise ContractError("GPU lease selects no device", "magpie_gpu_lease_mismatch")
    return selector, devices


def _container_process(pid: int, identifier: str, *, proc_root: Path) -> bool:
    path = proc_root / str(pid) / "cgroup"
    try:
        first = path.read_text(encoding="utf-8")
        second = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(
            "GPU process cgroup cannot be inspected", "magpie_gpu_process_unavailable"
        ) from error
    if first != second:
        raise ContractError("GPU process cgroup raced", "magpie_gpu_process_race")
    return identifier in first or identifier[:12] in first


__all__ = ["ContainerGpuObserver", "RocmContainerGpuObserver"]
