"""Use caller-assigned GPUs without inspecting host processes or acquiring them."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Callable, Mapping

from apex.core import ContractError, sha256_json

_PROBE = """
import json, torch
if not torch.cuda.is_available():
    raise RuntimeError('No GPU is available')
devices = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    x = torch.ones(1, device=f'cuda:{i}')
    x.add_(1)
    torch.cuda.synchronize(i)
    if x.item() != 2:
        raise RuntimeError('GPU smoke operation failed')
    devices.append({'index': i, 'name': p.name, 'arch': getattr(p, 'gcnArchName', '')})
print(json.dumps({'devices': devices, 'torch': torch.__version__, 'hip': torch.version.hip}))
"""


def probe_assigned_gpus() -> dict[str, object]:
    try:
        result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True,
                                text=True, check=True, timeout=60)
        value = json.loads(result.stdout)
        if not value.get("devices"):
            raise ValueError("empty GPU inventory")
        return value
    except (subprocess.SubprocessError, ValueError) as error:
        raise ContractError("Assigned GPU smoke failed", "assigned_gpu_unavailable") from error


@dataclass(frozen=True, slots=True)
class AssignedGpuReceipt:
    run_id: str
    runtime: Mapping[str, object]
    visibility: Mapping[str, str]
    acquired_unix_seconds: float
    schema_version: str = "apex.assigned-gpu/v1"
    allocation_owner: str = "caller"
    host_ownership_verified: bool = False

    def __post_init__(self) -> None:
        devices = self.runtime.get("devices")
        if (not self.run_id or self.schema_version != "apex.assigned-gpu/v1"
                or self.allocation_owner != "caller" or self.host_ownership_verified is not False
                or not isinstance(devices, list) or not devices
                or any(not isinstance(d, dict) or d.get("index") != i
                       for i, d in enumerate(devices))
                or not math.isfinite(self.acquired_unix_seconds) or self.acquired_unix_seconds <= 0):
            raise ValueError("invalid assigned GPU runtime")

    @property
    def execution_scope(self) -> str:
        return "caller-assigned-visible-gpus"

    @property
    def physical_scope(self) -> None:
        return None

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AssignedGpuMeasurement:
    run_id: str
    action_id: str
    lease_digest: str
    started_unix_seconds: float
    finished_unix_seconds: float

    def __post_init__(self) -> None:
        if (not self.run_id or not self.action_id or len(self.lease_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.lease_digest)
                or not math.isfinite(self.started_unix_seconds)
                or not math.isfinite(self.finished_unix_seconds)
                or not 0 < self.started_unix_seconds <= self.finished_unix_seconds):
            raise ValueError("invalid assigned GPU measurement")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "apex.assigned-gpu-measurement/v1", **asdict(self)}

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class AssignedGpuLease:
    """Serialize this controller; allocation and final teardown belong to its caller."""

    def __init__(self, run_id: str, probe: Callable[[], dict[str, object]], lock) -> None:
        self._run_id, self._probe = run_id, probe
        self._lock = lock
        self._active = False

    def __enter__(self) -> "AssignedGpuLease":
        if not self._lock.acquire(blocking=False):
            raise ContractError("This controller already has a GPU run", "gpu_lease_busy")
        try:
            runtime = self._probe()
            self.receipt = AssignedGpuReceipt(
                self._run_id, runtime,
                {key: os.environ[key] for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                                                 "CUDA_VISIBLE_DEVICES") if key in os.environ},
                time.time(),
            )
            self._active = True
            return self
        except BaseException:
            self._lock.release()
            raise

    def __exit__(self, *_: object) -> None:
        if self._active:
            self._active = False
            self._lock.release()

    @contextmanager
    def measurement(self, action_id: str):
        if not self._active or not action_id:
            raise ContractError("Assigned GPU run is inactive", "gpu_lease_inactive")
        start = time.time()
        boundary = SimpleNamespace(receipt=None)
        yield boundary
        boundary.receipt = AssignedGpuMeasurement(
            self._run_id, action_id, self.receipt.digest, start, time.time()
        )


class AssignedGpuLeaseManager:
    def __init__(self, probe: Callable[[], dict[str, object]] = probe_assigned_gpus) -> None:
        self._probe = probe
        self._lock = threading.Lock()

    def acquire(self, run_id: str, *, requested_devices: str | None = None) -> AssignedGpuLease:
        if requested_devices is not None:
            raise ContractError("Set GPU visibility before starting Apex", "assigned_gpu_selection")
        return AssignedGpuLease(run_id, self._probe, self._lock)
