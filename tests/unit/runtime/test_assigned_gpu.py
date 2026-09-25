from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.runtime.assigned_gpu import AssignedGpuLeaseManager


def test_assigned_run_probes_once_and_records_no_host_ownership():
    calls = []
    def probe():
        calls.append(True)
        return {"devices": [{"index": 0, "name": "test", "arch": "gfx950"}]}
    manager = AssignedGpuLeaseManager(probe)
    with manager.acquire("run-one") as lease:
        assert lease.receipt.host_ownership_verified is False
        with pytest.raises(ContractError, match="already"):
            with manager.acquire("run-two"):
                pass
        with lease.measurement("attempt-one") as measurement:
            pass
        assert measurement.receipt.action_id == "attempt-one"
        assert measurement.receipt.lease_digest == lease.receipt.digest
        assert len(calls) == 1
    with manager.acquire("run-two"):
        pass
    assert len(calls) == 2


def test_failed_smoke_releases_controller_lock():
    calls = []
    def fail():
        calls.append(True)
        if len(calls) == 1:
            raise ContractError("smoke failed", "assigned_gpu_unavailable")
        return {"devices": [{"index": 0}]}
    manager = AssignedGpuLeaseManager(fail)
    with pytest.raises(ContractError):
        with manager.acquire("run-one"):
            pass
    with manager.acquire("run-two"):
        pass


def test_probe_failure_prevents_run(monkeypatch):
    import subprocess
    from apex.runtime.assigned_gpu import probe_assigned_gpus
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="No GPU")
    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(ContractError) as error:
        probe_assigned_gpus()
    assert error.value.reason_code == "assigned_gpu_unavailable"
