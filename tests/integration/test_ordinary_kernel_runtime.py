"""Exercise real ordinary subprocesses; GPU inventory alone is a CPU fixture."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from apex.core import ContractError, TaskStatus
from apex.runtime import LocalGpuLeaseManager
from apex.runtime.assigned_gpu import AssignedGpuLeaseManager
from tests.integration import test_kernel_optimize_use_case as fixture


@pytest.fixture
def assigned_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary kernel runtime requested host isolation")
    monkeypatch.setattr("apex.execution.supervisor.prepare_pid_namespace", forbidden)
    monkeypatch.setattr("apex.runtime.gpu_ownership.RocmSmiGpuOwnershipInspector.inspect", forbidden)
    monkeypatch.setattr(fixture, "_gpu_leases", lambda _: AssignedGpuLeaseManager(
        lambda: {"devices": [{"index": 0, "name": "fixture", "arch": "gfx950"}]}
    ))


def test_single_kernel_without_host_processes_or_bubblewrap(tmp_path, assigned_runtime):
    task, result, path = fixture._run(tmp_path, fixture.EditingAgent())
    assert result.status is TaskStatus.CANDIDATE_READY, result.reason_code
    saved = json.loads(path.read_text())
    assert saved["gpu_lease"]["host_ownership_verified"] is False
    assert saved["external_verification_required"] is True
    assert saved["safety_certified"] is False
    assert saved["bundle_path"]
    assert saved["changed_files"]


def test_ordinary_measurement_and_reward_replay(tmp_path, assigned_runtime):
    fixture.test_raw_measurement_commits_robust_reward_and_rl_receipts(tmp_path)


@pytest.mark.parametrize("source,reason,reward,stage", [
    ("COMPILE_FAIL = True\ndef kernel(x):\n    return x\n", "compile_failed", 0.0, "compile"),
    ("CORRECTNESS_FAIL = True\ndef kernel(x):\n    return x\n", "wrong", 20.0, "correctness"),
])
def test_ordinary_failure_reward_replay(tmp_path, assigned_runtime, source, reason, reward, stage):
    fixture.test_trusted_gate_failure_has_attempt_and_terminal_reward(
        tmp_path, source, reason, reward, stage,
    )


def test_enabled_safety_tools_retain_host_gpu_ownership_gate(tmp_path, monkeypatch):
    def unavailable(self, run_id):
        raise ContractError("Host GPU ownership unavailable", "strict_gpu_unavailable")

    monkeypatch.setattr(fixture, "_gpu_leases", lambda _: None)
    monkeypatch.setattr(LocalGpuLeaseManager, "acquire", unavailable)
    policy, tools = fixture._configured_safety()
    _, result, _ = fixture._run(
        tmp_path, fixture.EditingAgent(), safety_policy=policy, safety_tools=tools,
    )
    assert result.reason_code == "strict_gpu_unavailable"
