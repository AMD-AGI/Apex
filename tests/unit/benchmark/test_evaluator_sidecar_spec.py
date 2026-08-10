from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.benchmark.evaluator_execution import LmEvalExecutionContract
from apex.benchmark.evaluator_sidecar_spec import build_evaluator_sidecar_spec
from apex.core import ConfigurationError, sha256_file


def _prepared(tmp_path: Path, *, launcher_sha256: str | None = None):
    launcher = (
        Path(__file__).resolve().parents[3]
        / "src/apex/benchmark/evaluator_sidecar_entry.py"
    )
    contract = LmEvalExecutionContract(
        run_id="baseline-measurement",
        config_sha256="1" * 64,
        model="Qwen/example",
        endpoint_port=8888,
        policy_sha256="2" * 64,
        policy_lock_sha256="3" * 64,
        task_definition_sha256="4" * 64,
        effective_task_definition_sha256="5" * 64,
        task_materialization_receipt_sha256="6" * 64,
        dataset_receipt_sha256="7" * 64,
        dataset_revision="8" * 40,
        runtime_sha256="9" * 64,
        runtime_manifest_sha256="a" * 64,
        runtime_lock_sha256="b" * 64,
        launcher_sha256=launcher_sha256 or sha256_file(launcher),
        image_repo_digest="example/eval@sha256:" + "c" * 64,
        image_id="sha256:" + "d" * 64,
        max_length=2248,
        max_gen_tokens=480,
        concurrent_requests=64,
        timeout_seconds=3600,
    )
    roots = {}
    for name in ("sidecar", "dataset", "runtime", "task"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    contract_path = tmp_path / "execution_contract.json"
    contract_path.write_text("{}\n")
    return SimpleNamespace(
        contract=contract,
        contract_path=contract_path,
        sidecar_root=roots["sidecar"],
        dataset_mount=roots["dataset"],
        runtime_mount=roots["runtime"],
        task_mount=roots["task"],
    )


def test_builds_no_network_no_gpu_read_only_sidecar_spec(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    broker = tmp_path / "broker"
    broker.mkdir()

    spec = build_evaluator_sidecar_spec(prepared, broker)
    argv = spec.create_argv

    assert ("--network", "none") == argv[argv.index("--network") : argv.index("--network") + 2]
    assert "--read-only" in argv
    assert ("--cap-drop", "ALL") == argv[argv.index("--cap-drop") : argv.index("--cap-drop") + 2]
    assert "--privileged" not in argv
    assert "--device" not in argv
    assert "--gpus" not in argv
    assert spec.image_repo_digest in argv
    assert spec.sidecar_argv == prepared.contract.sidecar_argv
    mounts = {item.role: item for item in spec.mounts}
    assert mounts["authority"].read_only is False
    assert all(item.read_only for role, item in mounts.items() if role != "authority")
    assert "PYTHONPATH=/evaluator/runtime/site-packages" in argv
    assert spec.sha256


def test_rejects_launcher_drift_after_contract_freeze(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, launcher_sha256="0" * 64)
    broker = tmp_path / "broker"
    broker.mkdir()

    with pytest.raises(ConfigurationError, match="launcher changed"):
        build_evaluator_sidecar_spec(prepared, broker)
