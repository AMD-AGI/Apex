from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from apex.benchmark.local_gpu_observation import LocalGpuLeaseAuthority
from apex.benchmark.local_magpie_attestor import LocalMagpieExecutionAttestor
from apex.benchmark.local_magpie_contract import load_local_contract
from apex.benchmark.local_process_observation import LocalProcessIdentity
from apex.benchmark.magpie_attestation import load_magpie_execution_attestation
from apex.benchmark.results import parse_benchmark_report
from apex.core import ContractError, sha256_file
from apex.ports import BenchmarkPass, MagpieAttestationRequest
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt


_GPU_ID = "GPU-0000000000000001"


class FakeProcesses:
    def __init__(self, owner: LocalProcessIdentity) -> None:
        self.current = {owner.pid: owner}

    def snapshot(self) -> tuple[LocalProcessIdentity, ...]:
        return tuple(self.current[pid] for pid in sorted(self.current))

    def process(self, pid: int) -> LocalProcessIdentity | None:
        return self.current.get(pid)


class FakePorts:
    def listener_owners(self, port, processes):
        assert port == 8888
        return tuple(item for item in processes if item.pid in {2001, 3001})


class FakeGpu:
    def __init__(self) -> None:
        self.benchmark_seen = threading.Event()
        self.quiescent_calls = 0

    def observe(self, roots, authority):
        assert authority.devices == ((0, _GPU_ID),)
        if any(item.pid == 2001 for item in roots):
            self.benchmark_seen.set()
        if not roots:
            return None
        owner = roots[-1]
        return {
            "devices": [{"rsmi_index": 0, "unique_id": _GPU_ID}],
            "processes": [
                {
                    "pid": owner.pid,
                    "uid": owner.uid,
                    "start_time_ticks": owner.start_time_ticks,
                    "cmdline_sha256": owner.cmdline_sha256,
                    "rsmi_device_indices": [0],
                    "root_pid": owner.pid,
                    "cgroup_sha256": owner.cgroup_sha256,
                }
            ],
            "ownership_receipt_sha256": "9" * 64,
        }

    def require_quiescent(self, authority):
        assert authority.devices == ((0, _GPU_ID),)
        self.quiescent_calls += 1
        return {
            "devices": [{"rsmi_index": 0, "unique_id": _GPU_ID}],
            "ownership_receipt_sha256": "8" * 64,
            "verified": True,
        }


def _identity(
    pid: int,
    *,
    ppid: int,
    cwd: Path,
    argv: tuple[str, ...],
    start: int,
) -> LocalProcessIdentity:
    return LocalProcessIdentity(
        pid,
        os.getuid(),
        ppid,
        pid,
        pid,
        start,
        f"{pid % 10}" * 64,
        argv,
        cwd.resolve(),
        "7" * 64,
        ("0::/apex.slice",),
    )


def _receipt(tmp_path: Path) -> DependencyReceipt:
    roots = {name: tmp_path / name for name in ("magpie", "tracelens", "inferencex")}
    for root in roots.values():
        root.mkdir()
    return DependencyReceipt(
        "apex.dependencies.receipt/v1",
        "3" * 64,
        Path("/usr/bin/python3"),
        roots,
        {name: "4" * 40 for name in roots},
        {},
    )


def _dependencies(receipt: DependencyReceipt):
    return {
        "lock_sha256": receipt.lock_sha256,
        "dependencies": {
            name: {
                "root": str(root.resolve()),
                "commit": "4" * 40,
                "tree": "5" * 40,
            }
            for name, root in receipt.roots.items()
        },
    }


def _authority(owner: LocalProcessIdentity, run_id: str) -> LocalGpuLeaseAuthority:
    return LocalGpuLeaseAuthority(
        run_id,
        "1" * 64,
        "0",
        ((0, _GPU_ID),),
        owner,
        "2" * 64,
        time.time() + 300,
    )


def _config(
    path: Path,
    receipt: DependencyReceipt,
    *,
    lifecycle: str,
    pid_dir: Path,
    force_reuse: bool = False,
) -> None:
    lifecycle_yaml = ""
    if lifecycle != "one_shot":
        lifecycle_yaml = (
            "  server_lifecycle:\n"
            "    enabled: true\n"
            f"    cleanup: {'true' if lifecycle == 'cleanup' else 'false'}\n"
            f"    force_reuse: {'true' if force_reuse else 'false'}\n"
            f"    pid_dir: {pid_dir}\n"
        )
    path.write_text(
        "benchmark:\n"
        "  framework: vllm\n"
        "  model: Qwen/example\n"
        "  run_mode: local\n"
        f"  inferencex_path: {receipt.root('inferencex')}\n"
        "  benchmark_script: vllm_mi355x.sh\n"
        "  envs:\n"
        "    TP: 1\n"
        "    PORT: 8888\n"
        + lifecycle_yaml,
        encoding="utf-8",
    )


def _request(
    run_root: Path,
    config: Path,
    *,
    lifecycle: str,
    argv: tuple[str, ...],
    pass_type: BenchmarkPass = BenchmarkPass.MEASUREMENT,
) -> MagpieAttestationRequest:
    return MagpieAttestationRequest(
        run_id="baseline-measurement",
        pass_type=pass_type,
        config_path=config,
        run_root=run_root,
        benchmark_argv=argv,
        config_sha256=sha256_file(config),
        execution_mode="local",
        lifecycle=lifecycle,
        requested_image=None,
        gpu_lease={"synthetic": True},
    )


def _report(workspace: Path, *, profiling_enabled: bool = False) -> Path:
    workspace.mkdir(parents=True)
    path = workspace / "benchmark_report.json"
    path.write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "Qwen/example",
                "workspace_dir": str(workspace.resolve()),
                "profiling_enabled": profiling_enabled,
                "throughput": {"total_token_throughput": 10.0},
                "latency": {"ttft": {"p99_ms": 3.0}, "tpot": {"p99_ms": 1.0}},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _quality_artifacts(workspace: Path) -> None:
    quality = workspace / "lm_eval"
    quality.mkdir()
    (quality / "results.json").write_text(
        json.dumps(
            {"results": {"gsm8k": {"exact_match,strict-match": 1.0}}}
        ),
        encoding="utf-8",
    )
    (quality / "samples_gsm8k.jsonl").write_text(
        '{"doc_id": 1, "exact_match": true}\n', encoding="utf-8"
    )


def _server_files(pid_dir: Path, receipt: DependencyReceipt) -> tuple[Path, Path]:
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "vllm_8888.pid"
    meta_file = pid_dir / "vllm_8888.json"
    pid_file.write_text("3001\n", encoding="ascii")
    meta_file.write_text(
        json.dumps(
            {
                "server_pid": 3001,
                "framework": "vllm",
                "model": "Qwen/example",
                "tp": "1",
                "port": 8888,
                "extra_vllm_args": "",
                "extra_sglang_args": "",
                "extra_atom_args": "",
                "max_model_len": "",
                "inferencex_path": str(receipt.root("inferencex").resolve()),
                "started_at": 1.0,
                "last_used_at": 2.0,
            }
        ),
        encoding="utf-8",
    )
    return pid_file, meta_file


def _attestor(tmp_path: Path, lifecycle: str):
    receipt = _receipt(tmp_path)
    owner = _identity(
        os.getpid(), ppid=1, cwd=tmp_path, argv=("pytest",), start=100
    )
    processes = FakeProcesses(owner)
    gpu = FakeGpu()
    attestor = LocalMagpieExecutionAttestor(
        receipt,
        processes=processes,
        ports=FakePorts(),
        gpu=gpu,
        dependency_observer=_dependencies,
        lease_validator=lambda value, run_id: _authority(owner, run_id),
        poll_seconds=0.001,
    )
    pid_dir = tmp_path / "pids"
    config = tmp_path / "config.yaml"
    _config(config, receipt, lifecycle=lifecycle, pid_dir=pid_dir)
    run_root = tmp_path / "results" / "baseline-measurement" / "measurement"
    run_root.mkdir(parents=True)
    argv = ("/usr/bin/python3", "-m", "Magpie", "benchmark")
    return receipt, owner, processes, gpu, attestor, pid_dir, config, run_root, argv


def _observe_top(processes, owner, receipt, argv, gpu):
    top = _identity(2001, ppid=owner.pid, cwd=receipt.root("magpie"), argv=argv, start=200)
    processes.current[top.pid] = top
    assert gpu.benchmark_seen.wait(timeout=2)
    return top


def test_local_one_shot_binds_process_gpu_and_preserves_report(tmp_path: Path) -> None:
    receipt, owner, processes, gpu, attestor, _, config, run_root, argv = _attestor(
        tmp_path, "one_shot"
    )
    session = attestor.prepare(_request(run_root, config, lifecycle="one_shot", argv=argv))
    top = _observe_top(processes, owner, receipt, argv, gpu)
    del processes.current[top.pid]
    report = _report(run_root / "benchmark_vllm_1")
    original = report.read_bytes()

    path = attestor.complete(session, report_path=report, command_exit_code=0, timed_out=False)

    assert report.read_bytes() == original
    evidence = load_magpie_execution_attestation(
        path,
        report_path=report,
        report=json.loads(original),
        expected_config_sha256=sha256_file(config),
        expected_run_id="baseline-measurement",
        expected_pass_type=BenchmarkPass.MEASUREMENT,
        command_exit_code=0,
        timed_out=False,
    )
    assert evidence.runtime["inferencex_runtime_receipt"] is None
    runtime = evidence.runtime["serving_runtime_receipt"]
    assert runtime["schema"] == "apex.magpie-local-runtime-observation/v2"
    assert runtime["execution_mode"] == "local"
    assert runtime["benchmark_process"]["start_time_ticks"] == 200
    assert evidence.gpu_engagement["verified"] is True
    assert gpu.quiescent_calls == 1


def test_abort_stops_local_observer_without_attestation(tmp_path: Path) -> None:
    _, _, _, _, attestor, _, config, run_root, argv = _attestor(
        tmp_path, "one_shot"
    )
    session = attestor.prepare(
        _request(run_root, config, lifecycle="one_shot", argv=argv)
    )

    attestor.abort(session, reason="magpie_process_start_failed")

    assert session.thread is not None and not session.thread.is_alive()
    assert not (run_root / "evaluator" / "execution_attestation.json").exists()


def test_local_diagnostic_is_consumed_by_strict_result_parser(tmp_path: Path) -> None:
    receipt, owner, processes, gpu, attestor, _, config, run_root, argv = _attestor(
        tmp_path, "one_shot"
    )
    request = _request(
        run_root,
        config,
        lifecycle="one_shot",
        argv=argv,
        pass_type=BenchmarkPass.DIAGNOSTIC,
    )
    session = attestor.prepare(request)
    top = _observe_top(processes, owner, receipt, argv, gpu)
    del processes.current[top.pid]
    report = _report(
        run_root / "benchmark_vllm_diagnostic", profiling_enabled=True
    )
    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )

    result = parse_benchmark_report(
        report,
        run_id="baseline-measurement",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        quality_required=False,
        command_exit_code=0,
        timed_out=False,
        expected_model="Qwen/example",
        expected_inferencex_root=receipt.root("inferencex").resolve(),
        expected_inferencex_commit="4" * 40,
        expected_inferencex_tree="5" * 40,
        expected_lm_eval_execution_mode="not_requested",
        expected_config_sha256=sha256_file(config),
        expected_gpu_lease_digest="1" * 64,
        expected_execution_mode="local",
        expected_lifecycle="one_shot",
        execution_attestation_path=path,
    )

    assert result.succeeded is True
    assert result.inferencex_runtime.required is False
    assert result.local_runtime.required is True
    assert result.local_runtime.passed is True
    assert result.local_runtime.lifecycle == "one_shot"
    assert result.local_runtime.source_root == receipt.root("inferencex").resolve()


def test_local_measurement_fails_without_exact_lm_eval_engagement(
    tmp_path: Path,
) -> None:
    receipt, owner, processes, gpu, attestor, _, config, run_root, argv = _attestor(
        tmp_path, "one_shot"
    )
    session = attestor.prepare(
        _request(run_root, config, lifecycle="one_shot", argv=argv)
    )
    top = _observe_top(processes, owner, receipt, argv, gpu)
    del processes.current[top.pid]
    report = _report(run_root / "benchmark_vllm_measurement")
    _quality_artifacts(report.parent)
    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )
    expected_lm_eval = LmEvalRuntimeReceipt(
        tmp_path / "lm-eval-runtime",
        "a" * 64,
        "b" * 64,
        {"python_abi": "cp312", "lm_eval_version": "0.4.9.2"},
        1,
        "c" * 64,
    )

    result = parse_benchmark_report(
        report,
        run_id="baseline-measurement",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_quality_kind="lm_eval",
        expected_model="Qwen/example",
        expected_inferencex_root=receipt.root("inferencex").resolve(),
        expected_inferencex_commit="4" * 40,
        expected_inferencex_tree="5" * 40,
        expected_lm_eval_runtime=expected_lm_eval,
        expected_lm_eval_execution_mode="local",
        expected_config_sha256=sha256_file(config),
        expected_gpu_lease_digest="1" * 64,
        expected_execution_mode="local",
        expected_lifecycle="one_shot",
        execution_attestation_path=path,
    )

    assert result.local_runtime.passed is True
    assert result.succeeded is False
    assert "lm_eval_runtime_evidence_missing" in result.errors


def test_reuse_binds_existing_server_listener_and_leaves_exact_identity(
    tmp_path: Path,
) -> None:
    receipt, owner, processes, gpu, attestor, pid_dir, config, run_root, argv = _attestor(
        tmp_path, "reuse"
    )
    _server_files(pid_dir, receipt)
    server = _identity(3001, ppid=1, cwd=receipt.root("inferencex"), argv=("vllm",), start=300)
    processes.current[server.pid] = server
    session = attestor.prepare(_request(run_root, config, lifecycle="reuse", argv=argv))
    top = _observe_top(processes, owner, receipt, argv, gpu)
    del processes.current[top.pid]
    report = _report(run_root / "benchmark_vllm_1")

    path = attestor.complete(session, report_path=report, command_exit_code=0, timed_out=False)
    value = json.loads(path.read_text(encoding="utf-8"))
    lifecycle = value["runtime"]["serving_runtime_receipt"]["lifecycle_receipt"]

    assert lifecycle["mode"] == "reuse"
    assert lifecycle["server_state"]["process"]["start_time_ticks"] == 300
    assert lifecycle["server_state"]["listener_pids"] == [3001]
    assert gpu.quiescent_calls == 0


def test_cleanup_requires_and_removes_exact_server_and_gpu_owners(tmp_path: Path) -> None:
    receipt, owner, processes, gpu, attestor, pid_dir, config, run_root, argv = _attestor(
        tmp_path, "cleanup"
    )
    pid_file, meta_file = _server_files(pid_dir, receipt)
    server = _identity(3001, ppid=1, cwd=receipt.root("inferencex"), argv=("vllm",), start=300)
    processes.current[server.pid] = server
    session = attestor.prepare(_request(run_root, config, lifecycle="cleanup", argv=argv))
    top = _observe_top(processes, owner, receipt, argv, gpu)
    del processes.current[top.pid]
    del processes.current[server.pid]
    pid_file.unlink()
    meta_file.unlink()
    report = _report(run_root / "benchmark_vllm_1")

    path = attestor.complete(session, report_path=report, command_exit_code=0, timed_out=False)
    value = json.loads(path.read_text(encoding="utf-8"))
    lifecycle = value["runtime"]["serving_runtime_receipt"]["lifecycle_receipt"]

    assert lifecycle["mode"] == "cleanup"
    assert lifecycle["quiescence_receipt"]["verified"] is True
    assert value["reward_eligible"] is False
    assert value["errors"] == []


def test_one_shot_residual_process_fails_closed(tmp_path: Path) -> None:
    receipt, owner, processes, gpu, attestor, _, config, run_root, argv = _attestor(
        tmp_path, "one_shot"
    )
    session = attestor.prepare(_request(run_root, config, lifecycle="one_shot", argv=argv))
    _observe_top(processes, owner, receipt, argv, gpu)
    report = _report(run_root / "benchmark_vllm_1")

    path = attestor.complete(session, report_path=report, command_exit_code=0, timed_out=False)
    value = json.loads(path.read_text(encoding="utf-8"))

    assert "magpie_local_residual_process:2001" in value["errors"]
    assert value["reward_eligible"] is False


def test_cleanup_without_preexisting_server_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    _, _, _, _, attestor, _, config, run_root, argv = _attestor(tmp_path, "cleanup")

    with pytest.raises(ContractError) as caught:
        attestor.prepare(_request(run_root, config, lifecycle="cleanup", argv=argv))

    assert caught.value.reason_code == "magpie_local_cleanup_target_missing"


def test_force_reuse_is_rejected_before_execution(tmp_path: Path) -> None:
    receipt, _, _, _, attestor, pid_dir, config, run_root, argv = _attestor(
        tmp_path, "reuse"
    )
    _config(
        config,
        receipt,
        lifecycle="reuse",
        pid_dir=pid_dir,
        force_reuse=True,
    )

    with pytest.raises(ContractError) as caught:
        attestor.prepare(_request(run_root, config, lifecycle="reuse", argv=argv))

    assert caught.value.reason_code == "magpie_local_force_reuse_forbidden"


def test_server_source_generation_ignores_client_only_cleanup_config(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    dependencies = _dependencies(receipt)
    pid_dir = tmp_path / "pids"
    reuse_config = tmp_path / "reuse.yaml"
    cleanup_config = tmp_path / "cleanup.yaml"
    _config(reuse_config, receipt, lifecycle="reuse", pid_dir=pid_dir)
    _config(cleanup_config, receipt, lifecycle="cleanup", pid_dir=pid_dir)
    cleanup_config.write_text(
        cleanup_config.read_text(encoding="utf-8").replace(
            "    TP: 1\n", "    TP: 1\n    ISL: 1024\n    OSL: 1024\n"
        ),
        encoding="utf-8",
    )
    reuse_config.write_text(
        reuse_config.read_text(encoding="utf-8").replace(
            "    TP: 1\n", "    TP: 1\n    ISL: 8192\n    OSL: 1024\n"
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "results"
    run_root.mkdir()
    argv = ("/usr/bin/python3", "-m", "Magpie", "benchmark")
    reuse = load_local_contract(
        _request(run_root, reuse_config, lifecycle="reuse", argv=argv),
        receipt,
        dependencies,
    )
    cleanup = load_local_contract(
        _request(run_root, cleanup_config, lifecycle="cleanup", argv=argv),
        receipt,
        dependencies,
    )

    assert sha256_file(reuse_config) != sha256_file(cleanup_config)
    assert (
        reuse.server_source_generation_sha256
        == cleanup.server_source_generation_sha256
    )


def test_support_matrix_is_local_only() -> None:
    supports = LocalMagpieExecutionAttestor.supports

    assert supports("local", "one_shot") is True
    assert supports("local", "reuse") is True
    assert supports("local", "cleanup") is True
    assert supports("docker", "one_shot") is False
    assert supports("ray", "one_shot") is False
    formal = LocalMagpieExecutionAttestor.formal_measurement_support(
        "local", "one_shot"
    )
    assert formal.reason_code == "magpie_local_quality_execution_unavailable"
    assert "magpie_inferencex_eval_argument_mismatch" in formal.blockers
    assert "local_remote_eval_task_contract_mismatch" in formal.blockers
