from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.benchmark.magpie_attestation import load_magpie_execution_attestation
from apex.benchmark.magpie import _ray_contract
from apex.benchmark.results import parse_benchmark_report
from apex.benchmark.ray_gpu_observation import RocmRayGpuObserver
from apex.benchmark.ray_magpie_attestor import RayOneShotMagpieExecutionAttestor
from apex.benchmark.ray_observation import (
    LocalRayDriverProcessObserver,
    LocalRayWorkerProcessObserver,
    RayCliObservationClient,
    RayDriverProcessObservation,
    RayJobObservation,
    RayJobSnapshot,
    RayTaskObservation,
    RayTaskSnapshot,
    RayWorkerProcessObservation,
)
from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file, sha256_json
from apex.execution import ProcessResult
from apex.ports import (
    BenchmarkPass,
    MagpieAttestationRequest,
    RayArtifactClaim,
    RayExecutionContract,
    RayNodeEvidenceReceipt,
)
from apex.runtime import DependencyReceipt


_TASK = "1" * 24
_JOB = "2" * 8
_WORKER = "3" * 56
_NODE = "4" * 56


class FakeRay:
    def __init__(self) -> None:
        self.active = False
        self.finished = False
        self.extra = False
        self.stale = False
        self.list_calls = 0
        self.address = "auto"

    def tasks(self) -> RayTaskSnapshot:
        self.list_calls += 1
        tasks: tuple[RayTaskObservation, ...] = (
            (
                _task(
                    task_id="8" * 24,
                    worker_pid=221,
                    job_id="f" * 8,
                ),
            )
            if self.stale
            else ()
        )
        if self.active:
            state = "FINISHED" if self.finished else "RUNNING"
            tasks += (_task(state=state),)
            if self.extra:
                tasks += (_task(task_id="9" * 24, worker_pid=223),)
        return RayTaskSnapshot(
            sha256_bytes(self.address.encode()),
            "/usr/bin/ray",
            "6" * 64,
            time.time_ns(),
            tasks,
        )

    def jobs(self) -> RayJobSnapshot:
        jobs: tuple[RayJobObservation, ...] = (
            (RayJobObservation("f" * 8, 110, False),) if self.stale else ()
        )
        if self.active:
            jobs += (RayJobObservation(_JOB, 111, self.finished),)
        return RayJobSnapshot(
            sha256_bytes(self.address.encode()),
            "/usr/bin/ray",
            "6" * 64,
            time.time_ns(),
            jobs,
        )


class FakeDriver:
    def freeze(self, job, benchmark_argv):
        del benchmark_argv
        return RayDriverProcessObservation(
            RayWorkerProcessObservation(
                job.driver_pid,
                os.getuid(),
                600,
                "1" * 64,
                "2" * 64,
                ("/apex/driver",),
            ),
            "3" * 64,
        )


class FakeRayFactory:
    is_available = True

    def __init__(self, ray: FakeRay) -> None:
        self.ray = ray
        self.addresses: list[str] = []

    def create(self, address: str) -> FakeRay:
        self.addresses.append(address)
        self.ray.address = address
        return self.ray


class FakeNodeAuthority:
    is_available = True

    def __init__(self, receipt: DependencyReceipt, workspace: Path) -> None:
        self.receipt = receipt
        self.workspace = workspace
        self.aborted: str | None = None
        self.bad_binding = False
        self.bad_digest = False
        self.missing_gpu = False

    def prepare(self, request, *, ray_contract, cluster_identity_sha256):
        return (request.run_id, ray_contract.ray_config_sha256, cluster_identity_sha256)

    def complete(self, session, *, binding):
        assert session[0] == binding.run_id
        claims = tuple(
            RayArtifactClaim(
                "benchmark_report" if path.name == "benchmark_report.json" else "quality_artifact",
                path.relative_to(self.workspace).as_posix(),
                path.stat().st_size,
                ("0" * 64 if self.bad_digest else sha256_file(path)),
            )
            for path in sorted(self.workspace.rglob("*"))
            if path.is_file()
        )
        dependencies = _dependency_snapshot(self.receipt)
        node = {
            "schema": "apex.magpie-ray-worker-node/v1",
            "node_id": _NODE,
            "binding_sha256": binding.digest,
            "procfs": {
                "worker_pid": 222,
                "ray_task_id": _TASK,
                "ray_worker_id": _WORKER,
                "verified": True,
            },
            "dependencies_sha256": sha256_json(dependencies),
            "kfd": {"verified": True, "cleanup_verified": True},
            "verified": True,
        }
        devices = () if self.missing_gpu else (
            {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"},
        )
        processes = () if self.missing_gpu else (
            {
                "pid": 321,
                "uid": os.getuid(),
                "start_time_ticks": 900,
                "cmdline_sha256": "a" * 64,
                "rsmi_device_indices": [0],
                "node_id": _NODE,
                "ray_job_id": _JOB,
                "ray_task_id": _TASK,
                "ray_worker_id": _WORKER,
                "ray_worker_pid": 222,
            },
        )
        return RayNodeEvidenceReceipt(
            "apex.magpie-ray-node-evidence/v1",
            "9" * 64,
            ("8" * 64 if self.bad_binding else binding.digest),
            "bench_1234567890",
            self.workspace,
            claims,
            (node,),
            dependencies,
            devices,
            processes,
            {
                "model_revision_receipt": None,
                "inferencex_runtime_receipt": None,
                "lm_eval_runtime_receipt": None,
                "verified": True,
            },
        )

    def abort(self, session, *, reason: str) -> None:
        del session
        self.aborted = reason


class CaptureSupervisor:
    def __init__(self, output: str, *, truncated: bool = False) -> None:
        self.output = output
        self.truncated = truncated
        self.argv: tuple[str, ...] | None = None
        self.kwargs = None

    def run(self, argv, **kwargs):
        self.argv = tuple(argv)
        self.kwargs = kwargs
        return ProcessResult(
            tuple(argv),
            0,
            False,
            self.output,
            "",
            self.truncated,
            False,
            0.01,
        )


def _task(
    *,
    state: str = "RUNNING",
    task_id: str = _TASK,
    worker_pid: int = 222,
    job_id: str = _JOB,
) -> RayTaskObservation:
    return RayTaskObservation(
        task_id,
        0,
        job_id,
        _WORKER if task_id == _TASK else "e" * 56,
        worker_pid,
        _NODE,
        "run_task",
        "Magpie.remote.tasks.run_task",
        state,
    )


def _receipt(tmp_path: Path) -> DependencyReceipt:
    roots = {name: tmp_path / name for name in ("magpie", "tracelens", "inferencex")}
    for root in roots.values():
        root.mkdir(parents=True)
    raw = {
        "dependencies": {
            name: {"tree": "e" * 40} for name in roots
        }
    }
    return DependencyReceipt(
        "apex.dependencies.receipt/v1",
        "c" * 64,
        Path("/usr/bin/python3"),
        roots,
        {name: "d" * 40 for name in roots},
        raw,
    )


def _dependency_snapshot(receipt: DependencyReceipt):
    return {
        "lock_sha256": receipt.lock_sha256,
        "dependencies": {
            name: {
                "root": str(root.resolve()),
                "commit": "d" * 40,
                "tree": "e" * 40,
            }
            for name, root in receipt.roots.items()
        },
    }


def _lease() -> dict[str, object]:
    return {
        "owner_pid": os.getpid(),
        "ownership": {
            "selector_scope": "0",
            "selected_devices": [
                {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}
            ],
        },
    }


def _request(root: Path, config: Path, **changes) -> MagpieAttestationRequest:
    shared = changes.pop("shared_storage_path", root.parent / "shared")
    ray_contract = RayExecutionContract(
        "ray://cluster:10001",
        Path(shared),
        "f" * 64,
        False,
        1,
        8,
        8,
    )
    values = {
        "run_id": "baseline-measurement",
        "pass_type": BenchmarkPass.MEASUREMENT,
        "config_path": config,
        "run_root": root,
        "benchmark_argv": ("python", "-m", "Magpie"),
        "config_sha256": sha256_file(config),
        "execution_mode": "ray",
        "lifecycle": "one_shot",
        "requested_image": None,
        "gpu_lease": _lease(),
        "ray_contract": ray_contract,
    }
    values.update(changes)
    return MagpieAttestationRequest(**values)


def _report(workspace: Path, *, profiling: bool = False) -> Path:
    workspace.mkdir(parents=True)
    quality = workspace / "lm_eval"
    quality.mkdir()
    (quality / "results.json").write_text(
        json.dumps({"results": {"quality_task": {"acc,none": 1.0}}}),
        encoding="utf-8",
    )
    (quality / "samples.jsonl").write_text('{"sample": 1}\n', encoding="utf-8")
    path = workspace / "benchmark_report.json"
    path.write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "Qwen/example",
                "workspace_dir": str(workspace.resolve()),
                "profiling_enabled": profiling,
                "throughput": {"total_token_throughput": 10.0},
                "latency": {"ttft": {"p99_ms": 3.0}, "tpot": {"p99_ms": 1.0}},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _attestor(tmp_path: Path, workspace: Path):
    receipt = _receipt(tmp_path)
    ray = FakeRay()
    authority = FakeNodeAuthority(receipt, workspace)
    attestor = RayOneShotMagpieExecutionAttestor(
        receipt,
        ray_factory=FakeRayFactory(ray),
        node_authority=authority,
        driver=FakeDriver(),
        poll_seconds=0.001,
    )
    return attestor, ray, authority


def _wait(predicate) -> None:
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert predicate()


def _case(tmp_path: Path, *, profiling: bool = False):
    root = tmp_path / "run"
    root.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    shared = tmp_path / "shared"
    workspace = shared / "results" / "bench_1234567890" / "benchmark_vllm_20260810_000000"
    report = _report(workspace, profiling=profiling)
    attestor, ray, authority = _attestor(tmp_path / "deps", workspace)
    request = _request(root, config, shared_storage_path=shared)
    return root, config, report, attestor, ray, authority, request


def _finish(attestor, ray, request):
    session = attestor.prepare(request)
    ray.active = True
    _wait(lambda: session.task is not None)
    ray.finished = True
    location = attestor.locate_report(session)
    assert location.error is None and location.path is not None
    sidecar = attestor.complete(
        session,
        report_path=location.path,
        command_exit_code=0,
        timed_out=False,
    )
    assert sidecar is not None
    return location.path, sidecar


def test_abort_stops_ray_observer_and_node_authority(tmp_path: Path) -> None:
    root, _, _, attestor, _, authority, request = _case(tmp_path)
    session = attestor.prepare(request)

    attestor.abort(session, reason="magpie_launch_argv_invalid")

    assert session.thread is not None and not session.thread.is_alive()
    assert session.finalized
    assert authority.aborted == "magpie_launch_argv_invalid"
    assert not (root / "evaluator" / "execution_attestation.json").exists()


def test_ray_task_authority_and_unchanged_shared_report_are_bound(tmp_path: Path) -> None:
    _, config, remote, attestor, ray, _, request = _case(tmp_path)
    original = remote.read_bytes()

    report, path = _finish(attestor, ray, request)

    assert report.read_bytes() == original == remote.read_bytes()
    assert report != remote
    value = json.loads(path.read_text(encoding="utf-8"))
    runtime = value["runtime"]["serving_runtime_receipt"]
    assert value["reward_eligible"] is False
    assert value["quality_gate"]["verified"] is False
    assert runtime["schema"] == "apex.magpie-ray-runtime-observation/v2"
    assert runtime["task"]["task_id"] == _TASK
    assert runtime["task"]["worker_id"] == _WORKER
    assert runtime["job"]["job_id"] == _JOB
    assert runtime["magpie_task_id"] == "bench_1234567890"
    assert runtime["artifact_import"]["origin_workspace_path"] == str(remote.parent)
    assert ray.address == "ray://cluster:10001"

    evidence = load_magpie_execution_attestation(
        path,
        report_path=report,
        report=json.loads(report.read_text(encoding="utf-8")),
        expected_config_sha256=sha256_file(config),
        expected_run_id="baseline-measurement",
        expected_pass_type=BenchmarkPass.MEASUREMENT,
        command_exit_code=0,
        timed_out=False,
    )
    assert evidence.imported_workspace_origin == remote.parent
    assert evidence.gpu_engagement["verified"] is True
    assert evidence.verdict_errors() == (
        "execution_attestation_quality_unverified",
    )
    normalized = parse_benchmark_report(
        report,
        run_id="baseline-measurement",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_quality_kind="lm_eval",
        expected_config_sha256=sha256_file(config),
        expected_execution_mode="ray",
        execution_attestation_path=path,
    )
    assert normalized.succeeded is False
    assert "execution_attestation_quality_unverified" in normalized.errors
    assert report in normalized.artifacts


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"execution_mode": "docker"}, "magpie_observer_mode_unavailable"),
        ({"lifecycle": "reuse"}, "magpie_observer_lifecycle_unavailable"),
        ({"requested_image": "image:tag"}, "magpie_ray_image_invalid"),
    ],
)
def test_rejects_unsupported_contract_before_ray_query(
    tmp_path: Path, changes: dict[str, object], reason: str
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    workspace = tmp_path / "shared" / "results" / "bench_123" / "benchmark_vllm_x"
    workspace.mkdir(parents=True)
    attestor, ray, _ = _attestor(tmp_path / "deps", workspace)

    with pytest.raises(ContractError) as caught:
        attestor.prepare(_request(root, config, **changes))

    assert caught.value.reason_code == reason
    assert ray.list_calls == 0


def test_support_matrix_is_ray_one_shot_only() -> None:
    supports = RayOneShotMagpieExecutionAttestor.supports
    assert supports("ray", "one_shot") is True
    assert supports("docker", "one_shot") is False
    assert supports("local", "one_shot") is False
    assert supports("ray", "reuse") is False
    formal = RayOneShotMagpieExecutionAttestor.formal_measurement_support(
        "ray", "one_shot"
    )
    assert formal.blockers == ("ray_node_quality_execution_unproven",)


def test_preexisting_ray_task_is_not_misattributed(tmp_path: Path) -> None:
    _, _, _, attestor, ray, _, request = _case(tmp_path)
    ray.stale = True
    _, path = _finish(attestor, ray, request)
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["runtime"]["serving_runtime_receipt"]["task"]["task_id"] == _TASK
    assert value["reward_eligible"] is False
    assert value["quality_gate"]["verified"] is False


@pytest.mark.parametrize("failure", ["ambiguous", "authority_binding", "artifact_digest", "missing_gpu"])
def test_observer_identity_failures_never_become_reward(
    tmp_path: Path, failure: str
) -> None:
    _, _, _, attestor, ray, authority, request = _case(tmp_path)
    if failure == "ambiguous":
        ray.extra = True
    elif failure == "authority_binding":
        authority.bad_binding = True
    elif failure == "artifact_digest":
        authority.bad_digest = True
    else:
        authority.missing_gpu = True

    session = attestor.prepare(request)
    ray.active = True
    _wait(lambda: bool(session.errors) if failure == "ambiguous" else session.task is not None)
    ray.finished = True
    location = attestor.locate_report(session)

    assert location.path is None
    assert location.error is not None
    assert attestor.complete(
        session, report_path=None, command_exit_code=0, timed_out=False
    ) is None


def test_diagnostic_trace_is_never_scoring_evidence(tmp_path: Path) -> None:
    _, _, _, attestor, ray, _, request = _case(tmp_path, profiling=True)
    request = _request(
        request.run_root,
        request.config_path,
        shared_storage_path=request.ray_contract.shared_storage_path,
        pass_type=BenchmarkPass.DIAGNOSTIC,
    )
    _, path = _finish(attestor, ray, request)
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["lane_verified"] is True
    assert value["profiling_enabled"] is True
    assert value["reward_eligible"] is False


def test_ray_is_unavailable_without_injected_node_authority(tmp_path: Path) -> None:
    root, _, remote, _, ray, _, request = _case(tmp_path)
    attestor = RayOneShotMagpieExecutionAttestor(
        _receipt(tmp_path / "other-deps"),
        ray_factory=FakeRayFactory(ray),
        driver=FakeDriver(),
    )

    assert attestor.is_available is False
    with pytest.raises(ContractError) as caught:
        attestor.prepare(replace(request, run_root=root))
    assert caught.value.reason_code == "ray_node_authority_unavailable"
    assert remote.is_file()


def test_resolved_ray_contract_preserves_per_run_address_and_shared_results() -> None:
    raw = {
        "cluster_address": "ray://10.0.0.8:10001",
        "shared_storage_path": "/cluster/shared/magpie",
        "multi_node": True,
        "num_nodes": 2,
        "total_num_gpus": 16,
        "gpus_per_node": 8,
    }

    contract = _ray_contract({"run_mode": "ray", "ray_config": raw})

    assert contract is not None
    assert contract.cluster_address == raw["cluster_address"]
    assert contract.results_path == Path("/cluster/shared/magpie/results")
    assert contract.ray_config_sha256 == sha256_json(raw)
    assert contract.num_nodes == 2


def test_multinode_requires_exact_node_authority_coverage(tmp_path: Path) -> None:
    _, _, _, attestor, ray, authority, request = _case(tmp_path)
    contract = replace(request.ray_contract, multi_node=True, num_nodes=2)
    session = attestor.prepare(replace(request, ray_contract=contract))
    ray.active = True
    _wait(lambda: session.task is not None)
    ray.finished = True

    location = attestor.locate_report(session)

    assert location.path is None
    assert "ray_node_evidence" in str(location.error)
    assert authority.aborted is None


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_shared_artifact_links_fail_closed(tmp_path: Path, link_kind: str) -> None:
    _, _, remote, attestor, ray, _, request = _case(tmp_path)
    source = remote.parent / "lm_eval" / "results.json"
    linked = remote.parent / "linked.json"
    if link_kind == "symlink":
        linked.symlink_to(source)
    else:
        os.link(source, linked)
    session = attestor.prepare(request)
    ray.active = True
    _wait(lambda: session.task is not None)
    ray.finished = True

    location = attestor.locate_report(session)

    assert location.path is None
    assert "ray_artifact" in str(location.error)


def test_shared_artifact_double_read_detects_mutation(tmp_path: Path, monkeypatch) -> None:
    from apex.benchmark import ray_artifacts

    _, _, _, attestor, ray, _, request = _case(tmp_path)
    original = ray_artifacts._digest_fd
    mutated = False

    def race(descriptor: int):
        nonlocal mutated
        result = original(descriptor)
        if not mutated:
            Path(os.readlink(f"/proc/self/fd/{descriptor}")).write_bytes(b"changed")
            mutated = True
        return result

    monkeypatch.setattr(ray_artifacts, "_digest_fd", race)
    session = attestor.prepare(request)
    ray.active = True
    _wait(lambda: session.task is not None)
    ray.finished = True

    location = attestor.locate_report(session)

    assert location.path is None
    assert "ray_artifact" in str(location.error)


@pytest.mark.parametrize("drift", ["task", "origin"])
def test_restored_sidecar_rejects_ray_identity_drift(tmp_path: Path, drift: str) -> None:
    _, config, _, attestor, ray, _, request = _case(tmp_path)
    report, path = _finish(attestor, ray, request)
    value = json.loads(path.read_text(encoding="utf-8"))
    runtime = value["runtime"]["serving_runtime_receipt"]
    if drift == "task":
        runtime["task"]["worker_id"] = "e" * 56
    else:
        runtime["artifact_import"]["origin_workspace_path"] = "/other/workspace"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(IntegrityError):
        load_magpie_execution_attestation(
            path,
            report_path=report,
            report=json.loads(report.read_text(encoding="utf-8")),
            expected_config_sha256=sha256_file(config),
            expected_run_id="baseline-measurement",
            expected_pass_type=BenchmarkPass.MEASUREMENT,
            command_exit_code=0,
            timed_out=False,
        )


def test_ray_cli_uses_fixed_bounded_state_query(tmp_path: Path) -> None:
    executable = tmp_path / "ray"
    executable.write_text("ray-cli-placeholder\n", encoding="utf-8")
    task = _task().to_dict()
    supervisor = CaptureSupervisor(json.dumps([task]))
    client = RayCliObservationClient(
        "ray://127.0.0.1:10001",
        executable=str(executable),
        supervisor=supervisor,
    )

    snapshot = client.tasks()

    assert snapshot.tasks == (_task(),)
    assert supervisor.argv == (
        str(executable.resolve()),
        "list",
        "tasks",
        "--detail",
        "--format=json",
        "--filter",
        "name=run_task",
        "--limit",
        "10000",
        "--address",
        "ray://127.0.0.1:10001",
    )
    assert supervisor.kwargs["timeout_seconds"] == 15
    assert supervisor.kwargs["cwd"] == Path("/")


def test_ray_cli_rejects_truncated_state(tmp_path: Path) -> None:
    executable = tmp_path / "ray"
    executable.write_text("ray-cli-placeholder\n", encoding="utf-8")
    client = RayCliObservationClient(
        "auto",
        executable=str(executable),
        supervisor=CaptureSupervisor("[]", truncated=True),
    )
    with pytest.raises(ContractError) as caught:
        client.tasks()
    assert caught.value.reason_code == "ray_observer_failed"


def test_ray_cli_jobs_query_binds_driver_identity(tmp_path: Path) -> None:
    executable = tmp_path / "ray"
    executable.write_text("ray-cli-placeholder\n", encoding="utf-8")
    supervisor = CaptureSupervisor(
        json.dumps([{"job_id": _JOB, "driver_pid": 111, "is_dead": False}])
    )
    client = RayCliObservationClient(
        "auto", executable=str(executable), supervisor=supervisor
    )

    snapshot = client.jobs()

    assert snapshot.jobs == (RayJobObservation(_JOB, 111, False),)
    assert supervisor.argv == (
        str(executable.resolve()),
        "list",
        "jobs",
        "--detail",
        "--format=json",
        "--limit",
        "10000",
        "--address",
        "auto",
    )


def _write_proc(root: Path, pid: int, parent: int, start: int, cgroup: bytes) -> None:
    process = root / str(pid)
    process.mkdir(parents=True)
    fields = ["S", str(parent), *("0" for _ in range(50))]
    fields[19] = str(start)
    (process / "stat").write_text(
        f"{pid} (ray::run_task) {' '.join(fields)}\n", encoding="utf-8"
    )
    (process / "cgroup").write_bytes(cgroup)


def test_ray_job_driver_is_exact_supervised_magpie_child(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = b"0::/apex\n"
    owner = os.getpid()
    _write_proc(proc, owner, 1, 500, cgroup)
    _write_proc(proc, 111, owner, 600, cgroup)
    argv = ("/usr/bin/python3", "-m", "Magpie", "benchmark")
    (proc / "111" / "cmdline").write_bytes(
        b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    )
    observer = LocalRayDriverProcessObserver(proc_root=proc)

    receipt = observer.freeze(RayJobObservation(_JOB, 111, False), argv)

    assert receipt.process.pid == 111
    assert receipt.process.start_time_ticks == 600
    assert receipt.process.cgroup_sha256 == sha256_bytes(cgroup)
    with pytest.raises(ContractError) as caught:
        observer.freeze(
            RayJobObservation(_JOB, 111, False),
            ("/usr/bin/python3", "-m", "Other"),
        )
    assert caught.value.reason_code == "ray_driver_process_mismatch"


def test_local_worker_and_gpu_owner_are_pid_cgroup_and_lease_bound(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = b"0::/ray/workers\n"
    _write_proc(proc, 222, 1, 700, cgroup)
    (proc / "222" / "cmdline").write_bytes(
        f"python\0ray-worker\0--node-id={_NODE}\0".encode()
    )
    _write_proc(proc, 321, 222, 900, cgroup)
    worker = LocalRayWorkerProcessObserver(proc_root=proc).freeze(_task())
    owner = SimpleNamespace(
        pid=321,
        uid=os.getuid(),
        start_time_ticks=900,
        cmdline_sha256="a" * 64,
        rsmi_device_indices=(0,),
    )
    receipt = SimpleNamespace(
        selected_devices=(
            SimpleNamespace(rsmi_index=0, unique_id="GPU-0000000000000001"),
        ),
        allowed_owners=(),
        foreign_owners=(owner,),
        digest="b" * 64,
    )
    observer = RocmRayGpuObserver(
        inspector=SimpleNamespace(inspect=lambda selector: receipt),
        proc_root=proc,
    )

    evidence = observer.observe(_task(), worker, _lease())

    assert evidence["devices"] == [
        {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}
    ]
    assert evidence["processes"][0]["start_time_ticks"] == 900
    assert evidence["processes"][0]["cgroup_sha256"] == sha256_bytes(cgroup)


def test_gpu_owner_outside_ray_worker_is_rejected(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = b"0::/ray/workers\n"
    _write_proc(proc, 222, 1, 700, cgroup)
    _write_proc(proc, 321, 1, 900, cgroup)
    worker = RayWorkerProcessObservation(
        222, os.getuid(), 700, "7" * 64, sha256_bytes(cgroup), ("/ray/workers",)
    )
    owner = SimpleNamespace(
        pid=321,
        uid=os.getuid(),
        start_time_ticks=900,
        cmdline_sha256="a" * 64,
        rsmi_device_indices=(0,),
    )
    receipt = SimpleNamespace(
        selected_devices=(
            SimpleNamespace(rsmi_index=0, unique_id="GPU-0000000000000001"),
        ),
        allowed_owners=(),
        foreign_owners=(owner,),
        digest="b" * 64,
    )
    observer = RocmRayGpuObserver(
        inspector=SimpleNamespace(inspect=lambda selector: receipt), proc_root=proc
    )

    with pytest.raises(ContractError) as caught:
        observer.observe(_task(), worker, _lease())

    assert caught.value.reason_code == "magpie_ray_gpu_process_escape"
