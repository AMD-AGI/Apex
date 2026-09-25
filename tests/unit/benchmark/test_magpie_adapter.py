from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import yaml

from apex.benchmark import MagpieBenchmarkAdapter
from apex.benchmark import build_config_views as _build_config_views
from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_execution import LmEvalExecutionReceipt
from apex.core import sha256_file, sha256_json
from apex.execution import ProcessResult
from apex.ports import (
    BenchmarkPass,
    BenchmarkRequest,
    MagpieAttestationRequest,
    MagpieFormalMeasurementSupport,
    MagpieReportLocation,
)
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt
from apex.benchmark.magpie import _lm_eval_expectation
from apex.benchmark.quality import PRIMARY_METRICS
from tests.support.magpie_contract import resolved_contract


def build_config_views(source: Path, output: Path, **kwargs):
    receipt = kwargs["dependency_receipt"]
    return _build_config_views(
        source,
        output,
        resolved_contract=resolved_contract(source, receipt),
        **kwargs,
    )


def _receipt(tmp_path: Path) -> DependencyReceipt:
    magpie = tmp_path / "Magpie"
    tracelens = tmp_path / "TraceLens"
    inferencex = tmp_path / "InferenceX"
    magpie.mkdir()
    tracelens.mkdir()
    inferencex.mkdir()
    runtime = tmp_path / "lm-eval-runtime"
    runtime.mkdir()
    identity = {
        "lm_eval_commit": "6" * 40,
        "lm_eval_tree": "9" * 40,
        "lm_eval_version": "0.4.9.2",
        "python_abi": "cpython-312",
        "python_soabi": "cpython-312-x86_64-linux-gnu",
        "base_image_id": "sha256:" + "a" * 64,
        "base_image_repo_digest": "example/image@sha256:" + "b" * 64,
        "inferencex_commit": "3" * 40,
        "inferencex_tree": "8" * 40,
    }
    manifest = json.dumps(
        {
            "schema": "apex.lm-eval-runtime/v1",
            "runtime_sha256": "4" * 64,
            "site_packages": "site-packages",
            "identity": identity,
            "files": [],
        },
        sort_keys=True,
    ).encode("utf-8")
    (runtime / "lm_eval_runtime_manifest.json").write_bytes(manifest)
    return DependencyReceipt(
        schema="apex.dependency-receipt.v1",
        lock_sha256="a" * 64,
        python=Path("/verified/python"),
        roots={
            "magpie": magpie,
            "tracelens": tracelens,
            "inferencex": inferencex,
        },
        commits={
            "magpie": "1" * 40,
            "tracelens": "2" * 40,
            "inferencex": "3" * 40,
        },
        raw={"dependencies": {"tracelens": {"tree": "5" * 40}}},
        lm_eval_runtime=LmEvalRuntimeReceipt(
            runtime,
            "4" * 64,
            hashlib.sha256(manifest).hexdigest(),
            identity,
            1,
            "7" * 64,
        ),
    )


def _config(tmp_path: Path, receipt: DependencyReceipt) -> Path:
    source = tmp_path / "source.yaml"
    source.write_text(
        """benchmark:
  framework: vllm
  model: Qwen/example
  run_mode: docker
  envs: {TP: 1, CONC: 1, ISL: 8, OSL: 8}
  docker_image: example:image
""",
        encoding="utf-8",
    )
    return build_config_views(
        source, tmp_path / "views", dependency_receipt=receipt
    ).measurement


def _serving_runtime(config_path: Path) -> dict[str, object]:
    requested = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"][
        "docker_image"
    ]
    resolved = requested if requested.startswith("sha256:") else "sha256:" + "d" * 64
    return {
        "schema": "apex.magpie-serving-runtime-observation/v3",
        "execution_mode": "docker",
        "input_config_sha256": sha256_file(config_path),
        "input_image": requested,
        "input_image_id": resolved,
        "requested_image": requested,
        "resolved_image_id": resolved,
        "image_derivation": {
            "kind": "direct",
            "framework": "vllm",
            "runtime_schema": None,
            "base_image": requested,
            "base_image_id": resolved,
            "base_image_locator": requested,
            "derived_image": requested,
            "derived_image_id": resolved,
            "tracelens_source_commit": None,
            "tracelens_source_tree": None,
            "patch_version": None,
            "patch_path": None,
            "patch_sha256": None,
            "dependency_wheel_manifest_sha256": None,
            "validator": "docker-image-id",
            "verified": True,
        },
        "container_name": "magpie-benchmark-test",
        "container_spec_sha256": "e" * 64,
        "process_succeeded": True,
        "verified": True,
        "errors": [],
    }


def _tracelens_serving_runtime(
    config_path: Path,
    receipt: DependencyReceipt,
    *,
    source_commit: str | None = None,
) -> dict[str, object]:
    input_image = yaml.safe_load(config_path.read_text(encoding="utf-8"))["benchmark"][
        "docker_image"
    ]
    input_id = "sha256:" + "d" * 64
    requested = "magpie-tracelens-vllm:test"
    derived_id = "sha256:" + "9" * 64
    return {
        "schema": "apex.magpie-serving-runtime-observation/v3",
        "execution_mode": "docker",
        "input_config_sha256": sha256_file(config_path),
        "input_image": input_image,
        "input_image_id": input_id,
        "requested_image": requested,
        "resolved_image_id": derived_id,
        "image_derivation": {
            "kind": "tracelens-derived",
            "framework": "vllm",
            "runtime_schema": "magpie.tracelens-vllm-runtime/v1",
            "base_image": input_image,
            "base_image_id": input_id,
            "base_image_locator": input_id,
            "derived_image": requested,
            "derived_image_id": derived_id,
            "tracelens_source_commit": (
                source_commit or receipt.commits["tracelens"]
            ),
            "tracelens_source_tree": receipt.raw["dependencies"]["tracelens"][
                "tree"
            ],
            "patch_version": "v19",
            "patch_path": (
                "examples/custom_workflows/inference_analysis/vllm_patches/"
                "config_vllm_v0.19.0.patch"
            ),
            "patch_sha256": "6" * 64,
            "dependency_wheel_manifest_sha256": "7" * 64,
            "validator": "vllm-tracelens-runtime-validation/v1",
            "verified": True,
        },
        "container_name": "magpie-benchmark-test",
        "container_spec_sha256": "e" * 64,
        "process_succeeded": True,
        "verified": True,
        "errors": [],
    }


def _formal_quality_gate(
    workspace: Path,
    policy: dict[str, object],
    runtime: LmEvalRuntimeReceipt,
) -> dict[str, object]:
    results = workspace / "lm_eval" / "results.json"
    samples = workspace / "lm_eval" / "samples_gsm8k.jsonl"
    result_receipts = [
        {
            "path": "lm_eval/results.json",
            "size_bytes": results.stat().st_size,
            "sha256": sha256_file(results),
        }
    ]
    sample_receipts = [
        {
            "path": "lm_eval/samples_gsm8k.jsonl",
            "size_bytes": samples.stat().st_size,
            "sha256": sha256_file(samples),
        }
    ]
    outcomes = {
        "gsm8k": {
            "metric": "exact_match,strict-match",
            "value": 1.0,
            "source": "lm_eval/results.json",
        }
    }
    sample_digest = sha256_json(
        {"schema": "magpie.lm-eval-sample-set/v1", "artifacts": sample_receipts}
    )
    outcome_digest = sha256_json(
        {
            "schema": "magpie.lm-eval-outcomes/v1",
            "primary_metric_policy": list(PRIMARY_METRICS),
            "outcomes": outcomes,
            "result_artifacts": result_receipts,
            "sample_set_digest": sample_digest,
        }
    )
    execution = LmEvalExecutionReceipt(
        contract_sha256="1" * 64,
        config_sha256="6" * 64,
        policy_sha256=str(policy["sha256"]),
        policy_lock_sha256="7" * 64,
        task_definition_sha256=str(policy["task_definition_sha256"]),
        effective_task_definition_sha256="8" * 64,
        task_materialization_receipt_sha256="9" * 64,
        dataset_receipt_sha256="2" * 64,
        dataset_revision=str(policy["dataset_revision"]),
        runtime_sha256=runtime.runtime_sha256,
        runtime_manifest_sha256=runtime.manifest_sha256,
        runtime_lock_sha256=runtime.lock_sha256,
        launcher_sha256="1" * 64,
        image_repo_digest=runtime.identity["base_image_repo_digest"],
        image_id=runtime.identity["base_image_id"],
        container_id="3" * 64,
        listener_receipt_sha256="4" * 64,
        sidecar_spec_sha256="6" * 64,
        created_observation_sha256="7" * 64,
        exited_observation_sha256="8" * 64,
        broker_receipt_sha256="9" * 64,
        container_cleanup_sha256="a" * 64,
        runtime_probe_sha256="5" * 64,
        runtime_publication_sha256="b" * 64,
        result_artifacts=(EvaluatorArtifactReceipt.from_mapping(result_receipts[0]),),
        sample_artifacts=(EvaluatorArtifactReceipt.from_mapping(sample_receipts[0]),),
    )
    return {
        "requested": True,
        "status": "passed",
        "passed": True,
        "evidence_present": True,
        "evaluator_execution_receipt": execution.to_dict(),
        "primary_metric_policy": list(PRIMARY_METRICS),
        "primary_outcomes": outcomes,
        "result_artifact_receipts": result_receipts,
        "sample_artifact_receipts": sample_receipts,
        "outcome_digest": outcome_digest,
        "sample_set_digest": sample_digest,
        "task_count": 1,
        "tasks_truncated": False,
        "result_artifact_count": 1,
        "result_artifacts_truncated": False,
        "errors": [],
        "error_count": 0,
        "errors_truncated": False,
    }


def _write_execution_attestation(
    *,
    output_root: Path,
    report: Path,
    config_path: Path,
    profiling_enabled: bool,
    runtime: dict[str, object | None],
    quality_gate: dict[str, object] | None,
) -> None:
    evaluator = output_root / "evaluator"
    evaluator.mkdir()
    value = {
        "schema": "apex.magpie-execution-attestation/v1",
        "authority": "apex_evaluator",
        "official_report_path": report.relative_to(output_root).as_posix(),
        "official_report_size_bytes": report.stat().st_size,
        "report_sha256": sha256_file(report),
        "config_sha256": sha256_file(config_path),
        "run_id": output_root.parent.name,
        "pass_type": output_root.name,
        "lane_verified": True,
        "reward_eligible": output_root.name == "measurement",
        "profiling_enabled": profiling_enabled,
        "process": {
            "schema": "apex.magpie-process-attestation/v1",
            "argv_sha256": "f" * 64,
            "exit_code": 0,
            "timed_out": False,
            "succeeded": True,
            "verified": True,
        },
        "dependencies": {
            "schema": "apex.magpie-dependency-attestation/v1",
            "verified": True,
            "receipts": {
                "lock_sha256": "a" * 64,
                "dependencies": {
                    name: {
                        "root": f"/dependencies/{name}",
                        "commit": "b" * 40,
                        "tree": "c" * 40,
                    }
                    for name in ("magpie", "tracelens", "inferencex")
                },
            },
        },
        "runtime": {
            "schema": "apex.magpie-runtime-attestation/v1",
            "verified": True,
            **runtime,
        },
        "gpu_engagement": {
            "schema": "apex.magpie-gpu-engagement/v1",
            "verified": True,
            "devices": [
                {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}
            ],
            "processes": [{
                "pid": 123,
                "uid": 1000,
                "start_time_ticks": 456,
                "cmdline_sha256": "d" * 64,
                "rsmi_device_indices": [0],
            }],
        },
        "quality_gate": {
            "schema": "apex.magpie-quality-attestation/v1",
            "verified": True,
            "receipt": quality_gate,
        },
        "errors": [],
    }
    (evaluator / "execution_attestation.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


class FakeExecutionAttestor:
    is_available = True

    def __init__(self) -> None:
        self.prepared: MagpieAttestationRequest | None = None
        self.aborted: str | None = None

    @staticmethod
    def supports(execution_mode: str, lifecycle: str) -> bool:
        return execution_mode == "docker" and lifecycle == "one_shot"

    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        del execution_mode, lifecycle
        return MagpieFormalMeasurementSupport(True, None, "test")

    def prepare(self, request: MagpieAttestationRequest) -> object:
        self.prepared = request
        return request.run_root

    def launch_argv(self, session: object) -> tuple[str, ...]:
        assert Path(session).is_absolute()
        assert self.prepared is not None
        return self.prepared.benchmark_argv

    def abort(self, session: object, *, reason: str) -> None:
        assert self.prepared is not None
        assert session == self.prepared.run_root
        self.aborted = reason

    def locate_report(self, session: object) -> MagpieReportLocation:
        root = Path(session)
        reports = tuple(root.rglob("benchmark_report.json"))
        if len(reports) != 1:
            return MagpieReportLocation(None, "benchmark_report_missing")
        return MagpieReportLocation(reports[0].resolve())

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None:
        assert self.prepared is not None
        assert session == self.prepared.run_root
        assert command_exit_code == 0
        assert timed_out is False
        return (
            report_path.parent.parent / "evaluator" / "execution_attestation.json"
            if report_path
            else None
        )


class FakeSupervisor:
    def __init__(self, receipt: DependencyReceipt) -> None:
        self.call = None
        self.receipt = receipt

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        self.call = {
            "argv": tuple(argv),
            "cwd": cwd,
            "environment": environment,
            "timeout_seconds": timeout_seconds,
        }
        config_path = Path(argv[argv.index("--benchmark-config") + 1])
        output_root = Path(argv[argv.index("--output-dir") + 1])
        workspace = output_root / "benchmark_vllm_20260807_000000"
        workspace.mkdir()
        (workspace / "inferencex_runtime").mkdir()
        inferencex_receipt = {
            "schema": "magpie.inferencex-runtime-receipt/v1",
            "source_root": str(self.receipt.root("inferencex").resolve()),
            "source_is_git": True,
            "source_commit": self.receipt.commits["inferencex"],
            "source_tree": self.receipt.lm_eval_runtime.identity["inferencex_tree"],
            "source_clean": True,
            "source_status_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855"
            ),
            "source_status_unchanged": True,
            "runtime_path": "inferencex_runtime",
            "materialization_method": "git_private_index_checkout",
        }
        (workspace / "inferencex_runtime_receipt.json").write_text(
            json.dumps(inferencex_receipt), encoding="utf-8"
        )
        runtime = self.receipt.lm_eval_runtime
        assert runtime is not None
        manifest_bytes = runtime.root.joinpath(
            "lm_eval_runtime_manifest.json"
        ).read_bytes()
        manifest_path = workspace / "lm_eval_runtime_manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        runtime_receipt = {
            "schema": "magpie.lm-eval-runtime-receipt/v1",
            "runtime_sha256": runtime.runtime_sha256,
            "identity": dict(runtime.identity),
            "manifest_sha256": runtime.manifest_sha256,
            "site_packages": "site-packages",
            "python_abi": runtime.identity["python_abi"],
            "lm_eval_version": runtime.identity["lm_eval_version"],
            "lm_eval_module": "site-packages/lm_eval/__init__.py",
            "execution_mode": "docker",
            "read_only_mount": True,
            "verified": True,
        }
        runtime_receipt_path = workspace / "lm_eval_runtime_receipt.json"
        runtime_receipt_bytes = json.dumps(runtime_receipt).encode("utf-8")
        runtime_receipt_path.write_bytes(runtime_receipt_bytes)
        lm_eval_runtime_evidence = {
            "schema": "magpie.lm-eval-runtime-evidence/v1",
            "requested": True,
            "status": "verified",
            "verified": True,
            "evidence_present": True,
            "runtime_sha256": runtime.runtime_sha256,
            "identity": dict(runtime.identity),
            "mount_mode": "read_only",
            "manifest_artifact": {
                "path": manifest_path.name,
                "size_bytes": len(manifest_bytes),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            "receipt_artifact": {
                "path": runtime_receipt_path.name,
                "size_bytes": len(runtime_receipt_bytes),
                "sha256": hashlib.sha256(runtime_receipt_bytes).hexdigest(),
            },
            "errors": [],
        }
        (workspace / "lm_eval").mkdir()
        (workspace / "lm_eval" / "results.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 1.0}}}),
            encoding="utf-8",
        )
        (workspace / "lm_eval" / "samples_gsm8k.jsonl").write_text(
            '{"doc_id": 1, "exact_match": true}\n', encoding="utf-8"
        )
        report_path = workspace / "benchmark_report.json"
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        policy = resolved["apex"]["benchmark_view"]["quality_contract"][
            "evaluator_policy"
        ]
        quality_gate = _formal_quality_gate(workspace, policy, runtime)
        report_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "vllm",
                    "model": "Qwen/example",
                    "workspace_dir": str(workspace),
                    "profiling_enabled": False,
                    "throughput": {"output_throughput": 10.0},
                    "latency": {
                        "ttft": {"p99_ms": 3.0},
                        "tpot": {"p99_ms": 1.0},
                    },
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        _write_execution_attestation(
            output_root=output_root,
            report=report_path,
            config_path=config_path,
            profiling_enabled=False,
            runtime={
                "model_revision_receipt": None,
                "inferencex_runtime_receipt": inferencex_receipt,
                "lm_eval_runtime_receipt": lm_eval_runtime_evidence,
                "serving_runtime_receipt": _serving_runtime(config_path),
            },
            quality_gate=quality_gate,
        )
        return ProcessResult(
            argv=tuple(argv),
            exit_code=0,
            timed_out=False,
            stdout="ok",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
        )


class FakeDiagnosticSupervisor:
    def __init__(
        self,
        receipt: DependencyReceipt,
        *,
        tracelens_commit: str | None = None,
    ) -> None:
        self.receipt = receipt
        self.tracelens_commit = tracelens_commit

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        config_path = Path(argv[argv.index("--benchmark-config") + 1])
        benchmark = yaml.safe_load(config_path.read_text())["benchmark"]
        assert benchmark["envs"]["RUN_EVAL"] == "false"
        assert "lm_eval_runtime" not in benchmark
        output_root = Path(argv[argv.index("--output-dir") + 1])
        workspace = output_root / "benchmark_vllm_20260807_000001"
        workspace.mkdir()
        (workspace / "inferencex_runtime").mkdir()
        inferencex_receipt = {
            "schema": "magpie.inferencex-runtime-receipt/v1",
            "source_root": str(self.receipt.root("inferencex").resolve()),
            "source_is_git": True,
            "source_commit": self.receipt.commits["inferencex"],
            "source_tree": self.receipt.lm_eval_runtime.identity["inferencex_tree"],
            "source_clean": True,
            "source_status_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855"
            ),
            "source_status_unchanged": True,
            "runtime_path": "inferencex_runtime",
            "materialization_method": "git_private_index_checkout",
        }
        (workspace / "inferencex_runtime_receipt.json").write_text(
            json.dumps(inferencex_receipt), encoding="utf-8"
        )
        not_requested = {
            "schema": "magpie.lm-eval-runtime-evidence/v1",
            "requested": False,
            "status": "not_requested",
            "verified": False,
            "evidence_present": False,
            "runtime_sha256": None,
            "identity": None,
            "mount_mode": None,
            "manifest_artifact": None,
            "receipt_artifact": None,
            "errors": [],
        }
        report = {
            "success": True,
            "framework": "vllm",
            "model": "Qwen/example",
            "workspace_dir": str(workspace),
            "profiling_enabled": True,
            "throughput": {"output_throughput": 9.0},
            "latency": {},
            "errors": [],
        }
        report_path = workspace / "benchmark_report.json"
        report_path.write_text(
            json.dumps(report), encoding="utf-8"
        )
        _write_execution_attestation(
            output_root=output_root,
            report=report_path,
            config_path=config_path,
            profiling_enabled=True,
            runtime={
                "model_revision_receipt": None,
                "inferencex_runtime_receipt": inferencex_receipt,
                "lm_eval_runtime_receipt": not_requested,
                "serving_runtime_receipt": _tracelens_serving_runtime(
                    config_path,
                    self.receipt,
                    source_commit=self.tracelens_commit,
                ),
            },
            quality_gate=None,
        )
        return ProcessResult(
            argv=tuple(argv), exit_code=0, timed_out=False,
            stdout="ok", stderr="", stdout_truncated=False,
            stderr_truncated=False, duration_seconds=0.1,
        )


def test_adapter_uses_receipt_python_argv_and_normalizes_result(
    tmp_path: Path, monkeypatch
) -> None:
    receipt = _receipt(tmp_path)
    supervisor = FakeSupervisor(receipt)
    monkeypatch.setenv("PYTHONPATH", "/wrong/magpie")
    monkeypatch.setenv("BASH_ENV", "/tmp/injected-startup")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-agent-secret")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("HF_HOME", "/model-cache")
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", "registry-secret")
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    monkeypatch.setenv("DOCKER_CONFIG", "/home/test/.docker")
    monkeypatch.setenv("MAGPIE_PROTECT_BENCHMARK_CONTAINER", "true")
    attestor = FakeExecutionAttestor()
    adapter = MagpieBenchmarkAdapter(receipt, supervisor, attestor)
    request = BenchmarkRequest(
        run_id="baseline",
        config_path=_config(tmp_path, receipt),
        output_dir=tmp_path / "runs",
        pass_type=BenchmarkPass.MEASUREMENT,
        timeout_seconds=99,
        environment={"HF_TOKEN": "explicit-hf-token", "BENCHMARK_LANE": "acceptance"},
    )

    result = adapter.run(request)

    assert result.succeeded
    assert attestor.prepared is not None
    assert result.metrics["output_throughput"] == 10.0
    assert supervisor.call is not None
    assert supervisor.call["argv"][:4] == (
        "/verified/python",
        "-m",
        "Magpie",
        "benchmark",
    )
    assert "-c" not in supervisor.call["argv"]
    assert supervisor.call["cwd"] == receipt.root("magpie").resolve()
    assert "PYTHONPATH" not in supervisor.call["environment"]
    assert "BASH_ENV" not in supervisor.call["environment"]
    assert "LD_PRELOAD" not in supervisor.call["environment"]
    assert "OPENAI_API_KEY" not in supervisor.call["environment"]
    assert "DOCKER_AUTH_CONFIG" not in supervisor.call["environment"]
    assert supervisor.call["environment"]["DOCKER_HOST"] == (
        "unix:///run/user/1000/docker.sock"
    )
    assert supervisor.call["environment"]["DOCKER_CONFIG"] == "/home/test/.docker"
    assert supervisor.call["environment"][
        "MAGPIE_PROTECT_BENCHMARK_CONTAINER"
    ] == "true"
    assert supervisor.call["environment"]["ROCR_VISIBLE_DEVICES"] == "2"
    assert supervisor.call["environment"]["HF_HOME"] == "/model-cache"
    assert supervisor.call["environment"]["HF_TOKEN"] == "explicit-hf-token"
    assert supervisor.call["environment"]["BENCHMARK_LANE"] == "acceptance"
    assert supervisor.call["environment"]["PYTHONNOUSERSITE"] == "1"
    assert supervisor.call["environment"]["TRACELENS_REPO_PATH"] == str(
        receipt.root("tracelens").resolve()
    )


def test_adapter_runs_serving_diagnostic_without_lm_eval_runtime(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    measurement = _config(tmp_path, receipt)
    diagnostic = measurement.parent / "benchmark.diagnostic.resolved.yaml"
    adapter = MagpieBenchmarkAdapter(
        receipt, FakeDiagnosticSupervisor(receipt), FakeExecutionAttestor()
    )
    request = BenchmarkRequest(
        run_id="diagnostic",
        config_path=diagnostic,
        output_dir=tmp_path / "runs",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        timeout_seconds=99,
    )

    result = adapter.run_normalized(request)

    assert result.succeeded
    assert result.quality.required is False
    assert result.lm_eval_runtime.required is False
    assert result.lm_eval_runtime.passed
    assert result.serving_runtime.input_image == "example:image"
    assert result.serving_runtime.requested_image == "magpie-tracelens-vllm:test"
    assert result.serving_runtime.resolved_image_id == "sha256:" + "9" * 64
    assert result.serving_runtime.image_derivation is not None
    assert result.serving_runtime.image_derivation["kind"] == "tracelens-derived"


def test_adapter_rejects_diagnostic_with_unpinned_tracelens_lineage(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    measurement = _config(tmp_path, receipt)
    diagnostic = measurement.parent / "benchmark.diagnostic.resolved.yaml"
    adapter = MagpieBenchmarkAdapter(
        receipt,
        FakeDiagnosticSupervisor(receipt, tracelens_commit="8" * 40),
        FakeExecutionAttestor(),
    )
    request = BenchmarkRequest(
        run_id="diagnostic",
        config_path=diagnostic,
        output_dir=tmp_path / "runs",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        timeout_seconds=99,
    )

    result = adapter.run_normalized(request)

    assert not result.succeeded
    assert "serving_runtime_image_lineage_mismatch" in result.errors


def test_lm_eval_evidence_is_required_only_for_lm_eval_quality(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    benchmark = {"run_mode": "docker"}

    assert _lm_eval_expectation(
        benchmark, {"required": True, "kind": "framework_quality_gate"}, receipt
    ) == (None, None)
    assert _lm_eval_expectation(
        benchmark, {"required": False, "kind": "trace_only"}, receipt
    ) == (None, "not_requested")
    assert _lm_eval_expectation(
        benchmark, {"required": True, "kind": "lm_eval"}, receipt
    ) == (receipt.lm_eval_runtime, "docker")


def test_adapter_fails_before_supervisor_without_attestor(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    supervisor = FakeSupervisor(receipt)
    adapter = MagpieBenchmarkAdapter(receipt, supervisor)
    request = BenchmarkRequest(
        run_id="no-attestor",
        config_path=_config(tmp_path, receipt),
        output_dir=tmp_path / "runs",
        pass_type=BenchmarkPass.MEASUREMENT,
    )

    result = adapter.run_normalized(request)

    assert not result.succeeded
    assert result.errors == ("magpie_execution_attestor_unavailable",)
    assert supervisor.call is None
    assert not (tmp_path / "runs").exists()


def test_formal_measurement_support_requires_the_same_execution_lane(
    tmp_path: Path,
) -> None:
    adapter = MagpieBenchmarkAdapter(
        _receipt(tmp_path), execution_attestor=FakeExecutionAttestor()
    )

    assert adapter.formal_measurement_support("docker", "one_shot").available
    unsupported = adapter.formal_measurement_support("local", "one_shot")
    assert not unsupported.available
    assert unsupported.reason_code == "magpie_execution_attestor_unavailable"


class _InvalidLaunchAttestor(FakeExecutionAttestor):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def launch_argv(self, session: object) -> tuple[str, ...]:
        canonical = super().launch_argv(session)
        if self.failure == "raise":
            raise RuntimeError("launch projection failed")
        return ("/wrong/python", *canonical[1:])


class _RaisingSupervisor:
    def run(self, *args, **kwargs):
        del args, kwargs
        raise OSError("cannot start Magpie")


@pytest.mark.parametrize("failure", ["raise", "drift"])
def test_adapter_aborts_attestor_when_launch_projection_fails(
    tmp_path: Path, failure: str
) -> None:
    receipt = _receipt(tmp_path)
    attestor = _InvalidLaunchAttestor(failure)
    adapter = MagpieBenchmarkAdapter(receipt, FakeSupervisor(receipt), attestor)
    request = BenchmarkRequest(
        run_id=f"launch-{failure}",
        config_path=_config(tmp_path, receipt),
        output_dir=tmp_path / "runs",
        pass_type=BenchmarkPass.MEASUREMENT,
    )

    result = adapter.run_normalized(request)

    assert not result.succeeded
    assert result.errors[0].startswith("magpie_execution_attestor_prepare_failed:")
    assert attestor.aborted == result.errors[0]


def test_adapter_aborts_attestor_when_supervisor_cannot_start(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    attestor = FakeExecutionAttestor()
    adapter = MagpieBenchmarkAdapter(receipt, _RaisingSupervisor(), attestor)
    request = BenchmarkRequest(
        run_id="process-start-failure",
        config_path=_config(tmp_path, receipt),
        output_dir=tmp_path / "runs",
        pass_type=BenchmarkPass.MEASUREMENT,
    )

    result = adapter.run_normalized(request)

    assert not result.succeeded
    assert result.errors == (
        "magpie_process_start_failed:OSError:cannot start Magpie",
    )
    assert attestor.aborted == result.errors[0]
