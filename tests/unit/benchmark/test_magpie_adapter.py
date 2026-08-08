from __future__ import annotations

import json
import hashlib
from pathlib import Path

import yaml

from apex.benchmark import MagpieBenchmarkAdapter, build_config_views
from apex.execution import ProcessResult
from apex.ports import BenchmarkPass, BenchmarkRequest
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt
from apex.benchmark.magpie import _lm_eval_expectation


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
        raw={},
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
        (workspace / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "vllm",
                    "model": "Qwen/example",
                    "workspace_dir": str(workspace),
                    "profiling_enabled": False,
                    "run_kind": "measurement",
                    "reward_eligible": True,
                    "inferencex_runtime_receipt": inferencex_receipt,
                    "lm_eval_runtime_receipt": lm_eval_runtime_evidence,
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
    def __init__(self, receipt: DependencyReceipt) -> None:
        self.receipt = receipt

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
            "run_kind": "diagnostic",
            "reward_eligible": False,
            "inferencex_runtime_receipt": inferencex_receipt,
            "lm_eval_runtime_receipt": not_requested,
            "throughput": {"output_throughput": 9.0},
            "latency": {},
            "errors": [],
        }
        (workspace / "benchmark_report.json").write_text(
            json.dumps(report), encoding="utf-8"
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
    adapter = MagpieBenchmarkAdapter(receipt, supervisor)
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
    adapter = MagpieBenchmarkAdapter(receipt, FakeDiagnosticSupervisor(receipt))
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
