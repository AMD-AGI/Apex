from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from apex.benchmark import parse_benchmark_report as _parse_benchmark_report
from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_execution import LmEvalExecutionReceipt
from apex.benchmark.results import empty_result
from apex.core import sha256_file
from apex.ports import BenchmarkPass
from apex.runtime import LmEvalRuntimeReceipt


def parse_benchmark_report(report_path: Path, **kwargs):
    """Materialize the Apex evaluator sidecar used by result-parser tests."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluator_root = report_path.parent.parent / "evaluator"
    unique_root = evaluator_root / report_path.parent.name
    path = unique_root / "execution_attestation.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    )
    previous_runtime = previous.get("runtime", {})
    previous_quality = previous.get("quality_gate", {}).get("receipt")
    requested_pass = kwargs["pass_type"]
    expected_kind = requested_pass.value
    claimed_kind = report.pop("run_kind", expected_kind)
    reward_eligible = report.pop(
        "reward_eligible", requested_pass is BenchmarkPass.MEASUREMENT
    )
    runtime = {
        name: report.pop(name, previous_runtime.get(name))
        for name in (
            "model_revision_receipt",
            "inferencex_runtime_receipt",
            "lm_eval_runtime_receipt",
            "serving_runtime_receipt",
        )
    }
    quality_gate = report.pop("quality_gate", previous_quality)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    unique_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = kwargs.get("expected_config_sha256") or "1" * 64
    exit_code = kwargs.get("command_exit_code", 0)
    timed_out = kwargs.get("timed_out", False)
    attestation = {
        "schema": "apex.magpie-execution-attestation/v1",
        "authority": "apex_evaluator",
        "official_report_path": report_path.relative_to(
            evaluator_root.parent
        ).as_posix(),
        "official_report_size_bytes": report_path.stat().st_size,
        "report_sha256": sha256_file(report_path),
        "config_sha256": config_sha256,
        "run_id": kwargs["run_id"],
        "pass_type": requested_pass.value,
        "lane_verified": claimed_kind == expected_kind,
        "reward_eligible": reward_eligible,
        "profiling_enabled": report.get("profiling_enabled") is True,
        "process": {
            "schema": "apex.magpie-process-attestation/v1",
            "argv_sha256": "2" * 64,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "succeeded": exit_code == 0 and not timed_out,
            "verified": True,
        },
        "dependencies": {
            "schema": "apex.magpie-dependency-attestation/v1",
            "verified": True,
            "receipts": {
                "lock_sha256": "3" * 64,
                "dependencies": {
                    name: {
                        "root": f"/dependencies/{name}",
                        "commit": "4" * 40,
                        "tree": "5" * 40,
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
                "cmdline_sha256": "6" * 64,
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
    path.write_text(json.dumps(attestation), encoding="utf-8")
    return _parse_benchmark_report(
        report_path, execution_attestation_path=path, **kwargs
    )


def _report(
    workspace: Path, *, quality_gate: dict | None = None, framework: str = "vllm"
) -> Path:
    report = workspace / "benchmark_report.json"
    payload = {
        "success": True,
        "framework": framework,
        "model": "Qwen/example",
        "workspace_dir": str(workspace),
        "profiling_enabled": False,
        "run_kind": "measurement",
        "reward_eligible": True,
        "throughput": {
            "request_throughput": 12.5,
            "output_throughput": 512.0,
            "total_token_throughput": 768.0,
            "completed_requests": 160,
            "duration_seconds": 12.8,
        },
        "latency": {
            "ttft": {"mean_ms": 8.0, "median_ms": 7.0, "p99_ms": 13.0},
            "tpot": {"mean_ms": 2.0, "median_ms": 1.8, "p99_ms": 3.4},
        },
        "errors": [],
    }
    if quality_gate is not None:
        payload["quality_gate"] = quality_gate
    report.write_text(json.dumps(payload), encoding="utf-8")
    return report


def _policy() -> dict[str, object]:
    return {
        "primary_metric": "exact_match,strict-match",
        "tasks": "gsm8k",
        "sha256": "4" * 64,
        "task_definition_sha256": "5" * 64,
    }


def _execution_receipt(results: Path, samples: Path) -> dict[str, object] | None:
    if results.stat().st_size <= 0 or samples.stat().st_size <= 0:
        return None
    result = EvaluatorArtifactReceipt(
        "lm_eval/results.json", results.stat().st_size, sha256_file(results)
    )
    sample = EvaluatorArtifactReceipt(
        "lm_eval/samples_gsm8k.jsonl", samples.stat().st_size, sha256_file(samples)
    )
    return LmEvalExecutionReceipt(
        contract_sha256="1" * 64,
        config_sha256="3" * 64,
        policy_sha256="4" * 64,
        policy_lock_sha256="2" * 64,
        task_definition_sha256="5" * 64,
        effective_task_definition_sha256="d" * 64,
        task_materialization_receipt_sha256="e" * 64,
        dataset_receipt_sha256="6" * 64,
        dataset_revision="f" * 40,
        runtime_sha256="7" * 64,
        runtime_manifest_sha256="f" * 64,
        runtime_lock_sha256="0" * 64,
        launcher_sha256="1" * 64,
        image_repo_digest="example/image@sha256:" + "8" * 64,
        image_id="sha256:" + "9" * 64,
        container_id="a" * 64,
        listener_receipt_sha256="b" * 64,
        sidecar_spec_sha256="d" * 64,
        created_observation_sha256="e" * 64,
        exited_observation_sha256="f" * 64,
        broker_receipt_sha256="0" * 64,
        container_cleanup_sha256="1" * 64,
        runtime_probe_sha256="c" * 64,
        runtime_publication_sha256="2" * 64,
        result_artifacts=(result,),
        sample_artifacts=(sample,),
    ).to_dict()


def _formal_gate(first, results: Path, samples: Path) -> dict:
    gate = {
        "requested": True,
        "status": "passed",
        "passed": True,
        "evidence_present": True,
        "primary_metric_policy": [
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "exact_match,none",
            "exact_match",
            "acc_norm,none",
            "acc,none",
            "acc_norm",
            "acc",
            "pass@1,none",
            "pass@1",
        ],
        "primary_outcomes": {
            "gsm8k": {
                "metric": "exact_match,strict-match",
                "value": first.quality.primary_metrics[0].value,
                "source": "lm_eval/results.json",
            }
        },
        "result_artifact_receipts": [
            {
                "path": "lm_eval/results.json",
                "size_bytes": results.stat().st_size,
                "sha256": sha256_file(results),
            }
        ],
        "sample_artifact_receipts": [
            {
                "path": "lm_eval/samples_gsm8k.jsonl",
                "size_bytes": samples.stat().st_size,
                "sha256": sha256_file(samples),
            }
        ],
        "outcome_digest": first.quality.outcome_digest,
        "sample_set_digest": first.quality.sample_set_digest,
        "task_count": 1,
        "tasks_truncated": False,
        "result_artifact_count": 1,
        "result_artifacts_truncated": False,
        "errors": [],
        "error_count": 0,
        "errors_truncated": False,
    }
    receipt = _execution_receipt(results, samples)
    if receipt is not None:
        gate["evaluator_execution_receipt"] = receipt
    return gate


def _add_lm_eval_runtime_evidence(report_path: Path) -> LmEvalRuntimeReceipt:
    identity = {
        "lm_eval_commit": "1" * 40,
        "lm_eval_tree": "2" * 40,
        "lm_eval_version": "0.4.9.2",
        "python_abi": "cpython-312",
        "python_soabi": "cpython-312-x86_64-linux-gnu",
        "base_image_id": "sha256:" + "3" * 64,
        "base_image_repo_digest": "example/image@sha256:" + "4" * 64,
        "inferencex_commit": "5" * 40,
        "inferencex_tree": "6" * 40,
    }
    runtime_sha256 = "7" * 64
    manifest = {
        "schema": "apex.lm-eval-runtime/v1",
        "runtime_sha256": runtime_sha256,
        "site_packages": "site-packages",
        "identity": identity,
        "files": [{"path": "lm_eval/__init__.py", "size_bytes": 1, "mode": 292, "sha256": "8" * 64}],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path = report_path.parent / "lm_eval_runtime_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    runtime_receipt = {
        "schema": "magpie.lm-eval-runtime-receipt/v1",
        "runtime_sha256": runtime_sha256,
        "identity": identity,
        "manifest_sha256": manifest_sha256,
        "site_packages": "site-packages",
        "python_abi": "cpython-312",
        "lm_eval_version": "0.4.9.2",
        "lm_eval_module": "site-packages/lm_eval/__init__.py",
        "execution_mode": "docker",
        "read_only_mount": True,
        "verified": True,
    }
    receipt_bytes = json.dumps(runtime_receipt, sort_keys=True).encode("utf-8")
    receipt_path = report_path.parent / "lm_eval_runtime_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    evidence = {
        "schema": "magpie.lm-eval-runtime-evidence/v1",
        "requested": True,
        "status": "verified",
        "verified": True,
        "evidence_present": True,
        "runtime_sha256": runtime_sha256,
        "identity": identity,
        "mount_mode": "read_only",
        "manifest_artifact": {
            "path": manifest_path.name,
            "size_bytes": len(manifest_bytes),
            "sha256": manifest_sha256,
        },
        "receipt_artifact": {
            "path": receipt_path.name,
            "size_bytes": len(receipt_bytes),
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        "errors": [],
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["lm_eval_runtime_receipt"] = evidence
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return LmEvalRuntimeReceipt(
        report_path.parent / "runtime",
        runtime_sha256,
        manifest_sha256,
        identity,
        1,
        "9" * 64,
    )


def _not_requested_lm_eval_runtime_evidence() -> dict:
    return {
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


def test_normalizes_percentiles_and_lm_eval_quality(tmp_path: Path) -> None:
    eval_dir = tmp_path / "lm_eval" / "model"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.91,
                        "exact_match_stderr,strict-match": 0.01,
                        "alias": "gsm8k",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )

    assert result.succeeded
    assert result.throughput.output_tokens_per_second == 512.0
    assert result.latency.ttft.p99_ms == 13.0
    assert result.latency.tpot.p99_ms == 3.4
    assert [(item.task, item.name, item.value) for item in result.quality.metrics] == [
        ("gsm8k", "exact_match,strict-match", 0.91)
    ]
    assert result.metric_mapping()[
        "quality.gsm8k.exact_match,strict-match"
    ] == 0.91


def test_strict_match_wins_and_raw_samples_are_bound(tmp_path: Path) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    results = eval_dir / "results.json"
    results.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,flexible-extract": 0.99,
                        "exact_match,strict-match": 0.91,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    samples = eval_dir / "samples_gsm8k.jsonl"
    samples.write_text('{"doc_id": 1, "exact_match": true}\n', encoding="utf-8")
    first = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )
    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    report["quality_gate"] = _formal_gate(first, results, samples)
    first.report_path.write_text(json.dumps(report), encoding="utf-8")
    result = parse_benchmark_report(
        first.report_path,
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_evaluator_policy=_policy(),
    )

    assert result.succeeded
    assert result.quality.primary_metrics[0].name == "exact_match,strict-match"
    assert result.quality.primary_metrics[0].value == pytest.approx(0.91)
    assert samples.resolve() in result.quality.raw_artifact_paths
    assert samples.resolve() in result.artifacts
    assert len(result.quality.outcome_digest or "") == 64
    assert len(result.quality.sample_set_digest or "") == 64


def test_formal_quality_rejects_tampered_outcome_digest(tmp_path: Path) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    results = eval_dir / "results.json"
    results.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}),
        encoding="utf-8",
    )
    samples = eval_dir / "samples_gsm8k.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    first = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )
    gate = _formal_gate(first, results, samples)
    gate["outcome_digest"] = "0" * 64
    report_data = json.loads(first.report_path.read_text(encoding="utf-8"))
    report_data["quality_gate"] = gate
    first.report_path.write_text(json.dumps(report_data), encoding="utf-8")
    result = parse_benchmark_report(
        first.report_path,
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_evaluator_policy=_policy(),
    )

    assert not result.succeeded
    assert result.quality.error == "quality_outcome_digest_mismatch"


def test_formal_quality_rejects_failed_magpie_gate_with_matching_digests(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    results = eval_dir / "results.json"
    results.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}),
        encoding="utf-8",
    )
    samples = eval_dir / "samples_gsm8k.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    first = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )
    gate = _formal_gate(first, results, samples)
    gate.update({"status": "invalid", "passed": False})
    report_data = json.loads(first.report_path.read_text(encoding="utf-8"))
    report_data["quality_gate"] = gate
    first.report_path.write_text(json.dumps(report_data), encoding="utf-8")

    result = parse_benchmark_report(
        first.report_path,
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_evaluator_policy=_policy(),
    )

    assert not result.succeeded
    assert result.quality.error == "quality_gate_not_passed"
    assert result.quality.hard_failure is False


def test_formal_quality_marks_explicit_bound_failure_as_hard_failure(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    results = eval_dir / "results.json"
    results.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.7}}}),
        encoding="utf-8",
    )
    samples = eval_dir / "samples_gsm8k.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    first = parse_benchmark_report(
        _report(tmp_path),
        run_id="candidate",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )
    gate = _formal_gate(first, results, samples)
    gate.update({"status": "failed", "passed": False})
    report_data = json.loads(first.report_path.read_text(encoding="utf-8"))
    report_data["quality_gate"] = gate
    first.report_path.write_text(json.dumps(report_data), encoding="utf-8")

    result = parse_benchmark_report(
        first.report_path,
        run_id="candidate",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_evaluator_policy=_policy(),
    )

    assert not result.succeeded
    assert result.errors == ("quality_gate_not_passed",)
    assert result.quality.hard_failure is True


def test_formal_quality_rejects_empty_sample_without_execution_receipt(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    results = eval_dir / "results.json"
    results.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}),
        encoding="utf-8",
    )
    samples = eval_dir / "samples_gsm8k.jsonl"
    samples.write_bytes(b"")
    first = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )
    gate = _formal_gate(first, results, samples)
    report_data = json.loads(first.report_path.read_text(encoding="utf-8"))
    report_data["quality_gate"] = gate
    first.report_path.write_text(json.dumps(report_data), encoding="utf-8")

    result = parse_benchmark_report(
        first.report_path,
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_evaluator_policy=_policy(),
    )

    assert not result.succeeded
    assert result.quality.error == "quality_evaluator_execution_receipt_missing"


def test_formal_quality_requires_raw_samples(tmp_path: Path) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    (eval_dir / "results.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9}}}),
        encoding="utf-8",
    )
    result = parse_benchmark_report(
        _report(tmp_path, quality_gate={}),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_evaluator_policy=_policy(),
    )

    assert not result.succeeded
    assert result.quality.sample_set_digest is None


def test_quality_ignores_runtime_copy_outside_evaluator_directory(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    evidence = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.91}}
    }
    (eval_dir / "results_accepted.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    runtime_dir = tmp_path / "inferencex_runtime" / "benchmarks"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "results_copy.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )

    result = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )

    assert result.succeeded
    assert result.quality.source_paths == (
        (eval_dir / "results_accepted.json").resolve(),
    )


def test_missing_required_quality_fails_closed(tmp_path: Path) -> None:
    result = parse_benchmark_report(
        _report(tmp_path),
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )

    assert not result.succeeded
    assert result.quality.error == "quality_evidence_missing"
    assert "quality_evidence_missing" in result.errors


def test_empty_result_marks_a_requested_lm_eval_runtime_unverified(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, framework="xdit")
    expected = _add_lm_eval_runtime_evidence(report_path)

    result = empty_result(
        run_id="missing-report",
        pass_type=BenchmarkPass.MEASUREMENT,
        workspace=tmp_path,
        error="benchmark_report_missing",
        command_exit_code=1,
        timed_out=False,
        expected_lm_eval_runtime=expected,
        expected_lm_eval_execution_mode="docker",
    )

    assert result.lm_eval_runtime.required
    assert not result.lm_eval_runtime.passed
    assert result.lm_eval_runtime.error == "benchmark_report_missing"


def test_skipped_framework_quality_gate_fails_closed(tmp_path: Path) -> None:
    result = parse_benchmark_report(
        _report(
            tmp_path,
            quality_gate={"passed": True, "skipped": True, "ssim": 0.97},
            framework="xdit",
        ),
        run_id="diffusion",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )

    assert not result.succeeded
    assert result.quality.error == "quality_gate_not_passed"


def test_report_claiming_success_with_errors_fails_closed(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        quality_gate={"passed": True, "ssim": 0.99},
        framework="xdit",
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["errors"] = ["server emitted a partial result"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="candidate",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )

    assert not result.succeeded
    assert "server emitted a partial result" in result.errors


def test_measurement_rejects_diagnostic_or_ineligible_report(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, framework="xdit")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["quality_gate"] = {"passed": True, "ssim": 0.99}
    payload["run_kind"] = "diagnostic"
    payload["reward_eligible"] = False
    payload["profiling_enabled"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="candidate",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
    )

    assert not result.succeeded
    assert "execution_attestation_lane_unverified" in result.errors
    assert "execution_attestation_reward_eligibility_mismatch" in result.errors


def test_diagnostic_requires_non_reward_eligible_diagnostic_report(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path, framework="xdit")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["quality_gate"] = {"passed": True, "ssim": 0.99}
    payload["run_kind"] = "diagnostic"
    payload["reward_eligible"] = False
    payload["profiling_enabled"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="diagnostic",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        quality_required=True,
    )

    assert result.succeeded
    assert result.run_kind == "diagnostic"
    assert result.reward_eligible is False


def test_serving_diagnostic_accepts_explicit_trace_only_lm_eval_receipt(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "run_kind": "diagnostic",
            "reward_eligible": False,
            "profiling_enabled": True,
            "lm_eval_runtime_receipt": _not_requested_lm_eval_runtime_evidence(),
        }
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="trace-only",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        quality_required=False,
        expected_lm_eval_execution_mode="not_requested",
    )

    assert result.succeeded
    assert result.quality.required is False
    assert result.quality.passed
    assert result.quality.metrics == ()
    assert result.lm_eval_runtime.required is False
    assert result.lm_eval_runtime.passed
    assert result.lm_eval_runtime.manifest_path is None
    assert result.lm_eval_runtime.receipt_path is None


def test_trace_only_diagnostic_rejects_claimed_lm_eval_execution(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = _not_requested_lm_eval_runtime_evidence()
    evidence["requested"] = True
    payload.update(
        {
            "run_kind": "diagnostic",
            "reward_eligible": False,
            "profiling_enabled": True,
            "lm_eval_runtime_receipt": evidence,
        }
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="trace-only-tampered",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        quality_required=False,
        expected_lm_eval_execution_mode="not_requested",
    )

    assert not result.succeeded
    assert "lm_eval_not_requested_evidence_missing" in result.errors


def test_trace_only_diagnostic_rejects_a_missing_not_requested_receipt(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "run_kind": "diagnostic",
            "reward_eligible": False,
            "profiling_enabled": True,
        }
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="trace-only-missing-receipt",
        pass_type=BenchmarkPass.DIAGNOSTIC,
        quality_required=False,
        expected_lm_eval_execution_mode="not_requested",
    )

    assert not result.succeeded
    assert "lm_eval_not_requested_evidence_missing" in result.errors


def _add_model_revision_receipt(report_path: Path, revision: str) -> None:
    receipt_path = report_path.parent / "model_revision_receipt.json"
    snapshot = f"/root/.cache/huggingface/snapshots/{revision}"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "magpie.model-revision-receipt/v1",
                "model": "Qwen/example",
                "requested_revision": revision,
                "resolved_revision": revision,
                "snapshot_path": snapshot,
                "verified": True,
            }
        ),
        encoding="utf-8",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["model_revision_receipt"] = {
        "schema": "magpie.model-revision-evidence/v1",
        "requested": True,
        "status": "verified",
        "verified": True,
        "evidence_present": True,
        "model": "Qwen/example",
        "requested_revision": revision,
        "resolved_revision": revision,
        "snapshot_path": snapshot,
        "receipt_artifact": {
            "path": receipt_path.name,
            "size_bytes": receipt_path.stat().st_size,
            "sha256": sha256_file(receipt_path),
        },
        "errors": [],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")


def _add_inferencex_runtime_receipt(
    report_path: Path, source_root: Path, commit: str, tree: str
) -> None:
    (report_path.parent / "inferencex_runtime").mkdir()
    receipt_path = report_path.parent / "inferencex_runtime_receipt.json"
    receipt = {
        "schema": "magpie.inferencex-runtime-receipt/v1",
        "source_root": str(source_root.resolve()),
        "source_is_git": True,
        "source_commit": commit,
        "source_tree": tree,
        "source_clean": True,
        "source_status_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
        "source_status_unchanged": True,
        "runtime_path": "inferencex_runtime",
        "materialization_method": "git_private_index_checkout",
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["inferencex_runtime_receipt"] = receipt
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_formal_model_revision_receipt_is_rehashed(tmp_path: Path) -> None:
    revision = "a" * 40
    report_path = _report(tmp_path, framework="xdit")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["quality_gate"] = {"passed": True, "ssim": 0.99}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _add_model_revision_receipt(report_path, revision)

    result = parse_benchmark_report(
        report_path,
        run_id="formal",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_model="Qwen/example",
        expected_model_revision=revision,
    )

    assert result.succeeded
    assert result.model_revision.passed
    assert result.model_revision.resolved_revision == revision
    assert result.model_revision.source_path in result.artifacts

    receipt = tmp_path / "model_revision_receipt.json"
    receipt.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected = parse_benchmark_report(
        report_path,
        run_id="tampered",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_model="Qwen/example",
        expected_model_revision=revision,
    )
    assert not rejected.succeeded
    assert "model_revision_artifact_digest_mismatch" in rejected.errors


def test_formal_inferencex_runtime_receipt_is_independently_checked(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "InferenceX"
    source_root.mkdir(parents=True)
    commit = "b" * 40
    tree = "d" * 40
    report_path = _report(tmp_path, framework="xdit")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["quality_gate"] = {"passed": True, "ssim": 0.99}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _add_inferencex_runtime_receipt(report_path, source_root, commit, tree)

    result = parse_benchmark_report(
        report_path,
        run_id="formal",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_inferencex_root=source_root,
        expected_inferencex_commit=commit,
        expected_inferencex_tree=tree,
    )

    assert result.succeeded
    assert result.inferencex_runtime.passed
    assert result.inferencex_runtime.source_commit == commit
    assert result.inferencex_runtime.source_tree == tree
    assert result.inferencex_runtime.receipt_path in result.artifacts

    receipt_path = tmp_path / "inferencex_runtime_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_clean"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    rejected = parse_benchmark_report(
        report_path,
        run_id="tampered",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_inferencex_root=source_root,
        expected_inferencex_commit=commit,
        expected_inferencex_tree=tree,
    )
    assert not rejected.succeeded
    assert "inferencex_runtime_report_receipt_mismatch" in rejected.errors


def test_formal_lm_eval_runtime_evidence_is_rehashed_and_persisted(
    tmp_path: Path,
) -> None:
    report_path = _report(
        tmp_path,
        framework="xdit",
        quality_gate={"passed": True, "ssim": 0.99},
    )
    expected = _add_lm_eval_runtime_evidence(report_path)

    result = parse_benchmark_report(
        report_path,
        run_id="formal-runtime",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_lm_eval_runtime=expected,
        expected_lm_eval_execution_mode="docker",
    )

    assert result.succeeded
    assert result.lm_eval_runtime.passed
    assert result.lm_eval_runtime.runtime_sha256 == expected.runtime_sha256
    assert result.lm_eval_runtime.manifest_path in result.artifacts
    assert result.lm_eval_runtime.receipt_path in result.artifacts

    receipt_path = tmp_path / "lm_eval_runtime_receipt.json"
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    tampered = parse_benchmark_report(
        report_path,
        run_id="tampered-runtime",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_lm_eval_runtime=expected,
        expected_lm_eval_execution_mode="docker",
    )
    assert not tampered.succeeded
    assert any("invalid_lm_eval_runtime_evidence" in item for item in tampered.errors)


def test_formal_inferencex_runtime_rejects_a_wrong_source_tree(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "InferenceX"
    source_root.mkdir(parents=True)
    commit = "b" * 40
    expected_tree = "d" * 40
    report_path = _report(tmp_path, framework="xdit")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["quality_gate"] = {"passed": True, "ssim": 0.99}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _add_inferencex_runtime_receipt(
        report_path, source_root, commit, "f" * 40
    )

    result = parse_benchmark_report(
        report_path,
        run_id="wrong-tree",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_inferencex_root=source_root,
        expected_inferencex_commit=commit,
        expected_inferencex_tree=expected_tree,
    )

    assert not result.succeeded
    assert "invalid_inferencex_runtime_receipt" in result.errors


def test_formal_inferencex_runtime_rejects_unpinned_runtime(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "InferenceX"
    source_root.mkdir(parents=True)
    commit = "c" * 40
    tree = "e" * 40
    report_path = _report(tmp_path, framework="xdit")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["quality_gate"] = {"passed": True, "ssim": 0.99}
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _add_inferencex_runtime_receipt(report_path, source_root, commit, tree)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["inferencex_runtime_receipt"]["materialization_method"] = (
        "filesystem_copy"
    )
    receipt = tmp_path / "inferencex_runtime_receipt.json"
    receipt.write_text(
        json.dumps(report["inferencex_runtime_receipt"]), encoding="utf-8"
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = parse_benchmark_report(
        report_path,
        run_id="unpinned",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=True,
        expected_inferencex_root=source_root,
        expected_inferencex_commit=commit,
        expected_inferencex_tree=tree,
    )
    assert not result.succeeded
    assert "invalid_inferencex_runtime_receipt" in result.errors
