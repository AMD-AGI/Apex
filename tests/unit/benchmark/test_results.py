from __future__ import annotations

import json
import hashlib
from pathlib import Path

from apex.benchmark import parse_benchmark_report
from apex.benchmark.results import empty_result
from apex.core import sha256_file
from apex.ports import BenchmarkPass
from apex.runtime import LmEvalRuntimeReceipt


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


def test_normalizes_percentiles_and_lm_eval_quality(tmp_path: Path) -> None:
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
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
    assert "benchmark_report_run_kind_mismatch" in result.errors
    assert "benchmark_report_reward_eligibility_mismatch" in result.errors


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
