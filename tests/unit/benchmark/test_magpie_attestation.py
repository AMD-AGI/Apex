from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.benchmark import parse_benchmark_report
from apex.benchmark.magpie_attestation import load_magpie_execution_attestation
from apex.core import IntegrityError, sha256_file
from apex.ports import BenchmarkPass


def _official_report(root: Path) -> Path:
    workspace = root / "magpie-workspace"
    workspace.mkdir()
    report = workspace / "benchmark_report.json"
    report.write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "Qwen/example",
                "workspace_dir": str(workspace.resolve()),
                "profiling_enabled": False,
                "throughput": {"total_token_throughput": 10.0},
                "latency": {
                    "ttft": {"p99_ms": 3.0},
                    "tpot": {"p99_ms": 1.0},
                },
                "errors": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report


def _attestation(root: Path, report: Path, **changes: object) -> Path:
    evaluator = root / "evaluator"
    evaluator.mkdir(exist_ok=True)
    value = {
        "schema": "apex.magpie-execution-attestation/v1",
        "authority": "apex_evaluator",
        "official_report_path": report.relative_to(root).as_posix(),
        "official_report_size_bytes": report.stat().st_size,
        "report_sha256": sha256_file(report),
        "config_sha256": "1" * 64,
        "run_id": "baseline",
        "pass_type": "measurement",
        "lane_verified": True,
        "reward_eligible": True,
        "profiling_enabled": False,
        "process": {
            "schema": "apex.magpie-process-attestation/v1",
            "argv_sha256": "2" * 64,
            "exit_code": 0,
            "timed_out": False,
            "succeeded": True,
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
            "model_revision_receipt": None,
            "inferencex_runtime_receipt": None,
            "lm_eval_runtime_receipt": None,
            "serving_runtime_receipt": {"verified": True},
        },
        "gpu_engagement": {
            "schema": "apex.magpie-gpu-engagement/v1",
            "verified": True,
            "devices": [
                {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}
            ],
            "processes": [
                {
                    "pid": 123,
                    "uid": 1000,
                    "start_time_ticks": 456,
                    "cmdline_sha256": "6" * 64,
                    "rsmi_device_indices": [0],
                }
            ],
        },
        "quality_gate": {
            "schema": "apex.magpie-quality-attestation/v1",
            "verified": True,
            "receipt": {"status": "passed", "passed": True},
        },
        "errors": [],
    }
    value.update(changes)
    path = evaluator / "execution_attestation.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _load(path: Path, report: Path):
    return load_magpie_execution_attestation(
        path,
        report_path=report,
        report=json.loads(report.read_text(encoding="utf-8")),
        expected_config_sha256="1" * 64,
        expected_run_id="baseline",
        expected_pass_type=BenchmarkPass.MEASUREMENT,
        command_exit_code=0,
        timed_out=False,
    )


def test_loads_evaluator_attestation_from_sibling_root(tmp_path: Path) -> None:
    report = _official_report(tmp_path)
    result = _load(_attestation(tmp_path, report), report)

    assert result.run_kind == "measurement"
    assert result.reward_eligible is True
    assert result.source_path.parent == tmp_path / "evaluator"
    assert result.evaluator_evidence({})["serving_runtime_receipt"]["verified"] is True


def test_parser_uses_lane_only_from_execution_attestation(tmp_path: Path) -> None:
    report = _official_report(tmp_path)
    path = _attestation(tmp_path, report)

    result = parse_benchmark_report(
        report,
        run_id="baseline",
        pass_type=BenchmarkPass.MEASUREMENT,
        quality_required=False,
        expected_config_sha256="1" * 64,
        execution_attestation_path=path,
    )

    assert result.succeeded
    assert result.run_kind == "measurement"
    assert result.reward_eligible is True
    assert path.resolve() in result.artifacts


def test_rejects_private_fields_in_official_report(tmp_path: Path) -> None:
    report = _official_report(tmp_path)
    value = json.loads(report.read_text(encoding="utf-8"))
    value["reward_eligible"] = True
    report.write_text(json.dumps(value), encoding="utf-8")
    path = _attestation(tmp_path, report)

    with pytest.raises(IntegrityError) as caught:
        _load(path, report)

    assert caught.value.reason_code == "private_magpie_report_fields_present"


def test_rejects_report_changed_after_attestation(tmp_path: Path) -> None:
    report = _official_report(tmp_path)
    path = _attestation(tmp_path, report)
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as caught:
        _load(path, report)

    assert caught.value.reason_code == "magpie_execution_attestation_report_mismatch"


def test_rejects_attestation_inside_magpie_workspace(tmp_path: Path) -> None:
    report = _official_report(tmp_path)
    safe = _attestation(tmp_path, report)
    unsafe = report.parent / "execution_attestation.json"
    unsafe.write_bytes(safe.read_bytes())

    with pytest.raises(IntegrityError) as caught:
        _load(unsafe, report)

    assert caught.value.reason_code == "unsafe_magpie_execution_attestation_location"


def test_rejects_verified_dependency_attestation_without_exact_receipts(
    tmp_path: Path,
) -> None:
    report = _official_report(tmp_path)
    path = _attestation(tmp_path, report)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["dependencies"]["receipts"] = {}
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(IntegrityError) as caught:
        _load(path, report)

    assert caught.value.reason_code == "magpie_execution_attestation_mismatch"


def test_rejects_verified_gpu_attestation_without_process_coverage(
    tmp_path: Path,
) -> None:
    report = _official_report(tmp_path)
    path = _attestation(tmp_path, report)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["gpu_engagement"]["processes"] = []
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(IntegrityError) as caught:
        _load(path, report)

    assert caught.value.reason_code == "magpie_execution_attestation_mismatch"
