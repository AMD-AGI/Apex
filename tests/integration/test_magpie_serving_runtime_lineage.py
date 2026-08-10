"""Contract boundary between published Magpie main and Apex evidence policy."""

from __future__ import annotations

from Magpie.modes.benchmark.result import BenchmarkResult

from apex.benchmark import parse_serving_runtime_evidence


def test_published_magpie_report_without_runtime_receipt_fails_closed() -> None:
    """Published main does not mint the superseded branch-only receipt."""

    report = BenchmarkResult(
        success=True,
        framework="vllm",
        model="Qwen/example",
    ).to_dict()

    assert "serving_runtime_receipt" not in report
    evidence = parse_serving_runtime_evidence(
        report,
        expected_config_sha256="a" * 64,
        expected_requested_image="example/vllm:latest",
        expected_execution_mode="docker",
    )

    assert evidence.required
    assert not evidence.passed
    assert evidence.error == "serving_runtime_receipt_missing"
