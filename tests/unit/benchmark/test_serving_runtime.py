from __future__ import annotations

from apex.benchmark import parse_serving_runtime_evidence


CONFIG = "a" * 64
IMAGE = "sha256:" + "b" * 64


def _report() -> dict[str, object]:
    return {
        "serving_runtime_receipt": {
            "schema": "magpie.serving-runtime-receipt/v1",
            "execution_mode": "docker",
            "input_config_sha256": CONFIG,
            "requested_image": IMAGE,
            "resolved_image_id": IMAGE,
            "container_name": "magpie-benchmark-attempt-1",
            "docker_argv_sha256": "c" * 64,
            "process_succeeded": True,
            "verified": True,
            "errors": [],
        }
    }


def _parse(report: dict[str, object]):
    return parse_serving_runtime_evidence(
        report,
        expected_config_sha256=CONFIG,
        expected_requested_image=IMAGE,
        expected_execution_mode="docker",
    )


def test_accepts_exact_config_image_and_process_binding() -> None:
    evidence = _parse(_report())

    assert evidence.passed
    assert evidence.resolved_image_id == IMAGE
    assert evidence.process_succeeded is True


def test_rejects_missing_receipt_and_config_or_image_drift() -> None:
    assert _parse({}).error == "serving_runtime_receipt_missing"
    for key, value, reason in (
        ("input_config_sha256", "d" * 64, "serving_runtime_config_mismatch"),
        ("requested_image", "sha256:" + "e" * 64, "serving_runtime_image_mismatch"),
        ("resolved_image_id", "sha256:" + "f" * 64, "serving_runtime_image_id_mismatch"),
    ):
        report = _report()
        report["serving_runtime_receipt"][key] = value
        assert _parse(report).error == reason


def test_rejects_unverified_or_malformed_receipt() -> None:
    for key, value in (
        ("process_succeeded", False),
        ("verified", False),
        ("docker_argv_sha256", "not-a-digest"),
        ("errors", ["docker_image_inspect_failed"]),
    ):
        report = _report()
        report["serving_runtime_receipt"][key] = value
        assert _parse(report).error == "serving_runtime_receipt_invalid"

    report = _report()
    report["serving_runtime_receipt"]["extra"] = True
    assert _parse(report).error == "serving_runtime_receipt_key_set_mismatch"


def test_non_docker_parse_is_explicitly_not_required() -> None:
    evidence = parse_serving_runtime_evidence(
        {},
        expected_config_sha256=None,
        expected_requested_image=None,
        expected_execution_mode="local",
    )

    assert not evidence.required
    assert evidence.passed
