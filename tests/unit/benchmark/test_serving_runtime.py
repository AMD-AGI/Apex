from __future__ import annotations

import copy

import pytest

from apex.benchmark import parse_serving_runtime_evidence


CONFIG = "a" * 64
IMAGE = "sha256:" + "b" * 64
DERIVED_IMAGE = "magpie-tracelens-vllm:test"
DERIVED_ID = "sha256:" + "d" * 64
TRACELENS_COMMIT = "1" * 40
TRACELENS_TREE = "2" * 40


def _lineage(
    *,
    kind: str = "direct",
    input_image: str = IMAGE,
    input_image_id: str = IMAGE,
    requested_image: str = IMAGE,
    resolved_image_id: str = IMAGE,
) -> dict[str, object]:
    derived = kind == "tracelens-derived"
    return {
        "kind": kind,
        "framework": "vllm",
        "runtime_schema": "magpie.tracelens-vllm-runtime/v1" if derived else None,
        "base_image": input_image,
        "base_image_id": input_image_id,
        "base_image_locator": input_image_id if derived else input_image,
        "derived_image": requested_image,
        "derived_image_id": resolved_image_id,
        "tracelens_source_commit": TRACELENS_COMMIT if derived else None,
        "tracelens_source_tree": TRACELENS_TREE if derived else None,
        "patch_version": "v19" if derived else None,
        "patch_path": (
            "examples/custom_workflows/inference_analysis/vllm_patches/"
            "config_vllm_v0.19.0.patch"
            if derived
            else None
        ),
        "patch_sha256": "e" * 64 if derived else None,
        "dependency_wheel_manifest_sha256": "f" * 64 if derived else None,
        "validator": (
            "vllm-tracelens-runtime-validation/v1" if derived else "docker-image-id"
        ),
        "verified": True,
    }


def _report(*, derived: bool = False) -> dict[str, object]:
    input_image = "example/image:latest" if derived else IMAGE
    input_image_id = "sha256:" + "c" * 64 if derived else IMAGE
    requested_image = DERIVED_IMAGE if derived else IMAGE
    resolved_image_id = DERIVED_ID if derived else IMAGE
    return {
        "framework": "vllm",
        "serving_runtime_receipt": {
            "schema": "apex.magpie-serving-runtime-observation/v3",
            "execution_mode": "docker",
            "input_config_sha256": CONFIG,
            "input_image": input_image,
            "input_image_id": input_image_id,
            "requested_image": requested_image,
            "resolved_image_id": resolved_image_id,
            "image_derivation": _lineage(
                kind="tracelens-derived" if derived else "direct",
                input_image=input_image,
                input_image_id=input_image_id,
                requested_image=requested_image,
                resolved_image_id=resolved_image_id,
            ),
            "container_name": "magpie-benchmark-attempt-1",
            "container_spec_sha256": "c" * 64,
            "process_succeeded": True,
            "verified": True,
            "errors": [],
        }
    }


def _parse(
    report: dict[str, object],
    *,
    allow_derived: bool = False,
    expected_image: str = IMAGE,
):
    return parse_serving_runtime_evidence(
        report,
        expected_config_sha256=CONFIG,
        expected_requested_image=expected_image,
        expected_execution_mode="docker",
        allow_tracelens_derivation=allow_derived,
        expected_tracelens_commit=TRACELENS_COMMIT,
        expected_tracelens_tree=TRACELENS_TREE,
    )


def test_accepts_exact_direct_config_image_and_process_binding() -> None:
    evidence = _parse(_report())

    assert evidence.passed
    assert evidence.input_image == IMAGE
    assert evidence.input_image_id == IMAGE
    assert evidence.requested_image == IMAGE
    assert evidence.resolved_image_id == IMAGE
    assert evidence.container_spec_sha256 == "c" * 64
    assert evidence.image_derivation == _lineage()


def test_accepts_pinned_tracelens_derivation_only_when_explicitly_allowed() -> None:
    evidence = _parse(
        _report(derived=True),
        allow_derived=True,
        expected_image="example/image:latest",
    )

    assert evidence.passed
    assert evidence.input_image == "example/image:latest"
    assert evidence.requested_image == DERIVED_IMAGE
    assert evidence.resolved_image_id == DERIVED_ID
    assert evidence.image_derivation["kind"] == "tracelens-derived"
    assert (
        _parse(
            _report(derived=True),
            expected_image="example/image:latest",
        ).error
        == "serving_runtime_image_lineage_mismatch"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("base_image", "other:image"),
        ("base_image_id", "sha256:" + "9" * 64),
        ("base_image_locator", "sha256:" + "7" * 64),
        ("derived_image", "other:derived"),
        ("derived_image_id", "sha256:" + "8" * 64),
        ("tracelens_source_commit", "3" * 40),
        ("tracelens_source_tree", "4" * 40),
        ("runtime_schema", "wrong"),
        ("patch_version", "v18"),
        ("patch_path", "config_vllm_v0.19.0.patch"),
        ("patch_sha256", "bad"),
        ("dependency_wheel_manifest_sha256", "bad"),
        ("validator", "docker-image-id"),
        ("verified", False),
    ),
)
def test_rejects_tampered_tracelens_lineage(field: str, value: object) -> None:
    report = _report(derived=True)
    receipt = report["serving_runtime_receipt"]
    assert isinstance(receipt, dict)
    lineage = receipt["image_derivation"]
    assert isinstance(lineage, dict)
    lineage[field] = value

    assert (
        _parse(
            report,
            allow_derived=True,
            expected_image="example/image:latest",
        ).error
        == "serving_runtime_image_lineage_mismatch"
    )


def test_rejects_missing_receipt_and_config_or_input_image_drift() -> None:
    assert _parse({}).error == "serving_runtime_receipt_missing"
    for key, value, reason in (
        ("input_config_sha256", "d" * 64, "serving_runtime_config_mismatch"),
        ("input_image", "other:image", "serving_runtime_image_mismatch"),
        ("input_image_id", "sha256:" + "f" * 64, "serving_runtime_image_id_mismatch"),
    ):
        report = copy.deepcopy(_report())
        receipt = report["serving_runtime_receipt"]
        assert isinstance(receipt, dict)
        receipt[key] = value
        assert _parse(report).error == reason


def test_rejects_unverified_malformed_or_legacy_receipt() -> None:
    for key, value in (
        ("process_succeeded", False),
        ("verified", False),
        ("container_spec_sha256", "not-a-digest"),
        ("errors", ["docker_image_inspect_failed"]),
    ):
        report = copy.deepcopy(_report())
        receipt = report["serving_runtime_receipt"]
        assert isinstance(receipt, dict)
        receipt[key] = value
        assert _parse(report).error == "serving_runtime_receipt_invalid"

    report = copy.deepcopy(_report())
    receipt = report["serving_runtime_receipt"]
    assert isinstance(receipt, dict)
    receipt["extra"] = True
    assert _parse(report).error == "serving_runtime_receipt_key_set_mismatch"
    del receipt["extra"]
    receipt["schema"] = "magpie.serving-runtime-receipt/v2"
    assert _parse(report).error == "serving_runtime_receipt_invalid"


def test_non_docker_parse_is_explicitly_not_required() -> None:
    evidence = parse_serving_runtime_evidence(
        {},
        expected_config_sha256=None,
        expected_requested_image=None,
        expected_execution_mode="local",
    )

    assert not evidence.required
    assert evidence.passed
