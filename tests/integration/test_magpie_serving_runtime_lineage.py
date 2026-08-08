"""Cross-repository contract for Magpie's serving receipt and Apex's parser."""

from __future__ import annotations

from Magpie.modes.benchmark.serving_runtime import (
    finalize_serving_runtime_receipt,
    pending_serving_runtime_receipt,
)

from apex.benchmark import parse_serving_runtime_evidence


def test_pinned_magpie_tracelens_lineage_is_accepted_by_apex() -> None:
    config_sha256 = "a" * 64
    input_image = "example/vllm:v0.19.1"
    input_image_id = "sha256:" + "b" * 64
    derived_image = "magpie-tracelens-vllm:contract"
    derived_image_id = "sha256:" + "c" * 64
    tracelens_commit = "d" * 40
    tracelens_tree = "e" * 40
    container_name = "magpie-benchmark-contract"
    runtime = {
        "enabled": True,
        "framework": "vllm",
        "runtime_schema": "magpie.tracelens-vllm-runtime/v1",
        "base_image": input_image,
        "base_image_id": input_image_id,
        "base_image_locator": input_image_id,
        "image": derived_image,
        "public_runtime_image": derived_image,
        "public_runtime_image_id": derived_image_id,
        "tracelens_source_commit": tracelens_commit,
        "tracelens_source_tree": tracelens_tree,
        "patch_version": "v19",
        "tracelens_patch_path": (
            "examples/custom_workflows/inference_analysis/vllm_patches/"
            "config_vllm_v0.19.0.patch"
        ),
        "tracelens_patch_sha256": "f" * 64,
        "dependency_wheel_manifest_sha256": "1" * 64,
        "public_runtime_validation": {
            "valid": True,
            "image_id": derived_image_id,
        },
    }
    docker_argv = (
        "docker",
        "run",
        "--name",
        container_name,
        "--entrypoint",
        "bash",
        derived_image_id,
    )
    pending = pending_serving_runtime_receipt(
        execution_mode="docker",
        input_config_sha256=config_sha256,
        framework="vllm",
        input_image=input_image,
        input_image_id=input_image_id,
        requested_image=derived_image,
        resolved_image_id=derived_image_id,
        container_name=container_name,
        docker_argv=docker_argv,
        tracelens_runtime=runtime,
    )
    receipt = finalize_serving_runtime_receipt(
        pending,
        process_succeeded=True,
    )

    evidence = parse_serving_runtime_evidence(
        {"framework": "vllm", "serving_runtime_receipt": receipt},
        expected_config_sha256=config_sha256,
        expected_requested_image=input_image,
        expected_execution_mode="docker",
        allow_tracelens_derivation=True,
        expected_tracelens_commit=tracelens_commit,
        expected_tracelens_tree=tracelens_tree,
    )

    assert receipt["verified"] is True
    assert evidence.passed
    assert evidence.input_image == input_image
    assert evidence.input_image_id == input_image_id
    assert evidence.requested_image == derived_image
    assert evidence.resolved_image_id == derived_image_id

