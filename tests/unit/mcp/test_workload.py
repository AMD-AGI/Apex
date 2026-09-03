from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apex.core import ContractError, sha256_file, sha256_json
from apex.mcp import (
    CapabilityRegistry,
    CapabilityScope,
    WorkloadInspectHandler,
    workload_inspect_descriptor,
)
from apex.ports import CapabilityAuthority, CapabilityRequest
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt
from tests.support.magpie_contract import ResolvedPlanStub


def _config(
    workspace: Path,
    *,
    framework: str = "vllm",
    run_mode: str = "local",
    image: str | None = None,
    lifecycle: bool = True,
) -> Path:
    benchmark: dict[str, object] = {
        "framework": framework,
        "model": "org/private-model",
        "precision": "fp8",
        "run_mode": run_mode,
        "envs": {"TP": 1, "CONC": 8, "ISL": 128, "OSL": 32},
        "profiler": {},
    }
    if image is not None:
        benchmark["docker_image"] = image
    if run_mode == "ray":
        benchmark["ray_config"] = {"address": "ray://cluster"}
    if lifecycle:
        benchmark["server_lifecycle"] = {"enabled": True, "cleanup": False}
    path = workspace / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump({"benchmark": benchmark}, sort_keys=False), encoding="utf-8"
    )
    return path


def _manifest(path: Path, config: Path) -> tuple[Path, str]:
    payload = {
        "schema": "apex.magpie-benchmark-corpus/v1",
        "repository": "https://example.invalid/Magpie.git",
        "commit": "1" * 40,
        "repository_tree": "2" * 40,
        "benchmark_tree": "3" * 40,
        "files": [
            {
                "path": "examples/benchmarks/benchmark.yaml",
                "sha256": sha256_file(config),
            }
        ],
        "summary": {"config_count": 1},
    }
    digest = sha256_json(payload)
    path.write_text(
        json.dumps({**payload, "manifest_sha256": digest}, sort_keys=True),
        encoding="utf-8",
    )
    return path, digest


def _receipt(tmp_path: Path, config: Path) -> DependencyReceipt:
    roots = {}
    for name in ("magpie", "tracelens", "inferencex"):
        root = tmp_path / name
        root.mkdir()
        roots[name] = root
    runtime_root = tmp_path / "lm-eval-runtime"
    runtime_root.mkdir()
    manifest, manifest_digest = _manifest(tmp_path / "corpus.json", config)
    runtime = LmEvalRuntimeReceipt(
        runtime_root,
        "4" * 64,
        "5" * 64,
        {"lm_eval_commit": "6" * 40},
        1,
        "7" * 64,
    )
    return DependencyReceipt(
        schema="apex.dependencies.receipt/v1",
        lock_sha256="a" * 64,
        python=Path("/usr/bin/python3"),
        roots=roots,
        commits={
            "magpie": "1" * 40,
            "tracelens": "2" * 40,
            "inferencex": "3" * 40,
        },
        raw={
            "magpie_corpus": {
                "path": str(manifest),
                "benchmark_tree": "3" * 40,
                "manifest_sha256": manifest_digest,
                "summary": {"config_count": 1},
            }
        },
        lm_eval_runtime=runtime,
    )


def test_workload_inspection_is_lazy_scoped_and_reward_ineligible(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = _config(workspace)
    receipt = _receipt(tmp_path, config)
    calls = 0

    def dependencies() -> DependencyReceipt:
        nonlocal calls
        calls += 1
        return receipt

    registry = CapabilityRegistry()
    descriptor = workload_inspect_descriptor()
    registry.register(
        descriptor,
        WorkloadInspectHandler(
            CapabilityScope(workspace, tmp_path / "results"),
            dependencies,
            ResolvedPlanStub,
        ),
    )

    assert calls == 0
    assert registry.inventory()[0].descriptor.gpu_requirement.value == "none"
    result = registry.invoke(
        CapabilityRequest(
            "workload.inspect",
            {"run_id": "inspect-1", "config": "benchmark.yaml"},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    assert calls == 1
    assert result.reward_eligible is False
    assert result.content["reward_eligible"] is False
    assert result.content["corpus"] == {
        "benchmark_tree": "3" * 40,
        "manifest_sha256": receipt.raw["magpie_corpus"]["manifest_sha256"],
        "config_count": 1,
        "member": True,
        "matched_path": "examples/benchmarks/benchmark.yaml",
    }
    workload = result.content["workload"]
    assert workload["framework"] == "vllm"
    assert workload["run_mode"] == "local"
    assert workload["model_identity_sha256"] == sha256_json(
        {"model": "org/private-model"}
    )
    assert workload["image_status"] == "not_applicable"
    assert workload["server_lifecycle"] == {"enabled": True, "cleanup": False}
    assert workload["compatibility_status"] == "config_compatible"
    assert result.content["magpie_config_resolution"]["status"] == "config_compatible"
    assert result.content["magpie_config_resolution"]["raw_config_sha256"] == sha256_file(
        config
    )
    assert workload["runtime_requirements"] == [
        "gpu_topology_receipt",
        "local_runtime_engagement_receipt",
        "server_lifecycle_receipt",
    ]
    assert len(result.content["artifacts"]) == 4
    assert len(result.artifact_receipts) == 5
    assert result.artifact_receipts[0]["scope"] == "workspace"
    assert all(item["scope"] == "results" for item in result.artifact_receipts[1:])


@pytest.mark.parametrize(
    ("framework", "run_mode", "image", "image_status"),
    (
        ("atom", "docker", None, "runtime_selection_required"),
        ("sglang", "docker", "example/image:fixed", "mutable_locator"),
        ("vllm", "ray", None, "not_applicable"),
    ),
)
def test_workload_inspection_composes_current_dimensions(
    tmp_path: Path,
    framework: str,
    run_mode: str,
    image: str | None,
    image_status: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = _config(
        workspace,
        framework=framework,
        run_mode=run_mode,
        image=image,
        lifecycle=False,
    )
    receipt = _receipt(tmp_path, config)
    handler = WorkloadInspectHandler(
        CapabilityScope(workspace, tmp_path / "results"),
        lambda: receipt,
        ResolvedPlanStub,
    )
    registry = CapabilityRegistry()
    registry.register(workload_inspect_descriptor(), handler)

    result = registry.invoke(
        CapabilityRequest(
            "workload.inspect",
            {"run_id": "inspect-1", "config": "benchmark.yaml"},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    workload = result.content["workload"]
    assert workload["compatibility_status"] == "config_compatible"
    assert workload["image_status"] == image_status
    assert workload["unavailable_dimensions"] == []


def test_workload_inspection_reports_unknown_orthogonal_dimension(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = _config(
        workspace,
        framework="future-serving",
        run_mode="docker",
        image="sha256:" + "f" * 64,
        lifecycle=False,
    )
    receipt = _receipt(tmp_path, config)
    registry = CapabilityRegistry()
    registry.register(
        workload_inspect_descriptor(),
        WorkloadInspectHandler(
            CapabilityScope(workspace, tmp_path / "results"),
            lambda: receipt,
            lambda value: ResolvedPlanStub(
                value,
                status="capability_upgrade_required",
                blockers=("framework:future-serving",),
            ),
        ),
    )

    result = registry.invoke(
        CapabilityRequest(
            "workload.inspect",
            {"run_id": "inspect-1", "config": "benchmark.yaml"},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    assert result.content["workload"]["compatibility_status"] == (
        "capability_upgrade_required"
    )
    assert result.content["workload"]["unavailable_dimensions"] == [
        "framework:future-serving"
    ]
    assert result.content["magpie_config_resolution"]["status"] == (
        "capability_upgrade_required"
    )
    assert result.content["view_status"] == "capability_upgrade_required"
    assert result.content["workload_semantics_sha256"] is None
    assert result.content["artifacts"] == []
    assert len(result.artifact_receipts) == 1


def test_workload_inspection_rejects_non_file_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt = _receipt(tmp_path, _config(workspace))
    handler = WorkloadInspectHandler(
        CapabilityScope(workspace, tmp_path / "results"),
        lambda: receipt,
        ResolvedPlanStub,
    )

    with pytest.raises(ContractError) as caught:
        handler.invoke(
            CapabilityRequest(
                "workload.inspect",
                {"run_id": "inspect-1", "config": "."},
            )
        )

    assert caught.value.reason_code == "unsafe_capability_path"
