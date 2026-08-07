from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apex.benchmark import build_config_views, validate_resolved_view
from apex.core import ConfigurationError, IntegrityError
from apex.delivery import verify_replay_config_invariants
from apex.ports import BenchmarkPass
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt


def _receipt(tmp_path: Path) -> DependencyReceipt:
    magpie = tmp_path / "Magpie"
    tracelens = tmp_path / "TraceLens"
    inferencex = tmp_path / "InferenceX"
    magpie.mkdir()
    tracelens.mkdir()
    inferencex.mkdir()
    lm_eval_runtime = tmp_path / "lm-eval-runtime"
    lm_eval_runtime.mkdir()
    runtime = LmEvalRuntimeReceipt(
        root=lm_eval_runtime,
        runtime_sha256="4" * 64,
        manifest_sha256="5" * 64,
        identity={
            "lm_eval_commit": "6" * 40,
            "lm_eval_tree": "7" * 40,
            "lm_eval_version": "0.4.9.2",
            "python_abi": "cpython-312",
            "python_soabi": "cpython-312-x86_64-linux-gnu",
            "base_image_id": "sha256:" + "8" * 64,
            "base_image_repo_digest": "example/image@sha256:" + "9" * 64,
            "inferencex_commit": "3" * 40,
            "inferencex_tree": "a" * 40,
        },
        file_count=1,
        lock_sha256="b" * 64,
    )
    return DependencyReceipt(
        schema="apex.dependency-receipt.v1",
        lock_sha256="a" * 64,
        python=Path("/usr/bin/python3"),
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
        lm_eval_runtime=runtime,
    )


def _source(tmp_path: Path, *, run_eval: object | None = None) -> Path:
    envs: dict[str, object] = {
        "TP": 1,
        "CONC": 16,
        "ISL": 1024,
        "OSL": 1024,
        "RANDOM_RANGE_RATIO": 1,
    }
    if run_eval is not None:
        envs["RUN_EVAL"] = run_eval
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "Qwen/example",
                    "precision": "fp8",
                    "run_mode": "docker",
                    "envs": envs,
                    "profiler": {
                        "torch_profiler": {"enabled": True},
                        "system_profiler": {"enabled": True},
                    },
                    "gap_analysis": {"enabled": True, "top_k": 100},
                    "docker_image": "base:image",
                    "benchmark_script": "vllm_mi355x.sh",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return source


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workload(document: dict) -> dict:
    benchmark = dict(document["benchmark"])
    benchmark.pop("profiler", None)
    benchmark.pop("gap_analysis", None)
    benchmark.pop("docker_image", None)
    benchmark.pop("run_kind", None)
    return benchmark


def test_builds_four_immutable_semantically_equal_views(tmp_path: Path) -> None:
    source = _source(tmp_path)
    original_bytes = source.read_bytes()
    receipt = _receipt(tmp_path)

    views = build_config_views(
        source,
        tmp_path / "views",
        dependency_receipt=receipt,
        replay_image="derived@sha256:" + "f" * 64,
    )

    assert views.original.read_bytes() == original_bytes
    measurement = _load(views.measurement)
    diagnostic = _load(views.diagnostic)
    replay = _load(views.replay)
    assert _workload(measurement) == _workload(diagnostic) == _workload(replay)
    assert measurement["benchmark"]["envs"]["RUN_EVAL"] == "true"
    assert measurement["benchmark"]["envs"]["MAGPIE_EVAL_TASKS"] == "gsm8k"
    assert measurement["benchmark"]["run_kind"] == "measurement"
    assert replay["benchmark"]["run_kind"] == "measurement"
    assert diagnostic["benchmark"]["run_kind"] == "diagnostic"
    assert not any(
        value.get("enabled")
        for value in measurement["benchmark"]["profiler"].values()
        if isinstance(value, dict)
    )
    assert measurement["benchmark"]["gap_analysis"]["enabled"] is False
    assert diagnostic["benchmark"]["profiler"]["torch_profiler"]["enabled"]
    assert diagnostic["benchmark"]["profiler"]["tracelens"]["enabled"]
    assert diagnostic["benchmark"]["profiler"]["tracelens"][
        "tracelens_repo_path"
    ] == str(receipt.root("tracelens").resolve())
    assert diagnostic["benchmark"]["gap_analysis"]["enabled"]
    assert diagnostic["benchmark"]["gap_analysis"]["find_kernel_sources"] is False
    assert diagnostic["benchmark"]["gap_analysis"]["kernel_source_repos"] == []
    assert diagnostic["benchmark"]["gap_analysis"]["auto_clone_repos"] is False
    assert measurement["benchmark"]["inferencex_path"] == str(
        receipt.root("inferencex").resolve()
    )
    assert measurement["benchmark"]["lm_eval_runtime"] == {
        "path": str(receipt.lm_eval_runtime.root),
        "sha256": receipt.lm_eval_runtime.runtime_sha256,
        "identity": dict(receipt.lm_eval_runtime.identity),
    }
    assert replay["benchmark"]["docker_image"].startswith("derived@sha256:")
    _, _, delivery_semantics = verify_replay_config_invariants(
        views.measurement,
        views.replay,
        expected_image_locator=replay["benchmark"]["docker_image"],
    )
    assert delivery_semantics == views.workload_semantics_sha256

    validate_resolved_view(
        views.measurement,
        pass_type=BenchmarkPass.MEASUREMENT,
        dependency_receipt=receipt,
    )
    validate_resolved_view(
        views.diagnostic,
        pass_type=BenchmarkPass.DIAGNOSTIC,
        dependency_receipt=receipt,
    )
    validate_resolved_view(
        views.replay,
        pass_type=BenchmarkPass.MEASUREMENT,
        dependency_receipt=receipt,
    )


def test_refuses_to_replace_an_immutable_view(tmp_path: Path) -> None:
    source = _source(tmp_path)
    receipt = _receipt(tmp_path)
    output = tmp_path / "views"
    build_config_views(
        source,
        output,
        dependency_receipt=receipt,
        replay_image="first:image",
    )

    with pytest.raises(IntegrityError, match="different content"):
        build_config_views(
            source,
            output,
            dependency_receipt=receipt,
            replay_image="second:image",
        )


def test_explicitly_disabled_quality_contract_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        build_config_views(
            _source(tmp_path, run_eval=False),
            tmp_path / "views",
            dependency_receipt=_receipt(tmp_path),
        )

    assert caught.value.reason_code == "quality_contract_disabled"


def test_diagnostics_use_only_explicit_source_locks_without_auto_clone(
    tmp_path: Path,
) -> None:
    vllm = tmp_path / "vllm"
    aiter = tmp_path / "aiter"
    vllm.mkdir()
    aiter.mkdir()

    views = build_config_views(
        _source(tmp_path),
        tmp_path / "views",
        dependency_receipt=_receipt(tmp_path),
        source_repository_roots=(vllm, aiter),
    )

    gap = _load(views.diagnostic)["benchmark"]["gap_analysis"]
    assert gap["find_kernel_sources"] is True
    assert gap["kernel_source_repos"] == [str(vllm.resolve()), str(aiter.resolve())]
    assert gap["auto_clone_repos"] is False


def test_validation_rejects_profiler_or_receipt_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    receipt = _receipt(tmp_path)
    views = build_config_views(
        source, tmp_path / "views", dependency_receipt=receipt
    )
    tampered = _load(views.measurement)
    tampered["benchmark"]["profiler"]["torch_profiler"]["enabled"] = True
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError) as profiler_error:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.MEASUREMENT,
            dependency_receipt=receipt,
        )
    assert profiler_error.value.reason_code == "measurement_profiler_enabled"

    tampered["benchmark"]["profiler"]["torch_profiler"]["enabled"] = False
    tampered["apex"]["benchmark_view"]["dependencies"]["tracelens"][
        "commit"
    ] = "3" * 40
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError) as receipt_error:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.MEASUREMENT,
            dependency_receipt=receipt,
        )
    assert receipt_error.value.reason_code == "benchmark_dependency_mismatch"


def test_actual_qwen_config_generates_protected_views(tmp_path: Path) -> None:
    source = Path(
        "/home/viouyang/Magpie/examples/benchmarks/"
        "benchmark_vllm_qwen3_next_80b_fp8.yaml"
    )
    if not source.exists():
        pytest.skip("Qwen benchmark fixture is not available")

    views = build_config_views(
        source,
        tmp_path / "views",
        dependency_receipt=_receipt(tmp_path),
    )

    measurement = _load(views.measurement)["benchmark"]
    diagnostic = _load(views.diagnostic)["benchmark"]
    assert measurement["model"] == "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    assert measurement["run_kind"] == "measurement"
    assert measurement["envs"]["RUN_EVAL"] == "true"
    assert measurement["gap_analysis"]["enabled"] is False
    assert diagnostic["profiler"]["tracelens"]["enabled"] is True
    assert diagnostic["run_kind"] == "diagnostic"


def test_validation_rejects_run_kind_tampering(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    views = build_config_views(
        _source(tmp_path), tmp_path / "views", dependency_receipt=receipt
    )
    tampered = _load(views.measurement)
    tampered["benchmark"]["run_kind"] = "diagnostic"
    path = tmp_path / "run-kind-tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError) as caught:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.MEASUREMENT,
            dependency_receipt=receipt,
        )
    assert caught.value.reason_code == "benchmark_run_kind_mismatch"


def test_model_revision_and_cache_are_frozen_into_all_views(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    revision = "4" * 40
    views = build_config_views(
        _source(tmp_path),
        tmp_path / "views",
        dependency_receipt=receipt,
        model_revision=revision,
        hf_cache_path=cache,
    )

    for path in (views.measurement, views.diagnostic, views.replay):
        benchmark = _load(path)["benchmark"]
        assert benchmark["envs"]["MODEL_REVISION"] == revision
        assert benchmark["hf_cache_path"] == str(cache.resolve())
        assert benchmark["inferencex_path"] == str(
            receipt.root("inferencex").resolve()
        )

    tampered = _load(views.measurement)
    tampered["benchmark"]["envs"]["MODEL_REVISION"] = "5" * 40
    path = tmp_path / "revision-tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.MEASUREMENT,
            dependency_receipt=receipt,
        )
    assert caught.value.reason_code == "benchmark_semantics_mismatch"


def test_hf_offline_and_runtime_identity_are_frozen_in_every_view(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    views = build_config_views(
        _source(tmp_path),
        tmp_path / "views",
        dependency_receipt=receipt,
        hf_cache_path=cache,
        hf_offline=True,
    )

    for path in (views.measurement, views.diagnostic, views.replay):
        benchmark = _load(path)["benchmark"]
        assert benchmark["envs"]["HF_HUB_OFFLINE"] == "1"
        assert benchmark["envs"]["TRANSFORMERS_OFFLINE"] == "1"
        assert benchmark["envs"]["HF_DATASETS_OFFLINE"] == "1"
        assert benchmark["lm_eval_runtime"]["sha256"] == "4" * 64

    tampered = _load(views.measurement)
    tampered["benchmark"]["lm_eval_runtime"]["sha256"] = "f" * 64
    path = tmp_path / "runtime-tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.MEASUREMENT,
            dependency_receipt=receipt,
        )
    assert caught.value.reason_code in {
        "benchmark_semantics_mismatch",
        "benchmark_lm_eval_runtime_mismatch",
    }


def test_hf_offline_requires_an_explicit_verified_cache(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as caught:
        build_config_views(
            _source(tmp_path),
            tmp_path / "views",
            dependency_receipt=_receipt(tmp_path),
            hf_offline=True,
        )
    assert caught.value.reason_code == "hf_offline_cache_missing"
