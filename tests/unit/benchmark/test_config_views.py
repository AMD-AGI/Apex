from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apex.benchmark import build_config_views as _build_config_views
from apex.benchmark import validate_resolved_view
from apex.core import ConfigurationError, IntegrityError
from apex.delivery import verify_replay_config_invariants
from apex.optimization.e2e.overlay_config import derive_overlay_configs
from apex.ports import BenchmarkPass
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt
from tests.support.magpie_contract import resolved_contract


def build_config_views(source: Path, output: Path, **kwargs):
    receipt = kwargs["dependency_receipt"]
    return _build_config_views(
        source,
        output,
        resolved_contract=resolved_contract(source, receipt),
        **kwargs,
    )


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


def test_builds_trace_only_diagnostic_and_formal_measurement_views(
    tmp_path: Path,
) -> None:
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
    measurement_workload = _workload(measurement)
    diagnostic_workload = _workload(diagnostic)
    assert measurement_workload == _workload(replay)
    assert diagnostic_workload["envs"]["RUN_EVAL"] == "false"
    assert "lm_eval_runtime" not in diagnostic_workload
    diagnostic_workload["envs"]["RUN_EVAL"] = "true"
    diagnostic_workload["lm_eval_runtime"] = measurement_workload["lm_eval_runtime"]
    assert diagnostic_workload == measurement_workload
    assert measurement["benchmark"]["envs"]["RUN_EVAL"] == "true"
    assert measurement["benchmark"]["envs"]["MAGPIE_EVAL_TASKS"] == "gsm8k"
    assert measurement["benchmark"]["run_kind"] == "measurement"
    assert replay["benchmark"]["run_kind"] == "measurement"
    assert diagnostic["benchmark"]["run_kind"] == "diagnostic"
    policy = measurement["apex"]["benchmark_view"]["quality_contract"][
        "evaluator_policy"
    ]
    assert policy["policy_id"] == "apex-lm-eval-gsm8k-v2"
    assert policy["task_definition_sha256"] == (
        "c0e109ed6dc356e082aea80cd775c12d64dada787b88c602408a3b960e0b04a1"
    )
    assert policy["dataset_revision"] == "740312add88f781978c0658806c59bc2815b9866"
    assert policy["primary_metric"] == "exact_match,strict-match"
    assert policy["max_length"] == 2248
    assert policy["max_gen_tokens"] == 480
    assert diagnostic["apex"]["benchmark_view"]["quality_contract"] == {
        "required": False,
        "kind": "trace_only",
        "tasks": "gsm8k",
        "evaluator_policy": policy,
    }
    for document in (measurement, replay):
        assert document["apex"]["benchmark_view"]["quality_contract"] == {
            "required": True,
            "kind": "lm_eval",
            "tasks": "gsm8k",
            "evaluator_policy": policy,
        }
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
    assert "lm_eval_runtime" not in diagnostic["benchmark"]
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


def test_generated_trace_only_views_support_immutable_image_overlay(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    views = build_config_views(
        _source(tmp_path),
        tmp_path / "views",
        dependency_receipt=receipt,
        replay_image="derived@sha256:" + "f" * 64,
    )
    source_paths = (views.measurement, views.diagnostic, views.replay)
    originals = tuple(_load(path) for path in source_paths)

    derived = derive_overlay_configs(
        measurement=views.measurement,
        diagnostic=views.diagnostic,
        replay=views.replay,
        output_dir=tmp_path / "overlay",
        image_id="sha256:" + "e" * 64,
        workload_semantics_sha256=views.workload_semantics_sha256,
    )

    for original, path in zip(
        originals,
        (derived.measurement, derived.diagnostic, derived.replay),
        strict=True,
    ):
        observed = _load(path)
        assert observed["benchmark"]["docker_image"] == "sha256:" + "e" * 64
        observed["benchmark"]["docker_image"] = original["benchmark"]["docker_image"]
        assert observed == original


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


@pytest.mark.parametrize("run_mode", ("local", "ray"))
def test_non_docker_views_do_not_require_a_docker_image(
    tmp_path: Path, run_mode: str
) -> None:
    source = _source(tmp_path)
    document = _load(source)
    document["benchmark"]["run_mode"] = run_mode
    document["benchmark"].pop("docker_image")
    if run_mode == "ray":
        document["benchmark"]["ray_config"] = {"address": "ray://cluster"}
    source.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )

    views = build_config_views(
        source, tmp_path / "views", dependency_receipt=_receipt(tmp_path)
    )

    for path in (views.measurement, views.diagnostic, views.replay):
        benchmark = _load(path)["benchmark"]
        assert benchmark["run_mode"] == run_mode
        assert "docker_image" not in benchmark


def test_replay_image_is_rejected_for_non_docker_workload(tmp_path: Path) -> None:
    source = _source(tmp_path)
    document = _load(source)
    document["benchmark"].update({"run_mode": "local", "docker_image": ""})
    source.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError) as caught:
        build_config_views(
            source,
            tmp_path / "views",
            dependency_receipt=_receipt(tmp_path),
            replay_image="derived@sha256:" + "f" * 64,
        )

    assert caught.value.reason_code == "replay_image_not_applicable"


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


@pytest.mark.parametrize("run_eval", [True, "true", 1])
def test_validation_rejects_quality_execution_in_diagnostic_view(
    tmp_path: Path, run_eval: object
) -> None:
    receipt = _receipt(tmp_path)
    views = build_config_views(
        _source(tmp_path), tmp_path / "views", dependency_receipt=receipt
    )
    tampered = _load(views.diagnostic)
    tampered["benchmark"]["envs"]["RUN_EVAL"] = run_eval
    path = tmp_path / "diagnostic-quality-tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError) as caught:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.DIAGNOSTIC,
            dependency_receipt=receipt,
        )
    assert caught.value.reason_code == "quality_contract_missing"


def test_validation_rejects_weakened_measurement_quality_metadata(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    views = build_config_views(
        _source(tmp_path), tmp_path / "views", dependency_receipt=receipt
    )
    tampered = _load(views.measurement)
    quality = tampered["apex"]["benchmark_view"]["quality_contract"]
    quality.update({"required": False, "kind": "trace_only"})
    path = tmp_path / "measurement-quality-tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError) as caught:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.MEASUREMENT,
            dependency_receipt=receipt,
        )
    assert caught.value.reason_code == "quality_contract_missing"


def test_validation_rejects_lm_eval_runtime_in_trace_only_diagnostic(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    views = build_config_views(
        _source(tmp_path), tmp_path / "views", dependency_receipt=receipt
    )
    tampered = _load(views.diagnostic)
    tampered["benchmark"]["lm_eval_runtime"] = _load(views.measurement)[
        "benchmark"
    ]["lm_eval_runtime"]
    path = tmp_path / "diagnostic-runtime-tampered.yaml"
    path.write_text(yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigurationError) as caught:
        validate_resolved_view(
            path,
            pass_type=BenchmarkPass.DIAGNOSTIC,
            dependency_receipt=receipt,
        )
    assert caught.value.reason_code == "benchmark_lm_eval_runtime_mismatch"


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
    assert views.original.read_bytes() == source.read_bytes()
    assert views.original_sha256 == (
        "f97bda8e04655fbd1410bafb34072ec072de416ea7e24551d2618281e75deafb"
    )
    measurement = _load(views.measurement)
    envs = measurement["benchmark"]["envs"]
    assert envs["MAX_MODEL_LEN"] == 2248
    assert envs["MAGPIE_EVAL_MAX_LENGTH"] == "2248"
    assert envs["MAGPIE_EVAL_MAX_GEN_TOKENS"] == "480"
    assert envs["MAGPIE_EVAL_PRIMARY_METRIC"] == "exact_match,strict-match"
    assert views.evaluator_policy_sha256 == measurement["apex"]["benchmark_view"][
        "quality_contract"
    ]["evaluator_policy"]["sha256"]

    measurement = _load(views.measurement)["benchmark"]
    diagnostic = _load(views.diagnostic)["benchmark"]
    assert measurement["model"] == "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    assert measurement["run_kind"] == "measurement"
    assert measurement["envs"]["RUN_EVAL"] == "true"
    assert measurement["gap_analysis"]["enabled"] is False
    assert diagnostic["profiler"]["tracelens"]["enabled"] is True
    assert diagnostic["run_kind"] == "diagnostic"
    assert diagnostic["envs"]["RUN_EVAL"] == "false"


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


def test_hf_offline_is_global_but_runtime_is_absent_from_diagnostic(
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
    for path in (views.measurement, views.replay):
        benchmark = _load(path)["benchmark"]
        assert benchmark["lm_eval_runtime"]["sha256"] == "4" * 64
    assert "lm_eval_runtime" not in _load(views.diagnostic)["benchmark"]

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
