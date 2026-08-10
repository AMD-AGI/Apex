from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

import apex.benchmark.config_views as config_views_module
from apex.benchmark import (
    EvaluatorPolicy,
    build_config_views as _build_config_views,
    validate_phase_set_contract,
)
from apex.core import IntegrityError, sha256_json
from apex.runtime import DependencyReceipt
from tests.support.magpie_contract import resolved_contract


def build_config_views(source: Path, output: Path, **kwargs):
    receipt = kwargs["dependency_receipt"]
    return _build_config_views(
        source,
        output,
        resolved_contract=resolved_contract(source, receipt),
        **kwargs,
    )


def _profiler(*, diagnostic: bool) -> dict[str, Any]:
    return {
        "torch_profiler": {"enabled": diagnostic},
        "system_profiler": {"enabled": False},
        "gpu_monitor": {"enabled": diagnostic},
        "tracelens": {
            "enabled": diagnostic,
            "tracelens_repo_path": "/opt/apex/TraceLens",
        },
        "targeted_trace": {
            "enabled": diagnostic,
            "targets": ([{"target_id": "all"}] if diagnostic else []),
        },
    }


def _dependencies() -> dict[str, Any]:
    return {
        "receipt_schema": "apex.dependencies.receipt/v1",
        "lock_sha256": "a" * 64,
        "python": "/opt/apex/python",
        "magpie": {"root": "/opt/apex/Magpie", "commit": "1" * 40},
        "tracelens": {"root": "/opt/apex/TraceLens", "commit": "2" * 40},
        "inferencex": {"root": "/opt/apex/InferenceX", "commit": "3" * 40},
    }


def _magpie_config_resolution() -> dict[str, str]:
    return {
        "plan_schema": "apex.magpie-main-resolved-plan/v1",
        "plan_sha256": "6" * 64,
        "capability_schema": "apex.magpie-main-capability-receipt/v1",
        "capability_receipt_sha256": "7" * 64,
        "effective_config_sha256": "8" * 64,
        "scoring_config_sha256": "9" * 64,
        "phase_views_sha256": "a" * 64,
        "resolution_method_sha256": "b" * 64,
    }


def _phase_set(
    *, serving: bool = True, with_policy: bool = False
) -> tuple[list[dict[str, Any]], str]:
    envs: dict[str, Any] = {"TP": 1}
    framework = "vllm" if serving else "pytorch"
    quality_kind = "lm_eval" if serving else "framework_quality_gate"
    tasks = "gsm8k" if serving else ""
    if serving:
        envs.update({"RUN_EVAL": "true", "MAGPIE_EVAL_TASKS": tasks})
        typed = (
            EvaluatorPolicy(
                "strict-v2", tasks, "utils/evals/gsm8k.yaml", "c" * 64,
                "openai/gsm8k", "main", "d" * 40, "exact_match", 128, 32,
            )
            if with_policy
            else EvaluatorPolicy(
                "apex-lm-eval-gsm8k-v2", tasks,
                "utils/evals/gsm8k.yaml", "c" * 64,
                "openai/gsm8k", "main", "d" * 40,
                "exact_match,strict-match", 2248, 480,
            )
        )
        envs.update(typed.env())
        policy = typed.to_dict()
    else:
        policy = None
    benchmark: dict[str, Any] = {
        "framework": framework,
        "model": "Qwen/example",
        "docker_image": "example:v1",
        "run_kind": "measurement",
        "envs": envs,
        "profiler": _profiler(diagnostic=False),
        "gap_analysis": {"enabled": False},
    }
    if serving:
        benchmark["lm_eval_runtime"] = {
            "path": "/opt/apex/lm-eval",
            "sha256": "4" * 64,
            "identity": {"commit": "5" * 40},
        }
    projected = copy.deepcopy(benchmark)
    for key in ("profiler", "gap_analysis", "docker_image", "run_kind"):
        projected.pop(key)
    semantics = sha256_json(projected)

    documents: list[dict[str, Any]] = []
    for kind in ("measurement", "diagnostic", "replay"):
        selected = copy.deepcopy(benchmark)
        quality = {
            "required": True,
            "kind": quality_kind,
            "tasks": tasks,
            "evaluator_policy": copy.deepcopy(policy),
        }
        if kind == "diagnostic":
            selected["run_kind"] = "diagnostic"
            selected["profiler"] = _profiler(diagnostic=True)
            selected["gap_analysis"] = {"enabled": True}
            if serving:
                selected["envs"]["RUN_EVAL"] = "false"
                selected.pop("lm_eval_runtime")
                quality.update({"required": False, "kind": "trace_only"})
        elif kind == "replay":
            selected["docker_image"] = "derived@sha256:" + "f" * 64
        documents.append(
            {
                "benchmark": selected,
                "apex": {
                    "benchmark_view": {
                        "schema": "apex.benchmark-view.v2",
                        "kind": kind,
                        "original_sha256": "b" * 64,
                        "workload_semantics_sha256": semantics,
                        "dependencies": _dependencies(),
                        "magpie_config_resolution": _magpie_config_resolution(),
                        "quality_contract": quality,
                    }
                },
                "user_metadata": {"campaign": "same-phase-set"},
            }
        )
    return documents, semantics


def _set_nested(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _assert_set_drift(documents: list[dict[str, Any]], semantics: str) -> None:
    with pytest.raises(IntegrityError) as caught:
        validate_phase_set_contract(*documents, semantics)
    assert caught.value.reason_code == "benchmark_semantics_changed"


@pytest.mark.parametrize("with_policy", (False, True))
def test_accepts_self_consistent_trace_only_phase_set(with_policy: bool) -> None:
    documents, semantics = _phase_set(with_policy=with_policy)

    validate_phase_set_contract(*documents, semantics)


def test_accepts_self_consistent_framework_quality_phase_set() -> None:
    documents, semantics = _phase_set(serving=False)

    validate_phase_set_contract(*documents, semantics)


@pytest.mark.parametrize("run_mode", ("local", "ray"))
def test_accepts_phase_set_without_docker_image(run_mode: str) -> None:
    documents, semantics = _phase_set()
    for document in documents:
        document["benchmark"]["run_mode"] = run_mode
        document["benchmark"].pop("docker_image")
    projected = copy.deepcopy(documents[0]["benchmark"])
    for key in ("profiler", "gap_analysis", "run_kind"):
        projected.pop(key)
    semantics = sha256_json(projected)
    for document in documents:
        document["apex"]["benchmark_view"][
            "workload_semantics_sha256"
        ] = semantics

    validate_phase_set_contract(*documents, semantics)


@pytest.mark.parametrize(
    ("view", "path", "value"),
    (
        (0, ("apex", "benchmark_view", "schema"), "apex.benchmark-view.v3"),
        (1, ("apex", "benchmark_view", "kind"), "measurement"),
        (2, ("apex", "benchmark_view", "workload_semantics_sha256"), "0" * 64),
        (1, ("apex", "benchmark_view", "original_sha256"), "0" * 64),
        (
            1,
            ("apex", "benchmark_view", "dependencies", "magpie", "commit"),
            "0" * 40,
        ),
        (0, ("benchmark", "run_kind"), "diagnostic"),
        (1, ("benchmark", "run_kind"), "measurement"),
        (2, ("benchmark", "run_kind"), "diagnostic"),
        (1, ("benchmark", "docker_image"), "other:v2"),
        (0, ("benchmark", "profiler", "torch_profiler", "enabled"), True),
        (1, ("benchmark", "profiler", "targeted_trace", "enabled"), False),
        (2, ("benchmark", "gap_analysis", "enabled"), True),
        (1, ("benchmark", "envs", "TP"), 8),
        (1, ("benchmark", "lm_eval_runtime"), {"path": "/forbidden"}),
        (2, ("benchmark", "lm_eval_runtime", "sha256"), "0" * 64),
        (1, ("user_metadata", "campaign"), "different-phase-set"),
        (1, ("apex", "shared_receipt"), {"sha256": "0" * 64}),
    ),
)
def test_rejects_phase_metadata_role_and_workload_drift(
    view: int, path: tuple[str, ...], value: Any
) -> None:
    documents, semantics = _phase_set()
    _set_nested(documents[view], path, value)

    _assert_set_drift(documents, semantics)


def test_rejects_consistent_quality_tasks_that_disagree_with_environment() -> None:
    documents, semantics = _phase_set()
    for document in documents:
        document["apex"]["benchmark_view"]["quality_contract"]["tasks"] = "other"

    _assert_set_drift(documents, semantics)


def test_rejects_consistently_tampered_evaluator_policy_digest() -> None:
    documents, semantics = _phase_set(with_policy=True)
    for document in documents:
        policy = document["apex"]["benchmark_view"]["quality_contract"][
            "evaluator_policy"
        ]
        policy["sha256"] = "0" * 64

    _assert_set_drift(documents, semantics)


def test_rejects_trace_only_quality_shape_in_non_serving_phase_set() -> None:
    documents, semantics = _phase_set(serving=False)
    quality = documents[1]["apex"]["benchmark_view"]["quality_contract"]
    quality.update({"required": False, "kind": "trace_only"})

    _assert_set_drift(documents, semantics)


def test_consistent_external_identity_still_requires_receipt_validation() -> None:
    documents, semantics = _phase_set()
    for document in documents:
        metadata = document["apex"]["benchmark_view"]
        metadata["original_sha256"] = "0" * 64
        metadata["dependencies"]["magpie"]["commit"] = "9" * 40

    validate_phase_set_contract(*documents, semantics)


def _receipt(tmp_path: Path) -> DependencyReceipt:
    roots = {}
    for name in ("magpie", "tracelens", "inferencex"):
        root = tmp_path / name
        root.mkdir()
        roots[name] = root
    return DependencyReceipt(
        schema="apex.dependencies.receipt/v1",
        lock_sha256="a" * 64,
        python=Path("/usr/bin/python3"),
        roots=roots,
        commits={"magpie": "1" * 40, "tracelens": "2" * 40, "inferencex": "3" * 40},
        raw={},
    )


def test_build_config_views_validates_phase_set_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "pytorch",
                    "model": "example",
                    "docker_image": "example:v1",
                    "envs": {"TP": 1},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "views"

    def reject(*_args: object) -> None:
        raise IntegrityError("forced phase drift", "benchmark_semantics_changed")

    monkeypatch.setattr(
        config_views_module, "validate_phase_set_contract", reject
    )
    with pytest.raises(IntegrityError) as caught:
        build_config_views(source, output, dependency_receipt=_receipt(tmp_path))

    assert caught.value.reason_code == "benchmark_semantics_changed"
    assert not tuple(output.glob("*.yaml"))
