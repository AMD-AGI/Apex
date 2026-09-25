from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import yaml

from apex.benchmark import (
    CAPABILITY_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    MagpieMainConfigAdapter,
    validate_apex_magpie_config_documents,
)
from apex.core import ConfigurationError, sha256_file, sha256_json
from apex.runtime import (
    DependencyReceipt,
    EvaluatorDatasetLockFile,
    EvaluatorPolicyLock,
    MagpieMainPublicApi,
)


def _config(
    tmp_path: Path,
    *,
    run_mode: str = "docker",
    extra: str = "",
    secret: str | None = None,
) -> Path:
    env = ""
    if secret is not None:
        env = f"  envs:\n    TP: 1\n    HF_TOKEN: {secret}\n"
    value = (
        "benchmark:\n"
        "  framework: vllm\n"
        "  model: example/model\n"
        f"  run_mode: {run_mode}\n"
        f"{env}{extra}"
    )
    path = tmp_path / "benchmark.yaml"
    path.write_text(value, encoding="utf-8")
    return path


class _Model:
    def __init__(self, value: dict) -> None:
        self.value = copy.deepcopy(value)

    def to_dict(self) -> dict:
        result = {
            "framework": self.value.get("framework", "sglang").lower(),
            "model": self.value.get("model", ""),
            "precision": self.value.get("precision", "fp8"),
            "run_mode": self.value.get("run_mode", "docker").lower(),
            "envs": copy.deepcopy(self.value.get("envs") or {
                "TP": 1, "CONC": 32, "ISL": 1024, "OSL": 512,
                "RANDOM_RANGE_RATIO": 0.5,
            }),
            "profiler": copy.deepcopy(self.value.get("profiler") or {
                "torch_profiler": {"enabled": True},
                "system_profiler": {"enabled": False, "profile_args": []},
                "tracelens": {"enabled": False},
                "gpu_monitor": {"enabled": True},
            }),
            "gap_analysis": copy.deepcopy(
                self.value.get("gap_analysis") or {"enabled": False}
            ),
            "docker_image": self.value.get("docker_image"),
            "gpu_arch": self.value.get("gpu_arch"),
            "timeout_seconds": self.value.get("timeout_seconds", 3600.0),
            "inferencex_path": self.value.get("inferencex_path", ""),
            "hf_cache_path": self.value.get("hf_cache_path"),
            "runner_type": self.value.get("runner_type"),
            "benchmark_script": self.value.get("benchmark_script"),
            "gpu_selection": copy.deepcopy(
                self.value.get("gpu_selection") or {"enabled": True}
            ),
        }
        for key in ("ray_config", "server_lifecycle"):
            if key in self.value:
                result[key] = copy.deepcopy(self.value[key])
        return result


class _Loader:
    def __init__(self, *, mutate: bool = False) -> None:
        self.calls: list[Path] = []
        self.mutate = mutate

    def __call__(self, path: Path) -> object:
        self.calls.append(path)
        value = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
        if self.mutate:
            path.write_text("benchmark: {framework: sglang, model: changed}\n")
        return value


def _receipt(tmp_path: Path) -> DependencyReceipt:
    root = tmp_path / "Magpie"
    root.mkdir(exist_ok=True)
    evaluator = EvaluatorPolicyLock(
        tmp_path / "evaluator_policy.lock.json",
        "c" * 64,
        "apex-lm-eval-gsm8k-v2",
        "exact_match,strict-match",
        True,
        "gsm8k",
        "utils/evals/gsm8k.yaml",
        "d" * 64,
        "https://huggingface.co/datasets/openai/gsm8k",
        "openai/gsm8k",
        "main",
        "e" * 40,
        ("test", "train"),
        (
            EvaluatorDatasetLockFile("test", "main/test.parquet", 4, "f" * 64),
            EvaluatorDatasetLockFile("train", "main/train.parquet", 5, "1" * 64),
        ),
    )
    return DependencyReceipt(
        schema="apex.dependency-receipt.v1",
        lock_sha256="a" * 64,
        python=Path("/verified/venv/bin/python"),
        roots={"magpie": root},
        commits={"magpie": "b" * 40},
        raw={},
        evaluator_policy=evaluator,
    )


def _adapter(
    tmp_path: Path, *, loader: _Loader | None = None
) -> tuple[MagpieMainConfigAdapter, _Loader]:
    selected = loader or _Loader()
    receipt = _receipt(tmp_path)
    api = MagpieMainPublicApi(
        receipt.root("magpie"), loader=selected, model_factory=_Model
    )
    return MagpieMainConfigAdapter(receipt, public_api=api), selected


def test_adapter_uses_published_main_model_and_apex_owned_policy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter, loader = _adapter(tmp_path)

    resolved = adapter.resolve(config)

    assert loader.calls == [config.resolve()]
    assert resolved.status == "config_compatible"
    assert resolved.config_sha256 == sha256_file(config)
    assert resolved.magpie_commit == "b" * 40
    assert resolved.plan["schema"] == PLAN_SCHEMA
    assert resolved.capability_receipt["schema"] == CAPABILITY_SCHEMA
    assert resolved.plan["expected_result"]["schema"] == RESULT_SCHEMA
    assert resolved.capability_receipt["reward_contract"]["owner"] == "apex"
    assert resolved.capability_receipt["capabilities"]["benchmark_execution"] == (
        "published_magpie_main"
    )
    envs = resolved.scoring_config["envs"]
    assert envs["RUN_EVAL"] == "true"
    assert envs["MAGPIE_EVAL_POLICY_ID"] == "apex-lm-eval-gsm8k-v2"
    assert envs["MAGPIE_EVAL_TASK_DEFINITION_SHA256"] == "d" * 64
    assert envs["MAGPIE_EVAL_DATASET_REVISION"] == "e" * 40
    assert resolved.resolution_method_sha256 == sha256_json(
        {
            "method": "apex_magpie_main_public_config_projection_v1",
            "magpie_commit": "b" * 40,
            "public_apis": [
                "Magpie.main.load_benchmark_config",
                "Magpie.modes.benchmark.config.BenchmarkConfig.from_dict",
                "Magpie.modes.benchmark.config.BenchmarkConfig.to_dict",
            ],
        }
    )


def test_unknown_semantic_field_requires_capability_upgrade(tmp_path: Path) -> None:
    config = _config(tmp_path, extra="  future_semantics: enabled\n")
    adapter, _ = _adapter(tmp_path)

    resolved = adapter.resolve(config)

    assert resolved.status == "capability_upgrade_required"
    assert resolved.capability_receipt["blockers"] == [
        "unrecognized_benchmark_field:future_semantics"
    ]
    assert resolved.plan["extensions"]["unrecognized_benchmark_fields"] == {
        "future_semantics": "enabled"
    }


def test_unknown_nested_semantic_field_requires_upgrade(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        extra="  profiler:\n    torch_profiler:\n      enabled: false\n      future_mode: strict\n",
    )
    adapter, _ = _adapter(tmp_path)

    resolved = adapter.resolve(config)

    assert resolved.status == "capability_upgrade_required"
    assert resolved.capability_receipt["blockers"] == [
        "unrecognized_nested_field:profiler.torch_profiler.future_mode"
    ]
    assert resolved.plan["extensions"]["unrecognized_nested_fields"] == {
        "profiler.torch_profiler.future_mode": "strict"
    }


def test_ray_contract_requires_shared_quality_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path, run_mode="ray")
    adapter, _ = _adapter(tmp_path)

    resolved = adapter.resolve(config)

    assert resolved.capability_receipt["capabilities"]["quality_evaluation"] == (
        "shared_runtime_required"
    )
    validate_apex_magpie_config_documents(
        config, resolved.plan, resolved.capability_receipt
    )


def test_secret_values_are_redacted_but_reconstructable(tmp_path: Path) -> None:
    secret = "never-copy-me"
    config = _config(tmp_path, secret=secret)
    adapter, _ = _adapter(tmp_path)

    resolved = adapter.resolve(config)

    assert secret not in str(resolved.to_dict())
    assert resolved.scoring_config["envs"]["HF_TOKEN"] == "<redacted>"
    assert resolved.plan["redactions"]["paths"] == [
        "phase_views.requested.envs.HF_TOKEN",
        "phase_views.scoring_measurement.envs.HF_TOKEN",
    ]


def test_validator_rejects_rehashed_cross_document_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter, _ = _adapter(tmp_path)
    resolved = adapter.resolve(config)
    capability = copy.deepcopy(resolved.capability_receipt)
    capability["framework"] = "sglang"
    capability["receipt_sha256"] = sha256_json(
        {key: value for key, value in capability.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ConfigurationError, match="conflicts"):
        validate_apex_magpie_config_documents(config, resolved.plan, capability)


def test_adapter_rejects_duplicate_linked_and_hardlinked_input(tmp_path: Path) -> None:
    adapter, loader = _adapter(tmp_path)
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "benchmark:\n  framework: vllm\n  framework: sglang\n  model: x\n"
    )
    with pytest.raises(ConfigurationError) as duplicate_error:
        adapter.resolve(duplicate)
    assert duplicate_error.value.reason_code == "invalid_benchmark_config"

    config = _config(tmp_path)
    link = tmp_path / "linked.yaml"
    link.symlink_to(config)
    hardlink = tmp_path / "hardlinked.yaml"
    os.link(config, hardlink)
    for path in (link, hardlink):
        with pytest.raises(ConfigurationError) as raised:
            adapter.resolve(path)
        assert raised.value.reason_code == "invalid_benchmark_config"
    assert loader.calls == []


def test_adapter_rejects_config_toctou(tmp_path: Path) -> None:
    config = _config(tmp_path)
    adapter, _ = _adapter(tmp_path, loader=_Loader(mutate=True))

    with pytest.raises(ConfigurationError) as raised:
        adapter.resolve(config)
    assert raised.value.reason_code == "benchmark_config_changed"


def test_production_api_rejects_import_outside_receipt_root(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)

    with pytest.raises(ConfigurationError) as raised:
        MagpieMainPublicApi(receipt.root("magpie"))
    assert raised.value.reason_code == "magpie_main_import_mismatch"
