"""Freeze verified runtime locators and offline semantics into benchmark views."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from apex.core import ConfigurationError
from apex.runtime import DependencyReceipt


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SERVING_FRAMEWORKS = frozenset({"vllm", "sglang", "atom"})


def _environment(benchmark: dict[str, Any]) -> dict[str, Any]:
    envs = benchmark.setdefault("envs", {})
    if not isinstance(envs, dict):
        raise ConfigurationError(
            "benchmark.envs must be a mapping", "invalid_benchmark_config"
        )
    return envs


def _pin_inferencex(benchmark: dict[str, Any], receipt: DependencyReceipt) -> None:
    root = receipt.root("inferencex").resolve()
    commit = receipt.commits.get("inferencex", "")
    if not root.is_dir() or not _COMMIT.fullmatch(commit):
        raise ConfigurationError(
            "InferenceX receipt must identify an exact existing checkout",
            "invalid_inferencex_receipt",
        )
    benchmark["inferencex_path"] = str(root)


def _pin_lm_eval(benchmark: dict[str, Any], receipt: DependencyReceipt) -> None:
    framework = str(benchmark.get("framework", "")).strip().lower()
    if framework not in _SERVING_FRAMEWORKS:
        return
    runtime = receipt.lm_eval_runtime
    if runtime is None:
        raise ConfigurationError(
            "Serving quality evaluation requires a verified lm-eval runtime",
            "lm_eval_runtime_missing",
        )
    valid = (
        runtime.root.is_absolute()
        and not runtime.root.is_symlink()
        and runtime.root.is_dir()
        and bool(_DIGEST.fullmatch(runtime.runtime_sha256))
        and bool(_DIGEST.fullmatch(runtime.manifest_sha256))
        and bool(_DIGEST.fullmatch(runtime.lock_sha256))
        and bool(runtime.identity)
    )
    if not valid:
        raise ConfigurationError(
            "lm-eval runtime receipt must identify a verified immutable runtime",
            "invalid_lm_eval_runtime_receipt",
        )
    benchmark["lm_eval_runtime"] = {
        "path": str(runtime.root),
        "sha256": runtime.runtime_sha256,
        "identity": dict(runtime.identity),
    }


def _pin_hf_cache(
    benchmark: dict[str, Any],
    envs: dict[str, Any],
    cache: Path | None,
    offline: bool,
) -> None:
    if cache is not None:
        resolved = cache.resolve()
        if not cache.is_absolute() or cache.is_symlink() or not resolved.is_dir():
            raise ConfigurationError(
                "Hugging Face cache must be an existing absolute directory",
                "invalid_hf_cache_path",
            )
        benchmark["hf_cache_path"] = str(resolved)
    if offline and cache is None:
        raise ConfigurationError(
            "hf_offline requires a verified Hugging Face cache",
            "hf_offline_cache_missing",
        )
    if offline:
        envs.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )


def _pin_revision(envs: dict[str, Any], revision: str | None) -> None:
    if revision is None:
        return
    if not _COMMIT.fullmatch(revision):
        raise ConfigurationError(
            "Formal model revision must be a lowercase 40-hex commit",
            "invalid_model_revision",
        )
    envs["MODEL_REVISION"] = revision


def _pin_devices(envs: dict[str, Any], gpu_devices: str | None) -> None:
    if gpu_devices is None:
        return
    devices = tuple(part.strip() for part in gpu_devices.split(","))
    if not devices or any(not part.isdigit() for part in devices):
        raise ConfigurationError(
            "gpu_devices must be a comma-separated physical GPU index list",
            "invalid_gpu_devices",
        )
    envs["ROCR_VISIBLE_DEVICES"] = ",".join(devices)
    envs["HIP_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(len(devices)))


def pin_runtime_inputs(
    benchmark: dict[str, Any],
    receipt: DependencyReceipt,
    *,
    model_revision: str | None,
    hf_cache_path: Path | None,
    gpu_devices: str | None,
    hf_offline: bool,
) -> None:
    """Bind one receipt and caller-selected immutable runtime inputs."""

    envs = _environment(benchmark)
    _pin_inferencex(benchmark, receipt)
    _pin_lm_eval(benchmark, receipt)
    _pin_hf_cache(benchmark, envs, hf_cache_path, hf_offline)
    _pin_revision(envs, model_revision)
    _pin_devices(envs, gpu_devices)


__all__ = ["pin_runtime_inputs"]
