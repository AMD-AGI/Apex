"""Frozen config and dependency contract for local Magpie observation."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.core import (
    ContractError,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from apex.ports import MagpieAttestationRequest
from apex.runtime import DependencyReceipt
from apex.runtime.repositories import BootstrapError, inspect_repository


MAX_CONFIG_BYTES = 1024 * 1024
MAX_LIFECYCLE_BYTES = 256 * 1024
_BUILTIN_SCRIPTS = {
    framework: {f"{framework}_mi300x.sh", f"{framework}_mi355x.sh"}
    for framework in ("vllm", "sglang", "atom")
}


@dataclass(frozen=True, slots=True)
class LocalMagpieContract:
    framework: str
    model: str
    inferencex_root: Path
    lifecycle: str
    port: int
    pid_dir: Path | None
    pid_file: Path | None
    meta_file: Path | None
    metadata: Mapping[str, object]
    server_source_generation_sha256: str


def validate_local_request(request: MagpieAttestationRequest) -> None:
    if request.execution_mode != "local":
        raise ContractError(
            "Only local Magpie observation is available", "magpie_observer_mode_unavailable"
        )
    if request.lifecycle not in {"one_shot", "reuse", "cleanup"}:
        raise ContractError(
            "Local Magpie lifecycle is unavailable", "magpie_observer_lifecycle_unavailable"
        )
    if request.requested_image is not None:
        raise ContractError(
            "Local Magpie cannot bind a Docker image", "magpie_observer_image_mismatch"
        )
    if request.gpu_lease is None:
        raise ContractError("GPU lease authority is missing", "magpie_gpu_lease_missing")
    if not request.run_root.is_absolute() or request.run_root.is_symlink():
        raise ContractError("Magpie run root is unsafe", "invalid_benchmark_output")
    if sha256_file(request.config_path) != request.config_sha256:
        raise ContractError(
            "Benchmark config identity changed", "benchmark_config_changed_during_execution"
        )


def load_local_contract(
    request: MagpieAttestationRequest,
    receipt: DependencyReceipt,
    dependencies: Mapping[str, object],
) -> LocalMagpieContract:
    content = read_regular(request.config_path, MAX_CONFIG_BYTES)
    try:
        value = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise ContractError(
            "Local Magpie config is invalid", "invalid_benchmark_config"
        ) from error
    benchmark = value.get("benchmark") if isinstance(value, Mapping) else None
    if not isinstance(benchmark, Mapping):
        raise ContractError("Local Magpie config is invalid", "invalid_benchmark_config")
    framework = str(benchmark.get("framework", "")).lower()
    model = benchmark.get("model")
    inferencex = Path(str(benchmark.get("inferencex_path", ""))).resolve()
    script = benchmark.get("benchmark_script")
    if (
        framework not in _BUILTIN_SCRIPTS
        or str(benchmark.get("run_mode", "")).lower() != "local"
        or not isinstance(model, str)
        or not model
        or inferencex != receipt.root("inferencex").resolve()
        or script not in _BUILTIN_SCRIPTS[framework]
    ):
        raise ContractError(
            "Local Magpie runtime is unresolved", "magpie_local_runtime_unresolved"
        )
    lifecycle, pid_dir, pid_file, meta_file = _lifecycle_paths(benchmark, framework)
    if lifecycle != request.lifecycle:
        raise ContractError(
            "Local lifecycle differs from the request", "magpie_local_lifecycle_mismatch"
        )
    envs = benchmark.get("envs") if isinstance(benchmark.get("envs"), Mapping) else {}
    port = parse_port(envs.get("PORT", 8888))
    metadata = _desired_metadata(framework, model, inferencex, port, envs)
    generation = sha256_json(
        {
            "dependencies": dependencies,
            "framework": framework,
            "model": model,
            "script": script,
            "server_compatibility_metadata": metadata,
        }
    )
    return LocalMagpieContract(
        framework,
        model,
        inferencex,
        lifecycle,
        port,
        pid_dir,
        pid_file,
        meta_file,
        metadata,
        generation,
    )


def dependency_snapshot(receipt: DependencyReceipt) -> Mapping[str, object]:
    dependencies: dict[str, object] = {}
    for name in ("magpie", "tracelens", "inferencex"):
        try:
            state = inspect_repository(receipt.root(name))
        except BootstrapError as error:
            raise ContractError(
                "Dependency cannot be observed", "magpie_dependency_observation_failed"
            ) from error
        if state.commit != receipt.commits.get(name) or state.dirty_paths:
            raise ContractError(
                "Dependency receipt drifted", "magpie_dependency_observation_failed"
            )
        dependencies[name] = {
            "root": str(state.root),
            "commit": state.commit,
            "tree": state.tree,
        }
    return {"lock_sha256": receipt.lock_sha256, "dependencies": dependencies}


def read_regular(path: Path, limit: int) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
        raise ContractError("Evidence file is unsafe", "unsafe_magpie_local_evidence")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        first = os.read(descriptor, limit + 1)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(first) > limit or first != second:
        raise ContractError(
            "Evidence file is oversized or raced", "unsafe_magpie_local_evidence"
        )
    return first


def json_mapping(raw: bytes, reason: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("Evidence JSON is invalid", reason) from error
    if not isinstance(value, Mapping):
        raise ContractError("Evidence JSON is invalid", reason)
    return value


def write_new(path: Path, value: Mapping[str, object]) -> None:
    """Create one immutable evaluator artifact without replacing prior evidence."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lifecycle_paths(
    benchmark: Mapping[str, object], framework: str
) -> tuple[str, Path | None, Path | None, Path | None]:
    value = benchmark.get("server_lifecycle")
    if not isinstance(value, Mapping) or not enabled(value.get("enabled")):
        return "one_shot", None, None, None
    if enabled(value.get("force_reuse")):
        raise ContractError(
            "force_reuse bypasses exact server metadata binding",
            "magpie_local_force_reuse_forbidden",
        )
    raw_dir = value.get("pid_dir")
    pid_dir = (
        Path(str(raw_dir)).expanduser()
        if raw_dir not in (None, "")
        else Path.home() / ".cache" / "magpie" / "server"
    )
    if not pid_dir.is_absolute():
        raise ContractError(
            "Local lifecycle pid_dir is unsafe", "magpie_local_pid_dir_unsafe"
        )
    _reject_symlink_chain(pid_dir)
    envs = benchmark.get("envs") if isinstance(benchmark.get("envs"), Mapping) else {}
    port = parse_port(envs.get("PORT", 8888))
    lifecycle = "cleanup" if enabled(value.get("cleanup")) else "reuse"
    tag = f"{framework}_{port}"
    return lifecycle, pid_dir, pid_dir / f"{tag}.pid", pid_dir / f"{tag}.json"


def _desired_metadata(
    framework: str,
    model: str,
    inferencex: Path,
    port: int,
    envs: Mapping[str, object],
) -> Mapping[str, object]:
    upper = {str(key).upper(): str(value) for key, value in envs.items()}
    maximum = upper.get(
        "MAX_MODEL_LEN",
        upper.get("MAX_MODEL_LENGTH", upper.get("SGL_MEM_FRACTION_STATIC", "")),
    )
    return {
        "framework": framework,
        "model": model,
        "tp": upper.get("TP", "1"),
        "port": port,
        "extra_vllm_args": upper.get("EXTRA_VLLM_ARGS", ""),
        "extra_sglang_args": upper.get("EXTRA_SGLANG_ARGS", ""),
        "extra_atom_args": upper.get("EXTRA_ATOM_ARGS", ""),
        "max_model_len": maximum,
        "inferencex_path": str(inferencex),
    }


def parse_port(value: object) -> int:
    try:
        port = int(str(value))
    except (TypeError, ValueError) as error:
        raise ContractError("Local server port is invalid", "magpie_local_port_invalid") from error
    if not 0 < port < 65536:
        raise ContractError("Local server port is invalid", "magpie_local_port_invalid")
    return port


def enabled(value: object) -> bool:
    return value is True or value == 1 or (
        isinstance(value, str)
        and value.strip().lower() in {"1", "true", "yes", "on"}
    )


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ContractError(
                "Local lifecycle pid_dir is unsafe", "magpie_local_pid_dir_unsafe"
            )
        current = current.parent


__all__ = [
    "LocalMagpieContract",
    "MAX_CONFIG_BYTES",
    "MAX_LIFECYCLE_BYTES",
    "dependency_snapshot",
    "json_mapping",
    "load_local_contract",
    "read_regular",
    "validate_local_request",
    "write_new",
]
