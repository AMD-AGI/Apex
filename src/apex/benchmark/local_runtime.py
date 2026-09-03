"""Strict typed evidence for Apex-observed local Magpie runtimes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import sha256_json

from .local_runtime_receipt import LOCAL_RUNTIME_SCHEMA


_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GPU_UNIQUE_ID = re.compile(r"GPU-[0-9a-f]{16}")
_KEYS = frozenset(
    {
        "schema",
        "execution_mode",
        "lifecycle",
        "input_config_sha256",
        "gpu_lease_digest",
        "dependency_receipt_sha256",
        "inferencex_source",
        "benchmark_process",
        "runtime_processes",
        "lifecycle_receipt",
        "process_succeeded",
        "verified",
        "errors",
    }
)
_SOURCE_KEYS = frozenset({"root", "commit", "tree"})
_PROCESS_KEYS = frozenset(
    {
        "pid",
        "uid",
        "ppid",
        "process_group",
        "session_id",
        "start_time_ticks",
        "cmdline_sha256",
        "argv",
        "cwd",
        "cgroup_sha256",
        "cgroup_lines",
    }
)
_LIFECYCLE_KEYS = frozenset(
    {
        "mode",
        "port",
        "observed_listener_pids",
        "server_state",
        "quiescence_receipt",
        "server_source_generation_sha256",
        "server_generation_sha256",
    }
)
_SERVER_KEYS = frozenset(
    {"process", "listener_pids", "compatibility_sha256"}
)


@dataclass(frozen=True, slots=True)
class LocalRuntimeEvidence:
    """Normalized source, process, listener, and lifecycle evidence."""

    required: bool
    passed: bool
    lifecycle: str | None
    source_root: Path | None
    source_commit: str | None
    source_tree: str | None
    benchmark_pid: int | None
    runtime_process_count: int | None
    gpu_lease_digest: str | None
    server_source_generation_sha256: str | None
    server_generation_sha256: str | None
    quiescence_verified: bool | None
    error: str | None = None


def parse_local_runtime_evidence(
    report: Mapping[str, Any],
    *,
    expected_execution_mode: str | None,
    expected_lifecycle: str | None,
    expected_config_sha256: str | None,
    expected_gpu_lease_digest: str | None,
    expected_inferencex_root: Path | None,
    expected_inferencex_commit: str | None,
    expected_inferencex_tree: str | None,
    dependency_receipts: Mapping[str, Any] | None,
) -> LocalRuntimeEvidence:
    """Validate one local receipt minted by the pre/post execution observer."""

    receipt = report.get("serving_runtime_receipt")
    required = expected_execution_mode == "local"
    if not required:
        if (
            isinstance(receipt, Mapping)
            and receipt.get("schema") == LOCAL_RUNTIME_SCHEMA
        ):
            return _failed("unexpected_local_runtime_evidence")
        return _empty(required=False, passed=True)
    error = _receipt_error(
        receipt,
        expected_lifecycle=expected_lifecycle,
        expected_config_sha256=expected_config_sha256,
        expected_gpu_lease_digest=expected_gpu_lease_digest,
        expected_inferencex_root=expected_inferencex_root,
        expected_inferencex_commit=expected_inferencex_commit,
        expected_inferencex_tree=expected_inferencex_tree,
        dependency_receipts=dependency_receipts,
    )
    data = receipt if isinstance(receipt, Mapping) else {}
    source = data.get("inferencex_source")
    lifecycle = data.get("lifecycle_receipt")
    benchmark = data.get("benchmark_process")
    processes = data.get("runtime_processes")
    quiescence = (
        lifecycle.get("quiescence_receipt")
        if isinstance(lifecycle, Mapping)
        else None
    )
    return LocalRuntimeEvidence(
        True,
        error is None,
        _string(data.get("lifecycle")),
        _path(source.get("root")) if isinstance(source, Mapping) else None,
        _string(source.get("commit")) if isinstance(source, Mapping) else None,
        _string(source.get("tree")) if isinstance(source, Mapping) else None,
        _integer(benchmark.get("pid")) if isinstance(benchmark, Mapping) else None,
        len(processes) if isinstance(processes, list) else None,
        _string(data.get("gpu_lease_digest")),
        _string(lifecycle.get("server_source_generation_sha256"))
        if isinstance(lifecycle, Mapping)
        else None,
        _string(lifecycle.get("server_generation_sha256"))
        if isinstance(lifecycle, Mapping)
        else None,
        quiescence.get("verified")
        if isinstance(quiescence, Mapping)
        and isinstance(quiescence.get("verified"), bool)
        else None,
        error,
    )


def _receipt_error(
    value: object,
    *,
    expected_lifecycle: str | None,
    expected_config_sha256: str | None,
    expected_gpu_lease_digest: str | None,
    expected_inferencex_root: Path | None,
    expected_inferencex_commit: str | None,
    expected_inferencex_tree: str | None,
    dependency_receipts: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(value, Mapping):
        return "local_runtime_receipt_missing"
    errors = value.get("errors")
    valid = (
        frozenset(value) == _KEYS
        and value.get("schema") == LOCAL_RUNTIME_SCHEMA
        and value.get("execution_mode") == "local"
        and value.get("lifecycle") == expected_lifecycle
        and value.get("input_config_sha256") == expected_config_sha256
        and _digest(expected_gpu_lease_digest)
        and value.get("gpu_lease_digest") == expected_gpu_lease_digest
        and _digest(value.get("dependency_receipt_sha256"))
        and isinstance(dependency_receipts, Mapping)
        and value.get("dependency_receipt_sha256") == sha256_json(dependency_receipts)
        and value.get("process_succeeded") is True
        and value.get("verified") is True
        and isinstance(errors, list)
        and not errors
    )
    if not valid:
        return "local_runtime_receipt_invalid"
    source_error = _source_error(
        value.get("inferencex_source"),
        expected_root=expected_inferencex_root,
        expected_commit=expected_inferencex_commit,
        expected_tree=expected_inferencex_tree,
    )
    if source_error:
        return source_error
    benchmark = value.get("benchmark_process")
    if not _valid_process(benchmark):
        return "local_runtime_benchmark_process_invalid"
    return _runtime_error(value, benchmark)


def _source_error(
    value: object,
    *,
    expected_root: Path | None,
    expected_commit: str | None,
    expected_tree: str | None,
) -> str | None:
    if (
        not isinstance(value, Mapping)
        or frozenset(value) != _SOURCE_KEYS
        or expected_root is None
        or expected_commit is None
        or expected_tree is None
        or not expected_root.is_absolute()
        or value.get("root") != str(expected_root.resolve())
        or value.get("commit") != expected_commit
        or value.get("tree") != expected_tree
        or not _COMMIT.fullmatch(str(value.get("commit", "")))
        or not _COMMIT.fullmatch(str(value.get("tree", "")))
    ):
        return "local_runtime_source_mismatch"
    return None


def _runtime_error(
    value: Mapping[str, Any], benchmark: object
) -> str | None:
    processes = value.get("runtime_processes")
    if not isinstance(processes, list) or not processes or len(processes) > 4096:
        return "local_runtime_processes_invalid"
    if any(not _valid_process(process) for process in processes):
        return "local_runtime_processes_invalid"
    pids = [_integer(process.get("pid")) for process in processes]
    if None in pids or len(set(pids)) != len(pids):
        return "local_runtime_processes_invalid"
    cgroup = benchmark.get("cgroup_sha256") if isinstance(benchmark, Mapping) else None
    if any(process.get("cgroup_sha256") != cgroup for process in processes):
        return "local_runtime_process_cgroup_mismatch"
    lifecycle = value.get("lifecycle_receipt")
    return _lifecycle_error(lifecycle, frozenset(int(pid) for pid in pids if pid))


def _lifecycle_error(value: object, runtime_pids: frozenset[int]) -> str | None:
    if not isinstance(value, Mapping) or frozenset(value) != _LIFECYCLE_KEYS:
        return "local_runtime_lifecycle_invalid"
    mode, port = value.get("mode"), value.get("port")
    listeners = _pid_list(value.get("observed_listener_pids"))
    source_generation = value.get("server_source_generation_sha256")
    if (
        mode not in {"one_shot", "reuse", "cleanup"}
        or isinstance(port, bool)
        or not isinstance(port, int)
        or not 0 < port < 65536
        or listeners is None
        or not listeners
        or not set(listeners).issubset(runtime_pids)
        or not _digest(source_generation)
    ):
        return "local_runtime_lifecycle_invalid"
    return _lifecycle_mode_error(value, mode, source_generation, runtime_pids)


def _lifecycle_mode_error(
    value: Mapping[str, Any],
    mode: object,
    source_generation: object,
    runtime_pids: frozenset[int],
) -> str | None:
    server = value.get("server_state")
    quiescence = value.get("quiescence_receipt")
    generation = value.get("server_generation_sha256")
    if mode == "one_shot":
        valid = (
            server is None
            and generation is None
            and _valid_quiescence(quiescence)
        )
        return None if valid else "local_runtime_lifecycle_invalid"
    if not _valid_server(server, runtime_pids):
        return "local_runtime_server_state_invalid"
    expected = sha256_json(
        {
            "server_source_generation_sha256": source_generation,
            "server_process": server["process"],
            "compatibility_sha256": server["compatibility_sha256"],
            "port": value["port"],
        }
    )
    if generation != expected:
        return "local_runtime_server_generation_mismatch"
    quiescence_valid = (
        quiescence is None
        if mode == "reuse"
        else _valid_quiescence(quiescence)
    )
    return None if quiescence_valid else "local_runtime_lifecycle_invalid"


def _valid_server(value: object, runtime_pids: frozenset[int]) -> bool:
    if not isinstance(value, Mapping) or frozenset(value) != _SERVER_KEYS:
        return False
    process = value.get("process")
    listeners = _pid_list(value.get("listener_pids"))
    return bool(
        _valid_process(process)
        and process.get("pid") in runtime_pids
        and listeners
        and set(listeners).issubset(runtime_pids)
        and _digest(value.get("compatibility_sha256"))
    )


def _valid_quiescence(value: object) -> bool:
    if (
        not isinstance(value, Mapping)
        or frozenset(value) != {
            "devices", "ownership_receipt_sha256", "verified"
        }
        or value.get("verified") is not True
    ):
        return False
    devices = value.get("devices")
    if not isinstance(devices, list) or not devices:
        return False
    identities = []
    for device in devices:
        if not isinstance(device, Mapping) or frozenset(device) != {
            "rsmi_index", "unique_id"
        }:
            return False
        index, unique_id = device.get("rsmi_index"), device.get("unique_id")
        if (
            not _nonnegative(index)
            or not isinstance(unique_id, str)
            or not _GPU_UNIQUE_ID.fullmatch(unique_id)
        ):
            return False
        identities.append((index, unique_id))
    return len(identities) == len(set(identities)) and _digest(
        value.get("ownership_receipt_sha256")
    )


def _valid_process(value: object) -> bool:
    if not isinstance(value, Mapping) or frozenset(value) != _PROCESS_KEYS:
        return False
    argv, cgroups = value.get("argv"), value.get("cgroup_lines")
    cwd = value.get("cwd")
    return bool(
        _positive(value.get("pid"))
        and _nonnegative(value.get("uid"))
        and _nonnegative(value.get("ppid"))
        and _positive(value.get("process_group"))
        and _positive(value.get("session_id"))
        and _positive(value.get("start_time_ticks"))
        and _digest(value.get("cmdline_sha256"))
        and isinstance(argv, list)
        and argv
        and all(isinstance(item, str) and item for item in argv)
        and isinstance(cwd, str)
        and Path(cwd).is_absolute()
        and _digest(value.get("cgroup_sha256"))
        and isinstance(cgroups, list)
        and cgroups
        and all(isinstance(item, str) and item for item in cgroups)
    )


def _pid_list(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value or len(value) > 4096:
        return None
    if any(not _positive(item) for item in value) or len(set(value)) != len(value):
        return None
    return tuple(value)


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(_DIGEST.fullmatch(value))


def _positive(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _path(value: object) -> Path | None:
    return Path(value) if isinstance(value, str) and Path(value).is_absolute() else None


def _failed(error: str) -> LocalRuntimeEvidence:
    return _empty(required=False, passed=False, error=error)


def _empty(
    *, required: bool, passed: bool, error: str | None = None
) -> LocalRuntimeEvidence:
    return LocalRuntimeEvidence(
        required=required,
        passed=passed,
        lifecycle=None,
        source_root=None,
        source_commit=None,
        source_tree=None,
        benchmark_pid=None,
        runtime_process_count=None,
        gpu_lease_digest=None,
        server_source_generation_sha256=None,
        server_generation_sha256=None,
        quiescence_verified=None,
        error=error,
    )


__all__ = ["LocalRuntimeEvidence", "parse_local_runtime_evidence"]
