"""Canonical construction of Apex-owned local Magpie runtime receipts."""

from __future__ import annotations

from typing import Mapping, Sequence

from apex.core import sha256_json


LOCAL_RUNTIME_SCHEMA = "apex.magpie-local-runtime-observation/v2"


def build_local_runtime_receipt(
    *,
    config_sha256: str,
    gpu_lease_digest: str,
    dependencies: Mapping[str, object],
    lifecycle: str,
    port: int,
    server_source_generation_sha256: str,
    benchmark_process: Mapping[str, object] | None,
    runtime_processes: Sequence[Mapping[str, object]],
    server_process: Mapping[str, object] | None,
    server_listener_pids: Sequence[int],
    compatibility_sha256: str | None,
    observed_listener_pids: Sequence[int],
    quiescence_receipt: Mapping[str, object] | None,
    process_succeeded: bool,
    verified: bool,
    errors: Sequence[str],
) -> Mapping[str, object]:
    """Build the exact receipt schema from already observed immutable facts."""

    server_state = _server_state(
        server_process, server_listener_pids, compatibility_sha256
    )
    generation = _server_generation(
        server_process,
        compatibility_sha256,
        server_source_generation_sha256,
        port,
    )
    return {
        "schema": LOCAL_RUNTIME_SCHEMA,
        "execution_mode": "local",
        "lifecycle": lifecycle,
        "input_config_sha256": config_sha256,
        "gpu_lease_digest": gpu_lease_digest,
        "dependency_receipt_sha256": sha256_json(dependencies),
        "inferencex_source": _inferencex_source(dependencies),
        "benchmark_process": benchmark_process,
        "runtime_processes": list(runtime_processes),
        "lifecycle_receipt": {
            "mode": lifecycle,
            "port": port,
            "observed_listener_pids": list(observed_listener_pids),
            "server_state": server_state,
            "quiescence_receipt": quiescence_receipt,
            "server_source_generation_sha256": server_source_generation_sha256,
            "server_generation_sha256": generation,
        },
        "process_succeeded": process_succeeded,
        "verified": verified,
        "errors": list(dict.fromkeys(errors)),
    }


def _server_state(
    process: Mapping[str, object] | None,
    listener_pids: Sequence[int],
    compatibility_sha256: str | None,
) -> Mapping[str, object] | None:
    if process is None:
        return None
    return {
        "process": process,
        "listener_pids": list(listener_pids),
        "compatibility_sha256": compatibility_sha256,
    }


def _server_generation(
    process: Mapping[str, object] | None,
    compatibility_sha256: str | None,
    source_generation_sha256: str,
    port: int,
) -> str | None:
    if process is None:
        return None
    return sha256_json(
        {
            "server_source_generation_sha256": source_generation_sha256,
            "server_process": process,
            "compatibility_sha256": compatibility_sha256,
            "port": port,
        }
    )


def _inferencex_source(
    dependencies: Mapping[str, object],
) -> Mapping[str, object] | None:
    values = dependencies.get("dependencies")
    source = values.get("inferencex") if isinstance(values, Mapping) else None
    return dict(source) if isinstance(source, Mapping) else None


__all__ = ["LOCAL_RUNTIME_SCHEMA", "build_local_runtime_receipt"]
