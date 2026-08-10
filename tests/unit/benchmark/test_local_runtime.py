from __future__ import annotations

import copy
from pathlib import Path

import pytest

from apex.benchmark.local_runtime import parse_local_runtime_evidence
from apex.core import sha256_json


def _process(pid: int, *, ppid: int = 1) -> dict[str, object]:
    return {
        "pid": pid,
        "uid": 1000,
        "ppid": ppid,
        "process_group": pid,
        "session_id": pid,
        "start_time_ticks": pid * 10,
        "cmdline_sha256": "2" * 64,
        "argv": ["python", "-m", "Magpie"],
        "cwd": "/dependencies/magpie",
        "cgroup_sha256": "3" * 64,
        "cgroup_lines": ["0::/apex.slice"],
    }


def _dependencies(root: Path) -> dict[str, object]:
    return {
        "lock_sha256": "4" * 64,
        "dependencies": {
            name: {
                "root": str((root / name).resolve()),
                "commit": "5" * 40,
                "tree": "6" * 40,
            }
            for name in ("magpie", "tracelens", "inferencex")
        },
    }


def _cleanup() -> dict[str, object]:
    return {
        "devices": [
            {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}
        ],
        "ownership_receipt_sha256": "7" * 64,
        "verified": True,
    }


def _receipt(root: Path, dependencies: dict[str, object]) -> dict[str, object]:
    process = _process(123)
    return {
        "schema": "apex.magpie-local-runtime-observation/v2",
        "execution_mode": "local",
        "lifecycle": "one_shot",
        "input_config_sha256": "1" * 64,
        "gpu_lease_digest": "a" * 64,
        "dependency_receipt_sha256": sha256_json(dependencies),
        "inferencex_source": dependencies["dependencies"]["inferencex"],
        "benchmark_process": process,
        "runtime_processes": [_process(123)],
        "lifecycle_receipt": {
            "mode": "one_shot",
            "port": 8888,
            "observed_listener_pids": [123],
            "server_state": None,
            "quiescence_receipt": _cleanup(),
            "server_source_generation_sha256": "8" * 64,
            "server_generation_sha256": None,
        },
        "process_succeeded": True,
        "verified": True,
        "errors": [],
    }


def _parse(
    root: Path,
    receipt: dict[str, object],
    dependencies: dict[str, object],
    *,
    mode: str = "local",
    lifecycle: str = "one_shot",
):
    return parse_local_runtime_evidence(
        {"serving_runtime_receipt": receipt},
        expected_execution_mode=mode,
        expected_lifecycle=lifecycle,
        expected_config_sha256="1" * 64,
        expected_gpu_lease_digest="a" * 64,
        expected_inferencex_root=(root / "inferencex").resolve(),
        expected_inferencex_commit="5" * 40,
        expected_inferencex_tree="6" * 40,
        dependency_receipts=dependencies,
    )


def test_accepts_exact_one_shot_source_process_and_cleanup(tmp_path: Path) -> None:
    dependencies = _dependencies(tmp_path)
    result = _parse(tmp_path, _receipt(tmp_path, dependencies), dependencies)

    assert result.passed is True
    assert result.required is True
    assert result.source_root == (tmp_path / "inferencex").resolve()
    assert result.benchmark_pid == 123
    assert result.runtime_process_count == 1
    assert result.quiescence_verified is True


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value.update(input_config_sha256="9" * 64),
         "local_runtime_receipt_invalid"),
        (lambda value: value.update(gpu_lease_digest="9" * 64),
         "local_runtime_receipt_invalid"),
        (lambda value: value["inferencex_source"].update(root="/wrong"),
         "local_runtime_source_mismatch"),
        (lambda value: value.update(extra=True),
         "local_runtime_receipt_invalid"),
        (lambda value: value["benchmark_process"].update(pid=0),
         "local_runtime_benchmark_process_invalid"),
        (lambda value: value["runtime_processes"][0].update(cgroup_sha256="9" * 64),
         "local_runtime_process_cgroup_mismatch"),
        (lambda value: value["lifecycle_receipt"]["quiescence_receipt"].update(
            verified=False
        ), "local_runtime_lifecycle_invalid"),
    ],
)
def test_rejects_tampered_local_observer_facts(
    tmp_path: Path, mutation, error: str
) -> None:
    dependencies = _dependencies(tmp_path)
    receipt = copy.deepcopy(_receipt(tmp_path, dependencies))
    mutation(receipt)

    assert _parse(tmp_path, receipt, dependencies).error == error


def test_reuse_generation_binds_exact_server_process_and_metadata(
    tmp_path: Path,
) -> None:
    dependencies = _dependencies(tmp_path)
    receipt = _receipt(tmp_path, dependencies)
    server = _process(456)
    source_generation = "8" * 64
    metadata = "9" * 64
    receipt["lifecycle"] = "reuse"
    receipt["runtime_processes"] = [receipt["benchmark_process"], server]
    lifecycle = receipt["lifecycle_receipt"]
    lifecycle.update(
        {
            "mode": "reuse",
            "observed_listener_pids": [456],
            "server_state": {
                "process": server,
                "listener_pids": [456],
                "compatibility_sha256": metadata,
            },
            "quiescence_receipt": None,
            "server_generation_sha256": sha256_json(
                {
                    "server_source_generation_sha256": source_generation,
                    "server_process": server,
                    "compatibility_sha256": metadata,
                    "port": 8888,
                }
            ),
        }
    )

    assert _parse(
        tmp_path, receipt, dependencies, lifecycle="reuse"
    ).passed is True
    lifecycle["server_generation_sha256"] = "0" * 64
    assert _parse(
        tmp_path, receipt, dependencies, lifecycle="reuse"
    ).error == "local_runtime_server_generation_mismatch"


def test_rejects_local_receipt_on_a_docker_lane(tmp_path: Path) -> None:
    dependencies = _dependencies(tmp_path)
    result = _parse(
        tmp_path, _receipt(tmp_path, dependencies), dependencies, mode="docker"
    )

    assert result.required is False
    assert result.passed is False
    assert result.error == "unexpected_local_runtime_evidence"
