"""Immutable candidate image and same-process import/test execution for Qwen."""

from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    build_subprocess_environment,
)

from .oracles import ResolvedCorrectnessOracle
from .overlay_runtime import (
    BuiltOverlay,
    ContainerEngine,
    InstalledPythonTarget,
)


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_TEST_ROOT = "/opt/apex-oracle"
_RESULT_ROOT = "/opt/apex-result"
_RUNNER_NAME = "apex_oracle_runner.py"
_LOADED_RECEIPT = "loaded-candidate.json"
_ORACLE_RUNNER = r'''from __future__ import annotations
import hashlib
import importlib
import json
import os
import pathlib
import sys


def observe(module_name):
    module = importlib.import_module(module_name)
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise RuntimeError("module_file_missing")
    path = pathlib.Path(raw).resolve(strict=True)
    return module, str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def write_receipt(path, payload):
    target = pathlib.Path(path)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o444)
    os.replace(temporary, target)


def main():
    module_name, expected_path, expected_sha, receipt_path, pytest_json = sys.argv[1:]
    payload = {
        "schema": "apex.qwen-oracle-loaded-candidate/v1",
        "module_name": module_name,
        "expected_path": expected_path,
        "expected_sha256": expected_sha,
        "status": "failed",
        "reason_code": "runner_failed",
    }
    exit_code = 1
    try:
        module, before_path, before_sha = observe(module_name)
        payload["before"] = {"path": before_path, "sha256": before_sha}
        if before_path != expected_path:
            raise RuntimeError("loaded_module_path_mismatch")
        if before_sha != expected_sha:
            raise RuntimeError("loaded_module_bytes_mismatch")
        import pytest
        pytest_exit = int(pytest.main(json.loads(pytest_json)))
        payload["pytest_exit_code"] = pytest_exit
        same_module = sys.modules.get(module_name) is module
        _, after_path, after_sha = observe(module_name)
        payload["after"] = {
            "path": after_path,
            "sha256": after_sha,
            "same_module_object": same_module,
        }
        if not same_module or after_path != expected_path:
            raise RuntimeError("loaded_module_replaced")
        if after_sha != expected_sha:
            raise RuntimeError("loaded_module_bytes_changed")
        if pytest_exit != 0:
            raise RuntimeError("pytest_failed")
        payload["status"] = "passed"
        payload["reason_code"] = "same_process_import_and_tests_passed"
        exit_code = 0
    except BaseException as error:
        payload["reason_code"] = str(error) or type(error).__name__
    write_receipt(receipt_path, payload)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
'''


class CommandPort(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> ProcessResult: ...


class OracleOverlayBuildPort(Protocol):
    def build(
        self,
        *,
        parent_locator: str,
        parent_image_id: str,
        candidate_source: Path,
        target: InstalledPythonTarget,
        build_root: Path,
        cwd: Path,
    ) -> BuiltOverlay: ...


@dataclass(frozen=True, slots=True)
class OracleTestExecution:
    process: ProcessResult
    loaded_candidate: Mapping[str, Any]
    junit: Mapping[str, Any]
    runner_sha256: str


class DockerOracleOverlayBuilder:
    """Build one no-network overlay using the verified immutable repo digest."""

    def __init__(self, commands: CommandPort, engine: ContainerEngine) -> None:
        self._commands = commands
        self._engine = engine

    def build(
        self,
        *,
        parent_locator: str,
        parent_image_id: str,
        candidate_source: Path,
        target: InstalledPythonTarget,
        build_root: Path,
        cwd: Path,
    ) -> BuiltOverlay:
        if "@sha256:" not in parent_locator or not _IMAGE_ID.fullmatch(parent_image_id):
            raise ContractError("Oracle parent is mutable", "invalid_image_identity")
        if candidate_source.is_symlink() or not candidate_source.is_file():
            raise IntegrityError("Candidate source is unsafe", "invalid_frozen_candidate")
        context = build_root / "context"
        if build_root.exists() or build_root.is_symlink():
            raise IntegrityError("Oracle overlay exists", "immutable_delivery_artifact")
        context.mkdir(parents=True)
        copied = context / "candidate.py"
        shutil.copyfile(candidate_source, copied)
        copied.chmod(0o444)
        candidate_sha = sha256_file(candidate_source)
        if sha256_file(copied) != candidate_sha:
            raise IntegrityError("Overlay copy drifted", "candidate_lineage_mismatch")
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            f"FROM {parent_locator}\n"
            f"COPY {json.dumps(['candidate.py', target.container_path])}\n",
            encoding="utf-8",
        )
        iidfile = build_root / "derived-image.id"
        _checked(
            self._commands,
            (
                "docker",
                "build",
                "--no-cache",
                "--pull=false",
                "--network=none",
                "--iidfile",
                str(iidfile),
                "--file",
                str(dockerfile),
                str(context),
            ),
            cwd,
            1800,
            "overlay_build_failed",
        )
        image_id = iidfile.read_text(encoding="utf-8").strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise IntegrityError("Overlay image ID is invalid", "overlay_build_failed")
        image = self._engine.inspect_image(image_id, cwd=cwd)
        if image.image_id != image_id:
            raise IntegrityError("Overlay image identity drifted", "image_identity_mismatch")
        return BuiltOverlay(image, sha256_file(dockerfile), candidate_sha)


def runner_sha256() -> str:
    return sha256_bytes(_ORACLE_RUNNER.encode("utf-8"))


def materialize_runner(destination: Path) -> Mapping[str, str]:
    runner = destination / _RUNNER_NAME
    runner.write_text(_ORACLE_RUNNER, encoding="utf-8")
    runner.chmod(0o444)
    observed = sha256_file(runner)
    if observed != runner_sha256():
        raise IntegrityError("Oracle runner copy drifted", "oracle_runner_drift")
    return {"path": _RUNNER_NAME, "sha256": observed}


def run_oracle_tests(
    commands: CommandPort,
    *,
    image_id: str,
    module_name: str,
    target: InstalledPythonTarget,
    candidate_sha256: str,
    oracle: ResolvedCorrectnessOracle,
    tests: Path,
    results: Path,
    gpu_scope: str,
    cwd: Path,
    timeout_seconds: int,
) -> OracleTestExecution:
    if oracle.test_argv[:3] != ("python3", "-m", "pytest"):
        raise ContractError("Oracle pytest argv is invalid", "invalid_oracle_binding")
    gpu_args = _gpu_arguments(gpu_scope)
    receipt_path = results / _LOADED_RECEIPT
    result = commands.run(
        (
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add",
            "video",
            "--ipc=host",
            "--shm-size=16g",
            "--security-opt",
            "seccomp=unconfined",
            *gpu_args,
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "TRITON_CACHE_DIR=/tmp/triton-cache",
            "--mount",
            f"type=bind,src={tests},dst={_TEST_ROOT},readonly",
            "--mount",
            f"type=bind,src={results},dst={_RESULT_ROOT}",
            "--workdir",
            _TEST_ROOT,
            "--entrypoint",
            "python3",
            image_id,
            f"{_TEST_ROOT}/{_RUNNER_NAME}",
            module_name,
            target.container_path,
            candidate_sha256,
            f"{_RESULT_ROOT}/{_LOADED_RECEIPT}",
            json.dumps(list(oracle.test_argv[3:]), separators=(",", ":")),
        ),
        cwd=cwd.resolve(strict=True),
        environment=build_subprocess_environment(
            inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS
        ),
        timeout_seconds=timeout_seconds,
    )
    loaded = validate_loaded_receipt(
        receipt_path,
        module_name=module_name,
        expected_path=target.container_path,
        expected_sha256=candidate_sha256,
    )
    process = _process_receipt(result)
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise IntegrityError(
            "Oracle test process failed",
            "oracle_test_failed",
            {"process": process, "loaded_candidate": loaded},
        )
    junit = _validate_junit(results / "junit.xml", oracle.expected_test_count)
    return OracleTestExecution(result, loaded, junit, runner_sha256())


def validate_loaded_receipt(
    path: Path,
    *,
    module_name: str,
    expected_path: str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    metadata = path.lstat() if path.exists() else None
    if (
        metadata is None
        or path.is_symlink()
        or not path.is_file()
        or metadata.st_nlink != 1
        or metadata.st_size > 64 * 1024
        or metadata.st_mode & 0o222
    ):
        raise IntegrityError("Loaded receipt is unsafe", "loaded_byte_probe_failed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("Loaded receipt is invalid", "loaded_byte_probe_failed") from error
    expected = {
        "schema": "apex.qwen-oracle-loaded-candidate/v1",
        "module_name": module_name,
        "expected_path": expected_path,
        "expected_sha256": expected_sha256,
        "status": "passed",
        "reason_code": "same_process_import_and_tests_passed",
    }
    if not isinstance(payload, Mapping) or any(payload.get(k) != v for k, v in expected.items()):
        raise IntegrityError("Loaded receipt identity drifted", "loaded_byte_probe_failed")
    before = payload.get("before")
    after = payload.get("after")
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or before.get("path") != expected_path
        or after.get("path") != expected_path
        or before.get("sha256") != expected_sha256
        or after.get("sha256") != expected_sha256
        or after.get("same_module_object") is not True
        or payload.get("pytest_exit_code") != 0
    ):
        raise IntegrityError("Loaded receipt bytes drifted", "loaded_byte_probe_failed")
    return {**dict(payload), "receipt_sha256": sha256_file(path), "read_only": True}


def process_receipt(result: ProcessResult) -> Mapping[str, Any]:
    return _process_receipt(result)


def _gpu_arguments(scope: str) -> tuple[str, ...]:
    if scope == "all-visible-amd-gpus":
        return ()
    prefix = "amd-gpu-set="
    if not scope.startswith(prefix):
        raise ContractError("Oracle GPU scope is invalid", "invalid_gpu_device_scope")
    devices = tuple(scope.removeprefix(prefix).split(","))
    if not devices or any(not item or "\x00" in item for item in devices):
        raise ContractError("Oracle GPU scope is invalid", "invalid_gpu_device_scope")
    rocr = ",".join(devices)
    hip = ",".join(str(index) for index in range(len(devices)))
    return ("--env", f"ROCR_VISIBLE_DEVICES={rocr}", "--env", f"HIP_VISIBLE_DEVICES={hip}")


def _validate_junit(path: Path, expected: int) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError("Oracle JUnit receipt is missing", "oracle_test_receipt_missing")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise IntegrityError("Oracle JUnit receipt is oversized", "oracle_test_receipt_invalid")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise IntegrityError(
            "Oracle JUnit receipt is invalid", "oracle_test_receipt_invalid"
        ) from error
    suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
    values = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if values != {"tests": expected, "failures": 0, "errors": 0, "skipped": 0}:
        raise IntegrityError("Oracle test outcomes are incomplete", "oracle_test_failed", values)
    return {**values, "sha256": sha256_file(path)}


def _checked(
    commands: CommandPort,
    argv: tuple[str, ...],
    cwd: Path,
    timeout: int,
    reason: str,
) -> ProcessResult:
    result = commands.run(
        argv,
        cwd=cwd.resolve(strict=True),
        environment=build_subprocess_environment(
            inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS
        ),
        timeout_seconds=timeout,
    )
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise IntegrityError("Oracle Docker command failed", reason, _process_receipt(result))
    return result


def _process_receipt(result: ProcessResult) -> Mapping[str, Any]:
    return {
        "argv": list(result.argv),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(result.stderr.encode("utf-8")),
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "duration_seconds": result.duration_seconds,
    }


__all__ = [
    "CommandPort",
    "DockerOracleOverlayBuilder",
    "OracleOverlayBuildPort",
    "OracleTestExecution",
    "materialize_runner",
    "process_receipt",
    "run_oracle_tests",
    "runner_sha256",
    "validate_loaded_receipt",
]
