"""Strict Apex-owned execution evidence for unchanged published Magpie reports."""

from __future__ import annotations

import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ConfigurationError, IntegrityError, sha256_file
from apex.ports import (
    BenchmarkPass,
    MagpieAttestationRequest,
    MagpieFormalMeasurementSupport,
    MagpieReportLocation,
)
from apex.runtime.magpie_result_contract import EXECUTION_ATTESTATION_SCHEMA

from .ray_import_validation import (
    imported_artifact_paths,
    imported_workspace_origin,
    valid_ray_runtime,
    validate_imported_report,
)


_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_GPU_UNIQUE_ID = re.compile(r"GPU-[0-9a-f]{16}")
_PRIVATE_REPORT_FIELDS = frozenset(
    {
        "run_kind",
        "reward_eligible",
        "model_revision_receipt",
        "inferencex_runtime_receipt",
        "lm_eval_runtime_receipt",
        "serving_runtime_receipt",
        "quality_gate",
    }
)
_KEYS = frozenset(
    {
        "schema",
        "authority",
        "official_report_path",
        "official_report_size_bytes",
        "report_sha256",
        "config_sha256",
        "run_id",
        "pass_type",
        "lane_verified",
        "reward_eligible",
        "profiling_enabled",
        "process",
        "dependencies",
        "runtime",
        "gpu_engagement",
        "quality_gate",
        "errors",
    }
)
_PROCESS_KEYS = frozenset(
    {"schema", "argv_sha256", "exit_code", "timed_out", "succeeded", "verified"}
)
_DEPENDENCY_KEYS = frozenset({"schema", "verified", "receipts"})
_RUNTIME_KEYS = frozenset(
    {
        "schema",
        "verified",
        "model_revision_receipt",
        "inferencex_runtime_receipt",
        "lm_eval_runtime_receipt",
        "serving_runtime_receipt",
    }
)
_GPU_KEYS = frozenset({"schema", "verified", "devices", "processes"})
_QUALITY_KEYS = frozenset({"schema", "verified", "receipt"})
@dataclass(frozen=True, slots=True)
class MagpieExecutionAttestation:
    """Apex evaluator facts bound to one immutable official report."""

    source_path: Path
    config_sha256: str
    run_id: str
    pass_type: BenchmarkPass
    lane_verified: bool
    reward_eligible: bool
    profiling_enabled: bool
    process: Mapping[str, Any]
    dependencies: Mapping[str, Any]
    runtime: Mapping[str, Any]
    gpu_engagement: Mapping[str, Any]
    quality_gate: Mapping[str, Any]
    errors: tuple[str, ...]

    @property
    def run_kind(self) -> str:
        return self.pass_type.value

    @property
    def imported_workspace_origin(self) -> Path | None:
        """Return an attested Ray origin only as a local-copy consistency check."""

        return imported_workspace_origin(self.runtime)

    def imported_artifact_paths(self, workspace: Path) -> tuple[Path, ...]:
        """Rehash every authority-declared local copy before persistence."""

        return imported_artifact_paths(self.runtime, workspace)

    def evaluator_evidence(
        self, official_report: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Expose only evaluator-side values to existing focused validators."""

        return {
            "framework": official_report.get("framework"),
            "model_revision_receipt": self.runtime["model_revision_receipt"],
            "inferencex_runtime_receipt": self.runtime[
                "inferencex_runtime_receipt"
            ],
            "lm_eval_runtime_receipt": self.runtime["lm_eval_runtime_receipt"],
            "serving_runtime_receipt": self.runtime["serving_runtime_receipt"],
            "quality_gate": self.quality_gate["receipt"],
        }

    def verdict_errors(self) -> tuple[str, ...]:
        """Return fail-closed evaluator boundary errors without hiding detail."""

        errors = self.errors
        checks = (
            (self.lane_verified, "execution_attestation_lane_unverified"),
            (self.process["verified"], "execution_attestation_process_unverified"),
            (
                self.dependencies["verified"],
                "execution_attestation_dependencies_unverified",
            ),
            (self.runtime["verified"], "execution_attestation_runtime_unverified"),
            (
                self.gpu_engagement["verified"],
                "execution_attestation_gpu_engagement_unverified",
            ),
            (self.quality_gate["verified"], "execution_attestation_quality_unverified"),
        )
        return errors + tuple(error for valid, error in checks if valid is not True)


class UnavailableMagpieExecutionAttestor:
    """Explicit production default: no workload starts without an observer."""

    @property
    def is_available(self) -> bool:
        return False

    def formal_measurement_support(self, execution_mode: str, lifecycle: str) -> MagpieFormalMeasurementSupport:
        del execution_mode, lifecycle
        return MagpieFormalMeasurementSupport(
            False, "magpie_execution_attestor_unavailable", None,
            ("magpie_execution_attestor_unavailable",),
        )

    def prepare(self, request: MagpieAttestationRequest) -> object:
        raise RuntimeError("Magpie execution attestor is unavailable")

    def launch_argv(self, session: object) -> tuple[str, ...]:
        raise RuntimeError("Magpie execution attestor is unavailable")

    def abort(self, session: object, *, reason: str) -> None:
        del session, reason

    def locate_report(self, session: object) -> MagpieReportLocation:
        raise RuntimeError("Magpie execution attestor is unavailable")

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None:
        del session, report_path, command_exit_code, timed_out
        raise RuntimeError("Magpie execution attestor is unavailable")


def expected_attestation_path(report_path: Path) -> Path:
    """Return the sibling evaluator artifact for a Magpie workspace report."""

    return report_path.resolve().parent.parent / "evaluator" / "execution_attestation.json"


def locate_local_magpie_report(run_root: Path) -> MagpieReportLocation:
    """Locate one nonlinked official report under an Apex-created run root."""

    reports = tuple(sorted(run_root.rglob("benchmark_report.json")))
    if not reports:
        return MagpieReportLocation(None, "benchmark_report_missing")
    if len(reports) != 1:
        return MagpieReportLocation(None, "ambiguous_benchmark_reports")
    report = reports[0]
    try:
        info = report.lstat()
        resolved = report.resolve(strict=True)
        resolved.relative_to(run_root.resolve(strict=True))
    except (OSError, ValueError):
        return MagpieReportLocation(None, "unsafe_benchmark_report")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or report.is_symlink():
        return MagpieReportLocation(None, "unsafe_benchmark_report")
    return MagpieReportLocation(resolved)


def _load_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"Invalid Apex Magpie execution attestation {path}: {error}",
            "invalid_magpie_execution_attestation",
        ) from error
    if not isinstance(value, Mapping) or frozenset(value) != _KEYS:
        raise ConfigurationError(
            "Apex Magpie execution attestation has an invalid key set",
            "invalid_magpie_execution_attestation",
        )
    return value


def _regular_file(path: Path, reason: str) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise ConfigurationError(f"Missing evidence file {path}", reason) from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
        raise IntegrityError(f"Unsafe evidence file {path}", reason)
    return path.resolve()


def _evaluator_root(path: Path, workspace: Path) -> Path:
    current = path.parent.resolve()
    while current != current.parent and current.name != "evaluator":
        current = current.parent
    if current.name != "evaluator" or current.parent != workspace.parent:
        raise IntegrityError(
            "Execution attestation must be under an evaluator root sibling to Magpie",
            "unsafe_magpie_execution_attestation_location",
        )
    return current


def _validate_report_binding(
    value: Mapping[str, Any], path: Path, report_path: Path, report: Mapping[str, Any]
) -> None:
    workspace = report_path.parent.resolve()
    evaluator_root = _evaluator_root(path, workspace)
    expected_relative = report_path.relative_to(evaluator_root.parent).as_posix()
    if (
        value.get("official_report_path") != expected_relative
    ):
        raise IntegrityError(
            "Execution attestation does not bind the unchanged official report",
            "magpie_execution_attestation_report_mismatch",
        )
    _validate_report_document_binding(
        value,
        report_sha256=sha256_file(report_path),
        report_size_bytes=report_path.stat().st_size,
        report=report,
    )


def _validate_report_document_binding(
    value: Mapping[str, Any],
    *,
    report_sha256: str,
    report_size_bytes: int,
    report: Mapping[str, Any],
) -> None:
    relative = value.get("official_report_path")
    report_locator = PurePosixPath(relative) if isinstance(relative, str) else None
    size = value.get("official_report_size_bytes")
    if (
        report_locator is None
        or report_locator.is_absolute()
        or ".." in report_locator.parts
        or not report_locator.parts
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or size != report_size_bytes
        or value.get("report_sha256") != report_sha256
    ):
        raise IntegrityError(
            "Execution attestation does not bind the unchanged official report",
            "magpie_execution_attestation_report_mismatch",
        )
    private = _PRIVATE_REPORT_FIELDS.intersection(report)
    if private:
        raise IntegrityError(
            f"Official Magpie report contains Apex-private fields: {sorted(private)}",
            "private_magpie_report_fields_present",
        )
    validate_imported_report(value, report_sha256, report_size_bytes, report)


def _mapping(
    value: Mapping[str, Any], name: str, keys: frozenset[str], schema: str
) -> Mapping[str, Any]:
    item = value.get(name)
    if (
        not isinstance(item, Mapping)
        or frozenset(item) != keys
        or item.get("schema") != schema
        or not isinstance(item.get("verified"), bool)
    ):
        raise ConfigurationError(
            f"Execution attestation {name} mapping is invalid",
            "invalid_magpie_execution_attestation",
        )
    return item


def _validate_nested(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    process = _mapping(
        value, "process", _PROCESS_KEYS, "apex.magpie-process-attestation/v1"
    )
    dependencies = _mapping(
        value,
        "dependencies",
        _DEPENDENCY_KEYS,
        "apex.magpie-dependency-attestation/v1",
    )
    runtime = _mapping(
        value, "runtime", _RUNTIME_KEYS, "apex.magpie-runtime-attestation/v1"
    )
    gpu = _mapping(
        value, "gpu_engagement", _GPU_KEYS, "apex.magpie-gpu-engagement/v1"
    )
    quality = _mapping(
        value, "quality_gate", _QUALITY_KEYS, "apex.magpie-quality-attestation/v1"
    )
    return process, dependencies, runtime, gpu, quality


def _valid_containers(
    process: Mapping[str, Any], dependencies: Mapping[str, Any],
    runtime: Mapping[str, Any], gpu: Mapping[str, Any], quality: Mapping[str, Any],
) -> bool:
    receipt_values = tuple(runtime[key] for key in _RUNTIME_KEYS if key.endswith("receipt"))
    serving = runtime.get("serving_runtime_receipt")
    return bool(
        _DIGEST.fullmatch(str(process.get("argv_sha256", "")))
        and isinstance(process.get("exit_code"), (int, type(None)))
        and not isinstance(process.get("exit_code"), bool)
        and isinstance(process.get("timed_out"), bool)
        and isinstance(process.get("succeeded"), bool)
        and _valid_dependencies(dependencies)
        and all(item is None or isinstance(item, Mapping) for item in receipt_values)
        and valid_ray_runtime(serving)
        and _valid_gpu_engagement(gpu)
        and (
            quality.get("receipt") is None
            or isinstance(quality.get("receipt"), Mapping)
        )
    )


def _valid_dependencies(value: Mapping[str, Any]) -> bool:
    receipt = value.get("receipts")
    if not isinstance(receipt, Mapping):
        return False
    if not value["verified"]:
        return True
    dependencies = receipt.get("dependencies")
    if not _DIGEST.fullmatch(str(receipt.get("lock_sha256", ""))):
        return False
    if not isinstance(dependencies, Mapping) or frozenset(dependencies) != {
        "magpie", "tracelens", "inferencex"
    }:
        return False
    for dependency in dependencies.values():
        if not isinstance(dependency, Mapping):
            return False
        root = dependency.get("root")
        if (
            not isinstance(root, str)
            or not Path(root).is_absolute()
            or not _COMMIT.fullmatch(str(dependency.get("commit", "")))
            or not _COMMIT.fullmatch(str(dependency.get("tree", "")))
        ):
            return False
    return True


def _valid_gpu_engagement(value: Mapping[str, Any]) -> bool:
    devices, processes = value.get("devices"), value.get("processes")
    if not isinstance(devices, list) or not isinstance(processes, list):
        return False
    if not value["verified"]:
        return True
    expected: dict[int, str] = {}
    for device in devices:
        if not isinstance(device, Mapping):
            return False
        index, unique_id = device.get("rsmi_index"), device.get("unique_id")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in expected
            or not _GPU_UNIQUE_ID.fullmatch(str(unique_id))
        ):
            return False
        expected[index] = str(unique_id)
    observed: set[int] = set()
    seen_pids: set[int] = set()
    for process in processes:
        if not isinstance(process, Mapping):
            return False
        pid, indices = process.get("pid"), process.get("rsmi_device_indices")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or pid in seen_pids
            or isinstance(process.get("uid"), bool)
            or not isinstance(process.get("uid"), int)
            or process["uid"] < 0
            or isinstance(process.get("start_time_ticks"), bool)
            or not isinstance(process.get("start_time_ticks"), int)
            or process["start_time_ticks"] <= 0
            or not _DIGEST.fullmatch(str(process.get("cmdline_sha256", "")))
            or not isinstance(indices, list)
            or not indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index not in expected
                for index in indices
            )
        ):
            return False
        seen_pids.add(pid)
        observed.update(indices)
    return bool(expected) and bool(processes) and observed == set(expected)


def load_magpie_execution_attestation(
    attestation_path: Path,
    *,
    report_path: Path,
    report: Mapping[str, Any],
    expected_config_sha256: str | None,
    expected_run_id: str,
    expected_pass_type: BenchmarkPass,
    command_exit_code: int | None,
    timed_out: bool,
) -> MagpieExecutionAttestation:
    """Load and bind one evaluator artifact without trusting Magpie extensions."""

    path = _regular_file(
        attestation_path, "magpie_execution_attestation_missing"
    )
    official = _regular_file(report_path, "benchmark_report_missing")
    value = _load_object(path)
    _validate_report_binding(value, path, official, report)
    process, dependencies, runtime, gpu, quality, errors = _validate_document(
        value,
        expected_config_sha256=expected_config_sha256,
        expected_run_id=expected_run_id,
        expected_pass_type=expected_pass_type,
        command_exit_code=command_exit_code,
        timed_out=timed_out,
    )
    return MagpieExecutionAttestation(
        source_path=path,
        config_sha256=value["config_sha256"],
        run_id=expected_run_id,
        pass_type=expected_pass_type,
        lane_verified=value["lane_verified"],
        reward_eligible=value["reward_eligible"],
        profiling_enabled=value["profiling_enabled"],
        process=process,
        dependencies=dependencies,
        runtime=runtime,
        gpu_engagement=gpu,
        quality_gate=quality,
        errors=errors,
    )


def validate_magpie_execution_attestation_document(
    value: object,
    *,
    report_sha256: str,
    report_size_bytes: int,
    report: Mapping[str, Any],
    expected_config_sha256: str | None,
    expected_run_id: str,
    expected_pass_type: BenchmarkPass,
    command_exit_code: int | None,
    timed_out: bool,
) -> Mapping[str, Any]:
    """Validate a CAS-restored sidecar without trusting its original host path."""

    if not isinstance(value, Mapping) or frozenset(value) != _KEYS:
        raise IntegrityError(
            "Apex Magpie execution attestation has an invalid key set",
            "invalid_magpie_execution_attestation",
        )
    _validate_report_document_binding(
        value,
        report_sha256=report_sha256,
        report_size_bytes=report_size_bytes,
        report=report,
    )
    try:
        _validate_document(
            value,
            expected_config_sha256=expected_config_sha256,
            expected_run_id=expected_run_id,
            expected_pass_type=expected_pass_type,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
        )
    except ConfigurationError as error:
        raise IntegrityError(
            "CAS execution attestation is malformed",
            "invalid_magpie_execution_attestation",
        ) from error
    return value


def _validate_document(
    value: Mapping[str, Any],
    *,
    expected_config_sha256: str | None,
    expected_run_id: str,
    expected_pass_type: BenchmarkPass,
    command_exit_code: int | None,
    timed_out: bool,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[str, ...],
]:
    process, dependencies, runtime, gpu, quality = _validate_nested(value)
    raw_errors = value.get("errors")
    valid = (
        value.get("schema") == EXECUTION_ATTESTATION_SCHEMA
        and value.get("authority") == "apex_evaluator"
        and isinstance(value.get("config_sha256"), str)
        and bool(_DIGEST.fullmatch(value["config_sha256"]))
        and (
            expected_config_sha256 is None
            or value.get("config_sha256") == expected_config_sha256
        )
        and value.get("run_id") == expected_run_id
        and value.get("pass_type") == expected_pass_type.value
        and isinstance(value.get("lane_verified"), bool)
        and isinstance(value.get("reward_eligible"), bool)
        and isinstance(value.get("profiling_enabled"), bool)
        and process.get("exit_code") == command_exit_code
        and process.get("timed_out") is timed_out
        and process.get("succeeded") is (command_exit_code == 0 and not timed_out)
        and _valid_containers(
            process,
            dependencies,
            runtime,
            gpu,
            quality,
        )
        and isinstance(raw_errors, list)
        and all(isinstance(error, str) and error for error in raw_errors)
    )
    if not valid:
        raise IntegrityError(
            "Execution attestation differs from evaluator expectations",
            "magpie_execution_attestation_mismatch",
        )
    return (
        process,
        dependencies,
        runtime,
        gpu,
        quality,
        tuple(raw_errors),
    )

__all__ = [
    "MagpieExecutionAttestation",
    "UnavailableMagpieExecutionAttestor",
    "expected_attestation_path",
    "locate_local_magpie_report",
    "load_magpie_execution_attestation",
    "validate_magpie_execution_attestation_document",
]
