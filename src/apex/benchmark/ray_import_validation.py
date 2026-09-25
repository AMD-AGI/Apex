"""Path-independent validation for imported Magpie Ray workspaces."""

from __future__ import annotations

import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ConfigurationError, IntegrityError, sha256_file, sha256_json


_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUNTIME_KEYS = frozenset(
    {
        "schema", "execution_mode", "input_config_sha256", "ray_config_sha256",
        "ray_address_sha256", "gpu_lease_sha256", "cluster_identity_sha256", "node_authority_sha256",
        "node_evidence_binding_sha256", "magpie_task_id", "artifact_import", "job",
        "driver_process", "task", "node_receipts", "process_succeeded", "verified",
        "errors",
    }
)
_IMPORT_KEYS = frozenset(
    {"schema", "binding_sha256", "origin_workspace_path", "artifacts", "manifest_sha256", "verified"}
)
_ARTIFACT_KEYS = frozenset({"role", "path", "size_bytes", "sha256"})


def imported_workspace_origin(runtime: Mapping[str, Any]) -> Path | None:
    """Return a Ray origin solely to check unchanged report workspace bytes."""

    receipt = runtime.get("serving_runtime_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != (
        "apex.magpie-ray-runtime-observation/v2"
    ):
        return None
    imported = receipt.get("artifact_import")
    origin = imported.get("origin_workspace_path") if isinstance(imported, Mapping) else None
    return Path(origin) if isinstance(origin, str) and Path(origin).is_absolute() else None


def imported_artifact_paths(
    runtime: Mapping[str, Any], workspace: Path
) -> tuple[Path, ...]:
    """Rehash every authority-declared local copy before persistence."""

    receipt = runtime.get("serving_runtime_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("execution_mode") != "ray":
        return ()
    imported = receipt.get("artifact_import")
    artifacts = imported.get("artifacts") if isinstance(imported, Mapping) else None
    if not isinstance(artifacts, list):
        raise IntegrityError("Ray artifact import is missing", "unsafe_ray_artifact")
    paths: list[Path] = []
    root = workspace.resolve()
    for artifact in artifacts:
        assert isinstance(artifact, Mapping)
        candidate = root.joinpath(*PurePosixPath(str(artifact["path"])).parts)
        source = _regular_file(candidate)
        if source.stat().st_size != artifact["size_bytes"] or sha256_file(source) != artifact["sha256"]:
            raise IntegrityError("Ray artifact copy changed", "unsafe_ray_artifact")
        paths.append(source)
    return tuple(paths)


def validate_imported_report(
    value: Mapping[str, Any],
    report_sha256: str,
    report_size_bytes: int,
    report: Mapping[str, Any],
) -> None:
    """Bind unchanged official bytes to an authority-declared remote origin."""

    runtime = value.get("runtime")
    receipt = runtime.get("serving_runtime_receipt") if isinstance(runtime, Mapping) else None
    if not isinstance(receipt, Mapping) or receipt.get("schema") != (
        "apex.magpie-ray-runtime-observation/v2"
    ):
        return
    imported = receipt.get("artifact_import")
    artifacts = imported.get("artifacts") if isinstance(imported, Mapping) else None
    if not isinstance(artifacts, list):
        raise IntegrityError("Ray artifact import is invalid", "invalid_magpie_execution_attestation")
    reports = tuple(
        item for item in artifacts
        if isinstance(item, Mapping) and item.get("role") == "benchmark_report"
    )
    workspace = report.get("workspace_dir")
    binding_sha256 = _binding_digest(value, receipt)
    valid = (
        imported.get("schema") == "apex.magpie-ray-artifact-import/v1"
        and imported.get("verified") is True
        and _DIGEST.fullmatch(str(imported.get("binding_sha256", "")))
        and imported.get("origin_workspace_path") == workspace
        and isinstance(workspace, str)
        and Path(workspace).is_absolute()
        and imported.get("manifest_sha256") == sha256_json(artifacts)
        and receipt.get("node_evidence_binding_sha256") == binding_sha256
        and len(reports) == 1
        and reports[0].get("path") == "benchmark_report.json"
        and reports[0].get("size_bytes") == report_size_bytes
        and reports[0].get("sha256") == report_sha256
    )
    if not valid:
        raise IntegrityError("Ray import report binding is invalid", "magpie_execution_attestation_report_mismatch")


def _binding_digest(
    value: Mapping[str, Any], receipt: Mapping[str, Any]
) -> str:
    process = value.get("process")
    return sha256_json(
        {
            "run_id": value.get("run_id"),
            "pass_type": value.get("pass_type"),
            "config_sha256": value.get("config_sha256"),
            "benchmark_argv_sha256": (
                process.get("argv_sha256") if isinstance(process, Mapping) else None
            ),
            "gpu_lease_sha256": receipt.get("gpu_lease_sha256"),
            "ray_config_sha256": receipt.get("ray_config_sha256"),
            "ray_address_sha256": receipt.get("ray_address_sha256"),
            "cluster_identity_sha256": receipt.get("cluster_identity_sha256"),
            "job": receipt.get("job"),
            "driver_process": receipt.get("driver_process"),
            "task": receipt.get("task"),
        }
    )


def valid_ray_runtime(value: object) -> bool:
    """Structurally validate the path-independent v2 runtime receipt."""

    if not isinstance(value, Mapping) or value.get("execution_mode") != "ray":
        return True
    if value.get("schema") != "apex.magpie-ray-runtime-observation/v2":
        return False
    imported = value.get("artifact_import")
    if not isinstance(imported, Mapping) or frozenset(imported) != _IMPORT_KEYS:
        return False
    artifacts = imported.get("artifacts")
    nodes = value.get("node_receipts")
    job, task = value.get("job"), value.get("task")
    digest_fields = (
        "input_config_sha256", "ray_config_sha256", "ray_address_sha256", "gpu_lease_sha256",
        "cluster_identity_sha256", "node_authority_sha256", "node_evidence_binding_sha256",
    )
    return bool(
        frozenset(value) == _RUNTIME_KEYS
        and all(_DIGEST.fullmatch(str(value.get(field, ""))) for field in digest_fields)
        and imported.get("binding_sha256") == value.get("node_evidence_binding_sha256")
        and isinstance(imported.get("origin_workspace_path"), str)
        and Path(imported["origin_workspace_path"]).is_absolute()
        and isinstance(artifacts, list)
        and 0 < len(artifacts) <= 10_000
        and all(_valid_artifact(item) for item in artifacts)
        and imported.get("manifest_sha256") == sha256_json(artifacts)
        and imported.get("verified") is True
        and all(isinstance(value.get(field), Mapping) for field in ("job", "driver_process", "task"))
        and isinstance(nodes, list) and bool(nodes)
        and task.get("job_id") == job.get("job_id")
        and isinstance(task.get("node_id"), str)
        and any(
            isinstance(item, Mapping) and item.get("node_id") == task.get("node_id")
            for item in nodes
        )
        and all(
            isinstance(item, Mapping)
            and item.get("verified") is True
            and item.get("binding_sha256") == value.get("node_evidence_binding_sha256")
            for item in nodes
        )
        and isinstance(value.get("process_succeeded"), bool)
        and isinstance(value.get("verified"), bool)
        and isinstance(value.get("errors"), list)
        and all(isinstance(item, str) and item for item in value["errors"])
    )


def _valid_artifact(value: object) -> bool:
    if not isinstance(value, Mapping) or frozenset(value) != _ARTIFACT_KEYS:
        return False
    path = PurePosixPath(str(value.get("path", "")))
    size = value.get("size_bytes")
    return bool(
        isinstance(value.get("role"), str) and value["role"]
        and path.parts and not path.is_absolute() and ".." not in path.parts
        and isinstance(size, int) and not isinstance(size, bool) and size > 0
        and _DIGEST.fullmatch(str(value.get("sha256", "")))
    )


def _regular_file(path: Path) -> Path:
    try:
        info = path.lstat()
    except OSError as error:
        raise ConfigurationError("Missing imported Ray artifact", "unsafe_ray_artifact") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
        raise IntegrityError("Unsafe imported Ray artifact", "unsafe_ray_artifact")
    return path.resolve()


__all__ = [
    "imported_artifact_paths",
    "imported_workspace_origin",
    "valid_ray_runtime",
    "validate_imported_report",
]
