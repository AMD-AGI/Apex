"""Safely import authority-bound Magpie Ray artifacts from shared storage."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ContractError, sha256_json
from apex.ports import (
    MagpieAttestationRequest,
    RayArtifactClaim,
    RayNodeEvidenceBinding,
    RayNodeEvidenceReceipt,
)
from apex.runtime import DependencyReceipt


_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MAX_ARTIFACTS = 10_000
_MAX_FILE_BYTES = 1 << 30
_MAX_TOTAL_BYTES = 8 << 30
_CHUNK = 1 << 20
_NODE_KEYS = frozenset(
    {
        "schema", "node_id", "binding_sha256", "procfs",
        "dependencies_sha256", "kfd", "verified",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "model_revision_receipt",
        "inferencex_runtime_receipt",
        "lm_eval_runtime_receipt",
        "verified",
    }
)


@dataclass(frozen=True, slots=True)
class RayImportedWorkspace:
    """Local immutable copy plus path-independent import evidence."""

    origin_workspace: Path
    local_workspace: Path
    report_path: Path
    claims: tuple[RayArtifactClaim, ...]
    binding_sha256: str

    def to_dict(self) -> Mapping[str, Any]:
        claims = [_claim_dict(claim) for claim in self.claims]
        return {
            "schema": "apex.magpie-ray-artifact-import/v1",
            "binding_sha256": self.binding_sha256,
            "origin_workspace_path": str(self.origin_workspace),
            "artifacts": claims,
            "manifest_sha256": sha256_json(claims),
            "verified": True,
        }


class RaySharedArtifactImporter:
    """Copy only authority-declared regular files into the Apex run root."""

    def import_workspace(
        self,
        request: MagpieAttestationRequest,
        binding: RayNodeEvidenceBinding,
        receipt: RayNodeEvidenceReceipt,
    ) -> RayImportedWorkspace:
        validate_ray_node_evidence(receipt, binding, request)
        local = request.run_root / "ray_workspace"
        try:
            local.mkdir(mode=0o700, exist_ok=False)
        except OSError as error:
            raise ContractError(
                "Ray import workspace cannot be created", "ray_artifact_import_failed"
            ) from error
        for claim in receipt.artifacts:
            _copy_claim(receipt.workspace_path, local, claim)
        report_claim = next(
            claim for claim in receipt.artifacts if claim.role == "benchmark_report"
        )
        return RayImportedWorkspace(
            receipt.workspace_path,
            local.resolve(),
            (local / report_claim.relative_path).resolve(),
            receipt.artifacts,
            binding.digest,
        )


def validate_ray_node_evidence(
    receipt: RayNodeEvidenceReceipt,
    binding: RayNodeEvidenceBinding,
    request: MagpieAttestationRequest,
) -> None:
    """Require exact task/job/config/run authority and complete worker evidence."""

    contract = binding.ray_contract
    if (
        receipt.schema != "apex.magpie-ray-node-evidence/v1"
        or not _DIGEST.fullmatch(receipt.authority_sha256)
        or receipt.binding_sha256 != binding.digest
        or not _TASK_ID.fullmatch(receipt.magpie_task_id)
        or request.run_id != binding.run_id
        or request.pass_type is not binding.pass_type
    ):
        raise ContractError("Ray node evidence binding is invalid", "ray_node_evidence_invalid")
    _validate_workspace(receipt, contract.results_path)
    _validate_claims(receipt.artifacts)
    _validate_nodes(receipt, binding)
    _validate_dependencies(receipt.dependencies, receipt.node_receipts)
    _validate_gpu(receipt, binding, request)
    if frozenset(receipt.runtime) != _RUNTIME_KEYS or receipt.runtime.get("verified") is not True:
        raise ContractError("Ray runtime evidence is invalid", "ray_node_evidence_invalid")


def _validate_workspace(receipt: RayNodeEvidenceReceipt, results: Path) -> None:
    workspace = receipt.workspace_path
    if not workspace.is_absolute() or workspace == Path("/"):
        raise ContractError("Ray workspace is invalid", "ray_artifact_workspace_invalid")
    try:
        relative = workspace.relative_to(results / receipt.magpie_task_id)
    except ValueError as error:
        raise ContractError(
            "Ray workspace is outside the exact task result root",
            "ray_artifact_workspace_invalid",
        ) from error
    if len(relative.parts) != 1 or not relative.name.startswith("benchmark_"):
        raise ContractError("Ray workspace is ambiguous", "ray_artifact_workspace_invalid")


def _validate_claims(claims: tuple[RayArtifactClaim, ...]) -> None:
    if not claims or len(claims) > _MAX_ARTIFACTS:
        raise ContractError("Ray artifact manifest is invalid", "ray_artifact_manifest_invalid")
    total = 0
    keys: set[str] = set()
    report_count = 0
    for claim in claims:
        relative = PurePosixPath(claim.relative_path)
        valid_path = (
            bool(relative.parts)
            and not relative.is_absolute()
            and ".." not in relative.parts
            and all(part not in {"", "."} for part in relative.parts)
        )
        if (
            not claim.role
            or not valid_path
            or claim.relative_path in keys
            or isinstance(claim.size_bytes, bool)
            or claim.size_bytes <= 0
            or claim.size_bytes > _MAX_FILE_BYTES
            or not _DIGEST.fullmatch(claim.sha256)
        ):
            raise ContractError("Ray artifact manifest is invalid", "ray_artifact_manifest_invalid")
        keys.add(claim.relative_path)
        total += claim.size_bytes
        report_count += int(
            claim.role == "benchmark_report" and claim.relative_path == "benchmark_report.json"
        )
    if report_count != 1 or total > _MAX_TOTAL_BYTES:
        raise ContractError("Ray artifact manifest is invalid", "ray_artifact_manifest_invalid")


def _validate_nodes(
    receipt: RayNodeEvidenceReceipt, binding: RayNodeEvidenceBinding
) -> None:
    expected_count = binding.ray_contract.num_nodes if binding.ray_contract.multi_node else 1
    if len(receipt.node_receipts) != expected_count:
        raise ContractError("Ray node coverage is incomplete", "ray_node_evidence_invalid")
    node_ids: set[str] = set()
    for node in receipt.node_receipts:
        if (
            frozenset(node) != _NODE_KEYS
            or node.get("schema") != "apex.magpie-ray-worker-node/v1"
            or node.get("verified") is not True
            or node.get("binding_sha256") != binding.digest
            or not isinstance(node.get("node_id"), str)
            or node["node_id"] in node_ids
            or not isinstance(node.get("procfs"), Mapping)
            or not isinstance(node.get("kfd"), Mapping)
            or node["kfd"].get("verified") is not True
            or node["kfd"].get("cleanup_verified") is not True
            or not _DIGEST.fullmatch(str(node.get("dependencies_sha256", "")))
        ):
            raise ContractError("Ray node evidence is invalid", "ray_node_evidence_invalid")
        node_ids.add(str(node["node_id"]))
    if binding.task.get("node_id") not in node_ids:
        raise ContractError("Ray task node is not attested", "ray_node_evidence_invalid")
    task_node = next(node for node in receipt.node_receipts if node["node_id"] == binding.task["node_id"])
    procfs = task_node["procfs"]
    if (
        procfs.get("worker_pid") != binding.task.get("worker_pid")
        or procfs.get("ray_task_id") != binding.task.get("task_id")
        or procfs.get("ray_worker_id") != binding.task.get("worker_id")
    ):
        raise ContractError("Ray worker procfs binding is invalid", "ray_node_evidence_invalid")


def _validate_dependencies(
    observed: Mapping[str, Any],
    nodes: tuple[Mapping[str, Any], ...],
) -> None:
    dependencies = observed.get("dependencies")
    if (
        observed.get("lock_sha256") is None
        or not _DIGEST.fullmatch(str(observed.get("lock_sha256")))
        or not isinstance(dependencies, Mapping)
        or frozenset(dependencies) != {"magpie", "tracelens", "inferencex"}
    ):
        raise ContractError("Ray dependency evidence is invalid", "ray_node_evidence_invalid")
    digest = sha256_json(observed)
    if any(node.get("dependencies_sha256") != digest for node in nodes):
        raise ContractError("Ray node dependency binding is invalid", "ray_node_evidence_invalid")
    for value in dependencies.values():
        if (
            not isinstance(value, Mapping)
            or not Path(str(value.get("root", ""))).is_absolute()
            or not _COMMIT.fullmatch(str(value.get("commit", "")))
            or not _COMMIT.fullmatch(str(value.get("tree", "")))
        ):
            raise ContractError("Ray dependency evidence is invalid", "ray_node_evidence_invalid")
def _validate_gpu(
    receipt: RayNodeEvidenceReceipt,
    binding: RayNodeEvidenceBinding,
    request: MagpieAttestationRequest,
) -> None:
    if not receipt.gpu_devices or not receipt.gpu_processes:
        raise ContractError("Ray KFD evidence is missing", "ray_node_evidence_invalid")
    node_ids = {str(node["node_id"]) for node in receipt.node_receipts}
    expected = _lease_devices(request.gpu_lease)
    observed = {
        (device.get("rsmi_index"), device.get("unique_id"))
        for device in receipt.gpu_devices
        if isinstance(device, Mapping)
    }
    if observed != expected:
        raise ContractError("Ray GPU devices differ from lease", "ray_node_evidence_invalid")
    engaged: set[int] = set()
    for process in receipt.gpu_processes:
        if (
            not isinstance(process, Mapping)
            or process.get("node_id") not in node_ids
            or process.get("ray_job_id") != binding.task.get("job_id")
            or process.get("ray_task_id") != binding.task.get("task_id")
            or process.get("ray_worker_id") != binding.task.get("worker_id")
            or process.get("ray_worker_pid") != binding.task.get("worker_pid")
            or not isinstance(process.get("pid"), int)
            or isinstance(process.get("pid"), bool)
            or process["pid"] <= 0
            or not _DIGEST.fullmatch(str(process.get("cmdline_sha256", "")))
            or not process.get("rsmi_device_indices")
        ):
            raise ContractError("Ray KFD evidence is invalid", "ray_node_evidence_invalid")
        engaged.update(process["rsmi_device_indices"])
    if engaged != {index for index, _unique_id in expected}:
        raise ContractError("Ray KFD coverage is incomplete", "ray_node_evidence_invalid")


def _lease_devices(value: Mapping[str, object] | None) -> set[tuple[int, str]]:
    ownership = value.get("ownership") if isinstance(value, Mapping) else None
    devices = ownership.get("selected_devices") if isinstance(ownership, Mapping) else None
    if not isinstance(devices, list):
        raise ContractError("Ray GPU lease is invalid", "ray_node_evidence_invalid")
    expected = {
        (item.get("rsmi_index"), item.get("unique_id"))
        for item in devices if isinstance(item, Mapping)
    }
    if len(expected) != len(devices):
        raise ContractError("Ray GPU lease is invalid", "ray_node_evidence_invalid")
    return expected


def validate_remote_dependencies(
    observed: Mapping[str, Any], expected: DependencyReceipt
) -> None:
    """Bind worker dependency commits/trees to the installed Apex lock."""

    dependencies = observed.get("dependencies")
    raw = expected.raw.get("dependencies")
    if observed.get("lock_sha256") != expected.lock_sha256 or not isinstance(raw, Mapping):
        raise ContractError("Ray dependency lock differs", "ray_dependency_mismatch")
    assert isinstance(dependencies, Mapping)
    for name in ("magpie", "tracelens", "inferencex"):
        source = raw.get(name)
        remote = dependencies.get(name)
        if (
            not isinstance(source, Mapping)
            or not isinstance(remote, Mapping)
            or remote.get("commit") != expected.commits.get(name)
            or remote.get("tree") != source.get("tree")
        ):
            raise ContractError("Ray dependency identity differs", "ray_dependency_mismatch")


def _copy_claim(source_root: Path, destination_root: Path, claim: RayArtifactClaim) -> None:
    target = destination_root.joinpath(*PurePosixPath(claim.relative_path).parts)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_fd = _open_file_nofollow(source_root, claim.relative_path)
    target_fd: int | None = None
    try:
        first_info = _regular_info(source_fd, claim)
        first_digest, first_size = _digest_fd(source_fd)
        middle_info = _regular_info(source_fd, claim)
        os.lseek(source_fd, 0, os.SEEK_SET)
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        second_digest, second_size = _copy_digest(source_fd, target_fd)
        os.fsync(target_fd)
        final_info = _regular_info(source_fd, claim)
        stable = _identity(first_info) == _identity(middle_info) == _identity(final_info)
        valid = (
            stable
            and first_size == second_size == claim.size_bytes
            and first_digest == second_digest == claim.sha256
        )
        if not valid:
            raise ContractError("Ray artifact changed during import", "ray_artifact_import_race")
    except Exception:
        if target_fd is not None:
            os.close(target_fd)
            target_fd = None
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def _open_file_nofollow(root: Path, relative: str) -> int:
    directory = _open_directory_nofollow(root)
    parts = PurePosixPath(relative).parts
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    except OSError as error:
        raise ContractError("Ray artifact is unsafe", "unsafe_ray_artifact") from error
    finally:
        os.close(directory)


def _open_directory_nofollow(path: Path) -> int:
    if not path.is_absolute():
        raise ContractError("Ray artifact root is invalid", "unsafe_ray_artifact")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ContractError("Ray artifact root is unsafe", "unsafe_ray_artifact") from error


def _regular_info(descriptor: int, claim: RayArtifactClaim) -> os.stat_result:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != claim.size_bytes
    ):
        raise ContractError("Ray artifact is not a regular file", "unsafe_ray_artifact")
    return info


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _digest_fd(descriptor: int) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    while chunk := os.read(descriptor, _CHUNK):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _copy_digest(source: int, destination: int) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    while chunk := os.read(source, _CHUNK):
        digest.update(chunk)
        size += len(chunk)
        view = memoryview(chunk)
        while view:
            written = os.write(destination, view)
            view = view[written:]
    return digest.hexdigest(), size


def _claim_dict(claim: RayArtifactClaim) -> Mapping[str, Any]:
    return {
        "role": claim.role,
        "path": claim.relative_path,
        "size_bytes": claim.size_bytes,
        "sha256": claim.sha256,
    }


__all__ = [
    "RayImportedWorkspace",
    "RaySharedArtifactImporter",
    "validate_ray_node_evidence",
    "validate_remote_dependencies",
]
