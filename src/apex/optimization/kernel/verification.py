"""Evaluator-owned phased compile, correctness, and normal-performance execution."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from apex.core import IntegrityError, canonical_json_bytes, sha256_bytes, sha256_file
from apex.execution import (
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)
from apex.intake import CommandSpec, ResolvedTaskSpec
from apex.ports import AgentProcessContainmentReceipt


_PHASES = ("compile", "correctness", "performance")


@dataclass(frozen=True, slots=True)
class ExecutableIdentity:
    """Canonical executable bytes and filesystem identity frozen for one phase."""

    path: str
    size: int
    sha256: str
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("executable identity path must be absolute")
        if self.size < 0 or any(
            value < 0
            for value in (
                self.device,
                self.inode,
                self.mode,
                self.mtime_ns,
                self.ctime_ns,
            )
        ):
            raise ValueError("executable identity metadata must be non-negative")
        if self.mode & 0o111 == 0:
            raise ValueError("executable identity mode is not executable")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("executable identity SHA-256 is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    phase: str
    argv: tuple[str, ...]
    executable_identity: ExecutableIdentity
    executable_identity_reverified: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_seconds: float
    process_containment: AgentProcessContainmentReceipt

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"unsupported kernel verification phase: {self.phase}")
        if not self.argv or self.argv[0] != self.executable_identity.path:
            raise ValueError("command argv does not bind the executable identity")
        if not self.executable_identity_reverified:
            raise ValueError("command evidence requires post-phase executable revalidation")

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "argv": list(self.argv),
            "executable_identity": self.executable_identity.to_dict(),
            "executable_identity_reverified": self.executable_identity_reverified,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "process_containment": self.process_containment.to_dict(),
            "passed": self.passed,
        }


class CandidateVerifier:
    """Run one trusted argv phase at a time against unchanged candidate bytes."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor()

    def compile(
        self,
        resolved: ResolvedTaskSpec,
        *,
        candidate_root: Path,
        expected_source_digest: str,
    ) -> CommandEvidence:
        return self._run_checked(
            "compile", resolved, candidate_root, expected_source_digest
        )

    def correctness(
        self,
        resolved: ResolvedTaskSpec,
        *,
        candidate_root: Path,
        expected_source_digest: str,
    ) -> CommandEvidence:
        return self._run_checked(
            "correctness", resolved, candidate_root, expected_source_digest
        )

    def performance(
        self,
        resolved: ResolvedTaskSpec,
        *,
        candidate_root: Path,
        expected_source_digest: str,
    ) -> CommandEvidence:
        """Run only in the normal, uninstrumented candidate workspace."""

        return self._run_checked(
            "performance", resolved, candidate_root, expected_source_digest
        )

    def _run_checked(
        self,
        phase: str,
        resolved: ResolvedTaskSpec,
        candidate_root: Path,
        expected_source_digest: str,
    ) -> CommandEvidence:
        _assert_source_digest(
            candidate_root,
            resolved.task.editable_files,
            expected_source_digest,
            reason_code=f"candidate_changed_before_{phase}",
        )
        evidence = self._run(phase, resolved.task.commands[phase], candidate_root)
        _assert_source_digest(
            candidate_root,
            resolved.task.editable_files,
            expected_source_digest,
            reason_code=f"verifier_changed_candidate_during_{phase}",
        )
        return evidence

    def _run(self, phase: str, command: CommandSpec, candidate_root: Path) -> CommandEvidence:
        environment = build_subprocess_environment(
            command.env,
            inherit=(*GPU_RUNTIME_ENVIRONMENT_KEYS, *HF_RUNTIME_ENVIRONMENT_KEYS),
            reserved=("PATH",),
        )
        cwd = (
            candidate_root
            if command.cwd == "."
            else candidate_root.joinpath(*command.cwd.split("/"))
        )
        resolved_cwd = cwd.resolve(strict=True)
        executable = _freeze_executable(command.argv[0], resolved_cwd, environment["PATH"])
        argv = (executable.path, *command.argv[1:])
        try:
            process = self._supervisor.run(
                argv,
                cwd=resolved_cwd,
                environment=environment,
                timeout_seconds=command.timeout_seconds,
                require_pid_namespace=True,
            )
        finally:
            _revalidate_executable(executable, phase=phase)
        return _evidence(phase, process, executable)


def candidate_source_digest(root: Path, relative_paths: Sequence[str]) -> str:
    """Hash exact normal-runtime source bytes, independent of freeze-only mode bits."""

    resolved_root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    for relative in sorted(relative_paths):
        path = root.joinpath(*relative.split("/"))
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not resolved.is_relative_to(resolved_root)
        ):
            raise IntegrityError(
                f"candidate source is not an isolated regular file: {relative}",
                "candidate_source_integrity_failed",
            )
        entries.append(
            {"path": relative, "sha256": sha256_file(path), "size": metadata.st_size}
        )
    return sha256_bytes(
        canonical_json_bytes(
            {"schema_version": "apex.normal-candidate-source/v1", "files": entries}
        )
    )


def _assert_source_digest(
    root: Path,
    relative_paths: Sequence[str],
    expected: str,
    *,
    reason_code: str,
) -> None:
    observed = candidate_source_digest(root, relative_paths)
    if observed != expected:
        raise IntegrityError(
            "evaluator candidate source no longer matches the frozen bytes",
            reason_code,
            {"expected": expected, "observed": observed},
        )


def _freeze_executable(argv0: str, cwd: Path, search_path: str) -> ExecutableIdentity:
    located = _locate_executable(argv0, cwd, search_path)
    try:
        canonical = located.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise IntegrityError(
            "Kernel verifier executable cannot be resolved",
            "verifier_executable_not_found",
            {"argv0": argv0},
        ) from exc
    return _snapshot_executable(canonical)


def _locate_executable(argv0: str, cwd: Path, search_path: str) -> Path:
    if "\x00" in argv0:
        raise IntegrityError(
            "Kernel verifier executable contains a NUL byte",
            "verifier_executable_invalid",
        )
    if os.sep in argv0 or (os.altsep is not None and os.altsep in argv0):
        supplied = Path(argv0)
        return supplied if supplied.is_absolute() else cwd / supplied
    located = shutil.which(argv0, path=search_path)
    if located is None:
        raise IntegrityError(
            "Kernel verifier executable was not found on the evaluator PATH",
            "verifier_executable_not_found",
            {"argv0": argv0},
        )
    return Path(located)


def _snapshot_executable(path: Path) -> ExecutableIdentity:
    try:
        path_metadata = path.lstat()
    except OSError as exc:
        raise IntegrityError(
            "Kernel verifier executable path cannot be inspected",
            "verifier_executable_invalid",
            {"path": str(path)},
        ) from exc
    if not path.is_absolute() or stat.S_ISLNK(path_metadata.st_mode):
        raise IntegrityError(
            "Kernel verifier executable path must be absolute and non-symlink",
            "verifier_executable_invalid",
            {"path": str(path)},
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IntegrityError(
            "Kernel verifier executable is not a readable non-symlink file",
            "verifier_executable_invalid",
            {"path": str(path)},
        ) from exc
    try:
        before = os.fstat(descriptor)
        _validate_executable_metadata(path, before)
        digest = _digest_descriptor(descriptor)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _metadata_identity(before) != _metadata_identity(after):
        raise IntegrityError(
            "Kernel verifier executable changed while its identity was frozen",
            "verifier_executable_identity_unstable",
            {"path": str(path)},
        )
    return ExecutableIdentity(
        path=str(path),
        size=before.st_size,
        sha256=digest,
        device=before.st_dev,
        inode=before.st_ino,
        mode=stat.S_IMODE(before.st_mode),
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )


def _validate_executable_metadata(path: Path, metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111 == 0:
        raise IntegrityError(
            "Kernel verifier executable must be an executable regular file",
            "verifier_executable_invalid",
            {"path": str(path)},
        )


def _digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _revalidate_executable(expected: ExecutableIdentity, *, phase: str) -> None:
    try:
        observed = _snapshot_executable(Path(expected.path))
    except IntegrityError as exc:
        raise IntegrityError(
            "Kernel verifier executable disappeared or became invalid during the phase",
            f"verifier_executable_changed_during_{phase}",
            {"expected": expected.to_dict(), "observation_reason": exc.reason_code},
        ) from exc
    if observed != expected:
        raise IntegrityError(
            "Kernel verifier executable identity changed during the phase",
            f"verifier_executable_changed_during_{phase}",
            {"expected": expected.to_dict(), "observed": observed.to_dict()},
        )


def _evidence(
    phase: str, result: ProcessResult, executable: ExecutableIdentity
) -> CommandEvidence:
    if not result.argv or result.argv[0] != executable.path:
        raise IntegrityError(
            "Kernel verifier result does not bind the frozen executable",
            "verifier_executable_evidence_mismatch",
        )
    containment = result.process_containment
    if (
        containment is None
        or not containment.namespace_empty_verified
        or not result.cleanup_succeeded
    ):
        raise IntegrityError(
            "Kernel verifier process tree was not authoritatively contained",
            "verifier_process_containment_failed",
        )
    return CommandEvidence(
        phase=phase,
        argv=result.argv,
        executable_identity=executable,
        executable_identity_reverified=True,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=result.duration_seconds,
        process_containment=containment,
    )


__all__ = [
    "CandidateVerifier",
    "CommandEvidence",
    "ExecutableIdentity",
    "candidate_source_digest",
]
