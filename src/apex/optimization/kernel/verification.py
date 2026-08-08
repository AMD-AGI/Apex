"""Evaluator-owned phased compile, correctness, and normal-performance execution."""

from __future__ import annotations

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
class CommandEvidence:
    phase: str
    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_seconds: float
    process_containment: AgentProcessContainmentReceipt

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"unsupported kernel verification phase: {self.phase}")

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "argv": list(self.argv),
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
        )
        cwd = candidate_root if command.cwd == "." else candidate_root.joinpath(*command.cwd.split("/"))
        process = self._supervisor.run(
            command.argv,
            cwd=cwd.resolve(strict=True),
            environment=environment,
            timeout_seconds=command.timeout_seconds,
            require_pid_namespace=True,
        )
        return _evidence(phase, process)


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


def _evidence(phase: str, result: ProcessResult) -> CommandEvidence:
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
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_seconds=result.duration_seconds,
        process_containment=containment,
    )


__all__ = ["CandidateVerifier", "CommandEvidence", "candidate_source_digest"]
