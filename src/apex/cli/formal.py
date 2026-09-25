"""Shared pure overrides and exit semantics for formal CLI commands."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from apex.core import ApexError, TaskStatus
from apex.intake import TaskSpec
from apex.runtime import formal_results_validator


def kernel_budget_overrides(task: TaskSpec, args) -> TaskSpec:
    values = {
        "max_iterations": args.max_iterations,
        "max_turns": args.max_turns,
        "timeout_seconds": args.timeout_seconds,
    }
    selected = {key: value for key, value in values.items() if value is not None}
    return replace(task, budget=replace(task.budget, **selected)) if selected else task


def status_exit_code(status: TaskStatus) -> int:
    if status in {TaskStatus.CANDIDATE_READY, TaskStatus.SUCCEEDED, TaskStatus.NO_GAIN}:
        return 0
    if status in {TaskStatus.NEEDS_INPUT, TaskStatus.INVALID_REQUEST, TaskStatus.UNSUPPORTED}:
        return 2
    if status in {TaskStatus.REJECTED, TaskStatus.NO_MEASUREMENT, TaskStatus.VERIFICATION_FAILED}:
        return 3
    if status is TaskStatus.TIMEOUT:
        return 124
    return 1


def formal_results_root(path: Path, *, workspace: Path | None = None) -> Path:
    """Validate a live evidence root against source checkouts known to the CLI."""

    roots = (workspace,) if workspace is not None else ()
    apex_root = Path(__file__).resolve().parents[3]
    return formal_results_validator(
        apex_root=apex_root,
        workspace_roots=roots,
    ).validate(path.expanduser())


def formal_result_path(path: Path, results_root: Path) -> Path:
    """Confine the machine result to its validated formal root."""

    root = formal_results_root(results_root)
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ApexError("Formal result path must be absolute", "formal_result_not_absolute")
    if selected.is_symlink():
        raise ApexError("Formal result path cannot be a symlink", "unsafe_formal_result")
    resolved = selected.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ApexError(
            "Formal result path must be inside the formal results root",
            "formal_result_outside_results",
        ) from error
    return resolved


def regular_e2e_config(value: Path) -> Path:
    """Resolve one non-linked regular Magpie input config."""

    selected = value.expanduser()
    if selected.is_symlink():
        raise ApexError("E2E config cannot be a symlink", "unsafe_e2e_config")
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise ApexError("E2E config does not exist", "e2e_config_missing") from error
    if not resolved.is_file():
        raise ApexError("E2E config must be a regular file", "invalid_e2e_config")
    return resolved


__all__ = [
    "formal_result_path",
    "formal_results_root",
    "kernel_budget_overrides",
    "regular_e2e_config",
    "status_exit_code",
]
