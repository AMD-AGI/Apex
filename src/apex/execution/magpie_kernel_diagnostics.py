"""Pinned Magpie analyze/compare adapter for advisory kernel diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_file
from apex.ports import (
    KernelDiagnosticOutput,
    KernelDiagnosticRequest,
)

from .environment import GPU_RUNTIME_ENVIRONMENT_KEYS, build_subprocess_environment
from .supervisor import SubprocessSupervisor

MAGPIE_KERNEL_DIAGNOSTICS_ADAPTER_ID = "magpie-kernel-diagnostics-v1"
_MAX_OUTPUT_BYTES = 16 * 1024 * 1024


class _DependencyReceipt(Protocol):
    python: Path
    lock_sha256: str
    commits: Mapping[str, str]

    def root(self, name: str) -> Path: ...


class MagpieKernelDiagnosticsAdapter:
    """Run pinned Magpie after formal timing and return non-reward evidence."""

    adapter_id = MAGPIE_KERNEL_DIAGNOSTICS_ADAPTER_ID

    def __init__(
        self,
        dependency_receipt: Callable[[], _DependencyReceipt],
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._dependency_receipt = dependency_receipt
        self._verified_receipt: _DependencyReceipt | None = None
        self._supervisor = supervisor or SubprocessSupervisor(
            max_output_bytes=_MAX_OUTPUT_BYTES
        )

    def run(self, request: KernelDiagnosticRequest) -> KernelDiagnosticOutput:
        receipt = self._verified_receipt
        if receipt is None:
            receipt = self._dependency_receipt()
            self._verified_receipt = receipt
        python = receipt.python.resolve(strict=False)
        magpie_root = receipt.root("magpie").resolve(strict=True)
        output_root = request.output_root
        if output_root.exists():
            raise IntegrityError(
                "Magpie output already exists", "stale_kernel_diagnostic"
            )
        output_root.mkdir(parents=True, mode=0o700)
        config_path = output_root / "kernel_config.json"
        config_path.write_bytes(canonical_json_bytes(_config_document(request)))
        environment = build_subprocess_environment(
            {}, inherit=GPU_RUNTIME_ENVIRONMENT_KEYS
        )
        environment["PYTHONPATH"] = str(magpie_root.parent)
        argv = (
            str(python),
            "-m",
            "Magpie",
            request.mode,
            "--kernel-config",
            str(config_path),
            "--output-dir",
            str(output_root),
            *(("--baseline", "0") if request.mode == "compare" else ()),
        )
        process = self._supervisor.run(
            argv,
            cwd=magpie_root,
            environment=environment,
            timeout_seconds=request.timeout_seconds,
            require_pid_namespace=True,
        )
        _require_success(process)
        report_path = _single_report(output_root, request.mode)
        _load_report(report_path)
        execution = {
            "schema": "apex.magpie-kernel-diagnostic-execution/v1",
            "adapter_id": self.adapter_id,
            "mode": request.mode,
            "magpie_commit": receipt.commits["magpie"],
            "dependency_lock_sha256": receipt.lock_sha256,
            "config_sha256": sha256_file(config_path),
            "report_sha256": sha256_file(report_path),
            "exit_code": process.exit_code,
            "duration_seconds": process.duration_seconds,
            "process_containment": process.process_containment.to_dict(),
            "evidence_class": "diagnostic",
            "reward_eligible": False,
        }
        return KernelDiagnosticOutput(
            self.adapter_id, request.mode, report_path, config_path, execution
        )


def _config_document(request: KernelDiagnosticRequest) -> dict[str, object]:
    roots = (
        (("baseline", request.baseline_root), ("candidate", request.candidate_root))
        if request.mode == "compare"
        else (("candidate", request.candidate_root),)
    )
    kernels = [
        _kernel_entry(label, root, request) for label, root in roots if root is not None
    ]
    return {
        "kernels": kernels,
        "correctness": {"backend": "testcase"},
        "performance": {"timeout_seconds": request.timeout_seconds},
        "scheduler": {"environment": "local", "max_workers": 1},
    }


def _kernel_entry(
    label: str, root: Path, request: KernelDiagnosticRequest
) -> dict[str, object]:
    commands = (request.compile, request.correctness, request.performance)
    if len({command.cwd for command in commands}) != 1:
        raise ContractError(
            "Magpie requires one working directory for all kernel phases",
            "magpie_command_context_mismatch",
        )
    if any(command.env != commands[0].env for command in commands[1:]):
        raise ContractError(
            "Magpie requires one environment for all kernel phases",
            "magpie_command_context_mismatch",
        )
    cwd = root if commands[0].cwd == "." else root.joinpath(*commands[0].cwd.split("/"))
    return {
        "id": label,
        "type": "pytorch" if request.kernel_type == "python" else request.kernel_type,
        "source_files": [
            str(root.joinpath(*path.split("/"))) for path in request.source_files
        ],
        "working_dir": str(cwd),
        "env": dict(commands[0].env),
        "compile_command": list(request.compile.argv),
        "testcase_command": list(request.correctness.argv),
        "prof_command": list(request.performance.argv),
    }


def _require_success(process) -> None:
    containment = process.process_containment
    if (
        containment is None
        or not containment.namespace_empty_verified
        or not process.cleanup_succeeded
    ):
        raise IntegrityError(
            "Magpie process was not contained", "magpie_diagnostic_containment_failed"
        )
    if (
        process.timed_out
        or process.exit_code != 0
        or process.stdout_truncated
        or process.stderr_truncated
    ):
        raise ContractError(
            "Magpie diagnostic failed",
            "magpie_diagnostic_failed",
            {"exit_code": process.exit_code, "timed_out": process.timed_out},
        )


def _single_report(root: Path, mode: str) -> Path:
    matches = tuple(root.glob(f"**/{mode}_report.json"))
    if len(matches) != 1:
        raise ContractError("Magpie did not emit one report", "magpie_report_missing")
    return matches[0]


def _load_report(path: Path) -> object:
    if path.stat().st_size > _MAX_OUTPUT_BYTES:
        raise ContractError("Magpie report is too large", "magpie_report_invalid")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(
            "Magpie report is invalid", "magpie_report_invalid"
        ) from error
    if not isinstance(value, dict):
        raise ContractError("Magpie report is not an object", "magpie_report_invalid")
    return value


__all__ = ["MAGPIE_KERNEL_DIAGNOSTICS_ADAPTER_ID", "MagpieKernelDiagnosticsAdapter"]
