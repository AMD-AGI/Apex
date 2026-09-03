"""Trusted evaluator boundary for standalone kernel raw measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError


@dataclass(frozen=True, slots=True)
class KernelMeasurementRequest:
    """Frozen inputs and an evaluator-only destination for one measurement phase."""

    run_id: str
    attempt_id: str
    adapter_id: str
    candidate_root: Path
    report_path: Path
    harness_paths: tuple[Path, ...]
    runner_argv: tuple[str, ...]
    runner_cwd: Path
    runner_env: Mapping[str, str]
    runner_timeout_seconds: int
    candidate_source_sha256: str
    harness_sha256: str
    measurement_method_sha256: str
    measurement_policy_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.attempt_id or not self.adapter_id:
            raise ContractError(
                "Kernel measurement request identity is incomplete",
                "invalid_kernel_measurement_request",
            )
        if not self.candidate_root.is_absolute() or not self.report_path.is_absolute():
            raise ContractError(
                "Kernel measurement paths must be absolute",
                "invalid_kernel_measurement_request",
            )
        if self.report_path.is_relative_to(self.candidate_root):
            raise ContractError(
                "Evaluator measurement output must be outside the candidate workspace",
                "candidate_visible_measurement_output",
            )
        if not self.harness_paths or any(
            not path.is_absolute() or not path.is_relative_to(self.candidate_root)
            for path in self.harness_paths
        ):
            raise ContractError(
                "Kernel measurement harness paths are invalid",
                "invalid_kernel_measurement_request",
            )
        if (
            not self.runner_argv
            or any(
                not isinstance(value, str) or not value
                for value in self.runner_argv
            )
            or not self.runner_cwd.is_absolute()
            or not self.runner_cwd.is_relative_to(self.candidate_root)
            or self.runner_timeout_seconds <= 0
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in self.runner_env.items()
            )
        ):
            raise ContractError(
                "Kernel measurement runner contract is invalid",
                "invalid_kernel_measurement_request",
            )
        private_path = str(self.report_path)
        if any(private_path in value for value in self.runner_argv) or any(
            private_path in value for value in self.runner_env.values()
        ):
            raise ContractError(
                "Evaluator output path must not cross into the measurement runner",
                "candidate_visible_measurement_output",
            )


@dataclass(frozen=True, slots=True)
class KernelMeasurementOutput:
    """Raw report locator returned by the trusted adapter, without grading authority."""

    writer_id: str
    report_path: Path


class KernelMeasurementPort(Protocol):
    """A trusted adapter which keeps its evaluator output channel from candidate code."""

    @property
    def adapter_id(self) -> str: ...

    @property
    def measurement_method_sha256(self) -> str: ...

    def measure(self, request: KernelMeasurementRequest) -> KernelMeasurementOutput: ...


__all__ = [
    "KernelMeasurementOutput",
    "KernelMeasurementPort",
    "KernelMeasurementRequest",
]
