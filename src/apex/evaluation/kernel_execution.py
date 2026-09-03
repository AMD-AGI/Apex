"""Evaluator-authored provenance for standalone kernel measurement execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from apex.core import ContractError, canonical_json_bytes, sha256_bytes


EXECUTION_RECEIPT_SCHEMA = "apex.kernel-measurement-execution/v1"


@dataclass(frozen=True, slots=True)
class KernelMeasurementExecutionReceipt:
    """Seal writer, phase, timing, frozen inputs, and the raw report identity."""

    run_id: str
    attempt_id: str
    writer_id: str
    candidate_source_sha256: str
    harness_sha256: str
    measurement_method_sha256: str
    measurement_policy_sha256: str
    report_sha256: str
    report_size: int
    phase_started_monotonic_ns: int
    adapter_returned_monotonic_ns: int
    output_observed_monotonic_ns: int
    phase_completed_monotonic_ns: int
    schema: str = EXECUTION_RECEIPT_SCHEMA
    phase: str = "measurement"
    writer_kind: str = "trusted_evaluator_adapter"

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_RECEIPT_SCHEMA or self.phase != "measurement":
            raise ContractError(
                "Kernel measurement execution phase is invalid",
                "invalid_measurement_execution_receipt",
            )
        if self.writer_kind != "trusted_evaluator_adapter" or not all(
            (self.run_id, self.attempt_id, self.writer_id)
        ):
            raise ContractError(
                "Kernel measurement writer identity is invalid",
                "invalid_measurement_execution_receipt",
            )
        for value in (
            self.candidate_source_sha256,
            self.harness_sha256,
            self.measurement_method_sha256,
            self.measurement_policy_sha256,
            self.report_sha256,
        ):
            _digest(value)
        timeline = (
            self.phase_started_monotonic_ns,
            self.adapter_returned_monotonic_ns,
            self.output_observed_monotonic_ns,
            self.phase_completed_monotonic_ns,
        )
        if self.report_size <= 0 or any(value <= 0 for value in timeline):
            raise ContractError(
                "Kernel measurement execution receipt is incomplete",
                "invalid_measurement_execution_receipt",
            )
        if timeline != tuple(sorted(timeline)):
            raise ContractError(
                "Kernel measurement execution timeline is invalid",
                "invalid_measurement_execution_timeline",
            )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "writer_kind": self.writer_kind,
            "writer_id": self.writer_id,
            "phase": self.phase,
            "candidate_source_sha256": self.candidate_source_sha256,
            "harness_sha256": self.harness_sha256,
            "measurement_method_sha256": self.measurement_method_sha256,
            "measurement_policy_sha256": self.measurement_policy_sha256,
            "report_sha256": self.report_sha256,
            "report_size": self.report_size,
            "phase_started_monotonic_ns": self.phase_started_monotonic_ns,
            "adapter_returned_monotonic_ns": self.adapter_returned_monotonic_ns,
            "output_observed_monotonic_ns": self.output_observed_monotonic_ns,
            "phase_completed_monotonic_ns": self.phase_completed_monotonic_ns,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "KernelMeasurementExecutionReceipt":
        expected = {
            "schema", "run_id", "attempt_id", "writer_kind", "writer_id",
            "phase", "candidate_source_sha256", "harness_sha256",
            "measurement_method_sha256", "measurement_policy_sha256",
            "report_sha256", "report_size", "phase_started_monotonic_ns",
            "adapter_returned_monotonic_ns", "output_observed_monotonic_ns",
            "phase_completed_monotonic_ns",
        }
        if set(value) != expected:
            raise ContractError(
                "Kernel measurement execution receipt fields are invalid",
                "invalid_measurement_execution_receipt",
            )
        try:
            return cls(
                schema=str(value["schema"]),
                run_id=str(value["run_id"]),
                attempt_id=str(value["attempt_id"]),
                writer_kind=str(value["writer_kind"]),
                writer_id=str(value["writer_id"]),
                phase=str(value["phase"]),
                candidate_source_sha256=str(value["candidate_source_sha256"]),
                harness_sha256=str(value["harness_sha256"]),
                measurement_method_sha256=str(value["measurement_method_sha256"]),
                measurement_policy_sha256=str(value["measurement_policy_sha256"]),
                report_sha256=str(value["report_sha256"]),
                report_size=int(value["report_size"]),
                phase_started_monotonic_ns=int(value["phase_started_monotonic_ns"]),
                adapter_returned_monotonic_ns=int(value["adapter_returned_monotonic_ns"]),
                output_observed_monotonic_ns=int(value["output_observed_monotonic_ns"]),
                phase_completed_monotonic_ns=int(value["phase_completed_monotonic_ns"]),
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "Kernel measurement execution receipt cannot be decoded",
                "invalid_measurement_execution_receipt",
            ) from error


def _digest(value: str) -> None:
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ContractError(
            "Kernel measurement execution digest is invalid",
            "invalid_measurement_execution_receipt",
        )


__all__ = ["EXECUTION_RECEIPT_SCHEMA", "KernelMeasurementExecutionReceipt"]
