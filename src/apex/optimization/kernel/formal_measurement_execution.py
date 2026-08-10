"""Lease-bracketed execution of one formal kernel measurement capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apex.core import ApexError
from apex.ports import KernelMeasurementPort
from apex.runtime import GpuLeaseManager, require_gpu_measurement_guard
from apex.storage import ArtifactReceipt

from .formal_campaign import FormalKernelCampaign
from .gpu_recording import record_gpu_measurement_bracket
from .measurement import KernelMeasurementCapture, capture_kernel_measurement
from .verification import CandidateVerifier


@dataclass(frozen=True, slots=True)
class FormalMeasurementExecution:
    capture: KernelMeasurementCapture | None
    reason_code: str | None
    performance_receipt: ArtifactReceipt
    bracket_receipt: ArtifactReceipt


def execute_formal_measurement(
    campaign: FormalKernelCampaign,
    *,
    projection: Any,
    attempt_id: str,
    candidate_digest: str,
    requested_devices: str | None,
    gpu_leases: GpuLeaseManager,
    verifier: CandidateVerifier,
    evaluator: KernelMeasurementPort,
) -> FormalMeasurementExecution:
    """Capture only while the same live lease brackets the whole timing phase."""

    capture = None
    reason = None
    with gpu_leases.acquire(
        campaign.record.run_id, requested_devices=requested_devices
    ) as lease:
        campaign.record.record_gpu_lease(
            lease.receipt, attempt_id=attempt_id, phase="measurement"
        )
        with require_gpu_measurement_guard(lease, attempt_id) as bracket:
            performance = verifier.performance(
                projection.resolved,
                candidate_root=projection.root,
                expected_source_digest=candidate_digest,
            )
            performance_receipt = campaign.record.record_command(
                attempt_id, performance
            )
            if not performance.passed:
                reason = "performance_command_failed"
            else:
                try:
                    capture = capture_kernel_measurement(
                        projection.resolved,
                        candidate_root=projection.root,
                        run_id=campaign.record.run_id,
                        attempt_id=attempt_id,
                        output_root=(
                            campaign.record.root / "measurements" / attempt_id
                        ),
                        evaluator=evaluator,
                    )
                except ApexError as error:
                    reason = error.reason_code
        bracket_receipt = record_gpu_measurement_bracket(
            campaign.record, bracket.receipt, attempt_id=attempt_id
        )
    return FormalMeasurementExecution(
        capture, reason, performance_receipt, bracket_receipt
    )


__all__ = ["FormalMeasurementExecution", "execute_formal_measurement"]
