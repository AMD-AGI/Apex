"""Honest E2E-deferred qualification when no trusted micro harness exists."""

from __future__ import annotations

from .kernel_lane import KernelOpportunity
from .services import MicroQualification, MicroQualificationRequest


class E2EDeferredMicroQualifier:
    """Promote frozen source only to the unchanged Magpie E2E authority.

    This adapter deliberately emits no compile, correctness, timing, or kernel
    reward claim.  The search budget bounds how many deferred candidates reach
    the expensive quality-plus-throughput gate.
    """

    qualification_mode = "e2e_quality_deferred"

    def supports(self, opportunity: KernelOpportunity) -> bool:
        return (
            opportunity.eligible
            and opportunity.language in {"python", "triton"}
            and opportunity.origin_library in {"vllm", "aiter"}
        )

    def verify(self, request: MicroQualificationRequest) -> MicroQualification:
        candidate = request.candidate
        integrity = bool(
            candidate.succeeded
            and candidate.candidate_id
            and candidate.candidate_source_sha256
            and len(candidate.changed_files) == 1
            and candidate.changed_files == candidate.editable_files
            and candidate.changed_files[0].endswith(".py")
        )
        deferred = integrity
        return MicroQualification(
            candidate_id=candidate.candidate_id or candidate.attempt_id,
            grade=None,
            evidence={
                "schema_version": 1,
                "qualification_mode": "e2e_quality_deferred",
                "kernel_reward": {
                    "available": False,
                    "reason_code": "trusted_micro_harness_unavailable",
                },
                "claims": {
                    "compiled": "unmeasured",
                    "correct": "unmeasured",
                    "p50": "unmeasured",
                    "p99": "unmeasured",
                },
                "promotion_authority": {
                    "correctness": "unchanged_magpie_quality_gate",
                    "performance": "unchanged_magpie_e2e_measurement",
                },
                "anchor_generation": request.anchor_generation,
            },
            qualification_mode="e2e_quality_deferred",
            deferred_candidate_valid=deferred,
        )


__all__ = ["E2EDeferredMicroQualifier"]
