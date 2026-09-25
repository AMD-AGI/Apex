"""Exact provenance lock construction from evaluator-owned run identities."""

from __future__ import annotations

from apex.core import ContractError, sha256_file
from apex.delivery import BundleProvenanceLock

from .services import FinalDeliveryRequest


class ExactRequestProvenance:
    """Fail closed unless the finalization request contains exact identities."""

    def lock(self, request: FinalDeliveryRequest) -> BundleProvenanceLock:
        provenance = request.provenance
        if (
            provenance.container.image_id is None
            or provenance.model_revision is None
            or request.agent_backend is None
            or request.agent_model is None
            or request.accuracy_policy_sha256 is None
            or request.performance_policy_sha256 is None
            or request.safety_policy_sha256 is None
        ):
            raise ContractError(
                "Formal delivery provenance is incomplete",
                "source_provenance_unresolved",
            )
        if (
            sha256_file(request.benchmark_original)
            != provenance.benchmark_config_sha256
            or request.baseline.protocol_hash != request.final.protocol_hash
        ):
            raise ContractError(
                "Benchmark provenance changed before delivery",
                "benchmark_provenance_mismatch",
            )
        return BundleProvenanceLock(
            primary_run_id=request.run_id,
            framework=provenance.framework,
            model_id=provenance.model_id,
            model_revision=provenance.model_revision,
            gpu_arch=provenance.gpu_arch,
            baseline_image_digest=provenance.container.image_id,
            original_config_sha256=provenance.benchmark_config_sha256,
            workload_semantics_sha256=request.baseline.protocol_hash,
            accuracy_policy_sha256=request.accuracy_policy_sha256,
            performance_policy_sha256=request.performance_policy_sha256,
            safety_policy_sha256=request.safety_policy_sha256,
            agent_backend=request.agent_backend,
            agent_model=request.agent_model,
        )


__all__ = ["ExactRequestProvenance"]
