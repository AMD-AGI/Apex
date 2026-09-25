"""Kernel-only adaptive E2E optimization."""

from .candidate import AgentCandidateWorker, CandidateWorker, E2ECandidate, E2ECandidateRequest
from .component_deployment import (
    CandidateDeploymentRegistry,
    ComponentDeploymentBinding,
)
from .component_micro import ComponentMicroBinding, ComponentMicroQualifierRegistry
from .context import E2EContextBuilder
from .deferred import E2EDeferredMicroQualifier
from .docker_overlay import DockerOverlayDeployment, OverlayOnlyFinalDelivery
from .kernel_lane import (
    KernelOpportunity,
    KernelOpportunityPlan,
    KernelPlanningCoverage,
    build_kernel_opportunity_plan,
)
from .oracles import (
    CorrectnessOracleBinding,
    CorrectnessOracleRegistry,
    ResolvedCorrectnessOracle,
)
from .oracle_preflight import (
    DockerOracleMicroQualifier,
    DockerOraclePolicy,
    OracleDependencyLock,
    OracleSourceLock,
)
from .result import E2EOptimizationResult, build_e2e_result, write_e2e_result
from .services import (
    AcceptedCandidate,
    CandidateDeployment,
    CandidateDeploymentPort,
    CandidateDeploymentRequest,
    CandidateSafetyPort,
    FinalDeliveryPort,
    FinalDeliveryRequest,
    FinalDeliveryResult,
    MicroQualification,
    MicroQualificationPort,
    MicroQualificationRequest,
    SafetyQualification,
    SafetyQualificationRequest,
)
from .source_delivery import FormalDeliveryBinding, SourceRebuildFinalDelivery
from .source_delivery_models import (
    DeliveryProvenancePort,
    FormalRepositoryProfile,
    FormalSourceDeliveryProfile,
    PrimarySourceBuildOutput,
    PrimarySourceBuildPort,
    PrimarySourceBuildRequest,
)
from .source_delivery_provenance import ExactRequestProvenance
from .qwen_profile import (
    QWEN_ACCEPTANCE_PROFILE_ID,
    build_qwen_acceptance_bundle_verifier,
    build_qwen_acceptance_delivery,
    build_qwen_acceptance_provenance_resolver,
    build_qwen_correctness_oracles,
    build_qwen_oracle_micro_qualifier,
    default_qwen_source_roots,
    qwen_acceptance_recipe_sha256s,
)
from .preflight import E2EPreflightResult, write_preflight_result
from .use_case import BenchmarkAdapter, E2EOptimizeUseCase, ProvenancePort

__all__ = [
    "AcceptedCandidate",
    "AgentCandidateWorker",
    "BenchmarkAdapter",
    "CandidateDeployment",
    "CandidateDeploymentRegistry",
    "CandidateDeploymentPort",
    "CandidateDeploymentRequest",
    "CandidateSafetyPort",
    "CandidateWorker",
    "ComponentDeploymentBinding",
    "ComponentMicroBinding",
    "ComponentMicroQualifierRegistry",
    "CorrectnessOracleBinding",
    "CorrectnessOracleRegistry",
    "DockerOverlayDeployment",
    "DockerOracleMicroQualifier",
    "DockerOraclePolicy",
    "E2ECandidate",
    "E2ECandidateRequest",
    "E2EContextBuilder",
    "E2EDeferredMicroQualifier",
    "E2EOptimizationResult",
    "E2EOptimizeUseCase",
    "E2EPreflightResult",
    "FinalDeliveryPort",
    "FinalDeliveryRequest",
    "FinalDeliveryResult",
    "FormalDeliveryBinding",
    "FormalRepositoryProfile",
    "FormalSourceDeliveryProfile",
    "KernelOpportunity",
    "KernelOpportunityPlan",
    "KernelPlanningCoverage",
    "MicroQualification",
    "MicroQualificationPort",
    "MicroQualificationRequest",
    "OverlayOnlyFinalDelivery",
    "OracleDependencyLock",
    "OracleSourceLock",
    "PrimarySourceBuildOutput",
    "PrimarySourceBuildPort",
    "PrimarySourceBuildRequest",
    "ProvenancePort",
    "QWEN_ACCEPTANCE_PROFILE_ID",
    "ResolvedCorrectnessOracle",
    "DeliveryProvenancePort",
    "SafetyQualification",
    "SafetyQualificationRequest",
    "SourceRebuildFinalDelivery",
    "ExactRequestProvenance",
    "build_e2e_result",
    "build_kernel_opportunity_plan",
    "build_qwen_acceptance_bundle_verifier",
    "build_qwen_acceptance_delivery",
    "build_qwen_acceptance_provenance_resolver",
    "build_qwen_correctness_oracles",
    "build_qwen_oracle_micro_qualifier",
    "default_qwen_source_roots",
    "qwen_acceptance_recipe_sha256s",
    "write_e2e_result",
    "write_preflight_result",
]
