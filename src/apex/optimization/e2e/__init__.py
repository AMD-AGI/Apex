"""Kernel-only adaptive E2E optimization."""

from .candidate import AgentCandidateWorker, CandidateWorker, E2ECandidate, E2ECandidateRequest
from .context import E2EContextBuilder
from .deferred import E2EDeferredMicroQualifier
from .docker_overlay import DockerOverlayDeployment, OverlayOnlyFinalDelivery
from .kernel_lane import KernelOpportunity, KernelOpportunityPlan, build_kernel_opportunity_plan
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
    build_qwen_acceptance_delivery,
    build_qwen_acceptance_provenance_resolver,
    build_qwen_correctness_oracles,
    build_qwen_oracle_micro_qualifier,
    default_qwen_source_roots,
)
from .use_case import BenchmarkAdapter, E2EOptimizeUseCase, ProvenancePort

__all__ = [
    "AcceptedCandidate",
    "AgentCandidateWorker",
    "BenchmarkAdapter",
    "CandidateDeployment",
    "CandidateDeploymentPort",
    "CandidateDeploymentRequest",
    "CandidateSafetyPort",
    "CandidateWorker",
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
    "FinalDeliveryPort",
    "FinalDeliveryRequest",
    "FinalDeliveryResult",
    "FormalDeliveryBinding",
    "FormalRepositoryProfile",
    "FormalSourceDeliveryProfile",
    "KernelOpportunity",
    "KernelOpportunityPlan",
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
    "ResolvedCorrectnessOracle",
    "DeliveryProvenancePort",
    "SafetyQualification",
    "SafetyQualificationRequest",
    "SourceRebuildFinalDelivery",
    "ExactRequestProvenance",
    "build_e2e_result",
    "build_kernel_opportunity_plan",
    "build_qwen_acceptance_delivery",
    "build_qwen_acceptance_provenance_resolver",
    "build_qwen_correctness_oracles",
    "build_qwen_oracle_micro_qualifier",
    "default_qwen_source_roots",
    "write_e2e_result",
]
