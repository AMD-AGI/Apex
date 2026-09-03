"""Magpie benchmark boundary and immutable phase views."""

from .config_validation import validate_phase_set_contract
from .config_views import (
    BenchmarkConfigViews,
    TraceLensBinding,
    build_config_views,
    validate_resolved_view,
)
from .magpie import MagpieBenchmarkAdapter
from .docker_magpie_attestor import DockerOneShotMagpieExecutionAttestor
from .execution_attestor_registry import MagpieExecutionAttestorRegistry
from apex.runtime.magpie_config import (
    CAPABILITY_SCHEMA,
    PLAN_SCHEMA,
    REQUIRED_REWARD_METRICS,
    RESULT_SCHEMA,
    MagpieConfigContract,
    MagpieMainConfigAdapter,
    validate_apex_magpie_config_documents,
)
from .evaluator_policy import (
    EvaluatorPolicy,
    evaluator_policy_from_scoring,
)
from .inferencex_runtime import (
    InferenceXRuntimeEvidence,
    parse_inferencex_runtime_evidence,
)
from .model_revision import ModelRevisionEvidence, parse_model_revision_evidence
from .lm_eval_runtime import (
    LmEvalRuntimeEvidence,
    parse_lm_eval_runtime_evidence,
)
from .local_runtime import LocalRuntimeEvidence, parse_local_runtime_evidence
from .magpie_attestation import (
    MagpieExecutionAttestation,
    UnavailableMagpieExecutionAttestor,
    expected_attestation_path,
    load_magpie_execution_attestation,
    validate_magpie_execution_attestation_document,
)
from .serving_runtime import (
    ServingRuntimeEvidence,
    parse_serving_runtime_evidence,
)
from .results import (
    LatencyDistribution,
    LatencyMetrics,
    NormalizedBenchmarkResult,
    QualityEvidence,
    QualityMetric,
    ThroughputMetrics,
    parse_benchmark_report,
)

__all__ = [
    "BenchmarkConfigViews",
    "EvaluatorPolicy",
    "LatencyDistribution",
    "LatencyMetrics",
    "MagpieBenchmarkAdapter",
    "DockerOneShotMagpieExecutionAttestor",
    "MagpieExecutionAttestorRegistry",
    "MagpieConfigContract",
    "MagpieMainConfigAdapter",
    "MagpieExecutionAttestation",
    "UnavailableMagpieExecutionAttestor",
    "PLAN_SCHEMA",
    "CAPABILITY_SCHEMA",
    "RESULT_SCHEMA",
    "REQUIRED_REWARD_METRICS",
    "validate_apex_magpie_config_documents",
    "InferenceXRuntimeEvidence",
    "LmEvalRuntimeEvidence",
    "LocalRuntimeEvidence",
    "ModelRevisionEvidence",
    "ServingRuntimeEvidence",
    "NormalizedBenchmarkResult",
    "QualityEvidence",
    "QualityMetric",
    "ThroughputMetrics",
    "TraceLensBinding",
    "build_config_views",
    "parse_benchmark_report",
    "parse_inferencex_runtime_evidence",
    "parse_lm_eval_runtime_evidence",
    "parse_local_runtime_evidence",
    "expected_attestation_path",
    "load_magpie_execution_attestation",
    "validate_magpie_execution_attestation_document",
    "parse_model_revision_evidence",
    "parse_serving_runtime_evidence",
    "evaluator_policy_from_scoring",
    "validate_phase_set_contract",
    "validate_resolved_view",
]
