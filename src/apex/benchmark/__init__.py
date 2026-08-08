"""Magpie benchmark boundary and immutable phase views."""

from .config_validation import validate_phase_set_contract
from .config_views import (
    BenchmarkConfigViews,
    TraceLensBinding,
    build_config_views,
    validate_resolved_view,
)
from .magpie import MagpieBenchmarkAdapter
from .evaluator_policy import (
    EvaluatorPolicy,
    QWEN_CONFIG_SHA256,
    qwen_evaluator_policy,
    resolve_evaluator_policy,
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
    "InferenceXRuntimeEvidence",
    "LmEvalRuntimeEvidence",
    "ModelRevisionEvidence",
    "NormalizedBenchmarkResult",
    "QualityEvidence",
    "QualityMetric",
    "QWEN_CONFIG_SHA256",
    "ThroughputMetrics",
    "TraceLensBinding",
    "build_config_views",
    "parse_benchmark_report",
    "parse_inferencex_runtime_evidence",
    "parse_lm_eval_runtime_evidence",
    "parse_model_revision_evidence",
    "resolve_evaluator_policy",
    "qwen_evaluator_policy",
    "validate_phase_set_contract",
    "validate_resolved_view",
]
