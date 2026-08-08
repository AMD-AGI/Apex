"""Protocol boundaries between Apex domain logic and concrete adapters."""

from .agent import (
    AgentBackend,
    AgentCaptureStatus,
    AgentCost,
    AgentInvocationReceipt,
    AgentRequest,
    AgentResult,
    AgentSemanticEvent,
    AgentTranscriptEvent,
    AgentTerminationKind,
    AgentUsage,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)
from .agent_containment import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentProcessContainmentReceipt,
)
from .benchmark import BenchmarkPass, BenchmarkPort, BenchmarkRequest, BenchmarkResult
from .diagnostics import (
    DiagnosticsPort,
    DiagnosticsRequest,
    DiagnosticsResult,
    TraceComparisonArtifact,
    TraceComparisonPort,
    TraceComparisonRequest,
    TraceComparisonResult,
    TraceComparisonStatus,
    TraceDiagnosticEvidence,
)
from .knowledge import KnowledgePort, KnowledgeQuery, KnowledgeResult
from .kernel_measurement import (
    KernelMeasurementOutput,
    KernelMeasurementPort,
    KernelMeasurementRequest,
)
from .safety import (
    SafetyToolRunRequest,
    SafetyToolRunResult,
    SafetyToolRunner,
    SafetyVerificationPort,
)

__all__ = [
    "AGENT_PROCESS_CONTAINMENT_POLICY",
    "AgentBackend",
    "AgentCaptureStatus",
    "AgentCost",
    "AgentInvocationReceipt",
    "AgentProcessContainmentReceipt",
    "AgentRequest",
    "AgentResult",
    "AgentSemanticEvent",
    "AgentTranscriptEvent",
    "AgentTerminationKind",
    "AgentUsage",
    "BenchmarkPass",
    "BenchmarkPort",
    "BenchmarkRequest",
    "BenchmarkResult",
    "DiagnosticsPort",
    "DiagnosticsRequest",
    "DiagnosticsResult",
    "TraceComparisonArtifact",
    "TraceComparisonPort",
    "TraceComparisonRequest",
    "TraceComparisonResult",
    "TraceComparisonStatus",
    "TraceDiagnosticEvidence",
    "KnowledgePort",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KernelMeasurementOutput",
    "KernelMeasurementPort",
    "KernelMeasurementRequest",
    "SafetyToolRunRequest",
    "SafetyToolRunResult",
    "SafetyToolRunner",
    "SafetyVerificationPort",
    "STRUCTURED_TURN_CHECKPOINT_POLICY",
]
