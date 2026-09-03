"""Run-scoped local MCP façade for canonical Apex capabilities."""

from .benchmark import MagpieAcquisitionHandler
from .diagnostics import (
    HotspotRankHandler,
    TraceAnalyzeHandler,
    TraceCompareHandler,
    hotspot_rank_descriptor,
    trace_analyze_descriptor,
    trace_compare_descriptor,
)
from .delivery import BundleVerifyHandler
from .campaign import (
    CampaignCheckpointHandler,
    CampaignResumeHandler,
    CampaignStartHandler,
    CampaignStatusHandler,
    CampaignStopHandler,
)
from .experience import ExperienceRetrieveHandler, experience_retrieve_descriptor
from .evaluator import KernelEvaluatorHandler
from .knowledge import (
    KnowledgeExplainHandler,
    KnowledgeSearchHandler,
    knowledge_explain_descriptor,
    knowledge_search_descriptor,
)
from .registry import CapabilityRegistry
from .grants import (
    CapabilityGrantGate,
    capability_grant_required,
    granted_gpu_selector,
)
from .catalog import planned_capability_descriptors
from .scope import CapabilityScope
from .server import build_low_level_server, run_stdio_server
from .session_grants import KernelDraftSessionGrantAuthority
from .workload import WorkloadInspectHandler, workload_inspect_descriptor

__all__ = [
    "CapabilityRegistry",
    "CapabilityGrantGate",
    "planned_capability_descriptors",
    "CapabilityScope",
    "BundleVerifyHandler",
    "CampaignCheckpointHandler",
    "CampaignResumeHandler",
    "CampaignStartHandler",
    "CampaignStatusHandler",
    "CampaignStopHandler",
    "KnowledgeSearchHandler",
    "KernelEvaluatorHandler",
    "KernelDraftSessionGrantAuthority",
    "KnowledgeExplainHandler",
    "ExperienceRetrieveHandler",
    "HotspotRankHandler",
    "MagpieAcquisitionHandler",
    "TraceAnalyzeHandler",
    "TraceCompareHandler",
    "WorkloadInspectHandler",
    "build_low_level_server",
    "capability_grant_required",
    "granted_gpu_selector",
    "knowledge_search_descriptor",
    "knowledge_explain_descriptor",
    "experience_retrieve_descriptor",
    "hotspot_rank_descriptor",
    "run_stdio_server",
    "trace_analyze_descriptor",
    "trace_compare_descriptor",
    "workload_inspect_descriptor",
]
