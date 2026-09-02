"""The sole composition root for concrete Apex adapters and use cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from apex.benchmark import DockerOneShotMagpieExecutionAttestor, MagpieBenchmarkAdapter
from apex.delivery import (
    E2EBundleVerificationPort,
    E2EBundleVerifierRouter,
    E2EVerifierProfile,
)
from apex.core import ContractError
from apex.diagnostics import MagpieTraceEvidenceAdapter, PinnedTraceLensComparisonAdapter
from apex.evaluation import EvaluationContractAuthorizer
from apex.execution import (
    KernelTemplateMaterializer,
    KernelSkillPackage,
    MagpieKernelDiagnosticsAdapter,
    NativeBackendDoctor,
    NativeCodingSessionLauncher,
    StructuredKernelMeasurementAdapter,
    build_default_registry,
    load_kernel_skill_package,
)
from apex.knowledge import KnowledgeRetriever, load_knowledge_catalog
from apex.mcp import (
    BundleVerifyHandler,
    CampaignCheckpointHandler,
    CampaignResumeHandler,
    CampaignStartHandler,
    CampaignStatusHandler,
    CampaignStopHandler,
    CapabilityRegistry,
    CapabilityScope,
    KnowledgeSearchHandler,
    KnowledgeExplainHandler,
    KernelEvaluatorHandler,
    ExperienceRetrieveHandler,
    HotspotRankHandler,
    MagpieAcquisitionHandler,
    TraceAnalyzeHandler,
    TraceCompareHandler,
    WorkloadInspectHandler,
    knowledge_search_descriptor,
    knowledge_explain_descriptor,
    experience_retrieve_descriptor,
    hotspot_rank_descriptor,
    trace_analyze_descriptor,
    trace_compare_descriptor,
    workload_inspect_descriptor,
    planned_capability_descriptors,
)
from apex.optimization.e2e import (
    AgentCandidateWorker,
    CandidateDeploymentRegistry,
    ComponentDeploymentBinding,
    DockerOverlayDeployment,
    E2EContextBuilder,
    E2EDeferredMicroQualifier,
    E2EOptimizeUseCase,
    FinalDeliveryPort,
    QWEN_ACCEPTANCE_PROFILE_ID,
    build_qwen_acceptance_bundle_verifier,
    build_qwen_acceptance_delivery,
    build_qwen_acceptance_provenance_resolver,
    build_qwen_correctness_oracles,
    build_qwen_oracle_micro_qualifier,
    qwen_acceptance_recipe_sha256s,
)
from apex.optimization.kernel import (
    CandidateVerifier,
    FormalEvaluationAuthorityProvider,
    KernelCampaignDraftUseCase,
    KernelContextBuilder,
    KernelFormalEvaluator,
    KernelFormalCapabilityUseCase,
    KernelOptimizeUseCase,
    stop_formal_campaign,
)
from apex.ports import BenchmarkPass, CodingSessionLauncher
from apex.rl import backend_live_qualification_verifiers
from apex.runtime import (
    ApexExecutionIdentity,
    DependencyReceipt,
    EvaluatorQualificationArtifactAuthority,
    GpuDoctorInspector,
    LinuxGpuDoctorInspector,
    LocalGpuLeaseManager,
    collect_apex_execution_identity,
    formal_results_validator,
    verify_runtime_dependencies,
)


@dataclass(frozen=True, slots=True)
class Application:
    kernel_optimizer: KernelOptimizeUseCase | None
    e2e_optimizer: E2EOptimizeUseCase | None = None
    e2e_bundle_verifier: E2EBundleVerificationPort | None = None
    coding_session: CodingSessionLauncher | None = None
    capabilities: CapabilityRegistry | None = None
    kernel_template_materializer: KernelTemplateMaterializer | None = None
    backend_doctor: NativeBackendDoctor | None = None
    gpu_doctor: GpuDoctorInspector | None = None


def build_application(
    *,
    include_e2e: bool = False,
    include_e2e_verifier: bool = False,
    include_kernel: bool = True,
    include_coding_session: bool = False,
    include_capabilities: bool = False,
    include_kernel_templates: bool = False,
    include_backend_doctor: bool = False,
    include_gpu_doctor: bool = False,
    knowledge_catalog: Path | None = None,
    knowledge_enabled: bool = True,
    capability_workspace: Path | None = None,
    capability_results: Path | None = None,
    kernel_evaluation_authorizer: EvaluationContractAuthorizer | None = None,
    kernel_formal_authority_provider: FormalEvaluationAuthorityProvider | None = None,
    e2e_final_delivery: FinalDeliveryPort | None = None,
) -> Application:
    """Construct production adapters without import-time side effects."""
    skill_package, coding_session = _session_components(
        include_coding_session, include_capabilities, knowledge_catalog
    )
    backend_doctor = NativeBackendDoctor() if include_backend_doctor else None
    gpu_doctor = LinuxGpuDoctorInspector() if include_gpu_doctor else None
    template_materializer = (
        KernelTemplateMaterializer() if include_kernel_templates else None
    )
    if not (
        include_kernel
        or include_e2e
        or include_e2e_verifier
        or include_capabilities
        or include_kernel_templates
    ):
        return Application(
            None,
            coding_session=coding_session,
            backend_doctor=backend_doctor,
            gpu_doctor=gpu_doctor,
        )
    agents = build_default_registry() if include_kernel or include_e2e else None
    retriever = _knowledge_retriever(knowledge_catalog, enabled=knowledge_enabled)
    capabilities = (
        _capability_registry(
            retriever, workspace=capability_workspace, results=capability_results,
            skill_package=skill_package,
            formal_authority_provider=kernel_formal_authority_provider,
        )
        if include_capabilities
        else None
    )
    if include_kernel and agents is None:
        raise AssertionError("Kernel composition requires agent registry")
    kernel = (
        _kernel_optimizer(agents, retriever, kernel_evaluation_authorizer)
        if agents and include_kernel
        else None
    )
    application = Application(
        kernel_optimizer=kernel,
        coding_session=coding_session,
        capabilities=capabilities,
        kernel_template_materializer=template_materializer,
        backend_doctor=backend_doctor,
        gpu_doctor=gpu_doctor,
    )
    if not include_e2e and not include_e2e_verifier:
        return application
    receipt = verify_runtime_dependencies()
    bundle_verifier = _bundle_verifier(receipt) if include_e2e_verifier else None
    if not include_e2e:
        return replace(application, e2e_bundle_verifier=bundle_verifier)
    if agents is None:
        raise AssertionError("E2E composition requires agent registry")
    e2e = _e2e_optimizer(agents, retriever, receipt, e2e_final_delivery)
    return replace(
        application,
        e2e_optimizer=e2e,
        e2e_bundle_verifier=bundle_verifier,
    )

def build_qualification_artifact_authority(
    *, apex_root: Path, artifact_root: Path
) -> EvaluatorQualificationArtifactAuthority:
    """Compose the read-only release verifier from exact protected roots."""

    root = apex_root.expanduser().resolve(strict=True)
    receipt = verify_runtime_dependencies(apex_root=root)
    source_roots = (
        tuple(receipt.source_locks.roots.values())
        if receipt.source_locks is not None
        else ()
    )
    policy = formal_results_validator(
        apex_root=root,
        dependency_roots=tuple(receipt.roots.values()),
        source_roots=source_roots,
    )
    return EvaluatorQualificationArtifactAuthority(
        artifact_root=artifact_root,
        results_policy=policy,
        verifiers=backend_live_qualification_verifiers(),
    )

def _bundle_verifier(receipt: DependencyReceipt) -> E2EBundleVerifierRouter:
    def qwen_verifier() -> E2EBundleVerificationPort:
        verification_roots = {
            name: receipt.source_root(name) for name in ("vllm", "aiter")
        }
        return build_qwen_acceptance_bundle_verifier(
            receipt, source_roots=verification_roots
        )

    return E2EBundleVerifierRouter(
        (
            E2EVerifierProfile(
                QWEN_ACCEPTANCE_PROFILE_ID,
                qwen_acceptance_recipe_sha256s(),
                qwen_verifier,
            ),
        )
    )


def _e2e_optimizer(
    agents,
    retriever: KnowledgeRetriever,
    receipt: DependencyReceipt,
    final_delivery: FinalDeliveryPort | None,
) -> E2EOptimizeUseCase:
    provenance, micro, delivery, oracles = _e2e_runtime_capabilities(
        receipt, final_delivery
    )
    overlay = DockerOverlayDeployment()
    deployments = CandidateDeploymentRegistry(
        (
            ComponentDeploymentBinding(
                overlay.adapter_id,
                overlay.supported_components,
                overlay.supported_run_modes,
                overlay,
            ),
        )
    )
    return E2EOptimizeUseCase(
        dependency_receipt=receipt,
        execution_identity=collect_apex_execution_identity(
            _project_root(), dependency_lock_sha256=receipt.lock_sha256
        ),
        benchmark=MagpieBenchmarkAdapter(
            receipt, execution_attestor=DockerOneShotMagpieExecutionAttestor(receipt)
        ),
        candidate_worker=AgentCandidateWorker(agents),
        contexts=E2EContextBuilder(retriever),
        provenance=provenance,
        micro=micro,
        deployments=deployments,
        final_delivery=delivery,
        correctness_oracles=oracles,
    )


def _e2e_runtime_capabilities(
    receipt: DependencyReceipt, explicit_delivery: FinalDeliveryPort | None
):
    """Build the same exact-lock capability profile for start and resume."""

    if receipt.source_locks is None:
        return None, E2EDeferredMicroQualifier(), explicit_delivery, None
    roots = {name: receipt.source_root(name) for name in ("vllm", "aiter")}
    oracles = build_qwen_correctness_oracles(source_roots=roots)
    micro = build_qwen_oracle_micro_qualifier(oracles)
    provenance = build_qwen_acceptance_provenance_resolver(source_roots=roots)
    delivery = explicit_delivery or build_qwen_acceptance_delivery(
        receipt, source_roots=roots
    )
    return provenance, micro, delivery, oracles


def _knowledge_retriever(path: Path | None, *, enabled: bool) -> KnowledgeRetriever:
    if not enabled:
        return KnowledgeRetriever((), enabled=False)
    selected = path or _default_knowledge_catalog()
    if not selected.exists() and path is None:
        return KnowledgeRetriever((), enabled=False)
    catalog = load_knowledge_catalog(selected)
    return KnowledgeRetriever(catalog.cards)


def _kernel_optimizer(
    agents,
    retriever: KnowledgeRetriever,
    authorizer: EvaluationContractAuthorizer | None,
) -> KernelOptimizeUseCase:
    return KernelOptimizeUseCase(
        agents=agents,
        contexts=KernelContextBuilder(retriever),
        measurement_evaluator=StructuredKernelMeasurementAdapter(),
        diagnostics_evaluator=MagpieKernelDiagnosticsAdapter(verify_runtime_dependencies),
        evaluation_authorizer=authorizer,
        execution_identity=collect_apex_execution_identity(_project_root()),
    )

def _default_knowledge_catalog() -> Path:
    return Path(__file__).resolve().parent / "knowledge" / "data" / "cards.json"


def _session_components(
    include_coding_session: bool,
    include_capabilities: bool,
    knowledge_catalog: Path | None,
) -> tuple[KernelSkillPackage | None, NativeCodingSessionLauncher | None]:
    skill_package = (
        load_kernel_skill_package()
        if include_coding_session or include_capabilities
        else None
    )
    session = (
        NativeCodingSessionLauncher(
            mcp_command=_mcp_command(knowledge_catalog),
            skill_package=skill_package,
        )
        if include_coding_session
        else None
    )
    return skill_package, session


def _capability_registry(
    retriever: KnowledgeRetriever,
    *,
    workspace: Path | None,
    results: Path | None,
    skill_package: KernelSkillPackage | None,
    formal_authority_provider: FormalEvaluationAuthorityProvider | None,
) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    scope = (
        CapabilityScope(workspace, results)
        if workspace is not None and results is not None
        else None
    )
    registry.register(
        knowledge_search_descriptor(),
        KnowledgeSearchHandler(retriever),
    )
    registry.register(
        knowledge_explain_descriptor(),
        KnowledgeExplainHandler(retriever),
    )
    _register_planned_capabilities(
        registry, scope, skill_package, formal_authority_provider
    )
    if workspace is None or results is None:
        registry.register(
            trace_analyze_descriptor(),
            None,
            unavailable_reason="capability_scope_missing",
        )
        registry.register(
            workload_inspect_descriptor(),
            None,
            unavailable_reason="capability_scope_missing",
        )
        registry.register(
            experience_retrieve_descriptor(),
            None,
            unavailable_reason="capability_scope_missing",
        )
        registry.register(
            hotspot_rank_descriptor(),
            None,
            unavailable_reason="capability_scope_missing",
        )
        registry.register(
            trace_compare_descriptor(),
            None,
            unavailable_reason="capability_scope_missing",
        )
    else:
        assert scope is not None
        registry.register(
            trace_analyze_descriptor(),
            TraceAnalyzeHandler(scope, MagpieTraceEvidenceAdapter()),
        )
        registry.register(
            workload_inspect_descriptor(),
            WorkloadInspectHandler(scope, verify_runtime_dependencies),
        )
        registry.register(
            experience_retrieve_descriptor(),
            ExperienceRetrieveHandler(scope),
        )
        registry.register(
            hotspot_rank_descriptor(),
            HotspotRankHandler(scope),
        )
        registry.register(
            trace_compare_descriptor(),
            TraceCompareHandler(scope, _trace_comparator),
        )
    return registry


def _register_planned_capabilities(
    registry: CapabilityRegistry,
    scope: CapabilityScope | None,
    skill_package: KernelSkillPackage | None,
    formal_authority_provider: FormalEvaluationAuthorityProvider | None,
) -> None:
    execution_identity = (
        collect_apex_execution_identity(_project_root())
        if scope is not None
        else None
    )
    evaluator = (
        KernelFormalCapabilityUseCase(
            KernelFormalEvaluator(
                verifier=CandidateVerifier(),
                gpu_leases=LocalGpuLeaseManager(),
                measurement_evaluator=StructuredKernelMeasurementAdapter(),
                authority_provider=formal_authority_provider,
                execution_identity=execution_identity,
            )
        )
        if scope is not None
        else None
    )
    for descriptor in planned_capability_descriptors():
        if (
            skill_package is not None
            and descriptor.capability_id in skill_package.skill_paths
        ):
            registry.register_presentation(descriptor)
            continue
        handler = _planned_scoped_handler(
            descriptor.capability_id, scope, evaluator, execution_identity
        )
        if handler is not None:
            registry.register(descriptor, handler)
            continue
        if descriptor.capability_id in {
            "benchmark.run",
            "bundle.build",
            "bundle.verify",
            "campaign.status",
            "campaign.checkpoint",
            "campaign.start",
            "campaign.stop",
            "campaign.resume",
            "kernel.compile",
            "kernel.correctness",
            "kernel.grade",
            "kernel.measure",
            "profile.capture",
        }:
            registry.register(
                descriptor,
                None,
                unavailable_reason="capability_scope_missing",
            )
            continue
        registry.register(
            descriptor,
            None,
            unavailable_reason=_planned_unavailable_reason(
                descriptor.capability_id
            ),
        )


def _planned_scoped_handler(
    capability_id: str,
    scope: CapabilityScope | None,
    evaluator: KernelFormalCapabilityUseCase | None,
    execution_identity: ApexExecutionIdentity | None,
):
    if scope is None:
        return None
    if capability_id == "bundle.verify":
        return BundleVerifyHandler(scope)
    if capability_id == "campaign.status":
        return CampaignStatusHandler(scope)
    if capability_id == "campaign.checkpoint":
        return CampaignCheckpointHandler(scope)
    if capability_id == "campaign.start":
        assert execution_identity is not None
        return CampaignStartHandler(
            scope,
            KernelCampaignDraftUseCase(),
            execution_identity,
        )
    if capability_id == "campaign.stop":
        return CampaignStopHandler(scope, stop_formal_campaign)
    if capability_id == "campaign.resume":
        return CampaignResumeHandler(scope, _resume_e2e_campaign)
    if capability_id in {"benchmark.run", "profile.capture"}:
        pass_type = (
            BenchmarkPass.MEASUREMENT
            if capability_id == "benchmark.run"
            else BenchmarkPass.DIAGNOSTIC
        )
        return MagpieAcquisitionHandler(
            scope,
            _magpie_benchmark_adapter,
            LocalGpuLeaseManager(),
            pass_type=pass_type,
        )
    if capability_id in {
        "bundle.build",
        "kernel.compile",
        "kernel.correctness",
        "kernel.grade",
        "kernel.measure",
    }:
        assert evaluator is not None
        return KernelEvaluatorHandler(scope, evaluator)
    return None


def _trace_comparator() -> PinnedTraceLensComparisonAdapter:
    receipt = verify_runtime_dependencies()
    return PinnedTraceLensComparisonAdapter(
        root=receipt.root("tracelens"),
        commit=receipt.commits["tracelens"],
    )


def _magpie_benchmark_adapter():
    receipt = verify_runtime_dependencies()
    return MagpieBenchmarkAdapter(receipt, execution_attestor=DockerOneShotMagpieExecutionAttestor(receipt))


def _resume_e2e_campaign(run_root: Path):
    receipt = verify_runtime_dependencies()
    agents = build_default_registry()
    retriever = _knowledge_retriever(None, enabled=True)
    optimizer = _e2e_optimizer(agents, retriever, receipt, None)
    return optimizer.resume(run_root)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _planned_unavailable_reason(capability_id: str) -> str:
    return "capability_not_implemented"


def _mcp_command(knowledge_catalog: Path | None) -> tuple[str, ...]:
    command = [sys.executable, "-m", "apex.cli", "mcp-server"]
    if knowledge_catalog is not None:
        command.extend(["--knowledge-catalog", str(knowledge_catalog)])
    return tuple(command)


__all__ = [
    "Application",
    "build_application",
    "build_qualification_artifact_authority",
]
