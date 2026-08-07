"""The sole composition root for concrete Apex adapters and use cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.execution import build_default_registry
from apex.knowledge import KnowledgeRetriever, load_knowledge_catalog
from apex.optimization.e2e import (
    AgentCandidateWorker,
    DockerOverlayDeployment,
    E2EContextBuilder,
    E2EDeferredMicroQualifier,
    E2EOptimizeUseCase,
    FinalDeliveryPort,
    build_qwen_acceptance_delivery,
    build_qwen_acceptance_provenance_resolver,
    build_qwen_correctness_oracles,
)
from apex.optimization.kernel import KernelContextBuilder, KernelOptimizeUseCase
from apex.runtime import verify_runtime_dependencies


@dataclass(frozen=True, slots=True)
class Application:
    kernel_optimizer: KernelOptimizeUseCase
    e2e_optimizer: E2EOptimizeUseCase | None = None


def build_application(
    *,
    include_e2e: bool = False,
    knowledge_catalog: Path | None = None,
    knowledge_enabled: bool = True,
    e2e_final_delivery: FinalDeliveryPort | None = None,
) -> Application:
    """Construct production adapters without import-time side effects."""

    agents = build_default_registry()
    retriever = _knowledge_retriever(knowledge_catalog, enabled=knowledge_enabled)
    kernel = KernelOptimizeUseCase(
        agents=agents,
        contexts=KernelContextBuilder(retriever),
    )
    if not include_e2e:
        return Application(kernel_optimizer=kernel)
    receipt = verify_runtime_dependencies()
    final_delivery = e2e_final_delivery
    provenance = None
    correctness_oracles = None
    if final_delivery is None:
        roots = {name: receipt.source_root(name) for name in ("vllm", "aiter")}
        final_delivery = build_qwen_acceptance_delivery(receipt, source_roots=roots)
        provenance = build_qwen_acceptance_provenance_resolver(source_roots=roots)
        correctness_oracles = build_qwen_correctness_oracles(source_roots=roots)
    e2e = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        provenance=provenance,
        candidate_worker=AgentCandidateWorker(agents),
        contexts=E2EContextBuilder(retriever),
        micro=E2EDeferredMicroQualifier(),
        deployments=DockerOverlayDeployment(),
        final_delivery=final_delivery,
        correctness_oracles=correctness_oracles,
    )
    return Application(kernel_optimizer=kernel, e2e_optimizer=e2e)


def _knowledge_retriever(path: Path | None, *, enabled: bool) -> KnowledgeRetriever:
    if not enabled:
        return KnowledgeRetriever((), enabled=False)
    selected = path or _default_knowledge_catalog()
    if not selected.exists() and path is None:
        return KnowledgeRetriever((), enabled=False)
    catalog = load_knowledge_catalog(selected)
    return KnowledgeRetriever(catalog.cards)


def _default_knowledge_catalog() -> Path:
    return Path(__file__).resolve().parent / "knowledge" / "data" / "cards.json"


__all__ = ["Application", "build_application"]
