"""Compile the exact bounded observation used by a standalone candidate worker."""

from __future__ import annotations

from dataclasses import dataclass

from apex.context import (
    AnchorView,
    ArtifactReference,
    CompiledContext,
    ContextBudget,
    ContextCompileRequest,
    ContextCompiler,
    ContextContract,
    Hypothesis,
    TargetEvidence,
    freeze_metrics,
    render_context_packet,
)
from apex.core import canonical_json_bytes, sha256_json
from apex.intake import ResolvedTaskSpec
from apex.knowledge import (
    ExperienceIdentity,
    ExperienceView,
    KnowledgeRetriever,
    KnowledgeScope,
    normalize_operator_terms,
)
from apex.storage import ArtifactReceipt

from .run_record import KernelRunRecord


@dataclass(frozen=True, slots=True)
class KernelContext:
    """Compiled packet, rendered prompt, and immutable evidence bindings."""

    compiled: CompiledContext
    prompt: str
    packet_receipt: ArtifactReceipt
    source_receipts: tuple[ArtifactReceipt, ...]
    harness_receipt: ArtifactReceipt
    knowledge_receipt: ArtifactReceipt


class KernelContextBuilder:
    """Turn frozen source and policy into a backend-neutral ContextPacket."""

    def __init__(self, retriever: KnowledgeRetriever | None = None) -> None:
        self._compiler = ContextCompiler(retriever or KnowledgeRetriever((), enabled=False))

    def compile(
        self,
        resolved: ResolvedTaskSpec,
        *,
        record: KernelRunRecord,
        attempt_id: str,
        cycle: int,
    ) -> KernelContext:
        sources = _store_sources(resolved, record)
        harness = _store_harness(resolved, record)
        identity = kernel_experience_identity(resolved)
        experience = ExperienceView.from_events(record.iter_events())
        compile_request = _context_compile_request(
            resolved,
            record=record,
            cycle=cycle,
            sources=sources,
            harness=harness,
            identity=identity,
            experience=experience,
        )
        compiled = self._compiler.compile(compile_request)
        packet = record.artifacts.put_bytes(
            compiled.packet.canonical_bytes, media_type="application/json"
        )
        knowledge = record.artifacts.put_bytes(
            canonical_json_bytes(compiled.knowledge_selection.to_dict()),
            media_type="application/json",
        )
        prompt = render_context_packet(compiled.packet) + _workspace_contract(resolved)
        record.record_context(
            attempt_id,
            compiled=compiled,
            packet=packet,
            sources=sources,
            harness=harness,
            knowledge=knowledge,
            prompt=prompt,
        )
        return KernelContext(compiled, prompt, packet, sources, harness, knowledge)


def _store_sources(
    resolved: ResolvedTaskSpec,
    record: KernelRunRecord,
) -> tuple[ArtifactReceipt, ...]:
    return tuple(
        record.artifacts.put_file(path, media_type=_source_media_type(path.suffix))
        for path in resolved.editable_paths
    )


def _store_harness(
    resolved: ResolvedTaskSpec,
    record: KernelRunRecord,
) -> ArtifactReceipt:
    value = {
        "schema_version": 1,
        "commands": {
            name: command.to_dict()
            for name, command in sorted(resolved.task.commands.items())
        },
        "baseline_file_hashes": dict(sorted(resolved.baseline_file_hashes.items())),
    }
    return record.artifacts.put_bytes(
        canonical_json_bytes(value), media_type="application/json"
    )


def _context_compile_request(
    resolved: ResolvedTaskSpec,
    *,
    record: KernelRunRecord,
    cycle: int,
    sources: tuple[ArtifactReceipt, ...],
    harness: ArtifactReceipt,
    identity: ExperienceIdentity,
    experience: ExperienceView,
) -> ContextCompileRequest:
    task = resolved.task
    state = record.controller.state
    target = _target_evidence(resolved, sources)
    return ContextCompileRequest(
        run_id=record.run_id,
        workload_id=f"task-{task.task_id}",
        phase="executing",
        cycle=cycle,
        state_generation=state.sequence,
        role_kind="kernel_optimizer",
        role_objective=task.instructions,
        primary_metric="kernel_srobust",
        hard_constraints=_hard_constraints(resolved),
        target=target,
        hypothesis=_hypothesis(target),
        current_anchor=AnchorView(
            state.anchor_id,
            state.anchor_generation,
            freeze_metrics(
                {
                    "baseline_locked": True,
                    "attempts_completed": len(experience.compatible(identity)),
                }
            ),
        ),
        budget=ContextBudget(
            input_tokens=16_000,
            response_token_allocation=8_000,
            turns=task.budget.max_turns,
            wall_seconds=task.budget.timeout_seconds,
            gpu_seconds_remaining=task.budget.timeout_seconds,
        ),
        contract=_context_contract(task.editable_files),
        artifact_refs=tuple(
            _reference("baseline_source", receipt) for receipt in sources
        )
        + (_reference("protected_harness_contract", harness),),
        retrieval_scope=_scope(resolved),
        experience_identity=identity,
        experience_view=experience,
    )


def _target_evidence(
    resolved: ResolvedTaskSpec,
    sources: tuple[ArtifactReceipt, ...],
) -> TargetEvidence:
    return TargetEvidence(
        opportunity_id=(
            f"opportunity-{sha256_json(resolved.task.target_functions)[:16]}"
        ),
        source_and_symbol=_target_label(resolved),
        phase_shape_regime="standalone-protected-harness",
        evidence_receipts=tuple(item.digest for item in sources),
    )


def _hypothesis(target: TargetEvidence) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=(
            f"hypothesis-{sha256_json({'target': target.source_and_symbol})[:16]}"
        ),
        mechanism=(
            "The locked implementation may be limited by memory traffic, launch "
            "geometry, fusion boundaries, or target-specific code generation."
        ),
        falsification_condition=(
            "Independent compile/correctness gates or robust p50/p99 measurement "
            "fail to show an improvement under the protected harness."
        ),
    )


def _context_contract(editable_files: tuple[str, ...]) -> ContextContract:
    return ContextContract(
        allowed_actions=(
            "inspect_workspace",
            "edit_declared_source",
            "run_declared_verifier",
            "return_action_proposal",
        ),
        editable_files=editable_files,
        acceptance_policy=(
            "Evaluator-owned compile, correctness, integrity, safety, and robust "
            "p50/p99 evidence decide acceptance; agent claims have no authority."
        ),
        stop_policy=(
            "Stop this stateless invocation at its frozen turn/time budget; the "
            "controller alone decides whether another isolated attempt is allowed."
        ),
    )


def _reference(kind: str, receipt: ArtifactReceipt) -> ArtifactReference:
    return ArtifactReference(
        kind=kind,
        sha256=receipt.digest,
        locator=f"artifact://sha256/{receipt.digest}",
    )


def _scope(resolved: ResolvedTaskSpec) -> KnowledgeScope:
    return KnowledgeScope.from_mapping(
        {
            "operator": list(normalize_operator_terms(resolved.task.target_functions)),
            "gpu_arch": [resolved.task.gpu_arch],
            "dtype": list(resolved.task.scope.dtype),
            "regime": list(resolved.task.scope.regime),
            "language": [resolved.task.language],
            "framework": list(resolved.task.scope.framework),
            "versions": dict(resolved.task.scope.versions),
        }
    )


def kernel_experience_identity(resolved: ResolvedTaskSpec) -> ExperienceIdentity:
    task = resolved.task
    return ExperienceIdentity.from_mapping(
        {
            "task_id": task.task_id,
            "operator": task.target_functions[0],
            "gpu_arch": task.gpu_arch,
            "framework": task.language,
            "versions": {},
            "shape_hash": sha256_json({"targets": list(task.target_functions)}),
            "source_hash": resolved.resolution_hash,
            "harness_hash": sha256_json(
                {name: command.to_dict() for name, command in sorted(task.commands.items())}
            ),
            "policy_hash": sha256_json({"policy_id": "kernel_robust_v1"}),
        }
    )


def _hard_constraints(resolved: ResolvedTaskSpec) -> tuple[str, ...]:
    commands = "; ".join(
        f"{name} argv={list(command.argv)!r} cwd={command.cwd!r}"
        for name, command in sorted(resolved.task.commands.items())
    )
    return (
        f"Only edit: {', '.join(resolved.task.editable_files)}.",
        "Do not edit tests, harnesses, configs, result artifacts, or verifier policy.",
        "Correctness and anti-tampering are hard gates; sanitized runs never supply timing.",
        "Robust performance requires both p50 and p99 with at least 300 raw invocations.",
        commands,
    )


def _workspace_contract(resolved: ResolvedTaskSpec) -> str:
    files = "\n".join(
        f"- {path} sha256={resolved.baseline_file_hashes[path]}"
        for path in resolved.task.editable_files
    )
    return (
        "\n## Workspace contract\n\n"
        f"Editable source:\n{files}\n\n"
        "Leave the best candidate in those files. Apex freezes the bytes after the "
        "agent exits and reruns every declared verifier independently.\n"
    )


def _target_label(resolved: ResolvedTaskSpec) -> str:
    files = ",".join(resolved.task.editable_files)
    symbols = ",".join(resolved.task.target_functions)
    return f"{files}:{symbols}"


def _source_media_type(suffix: str) -> str:
    return "text/x-c++" if suffix in {".hip", ".cpp", ".cc", ".cxx"} else "text/x-python"


__all__ = [
    "KernelContext",
    "KernelContextBuilder",
    "kernel_experience_identity",
]
