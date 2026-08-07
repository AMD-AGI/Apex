"""Deterministic task-local ContextPacket compilation for E2E candidates."""

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
from apex.evaluation import E2EMeasurement
from apex.intake import E2EOptimizeSpec
from apex.knowledge import (
    ExperienceIdentity,
    ExperienceOutcome,
    ExperienceRecord,
    ExperienceView,
    KnowledgeRetriever,
    KnowledgeScope,
    normalize_operator_terms,
)
from apex.storage import ArtifactReceipt

from .kernel_lane import KernelOpportunity
from .run_record import E2ERunRecord


@dataclass(frozen=True, slots=True)
class E2EContext:
    compiled: CompiledContext
    prompt: str
    packet_receipt: ArtifactReceipt
    source_receipt: ArtifactReceipt
    harness_receipt: ArtifactReceipt
    knowledge_receipt: ArtifactReceipt
    prompt_receipt: ArtifactReceipt


@dataclass(frozen=True, slots=True)
class _ContextInputs:
    relative_source: str
    source: ArtifactReceipt
    harness: ArtifactReceipt
    identity: ExperienceIdentity


class E2EContextBuilder:
    """Project durable state and CAS receipts into one bounded observation."""

    def __init__(self, retriever: KnowledgeRetriever | None = None) -> None:
        self._compiler = ContextCompiler(retriever or KnowledgeRetriever((), enabled=False))

    def compile(
        self,
        *,
        spec: E2EOptimizeSpec,
        record: E2ERunRecord,
        opportunity: KernelOpportunity,
        attempt_id: str,
        anchor: E2EMeasurement,
        diagnostic_receipt: ArtifactReceipt,
        qualification_mode: str = "strict_micro",
    ) -> E2EContext:
        if qualification_mode not in {"strict_micro", "e2e_quality_deferred"}:
            raise ValueError(f"unsupported E2E qualification mode: {qualification_mode}")
        inputs = _context_inputs(spec, record, opportunity, qualification_mode)
        search = record.controller.state.e2e
        assert search is not None
        compiled = self._compiler.compile(
            _compile_request(
                spec,
                record,
                opportunity,
                anchor,
                diagnostic_receipt,
                inputs,
                qualification_mode,
            )
        )
        return _persist_context(record, attempt_id, compiled, inputs)


def _context_inputs(
    spec: E2EOptimizeSpec,
    record: E2ERunRecord,
    opportunity: KernelOpportunity,
    qualification_mode: str,
) -> _ContextInputs:
    if opportunity.source_path is None or opportunity.source_root is None:
        raise ValueError("E2E context requires an eligible source opportunity")
    source = record.artifacts.put_file(opportunity.source_path, media_type="text/x-python")
    relative = opportunity.source_path.resolve(strict=True).relative_to(
        opportunity.source_root.resolve(strict=True)
    ).as_posix()
    harness = record.artifacts.put_bytes(
        canonical_json_bytes(
            _harness_contract(opportunity, relative, qualification_mode)
        ),
        media_type="application/json",
    )
    identity = _identity(spec, opportunity, source.digest, harness.digest)
    return _ContextInputs(relative, source, harness, identity)


def _harness_contract(
    opportunity: KernelOpportunity,
    relative: str,
    qualification_mode: str,
) -> dict[str, object]:
    micro_policy = (
        "canonical KernelGrade only: compile+correct+integrity+valid p50/p99; "
        "Srobust > 1.05 plus confidence, CV, and worst-case gates"
        if qualification_mode == "strict_micro"
        else
        "frozen-source integrity only; compile, correctness, p50/p99, and kernel "
        "reward are explicitly unmeasured and deferred to unchanged Magpie quality "
        "plus E2E acceptance"
    )
    return {
        "schema_version": 1,
        "evidence_id": opportunity.evidence_id,
        "source": relative,
        "runtime_name": opportunity.runtime_name,
        "operation_name": opportunity.operation_name,
        "phase": opportunity.phase,
        "rank": opportunity.rank,
        "shape_summary": list(opportunity.shape_summary),
        "dtypes": list(opportunity.dtypes),
        "graph_mode": opportunity.graph_mode,
        "correctness_oracle": {
            "test_file": str(opportunity.test_file) if opportunity.test_file else None,
            "test_command": opportunity.test_command,
            "binding_sha256": opportunity.correctness_oracle_sha256,
        },
        "promotion_policy": {
            "qualification_mode": qualification_mode,
            "micro": micro_policy,
            "e2e": "throughput improves current live anchor; accuracy/TTFT/TPOT gates pass",
            "formal": "source rebuild, loaded-byte engagement, second clean replay",
        },
    }


def _compile_request(
    spec: E2EOptimizeSpec,
    record: E2ERunRecord,
    opportunity: KernelOpportunity,
    anchor: E2EMeasurement,
    diagnostic_receipt: ArtifactReceipt,
    inputs: _ContextInputs,
    qualification_mode: str,
) -> ContextCompileRequest:
    search = record.controller.state.e2e
    assert search is not None
    relative = inputs.relative_source
    return ContextCompileRequest(
        run_id=record.run_id,
        workload_id=search.workload_id,
        phase="executing",
        cycle=search.cycle,
        state_generation=search.state_generation,
        role_kind="e2e_kernel_candidate_generator",
        role_objective=(
            f"Optimize only {relative} for the measured kernel "
            f"{opportunity.runtime_name!r} on {spec.gpu_arch}."
        ),
        primary_metric=(
            "kernel_srobust_then_realized_e2e_throughput"
            if qualification_mode == "strict_micro"
            else "realized_e2e_throughput"
        ),
        hard_constraints=_hard_constraints(spec, relative, qualification_mode),
        target=TargetEvidence(
            opportunity_id=opportunity.opportunity_id,
            source_and_symbol=f"{relative}:{opportunity.runtime_name}",
            phase_shape_regime=_regime(opportunity),
            evidence_receipts=(diagnostic_receipt.digest, inputs.source.digest),
        ),
        hypothesis=_hypothesis(opportunity, search.cycle, qualification_mode),
        current_anchor=AnchorView(
            record.controller.state.anchor_id,
            record.controller.state.anchor_generation,
            freeze_metrics(_anchor_metrics(anchor)),
        ),
        budget=ContextBudget(
            input_tokens=spec.context_input_tokens,
            response_token_allocation=spec.context_response_token_allocation,
            turns=spec.max_turns,
            wall_seconds=spec.agent_timeout_seconds,
            gpu_seconds_remaining=spec.agent_timeout_seconds,
        ),
        contract=_agent_contract(relative, qualification_mode),
        artifact_refs=(
            _artifact("diagnostic_evidence", diagnostic_receipt),
            _artifact("baseline_kernel_source", inputs.source),
            _artifact("protected_harness_contract", inputs.harness),
        ),
        retrieval_scope=_scope(spec, opportunity),
        experience_identity=inputs.identity,
        experience_view=_experience(record, inputs.identity, opportunity.opportunity_id),
    )


def _hypothesis(
    opportunity: KernelOpportunity,
    cycle: int,
    qualification_mode: str,
) -> Hypothesis:
    digest = sha256_json({"target": opportunity.evidence_id, "cycle": cycle})[:24]
    falsification = (
        "The independent micro grader fails compile/correctness/integrity, either "
        "p50 or p99 does not improve robustly, or the unchanged workload E2E A/B "
        "does not pass promotion gates."
        if qualification_mode == "strict_micro"
        else
        "Frozen-source integrity fails, or the isolated immutable deployment fails "
        "loaded-byte engagement, unchanged Magpie quality, or current-anchor E2E "
        "acceptance. No kernel-level compile, correctness, timing, or reward claim is made."
    )
    return Hypothesis(
        hypothesis_id=f"hypothesis-{digest}",
        mechanism=_mechanism(opportunity),
        falsification_condition=falsification,
    )


def _agent_contract(relative: str, qualification_mode: str) -> ContextContract:
    acceptance = (
        "Only evaluator-owned micro, safety, source-delivery, and current-anchor "
        "E2E evidence can promote a candidate. Agent claims have no authority."
        if qualification_mode == "strict_micro"
        else
        "The pre-safety micro stage checks frozen-source integrity only. Only "
        "evaluator-owned safety, isolated source-delivery/loaded-byte evidence, unchanged "
        "Magpie quality, and current-anchor E2E evidence can promote a candidate. Kernel "
        "reward is unavailable; agent claims have no authority."
    )
    return ContextContract(
        allowed_actions=(
            "inspect_source",
            "edit_declared_kernel_source",
            "run_read_only_diagnostics",
            "return_action_proposal",
        ),
        editable_files=(relative,),
        acceptance_policy=acceptance,
        stop_policy=(
            "Stop when the declared source contains the best candidate or the frozen "
            "turn/wall budget is exhausted."
        ),
    )


def _persist_context(
    record: E2ERunRecord,
    attempt_id: str,
    compiled: CompiledContext,
    inputs: _ContextInputs,
) -> E2EContext:
    packet = record.artifacts.put_bytes(
        compiled.packet.canonical_bytes, media_type="application/json"
    )
    knowledge = record.artifacts.put_bytes(
        canonical_json_bytes(compiled.knowledge_selection.to_dict()),
        media_type="application/json",
    )
    prompt = render_context_packet(compiled.packet) + _workspace_contract(
        inputs.relative_source, inputs.source
    )
    prompt_receipt = record.record_context(
        attempt_id,
        compiled=compiled,
        packet=packet,
        source=inputs.source,
        harness=inputs.harness,
        knowledge=knowledge,
        prompt=prompt,
    )
    return E2EContext(
        compiled,
        prompt,
        packet,
        inputs.source,
        inputs.harness,
        knowledge,
        prompt_receipt,
    )


def _artifact(kind: str, receipt: ArtifactReceipt) -> ArtifactReference:
    return ArtifactReference(
        kind=kind,
        sha256=receipt.digest,
        locator=f"artifact://sha256/{receipt.digest}",
    )


def _anchor_metrics(anchor: E2EMeasurement) -> dict[str, float]:
    return {
        "throughput": anchor.throughput,
        "ttft_p99_ms": anchor.ttft_p99_ms,
        "tpot_p99_ms": anchor.tpot_p99_ms,
        "accuracy": anchor.accuracy,
    }


def _hard_constraints(
    spec: E2EOptimizeSpec,
    relative: str,
    qualification_mode: str,
) -> tuple[str, ...]:
    gates = spec.goal.gates
    micro_constraint = (
        "Micro promotion requires at least 300 raw samples and the lower of p50/p99 speedup."
        if qualification_mode == "strict_micro"
        else
        "No trusted micro timing harness is present: do not claim kernel compile, correctness, "
        "p50/p99, Srobust, or kernel reward; unchanged Magpie quality/E2E evidence is authoritative."
    )
    return (
        f"Edit only {relative}; config-only, server-flag, workload, and harness changes are forbidden.",
        "Preserve kernel semantics for all protected inputs; anti-tampering is a hard gate.",
        "Do not treat profiler-on diagnostics or sanitizer runs as performance evidence.",
        micro_constraint,
        f"E2E accuracy may not regress; TTFT p99 <= {gates.ttft_p99_regression_pct}% and "
        f"TPOT p99 <= {gates.tpot_p99_regression_pct}% regression.",
    )


def _regime(opportunity: KernelOpportunity) -> str:
    shapes = ",".join(opportunity.shape_summary) or "shape-unavailable"
    dtypes = ",".join(opportunity.dtypes) or "dtype-unavailable"
    return f"phase={opportunity.phase};rank={opportunity.rank};shapes={shapes};dtypes={dtypes};graph={opportunity.graph_mode}"


def _mechanism(opportunity: KernelOpportunity) -> str:
    bound = "unknown"
    if opportunity.roi_prior < opportunity.measured_gpu_pct:
        bound = "measured roofline headroom"
    return (
        f"The {opportunity.measured_gpu_pct:.4f}% GPU-time target has {bound}; "
        "launch geometry, memory traffic, fusion boundaries, or generated code may leave "
        "shape-specific headroom."
    )


def _scope(spec: E2EOptimizeSpec, opportunity: KernelOpportunity) -> KnowledgeScope:
    operator_terms = normalize_operator_terms(
        (opportunity.operation_name, opportunity.runtime_name)
    )
    return KnowledgeScope.from_mapping(
        {
            "operator": list(operator_terms),
            "gpu_arch": [spec.gpu_arch],
            "dtype": list(opportunity.dtypes),
            "regime": [opportunity.phase],
            "language": [opportunity.language],
            "framework": [opportunity.origin_library],
            "versions": {},
        }
    )


def _identity(
    spec: E2EOptimizeSpec,
    opportunity: KernelOpportunity,
    source_digest: str,
    harness_digest: str,
) -> ExperienceIdentity:
    return ExperienceIdentity.from_mapping(
        {
            "task_id": opportunity.opportunity_id,
            "operator": opportunity.operation_name or opportunity.runtime_name,
            "gpu_arch": spec.gpu_arch,
            "framework": opportunity.origin_library,
            "versions": {},
            "shape_hash": sha256_json(
                {
                    "shapes": list(opportunity.shape_summary),
                    "dtypes": list(opportunity.dtypes),
                    "phase": opportunity.phase,
                }
            ),
            "source_hash": source_digest,
            "harness_hash": harness_digest,
            "policy_hash": sha256_json({"policy_id": "e2e_kernel_promotion_v1"}),
        }
    )


def _experience(
    record: E2ERunRecord,
    identity: ExperienceIdentity,
    opportunity_id: str,
) -> ExperienceView:
    search = record.controller.state.e2e
    assert search is not None
    records: list[ExperienceRecord] = []
    for sequence, decision in enumerate(search.decisions, start=1):
        if decision.opportunity_id != opportunity_id:
            continue
        outcome = {
            "keep": ExperienceOutcome.SUCCESS,
            "revert": ExperienceOutcome.NO_GAIN,
            "reject": ExperienceOutcome.FAILURE,
            "needs_more_measurement": ExperienceOutcome.FAILURE,
        }[decision.verdict]
        records.append(
            ExperienceRecord(
                event_sequence=sequence,
                event_id=f"decision-{sequence}",
                candidate_id=decision.candidate_id,
                identity=identity,
                outcome=outcome,
                strategy_fingerprint=sha256_json(
                    {"candidate": decision.candidate_id, "reason": decision.reason}
                ),
                mechanism="prior source candidate for this measured opportunity",
                micro_verdict="qualified" if decision.verdict != "reject" else "rejected",
                e2e_verdict=decision.verdict,
                evidence_receipts=(decision.evidence_ref,),
                failure_reason=None if decision.verdict == "keep" else decision.reason,
                retry_condition=(
                    None if decision.verdict == "keep" else "new hypothesis or anchor generation"
                ),
            )
        )
    return ExperienceView(tuple(records), ())


def _workspace_contract(relative: str, source: ArtifactReceipt) -> str:
    return (
        "\n## Candidate workspace\n\n"
        f"The only editable file is `{relative}`. Its anchor receipt is {source.digest}.\n"
        "Leave the best source candidate in place. Apex freezes the bytes after this "
        "fresh backend exits; later evaluators independently decide every gate.\n"
    )


__all__ = ["E2EContext", "E2EContextBuilder"]
