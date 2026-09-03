"""Independent replay of one standalone kernel task-terminal reward."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_json,
)
from apex.evaluation import (
    GateVerdict,
    GradeAggregation,
    KernelMeasurementExecutionReceipt,
    MeasurementPolicy,
    grade_kernel,
    kernel_reward_policy_source,
    kernel_terminal_policy_source,
    load_kernel_measurement_report,
    load_kernel_terminal_grade,
)
from apex.storage import ArtifactReceipt, ArtifactStore

from .kernel_measurement_validation import kernel_command_executable_frozen
from .models import EpisodeEvent, EvidenceClass


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MISMATCH = "kernel_terminal_evidence_mismatch"


@dataclass(frozen=True, slots=True)
class KernelParentRewardProjection:
    task_reward: float | None
    reward_vector: Mapping[str, Any] | None
    reward_policy_id: str
    reward_policy_digest: str
    reward_source_receipt: str | None
    raw_measurement_receipts: tuple[str, ...]
    trainability: str
    untrainable_reason: str | None


def project_kernel_parent_reward(
    run_id: str,
    events: tuple[EpisodeEvent, ...],
    artifacts: ArtifactStore,
    result: Mapping[str, Any],
) -> KernelParentRewardProjection:
    """Replay a kernel terminal result from evaluator evidence, never child sums."""

    policy = kernel_terminal_policy_source()
    policy_id = str(policy["policy_id"])
    policy_digest = sha256_json(policy)
    _validate_result_shape(run_id, result, policy_id, policy_digest)
    contract = _evaluation_contract(
        events,
        artifacts,
        allow_unverified=result.get("trainability") == "untrainable",
    )
    if result.get("evaluation_contract_receipt_digest") != contract.digest:
        _reject("Kernel terminal result binds another evaluation contract")
    if result.get("trainability") == "untrainable":
        _validate_untrainable(events, result)
        return KernelParentRewardProjection(
            None,
            None,
            policy_id,
            policy_digest,
            None,
            (),
            "untrainable",
            str(result["untrainable_reason"]),
        )
    return _project_trainable(
        run_id,
        events,
        artifacts,
        result,
        contract,
        policy,
        policy_digest,
    )


def _project_trainable(
    run_id: str,
    events: tuple[EpisodeEvent, ...],
    artifacts: ArtifactStore,
    result: Mapping[str, Any],
    contract: ArtifactReceipt,
    policy: Mapping[str, Any],
    policy_digest: str,
) -> KernelParentRewardProjection:
    reward = _terminal_reward_event(events)
    source_receipt = _single_role(reward, "terminal_reward_source")
    vector_receipt = _single_role(reward, "kernel_terminal_grade")
    policy_receipt = _single_role(reward, "reward_policy")
    source = _read_json(artifacts, source_receipt)
    vector = _read_json(artifacts, vector_receipt)
    stored_policy = _read_json(artifacts, policy_receipt)
    try:
        grade = load_kernel_terminal_grade(vector)
    except ContractError as error:
        raise IntegrityError(
            "Kernel terminal grade cannot be decoded", _MISMATCH
        ) from error
    if stored_policy != policy or grade.policy_digest != policy_digest:
        _reject("Kernel terminal policy differs from the frozen formula")
    _validate_source(run_id, source, source_receipt, grade, contract)
    raw = _replay_outcome(reward, source, grade, artifacts)
    if not _reward_and_result_match(
        reward, result, grade.to_dict(), source_receipt, raw
    ):
        _reject("Kernel terminal result, reward, and evidence differ")
    return KernelParentRewardProjection(
        grade.scalar_reward,
        grade.to_dict(),
        grade.policy_id,
        grade.policy_digest,
        source_receipt.digest,
        raw,
        "complete",
        None,
    )


def _replay_outcome(
    reward: EpisodeEvent,
    source: Mapping[str, Any],
    grade: Any,
    artifacts: ArtifactStore,
) -> tuple[str, ...]:
    if grade.outcome in {"selected_candidate", "measured_noop"}:
        return _replay_measurement(reward, source, grade, artifacts)
    if grade.outcome == "compile_failure":
        compile_receipt = _single_role(reward, "compile_evidence")
        _validate_command(artifacts, compile_receipt, "compile", passed=False)
        _source_contains(source, (compile_receipt,))
        return ()
    if grade.outcome == "correctness_failure":
        compile_receipt = _single_role(reward, "compile_evidence")
        correctness = _single_role(reward, "correctness_evidence")
        _validate_command(artifacts, compile_receipt, "compile", passed=True)
        _validate_command(artifacts, correctness, "correctness", passed=False)
        _source_contains(source, (compile_receipt, correctness))
        return ()
    _reject("Trainable kernel terminal outcome is unsupported")


def _replay_measurement(
    reward: EpisodeEvent,
    source: Mapping[str, Any],
    terminal_grade: Any,
    artifacts: ArtifactStore,
) -> tuple[str, ...]:
    raw = _single_role(reward, "raw_measurement")
    execution_receipt = _single_role(reward, "measurement_execution")
    harness_receipt = _single_role(reward, "harness")
    grade_receipt = _single_role(reward, "kernel_grade")
    attempt_policy_receipt = _single_role(reward, "attempt_reward_policy")
    compile_receipt = _single_role(reward, "compile_evidence")
    correctness_receipt = _single_role(reward, "correctness_evidence")
    stored_grade = _read_json(artifacts, grade_receipt)
    attempt_policy = _read_json(artifacts, attempt_policy_receipt)
    policy = _measurement_policy(attempt_policy)
    aggregation = _aggregation(stored_grade)
    artifact = load_kernel_measurement_report(
        _artifact_path(artifacts, raw),
        aggregation=aggregation,
        measurement_policy=policy,
    )
    expected = grade_kernel(
        GateVerdict(True, True, True, True),
        artifact.cases,
        measurement_policy=policy,
        aggregation=aggregation,
    )
    if artifact.sha256 != raw.digest or stored_grade != expected.to_dict():
        _reject("Kernel terminal raw measurement does not reproduce its grade")
    execution = _execution(artifacts, execution_receipt)
    harness = _read_json(artifacts, harness_receipt)
    _validate_measurement_lineage(
        source, raw, execution, harness, policy, artifact.protocol.measurement_method_sha256
    )
    _validate_command(artifacts, compile_receipt, "compile", passed=True)
    _validate_command(artifacts, correctness_receipt, "correctness", passed=True)
    _validate_terminal_measurement_grade(terminal_grade, expected)
    _source_contains(
        source, (grade_receipt, compile_receipt, correctness_receipt)
    )
    return (raw.digest,)


def _validate_terminal_measurement_grade(terminal: Any, measured: Any) -> None:
    if measured.reward is None or measured.srobust is None:
        _reject("Kernel terminal measurement is not reward eligible")
    if terminal.outcome == "selected_candidate" and (
        not measured.promotion_eligible
        or terminal.reason_code != measured.promotion_reason_code
        or (terminal.s50, terminal.s99, terminal.srobust, terminal.scalar_reward)
        != (measured.s50, measured.s99, measured.srobust, measured.reward)
    ):
        _reject("Selected kernel terminal grade differs from raw measurement")
    if terminal.outcome == "measured_noop" and (
        terminal.scalar_reward != 120.0
        or terminal.srobust != 1.0
        or measured.measurement_status.value != "valid"
    ):
        _reject("Kernel no-op terminal lacks a valid frozen-reference measurement")


def _validate_measurement_lineage(
    source: Mapping[str, Any],
    raw: ArtifactReceipt,
    execution: KernelMeasurementExecutionReceipt,
    harness: Mapping[str, Any],
    policy: MeasurementPolicy,
    measurement_method: str,
) -> None:
    expected_source = source.get("measurement_candidate_source_sha256")
    if (
        execution.run_id != source.get("run_id")
        or execution.attempt_id != source.get("source_attempt_id")
        or execution.report_sha256 != raw.digest
        or execution.report_size != raw.size
        or execution.candidate_source_sha256 != expected_source
        or execution.harness_sha256 != harness.get("harness_sha256")
        or execution.measurement_policy_sha256 != sha256_json(policy.to_dict())
        or execution.measurement_method_sha256.removeprefix("sha256:")
        != measurement_method.removeprefix("sha256:")
    ):
        _reject("Kernel terminal measurement execution lineage differs")


def _measurement_policy(document: Mapping[str, Any]) -> MeasurementPolicy:
    value = document.get("measurement_policy")
    if not isinstance(value, Mapping) or set(value) != set(MeasurementPolicy().to_dict()):
        _reject("Kernel attempt reward policy lacks its measurement policy")
    try:
        policy = MeasurementPolicy(**dict(value))
    except (ContractError, TypeError, ValueError) as error:
        raise IntegrityError("Kernel measurement policy is invalid", _MISMATCH) from error
    if document != kernel_reward_policy_source(policy):
        _reject("Kernel attempt reward policy differs from canonical source")
    return policy


def _aggregation(grade: Mapping[str, Any]) -> GradeAggregation:
    try:
        return GradeAggregation(str(grade["aggregation"]))
    except (KeyError, ValueError) as error:
        raise IntegrityError("Kernel grade aggregation is invalid", _MISMATCH) from error


def _execution(
    artifacts: ArtifactStore, receipt: ArtifactReceipt
) -> KernelMeasurementExecutionReceipt:
    try:
        return KernelMeasurementExecutionReceipt.from_mapping(
            _read_json(artifacts, receipt)
        )
    except ContractError as error:
        raise IntegrityError("Kernel execution receipt is invalid", _MISMATCH) from error


def _validate_source(
    run_id: str,
    source: Mapping[str, Any],
    source_receipt: ArtifactReceipt,
    grade: Any,
    contract: ArtifactReceipt,
) -> None:
    expected_keys = {
        "schema",
        "run_id",
        "evaluation_contract_receipt_digest",
        "source_attempt_id",
        "implementation",
        "candidate_source_sha256",
        "measurement_candidate_source_sha256",
        "outcome",
        "reason_code",
        "attempt_evidence_receipts",
    }
    evidence = source.get("attempt_evidence_receipts")
    if (
        set(source) != expected_keys
        or source.get("schema") != "apex.kernel-terminal-reward-source/v1"
        or source.get("run_id") != run_id
        or source.get("evaluation_contract_receipt_digest") != contract.digest
        or source.get("source_attempt_id") != grade.source_attempt_id
        or source.get("outcome") != grade.outcome
        or source.get("reason_code") != grade.reason_code
        or not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not _DIGEST.fullmatch(item) for item in evidence)
        or len(evidence) != len(set(evidence))
        or source_receipt.digest == contract.digest
    ):
        _reject("Kernel terminal reward source is invalid")
    _validate_source_kind(source, grade.outcome)


def _validate_source_kind(source: Mapping[str, Any], outcome: str) -> None:
    candidate = source.get("candidate_source_sha256")
    measured = source.get("measurement_candidate_source_sha256")
    if outcome == "measured_noop":
        valid = source.get("implementation") == "frozen_reference" and candidate is None
    else:
        valid = source.get("implementation") == "candidate" and _is_digest(candidate)
    if outcome in {"selected_candidate", "measured_noop"}:
        valid = valid and _is_digest(measured)
    else:
        valid = valid and measured is None
    if not valid:
        _reject("Kernel terminal source kind is incoherent")


def _evaluation_contract(
    events: tuple[EpisodeEvent, ...],
    artifacts: ArtifactStore,
    *,
    allow_unverified: bool,
) -> ArtifactReceipt:
    frozen = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "dependency_verified"
        and event.payload.get("kind") == "evaluation_contract"
    )
    authorized = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "dependency_verified"
        and event.payload.get("kind") == "evaluation_contract_authorized"
    )
    if len(frozen) != 1 or len(authorized) > 1:
        _reject("Kernel parent lacks one frozen evaluation contract")
    base_receipt, base = _contract_document(frozen[0], artifacts)
    if authorized:
        receipt, document = _contract_document(authorized[0], artifacts)
        if (
            document.get("status") != "verified"
            or not _authority_valid(document, authorized[0])
            or document.get("draft") != base.get("draft")
            or document.get("draft_digest") != base.get("draft_digest")
            or authorized[0].payload.get("draft_digest")
            != base.get("draft_digest")
        ):
            _reject("Kernel evaluation contract authority is invalid")
        return receipt
    if base.get("status") == "verified" and _authority_valid(base, frozen[0]):
        return base_receipt
    if (
        allow_unverified
        and base.get("status") == "unverified"
        and base.get("authority") is None
        and isinstance(base.get("unverified_reason"), str)
        and base.get("unverified_reason")
    ):
        return base_receipt
    _reject("Kernel evaluation contract authority is invalid")


def _contract_document(event, artifacts) -> tuple[ArtifactReceipt, Mapping[str, Any]]:
    receipt = _single_role(event, "evaluation_contract")
    document = _read_json(artifacts, receipt)
    if (
        set(document)
        != {"schema", "status", "unverified_reason", "draft", "draft_digest", "authority"}
        or document.get("schema") != "apex.evaluation-contract-receipt/v1"
        or not isinstance(document.get("draft"), Mapping)
        or document.get("draft_digest") != sha256_json(document["draft"])
        or sha256_json(document) != receipt.digest
        or event.payload.get("contract_digest") != receipt.digest
        or event.payload.get("status") != document.get("status")
    ):
        _reject("Kernel evaluation contract authority is invalid")
    return receipt, document


def _authority_valid(document: Mapping[str, Any], event) -> bool:
    authority = document.get("authority")
    return bool(
        isinstance(authority, Mapping)
        and authority.get("schema") == "apex.evaluation-authority-receipt/v1"
        and authority.get("draft_digest") == document.get("draft_digest")
        and sha256_json(authority) == event.payload.get("authority_receipt_digest")
        and authority.get("authority_id") == event.payload.get("authority_id")
        and authority.get("authority_kind") == event.payload.get("authority_kind")
        and document.get("unverified_reason") is None
    )


def _validate_result_shape(
    run_id: str,
    result: Mapping[str, Any],
    policy_id: str,
    policy_digest: str,
) -> None:
    expected = {
        "schema",
        "task_kind",
        "run_id",
        "task_id",
        "evaluation_contract_receipt_digest",
        "task_reward",
        "reward_vector",
        "reward_policy_id",
        "reward_policy_digest",
        "reward_source_receipt",
        "raw_measurement_receipts",
        "trainability",
        "untrainable_reason",
    }
    if (
        set(result) != expected
        or result.get("schema") != "apex.kernel-terminal-result/v1"
        or result.get("task_kind") != "single_kernel"
        or result.get("run_id") != run_id
        or result.get("reward_policy_id") != policy_id
        or result.get("reward_policy_digest") != policy_digest
        or result.get("trainability") not in {"trainable", "untrainable"}
        or not _is_digest(result.get("evaluation_contract_receipt_digest"))
    ):
        _reject("Kernel terminal result is invalid")


def _validate_untrainable(
    events: tuple[EpisodeEvent, ...], result: Mapping[str, Any]
) -> None:
    if (
        _terminal_reward_events(events)
        or result.get("task_reward") is not None
        or result.get("reward_vector") is not None
        or result.get("reward_source_receipt") is not None
        or result.get("raw_measurement_receipts") != []
        or not isinstance(result.get("untrainable_reason"), str)
        or not result.get("untrainable_reason")
    ):
        _reject("Untrainable kernel parent fabricates terminal reward")


def _reward_and_result_match(
    reward: EpisodeEvent,
    result: Mapping[str, Any],
    vector: Mapping[str, Any],
    source: ArtifactReceipt,
    raw: tuple[str, ...],
) -> bool:
    scalar = vector.get("scalar_reward")
    policy_id = vector.get("policy_id")
    policy_digest = vector.get("policy_digest")
    return bool(
        reward.evidence_class is EvidenceClass.DERIVED
        and reward.payload.get("scope") == "task_terminal"
        and reward.payload.get("scalar_reward") == scalar
        and reward.payload.get("reward_vector") == vector
        and reward.payload.get("policy_id") == policy_id
        and reward.payload.get("policy_digest") == policy_digest
        and reward.payload.get("reward_source_receipt") == source.digest
        and reward.payload.get("raw_measurement_receipts") == list(raw)
        and result.get("trainability") == "trainable"
        and result.get("task_reward") == scalar
        and result.get("reward_vector") == vector
        and result.get("reward_source_receipt") == source.digest
        and result.get("raw_measurement_receipts") == list(raw)
        and result.get("untrainable_reason") is None
    )


def _validate_command(
    artifacts: ArtifactStore,
    receipt: ArtifactReceipt,
    phase: str,
    *,
    passed: bool,
) -> None:
    document = _read_json(artifacts, receipt)
    containment = document.get("process_containment")
    if (
        document.get("phase") != phase
        or document.get("passed") is not passed
        or not isinstance(containment, Mapping)
        or containment.get("namespace_empty_verified") is not True
        or not kernel_command_executable_frozen(document)
    ):
        _reject("Kernel terminal gate evidence is invalid")


def _source_contains(
    source: Mapping[str, Any], receipts: Sequence[ArtifactReceipt]
) -> None:
    evidence = source.get("attempt_evidence_receipts")
    if not isinstance(evidence, list) or any(item.digest not in evidence for item in receipts):
        _reject("Kernel terminal source omits selected attempt evidence")


def _terminal_reward_event(events: tuple[EpisodeEvent, ...]) -> EpisodeEvent:
    matches = _terminal_reward_events(events)
    if len(matches) != 1:
        _reject("Trainable kernel parent lacks one terminal reward")
    return matches[0]


def _terminal_reward_events(
    events: tuple[EpisodeEvent, ...],
) -> tuple[EpisodeEvent, ...]:
    return tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "reward_committed"
        and event.payload.get("scope") == "task_terminal"
    )


def _single_role(event: EpisodeEvent, role: str) -> ArtifactReceipt:
    values = tuple(item.receipt for item in event.artifacts if item.role == role)
    if len(values) != 1:
        _reject(f"Kernel terminal reward requires exactly one {role} artifact")
    return values[0]


def _artifact_path(artifacts: ArtifactStore, receipt: ArtifactReceipt):
    artifacts.verify(receipt)
    return artifacts.root / receipt.relative_path


def _read_json(
    artifacts: ArtifactStore, receipt: ArtifactReceipt
) -> Mapping[str, Any]:
    raw = artifacts.read_bytes(receipt)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise IntegrityError("Kernel terminal evidence is not JSON", _MISMATCH) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        _reject("Kernel terminal evidence is not a canonical object")
    return value


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and bool(_DIGEST.fullmatch(value))


def _reject(message: str) -> None:
    raise IntegrityError(message, _MISMATCH)


__all__ = ["KernelParentRewardProjection", "project_kernel_parent_reward"]
