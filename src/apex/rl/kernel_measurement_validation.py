"""Offline validation of standalone kernel measurement execution authority."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from apex.core import canonical_json_bytes, sha256_json
from apex.evaluation import (
    GateVerdict,
    KERNEL_REWARD_POLICY_ID,
    KernelMeasurementExecutionReceipt,
    kernel_reward,
    kernel_terminal_policy_source,
)
from apex.storage import ArtifactReceipt, ArtifactStore

from .models import EpisodeArtifact, EpisodeEvent
from .e2e_gpu_lease_validation import load_gpu_heartbeat, load_measurement_bracket


_MISMATCH = "kernel_measurement_execution_evidence_mismatch"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def kernel_measurement_evidence_reasons(
    events: Sequence[EpisodeEvent],
    artifacts: ArtifactStore,
) -> set[str]:
    """Verify reward and measurement bind the same typed execution receipt."""

    measurement = _single_event(events, "measurement_result")
    reward = _single_event(events, "reward_committed")
    if measurement is None or reward is None:
        return {_MISMATCH}
    if not _valid_gpu_bracket(events, measurement, reward, artifacts):
        return {_MISMATCH}
    measured_roles = _roles(measurement.artifacts)
    reward_roles = _roles(reward.artifacts)
    shared = {"raw_measurement", "measurement_execution", "harness"}
    if not shared <= set(measured_roles) or not (shared | {"reward_policy"}) <= set(
        reward_roles
    ):
        return {_MISMATCH}
    if any(
        len(measured_roles[role]) != 1 or len(reward_roles[role]) != 1
        for role in shared
    ):
        return {_MISMATCH}
    if len(reward_roles["reward_policy"]) != 1:
        return {_MISMATCH}
    if any(
        measured_roles[role][0].digest != reward_roles[role][0].digest
        for role in shared
    ):
        return {_MISMATCH}
    raw = measured_roles["raw_measurement"][0]
    execution_artifact = measured_roles["measurement_execution"][0]
    try:
        document = json.loads(artifacts.read_bytes(execution_artifact))
        if not isinstance(document, Mapping):
            return {_MISMATCH}
        execution = KernelMeasurementExecutionReceipt.from_mapping(document)
        raw_document = _document(artifacts, raw)
        harness_document = _document(artifacts, measured_roles["harness"][0])
        policy_document = _document(
            artifacts, reward_roles["reward_policy"][0]
        )
    except Exception:
        return {_MISMATCH}
    attempt_id = str(measurement.payload.get("attempt_id", ""))
    if (
        not attempt_id
        or attempt_id != str(reward.payload.get("attempt_id", ""))
        or execution.attempt_id != attempt_id
        or execution.report_sha256 != raw.digest
        or execution.report_size != raw.size
        or measurement.payload.get("measurement_execution_sha256")
        != execution.fingerprint
        or measurement.payload.get("measurement_writer_id") != execution.writer_id
        or measurement.payload.get("measurement_harness_sha256")
        != execution.harness_sha256
        or str(raw_document.get("measurement_method_sha256", "")).removeprefix(
            "sha256:"
        )
        != execution.measurement_method_sha256.removeprefix("sha256:")
        or harness_document.get("harness_sha256") != execution.harness_sha256
        or sha256_json(policy_document.get("measurement_policy"))
        != execution.measurement_policy_sha256
    ):
        return {_MISMATCH}
    return set()


def _valid_gpu_bracket(
    events: Sequence[EpisodeEvent],
    measurement: EpisodeEvent,
    reward: EpisodeEvent,
    artifacts: ArtifactStore,
) -> bool:
    brackets = tuple(
        event
        for event in events
        if event.payload.get("kind") == "gpu_measurement_bracket"
    )
    if len(brackets) != 1:
        return False
    event = brackets[0]
    roles = _roles(event.artifacts).get("gpu_measurement_bracket", ())
    if len(roles) != 1 or not (event.sequence < measurement.sequence < reward.sequence):
        return False
    try:
        document = _canonical_document(artifacts, roles[0])
        bracket = load_measurement_bracket(document)
    except Exception:
        return False
    attempt_id = str(measurement.payload.get("attempt_id", ""))
    return bool(
        bracket.digest == roles[0].digest
        and bracket.action_id == attempt_id
        and bracket.run_id
        and event.payload.get("attempt_id") == attempt_id
        and event.payload.get("bracket_digest") == bracket.digest
    )


def kernel_gate_reward_evidence_reasons(
    events: Sequence[EpisodeEvent],
    artifacts: ArtifactStore,
    stage: str,
) -> set[str]:
    """Replay trusted compile/correctness low rewards without timing evidence."""

    if stage not in {"compile", "correctness"}:
        return {_MISMATCH}
    reward = _single_event(events, "reward_committed")
    failed = _single_event(events, f"{stage}_result")
    compiled = _single_event(events, "compile_result")
    if reward is None or failed is None or compiled is None:
        return {_MISMATCH}
    if not _valid_gate_heartbeat(events, reward, artifacts, stage):
        return {_MISMATCH}
    reward_roles = _roles(reward.artifacts)
    failed_roles = _roles(failed.artifacts)
    evidence_role = f"{stage}_evidence"
    if any(
        len(values) != 1
        for values in (
            reward_roles.get(evidence_role, ()),
            reward_roles.get("reward_policy", ()),
            failed_roles.get(evidence_role, ()),
        )
    ):
        return {_MISMATCH}
    if (
        reward_roles[evidence_role][0].digest
        != failed_roles[evidence_role][0].digest
    ):
        return {_MISMATCH}
    try:
        evidence = _canonical_document(artifacts, failed_roles[evidence_role][0])
        policy = _canonical_document(artifacts, reward_roles["reward_policy"][0])
    except Exception:
        return {_MISMATCH}
    vector = reward.payload.get("reward_vector")
    expected_gates = GateVerdict(
        compiled=stage == "correctness",
        correct=False,
        integrity_passed=stage == "correctness",
        tampering_passed=stage == "correctness",
    )
    expected_scalar = kernel_reward(expected_gates, None)
    if (
        failed.payload.get("passed") is not False
        or evidence.get("phase") != stage
        or evidence.get("passed") is not False
        or not _contained(evidence)
        or not kernel_command_executable_frozen(evidence)
        or policy != kernel_terminal_policy_source()
        or reward.payload.get("scope") != "attempt"
        or reward.payload.get("policy_id") != KERNEL_REWARD_POLICY_ID
        or reward.payload.get("policy_digest") != sha256_json(policy)
        or reward.payload.get("scalar_reward") != expected_scalar
        or vector != _gate_vector(stage, expected_gates, expected_scalar)
    ):
        return {_MISMATCH}
    if stage == "correctness" and not _compile_passed(compiled, artifacts):
        return {_MISMATCH}
    return set()


def _valid_gate_heartbeat(
    events: Sequence[EpisodeEvent],
    reward: EpisodeEvent,
    artifacts: ArtifactStore,
    stage: str,
) -> bool:
    matching = tuple(
        event
        for event in events
        if event.payload.get("kind") == "gpu_lease_heartbeat"
        and event.payload.get("phase") == stage
    )
    if len(matching) != 1 or matching[0].sequence >= reward.sequence:
        return False
    event = matching[0]
    roles = _roles(event.artifacts).get("gpu_lease_heartbeat", ())
    if len(roles) != 1:
        return False
    try:
        document = _canonical_document(artifacts, roles[0])
        heartbeat = load_gpu_heartbeat(document)
    except Exception:
        return False
    attempt_id = reward.payload.get("attempt_id")
    return bool(
        heartbeat.digest == roles[0].digest
        and heartbeat.reason == "manual"
        and event.payload.get("attempt_id") == attempt_id
        and event.payload.get("heartbeat_digest") == heartbeat.digest
    )


def _gate_vector(
    stage: str, gates: GateVerdict, scalar: float | None
) -> dict[str, object]:
    return {
        "kernel_reward_stage": stage,
        "compile": gates.compiled,
        "correctness": gates.correct,
        "integrity": gates.integrity_passed,
        "anti_tampering": gates.tampering_passed,
        "safety": {"finding": gates.safety_finding},
        "kernel_srobust": None,
        "kernel_robust_reward": scalar,
    }


def _compile_passed(event: EpisodeEvent, artifacts: ArtifactStore) -> bool:
    roles = _roles(event.artifacts)
    values = roles.get("compile_evidence", ())
    if len(values) != 1 or event.payload.get("passed") is not True:
        return False
    try:
        document = _canonical_document(artifacts, values[0])
    except Exception:
        return False
    return bool(
        document.get("phase") == "compile"
        and document.get("passed") is True
        and _contained(document)
        and kernel_command_executable_frozen(document)
    )


def _contained(document: Mapping[str, object]) -> bool:
    value = document.get("process_containment")
    return isinstance(value, Mapping) and value.get("namespace_empty_verified") is True


def kernel_command_executable_frozen(document: Mapping[str, object]) -> bool:
    """Return whether command evidence binds one revalidated absolute executable."""

    identity = document.get("executable_identity")
    argv = document.get("argv")
    if not isinstance(identity, Mapping) or not isinstance(argv, list) or not argv:
        return False
    path = identity.get("path")
    size = identity.get("size")
    digest = identity.get("sha256")
    integers = tuple(
        identity.get(key)
        for key in ("device", "inode", "mode", "mtime_ns", "ctime_ns")
    )
    return bool(
        document.get("executable_identity_reverified") is True
        and isinstance(path, str)
        and path.startswith("/")
        and argv[0] == path
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size >= 0
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in integers)
        and all(value >= 0 for value in integers)
        and integers[2] & 0o111 != 0
    )


def _canonical_document(
    artifacts: ArtifactStore, receipt: ArtifactReceipt
) -> Mapping[str, object]:
    raw = artifacts.read_bytes(receipt)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise ValueError("artifact is not canonical JSON")
    return value


def _document(
    artifacts: ArtifactStore, receipt: ArtifactReceipt
) -> Mapping[str, object]:
    value = json.loads(artifacts.read_bytes(receipt))
    if not isinstance(value, Mapping):
        raise ValueError("artifact document is not an object")
    return value


def _single_event(
    events: Sequence[EpisodeEvent], event_type: str
) -> EpisodeEvent | None:
    found = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == event_type
    )
    return found[0] if len(found) == 1 else None


def _roles(
    bindings: Sequence[EpisodeArtifact],
) -> dict[str, tuple[ArtifactReceipt, ...]]:
    result: dict[str, list[ArtifactReceipt]] = {}
    for binding in bindings:
        result.setdefault(binding.role, []).append(binding.receipt)
    return {role: tuple(receipts) for role, receipts in result.items()}


__all__ = [
    "kernel_command_executable_frozen",
    "kernel_gate_reward_evidence_reasons",
    "kernel_measurement_evidence_reasons",
]
