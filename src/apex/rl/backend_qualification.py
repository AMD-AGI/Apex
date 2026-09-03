"""Strict replay of live backend qualification campaigns from raw CAS evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from apex.core import ContractError, IntegrityError, sha256_json
from apex.evaluation import (
    MeasurementPolicy,
    kernel_reward_policy_source,
    load_kernel_terminal_grade,
)
from apex.ports import (
    AgentInvocationReceipt,
    AgentProcessContainmentReceipt,
)
from apex.runtime.qualification_artifacts import QualificationArtifactSet
from apex.runtime.release_qualification import (
    QualificationEvidence,
    build_qualification_evidence,
)
from apex.storage import ArtifactReceipt

from .backend_qualification_agent import (
    load_agent_containment,
    load_agent_invocation,
)
from .graph_loader import load_episode_graph
from .graph_validation import validate_episode_graph


MANIFEST_SCHEMA = "apex.backend-live-qualification-artifacts/v1"
VERIFIER_SCHEMA = "apex.backend-live-qualification-verifier/v1"
_BACKENDS = ("claude", "codex", "cursor")
_BACKEND_CLIS = {"claude": "claude", "codex": "codex", "cursor": "cursor-agent"}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TREE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class BackendLiveQualificationArtifactVerifier:
    """Replay one backend's coding and formal kernel evidence without summaries."""

    backend: str

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            _reject("Backend qualification verifier identity is unsupported")

    @property
    def qualification_id(self) -> str:
        return f"backend-{self.backend}-gfx950"

    @property
    def verifier_identity_sha256(self) -> str:
        return sha256_json(_verifier_identity())

    def recompute(self, artifacts: QualificationArtifactSet) -> QualificationEvidence:
        try:
            return self._recompute(artifacts)
        except IntegrityError as error:
            raise ContractError(
                "Backend qualification raw evidence replay failed",
                "qualification_artifacts_invalid",
                {"cause": error.reason_code},
            ) from error

    def _recompute(self, artifact_set: QualificationArtifactSet) -> QualificationEvidence:
        manifest = _manifest(artifact_set, self.qualification_id, self.backend)
        episode_receipt = _receipt(manifest["episode_receipt"])
        result_receipt = _receipt(manifest["kernel_terminal_result_receipt"])
        coding_receipt = _receipt(manifest["coding_receipt"])
        episode = load_episode_graph(artifact_set.artifacts.read_json(episode_receipt))
        validation = validate_episode_graph(
            episode, artifact_set.artifacts, replay_reward=True
        )
        if not validation.reward_replayed or episode.parent.kind != "single_kernel":
            _reject("Backend qualification lacks a replayed kernel episode")
        result = _terminal_result(
            episode.parent.events, result_receipt, artifact_set
        )
        grade = _qualified_terminal_grade(result)
        contract_receipt = _evaluation_contract_receipt(
            episode.parent.events, result, artifact_set
        )
        _contract_gpu_identity(contract_receipt, artifact_set)
        apex_tree = _execution_apex_tree(
            episode.parent.events, manifest, artifact_set
        )
        agent_identity = _coding_identity(
            episode,
            episode.run_id,
            grade.source_attempt_id,
            coding_receipt,
            contract_receipt,
            self.backend,
            artifact_set,
        )
        policy_digest = _measurement_policy_digest(
            episode.parent.events, artifact_set
        )
        return _evidence(
            artifact_set,
            apex_tree=apex_tree,
            backend=self.backend,
            coding_receipt=coding_receipt,
            kernel_receipt=result_receipt,
            agent_identity=agent_identity,
            measurement_policy=policy_digest,
        )


def backend_live_qualification_verifiers(
) -> tuple[BackendLiveQualificationArtifactVerifier, ...]:
    """Return the only production verifier instances for backend live gates."""

    return tuple(BackendLiveQualificationArtifactVerifier(item) for item in _BACKENDS)


def _manifest(
    artifact_set: QualificationArtifactSet,
    qualification_id: str,
    backend: str,
) -> Mapping[str, Any]:
    value = artifact_set.manifest
    expected = {
        "schema", "qualification_id", "backend", "gpu_arch", "apex_tree",
        "coding_receipt", "episode_receipt", "kernel_terminal_result_receipt",
    }
    if (
        set(value) != expected
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("qualification_id") != qualification_id
        or artifact_set.qualification_id != qualification_id
        or value.get("backend") != backend
        or value.get("gpu_arch") != "gfx950"
        or not isinstance(value.get("apex_tree"), str)
        or _TREE.fullmatch(str(value["apex_tree"])) is None
    ):
        _reject("Backend qualification artifact manifest is invalid")
    receipts = (
        _receipt(value["coding_receipt"]),
        _receipt(value["episode_receipt"]),
        _receipt(value["kernel_terminal_result_receipt"]),
    )
    if len({item.digest for item in receipts}) != len(receipts):
        _reject("Backend qualification raw receipts are aliased")
    for receipt in receipts:
        artifact_set.artifacts.verify(receipt)
    return value


def _terminal_result(
    events: Sequence[Any],
    expected_receipt: ArtifactReceipt,
    artifact_set: QualificationArtifactSet,
) -> Mapping[str, Any]:
    matches = tuple(
        event for event in events
        if event.event_type.replace(".", "_") == "delivery_result"
        and event.payload.get("kind") == "kernel_terminal_result"
    )
    if len(matches) != 1:
        _reject("Backend qualification lacks one kernel terminal result")
    receipts = _role_receipts(matches[0], "kernel_terminal_result")
    if receipts != (expected_receipt,):
        _reject("Kernel terminal result receipt differs from the manifest")
    return artifact_set.artifacts.read_json(expected_receipt)


def _qualified_terminal_grade(result: Mapping[str, Any]) -> Any:
    vector = result.get("reward_vector")
    if not isinstance(vector, Mapping):
        _reject("Backend qualification terminal reward vector is missing")
    try:
        grade = load_kernel_terminal_grade(vector)
    except ContractError as error:
        raise ContractError(
            "Backend qualification terminal grade is invalid",
            "qualification_artifacts_invalid",
        ) from error
    if (
        result.get("schema") != "apex.kernel-terminal-result/v1"
        or result.get("task_kind") != "single_kernel"
        or result.get("trainability") != "trainable"
        or grade.outcome != "selected_candidate"
        or grade.scalar_reward is None
        or not grade.gates
        or not grade.gates.correctness_gate_passed
        or grade.gates.safety_finding
    ):
        _reject("Backend qualification terminal result is not a formal winner")
    return grade


def _evaluation_contract_receipt(
    events: Sequence[Any],
    result: Mapping[str, Any],
    artifact_set: QualificationArtifactSet,
) -> ArtifactReceipt:
    digest = result.get("evaluation_contract_receipt_digest")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        _reject("Kernel result evaluation contract digest is invalid")
    receipts = tuple(
        receipt
        for event in events
        if event.event_type.replace(".", "_") == "dependency_verified"
        and event.payload.get("kind") in {
            "evaluation_contract", "evaluation_contract_authorized"
        }
        for receipt in _role_receipts(event, "evaluation_contract")
        if receipt.digest == digest
    )
    if len(receipts) != 1:
        _reject("Kernel result evaluation contract receipt is ambiguous")
    artifact_set.artifacts.verify(receipts[0])
    return receipts[0]


def _contract_gpu_identity(
    receipt: ArtifactReceipt,
    artifact_set: QualificationArtifactSet,
) -> None:
    contract = artifact_set.artifacts.read_json(receipt)
    draft = _mapping(contract.get("draft"), "evaluation contract draft")
    repository = _mapping(draft.get("repository"), "evaluation repository")
    if (
        contract.get("schema") != "apex.evaluation-contract-receipt/v1"
        or contract.get("status") != "verified"
        or contract.get("unverified_reason") is not None
        or contract.get("draft_digest") != sha256_json(draft)
        or draft.get("schema") != "apex.evaluation-contract-draft/v1"
        or draft.get("gpu_arch") != "gfx950"
        or repository.get("status") != "resolved"
        or repository.get("dirty_paths") != []
        or not isinstance(contract.get("authority"), Mapping)
    ):
        _reject("Backend qualification contract identity differs")


def _execution_apex_tree(
    events: Sequence[Any],
    manifest: Mapping[str, Any],
    artifact_set: QualificationArtifactSet,
) -> str:
    matches = tuple(
        event for event in events
        if event.event_type.replace(".", "_") == "provenance_observed"
        and event.payload.get("kind") == "apex_execution_identity"
    )
    if len(matches) != 1:
        _reject("Backend qualification lacks one Apex execution identity")
    receipts = _role_receipts(matches[0], "apex_execution_identity")
    if len(receipts) != 1:
        _reject("Backend qualification Apex execution identity is missing")
    document = artifact_set.artifacts.read_json(receipts[0])
    internal_digest = document.get("receipt_sha256")
    payload = {key: value for key, value in document.items() if key != "receipt_sha256"}
    repository = _mapping(document.get("repository"), "Apex execution repository")
    package = _mapping(document.get("package"), "Apex execution package")
    tree = manifest["apex_tree"]
    expected = {
        "schema", "repository", "package", "dependency_lock_sha256",
        "receipt_sha256",
    }
    if (
        set(document) != expected
        or document.get("schema") != "apex.execution-identity/v1"
        or internal_digest != sha256_json(payload)
        or repository.get("status") != "resolved"
        or repository.get("tree") != tree
        or repository.get("dirty_paths") != []
        or matches[0].payload.get("execution_identity_sha256")
        != internal_digest
        or matches[0].payload.get("apex_tree") != tree
        or matches[0].payload.get("source_manifest_sha256")
        != package.get("source_manifest_sha256")
    ):
        _reject("Backend qualification Apex execution identity differs")
    return str(tree)


def _coding_identity(
    episode: Any,
    run_id: str,
    attempt_id: str | None,
    expected_receipt: ArtifactReceipt,
    contract_receipt: ArtifactReceipt,
    backend: str,
    artifact_set: QualificationArtifactSet,
) -> str:
    children = tuple(item for item in episode.children if item.attempt_id == attempt_id)
    if len(children) != 1:
        _reject("Terminal source attempt has no unique coding episode")
    events = tuple(
        event for event in children[0].events
        if event.event_type.replace(".", "_") == "agent_completed"
    )
    if len(events) != 1 or _role_receipts(events[0], "agent_transcript") != (
        expected_receipt,
    ):
        _reject("Coding receipt is not bound to the terminal source attempt")
    transcript = artifact_set.artifacts.read_json(expected_receipt)
    invocation = load_agent_invocation(transcript.get("invocation"))
    containment = load_agent_containment(
        _mapping(transcript.get("termination"), "agent termination").get(
            "process_containment"
        )
    )
    _validate_coding_event(
        events[0], transcript, invocation, containment, backend, run_id,
        contract_receipt,
    )
    return sha256_json({
        "schema": "apex.backend-agent-identity/v1",
        "backend": backend,
        "model": transcript["model"],
        "effort": transcript["effort"],
        "invocation": invocation.to_dict(),
    })


def _validate_coding_event(
    event: Any,
    transcript: Mapping[str, Any],
    invocation: AgentInvocationReceipt,
    containment: AgentProcessContainmentReceipt,
    backend: str,
    run_id: str,
    contract_receipt: ArtifactReceipt,
) -> None:
    expected = {
        "schema", "backend", "model", "effort", "invocation", "termination",
        "events", "semantic_events", "usage", "cost",
    }
    termination = _mapping(transcript.get("termination"), "agent termination")
    authority = invocation.execution_authority
    if (
        set(transcript) != expected
        or transcript.get("schema") != "apex.agent-transcript/v3"
        or transcript.get("backend") != backend
        or not isinstance(transcript.get("model"), str)
        or not transcript["model"].strip()
        or invocation.cli_name != _BACKEND_CLIS[backend]
        or not containment.namespace_empty_verified
        or termination.get("kind") != "completed"
        or termination.get("capture_status") != "complete"
        or termination.get("candidate_capture_allowed") is not True
        or termination.get("credential_redaction_count") != 0
        or termination.get("observed_turns") not in range(1, invocation.max_turns + 1)
        or not _real_agent_events(transcript)
        or event.payload.get("backend") != backend
        or event.payload.get("model") != transcript["model"]
        or event.payload.get("exit_code") != 0
        or event.payload.get("timed_out") is not False
        or event.payload.get("termination_kind") != "completed"
        or event.payload.get("candidate_capture_allowed") is not True
        or event.payload.get("capture_status") != "complete"
        or event.payload.get("process_containment") != containment.to_dict()
        or authority.backend != backend
        or authority.run_id != run_id
        or authority.attempt_id != event.payload.get("attempt_id")
        or authority.parent_receipt_sha256 != contract_receipt.digest
    ):
        _reject("Coding receipt does not prove one real formal backend invocation")


def _real_agent_events(transcript: Mapping[str, Any]) -> bool:
    events = transcript.get("events")
    semantic = transcript.get("semantic_events")
    return bool(
        isinstance(events, list)
        and events
        and isinstance(semantic, list)
        and any(
            isinstance(item, Mapping)
            and item.get("kind") == "agent_message"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
            for item in semantic
        )
    )


def _measurement_policy_digest(
    events: Sequence[Any], artifact_set: QualificationArtifactSet
) -> str:
    rewards = tuple(
        event for event in events
        if event.event_type.replace(".", "_") == "reward_committed"
        and event.payload.get("scope") == "task_terminal"
    )
    if len(rewards) != 1:
        _reject("Backend qualification terminal reward is missing")
    receipts = _role_receipts(rewards[0], "attempt_reward_policy")
    if len(receipts) != 1:
        _reject("Backend qualification measurement policy is missing")
    document = artifact_set.artifacts.read_json(receipts[0])
    policy = MeasurementPolicy()
    if document != kernel_reward_policy_source(policy):
        _reject("Backend qualification measurement policy is not exact")
    return sha256_json(policy.to_dict())


def _evidence(
    artifact_set: QualificationArtifactSet,
    *,
    apex_tree: str,
    backend: str,
    coding_receipt: ArtifactReceipt,
    kernel_receipt: ArtifactReceipt,
    agent_identity: str,
    measurement_policy: str,
) -> QualificationEvidence:
    subject = artifact_set.manifest_receipt.digest
    return build_qualification_evidence(
        qualification_id=artifact_set.qualification_id,
        apex_tree=apex_tree,
        subject_sha256=subject,
        status="qualified",
        coverage_count=2,
        formal_delivery_count=1,
        details={
            "schema": "apex.backend-live-qualification/v1",
            "qualification_manifest_sha256": subject,
            "backend": backend,
            "gpu_arch": "gfx950",
            "agent_identity_sha256": agent_identity,
            "coding_receipt_sha256": coding_receipt.digest,
            "kernel_receipt_sha256": kernel_receipt.digest,
            "measurement_policy_sha256": measurement_policy,
        },
    )


def _role_receipts(event: Any, role: str) -> tuple[ArtifactReceipt, ...]:
    return tuple(item.receipt for item in event.artifacts if item.role == role)


def _receipt(value: object) -> ArtifactReceipt:
    raw = _mapping(value, "artifact receipt")
    if set(raw) != {"digest", "size", "media_type", "relative_path"}:
        _reject("Artifact receipt fields differ")
    if type(raw.get("size")) is not int:
        _reject("Artifact receipt size is invalid")
    return ArtifactReceipt(
        _text(raw["digest"]), raw["size"], _text(raw["media_type"]),
        _text(raw["relative_path"]),
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _reject(f"{label.title()} must be an object")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        _reject("Expected non-empty text")
    return value


def _verifier_identity() -> dict[str, object]:
    return {
        "schema": VERIFIER_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "episode_schema": "apex.episode_graph/v1",
        "execution_identity_schema": "apex.execution-identity/v1",
        "terminal_result_schema": "apex.kernel-terminal-result/v1",
        "agent_transcript_schema": "apex.agent-transcript/v3",
        "agent_invocation_schema": "apex.agent-invocation/v4",
        "agent_identity_schema": "apex.backend-agent-identity/v1",
        "backends": list(_BACKENDS),
        "gpu_arch": "gfx950",
        "measurement_policy": MeasurementPolicy().to_dict(),
        "terminal_outcome": "selected_candidate",
        "required_terminal_roles": [
            "compile_evidence", "correctness_evidence", "raw_measurement",
            "measurement_execution", "attempt_reward_policy",
        ],
        "raw_artifact_replay_required": True,
    }


def _reject(message: str) -> None:
    raise ContractError(message, "qualification_artifacts_invalid")


__all__ = [
    "BackendLiveQualificationArtifactVerifier",
    "MANIFEST_SCHEMA",
    "VERIFIER_SCHEMA",
    "backend_live_qualification_verifiers",
]
