"""Independent replay of local persistent-server generations in E2E episodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from apex.core import IntegrityError, canonical_json_bytes, sha256_json
from apex.storage import ArtifactReceipt, ArtifactStore

from .models import EpisodeEvent, SemanticRole


_LINEAGE_SCHEMA = "apex.e2e-local-server-lineage/v1"
_REF_SCHEMA = "apex.e2e-local-server-lineage-ref/v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MODES = frozenset({"reuse", "cleanup"})
_KEYS = frozenset(
    {
        "schema", "run_id", "action_id", "lineage_sequence", "lifecycle",
        "framework", "model", "owner", "client_config_sha256",
        "server_source_generation_sha256",
        "server_generation_sha256", "server_identity_sha256",
        "dependency_receipt_sha256", "gpu_lease_digest",
        "execution_attestation_sha256", "local_runtime_receipt_sha256",
        "reward_eligible", "cleanup_verified", "cleanup_succeeded",
    }
)
_OWNER_KEYS = frozenset({"kind", "id", "anchor_id", "anchor_generation"})


@dataclass(frozen=True, slots=True)
class ServerLineageReplay:
    observation_count: int = 0
    active: Mapping[str, Any] | None = None
    retired_generations: tuple[str, ...] = ()
    retired_identities: tuple[str, ...] = ()
    cleanup_artifacts: tuple[str, ...] = ()
    anchor_id: str | None = None
    anchor_generation: int = 0


def validate_e2e_server_lineage(
    run_id: str,
    events: Iterable[EpisodeEvent],
    artifacts: ArtifactStore | None,
) -> ServerLineageReplay:
    """Replay exact server state; cleanup is lifecycle evidence, never reward."""

    ordered = tuple(sorted(events, key=lambda item: item.sequence))
    state = ServerLineageReplay()
    for event in ordered:
        normalized = event.event_type.replace(".", "_")
        if normalized == "run_started":
            state = replace(
                state, anchor_id=str(event.payload.get("initial_anchor_id", ""))
            )
        elif normalized == "measurement_result":
            state = _measurement(run_id, state, event, artifacts)
        elif normalized == "e2e_candidate_decided":
            state = _decision(state, event.payload)
    _reject_cleanup_reward(ordered, set(state.cleanup_artifacts))
    return state


def _measurement(
    run_id: str,
    state: ServerLineageReplay,
    event: EpisodeEvent,
    artifacts: ArtifactStore | None,
) -> ServerLineageReplay:
    lineage_receipt = _single(event, "local_server_lineage")
    attestation_receipt = _single(event, "benchmark_execution_attestation")
    local = None
    if artifacts is not None and attestation_receipt is not None:
        local = _local_runtime(_read(artifacts, attestation_receipt, canonical=False))
    ref = event.payload.get("server_lineage")
    persistent = bool(local is not None and local.get("lifecycle") in _MODES)
    if artifacts is not None:
        normalized_receipt = _single(event, "normalized_benchmark")
        if normalized_receipt is not None:
            normalized = _read(artifacts, normalized_receipt, canonical=True)
            normalized_local = normalized.get("local_runtime")
            persistent = persistent or bool(
                isinstance(normalized_local, Mapping)
                and normalized_local.get("required") is True
                and normalized_local.get("passed") is True
                and normalized_local.get("lifecycle") in _MODES
            )
    if lineage_receipt is None and ref is None and not persistent:
        return state
    if lineage_receipt is None or attestation_receipt is None:
        _reject("Local persistent measurement has incomplete server lineage")
    if artifacts is None:
        document = _document_from_reference(run_id, event, ref, lineage_receipt)
    else:
        document = _read(artifacts, lineage_receipt, canonical=True)
        _validate_raw_derivation(event, document, attestation_receipt, local, artifacts)
    _validate_document(run_id, document)
    _validate_reference(event, document, lineage_receipt)
    return _apply(state, document, event)


def _validate_raw_derivation(
    event: EpisodeEvent,
    value: Mapping[str, Any],
    attestation: ArtifactReceipt,
    runtime: Mapping[str, Any] | None,
    artifacts: ArtifactStore,
) -> None:
    if runtime is None or runtime.get("lifecycle") not in _MODES:
        _reject("Server lineage is not backed by a local persistent runtime")
    lifecycle = _mapping(runtime.get("lifecycle_receipt"), "lifecycle receipt")
    server_state = _mapping(lifecycle.get("server_state"), "server state")
    normalized_receipt = _required_single(event, "normalized_benchmark")
    normalized = _read(artifacts, normalized_receipt, canonical=True)
    config = _required_single(event, "benchmark_config")
    framework, model = normalized.get("framework"), normalized.get("model")
    dependency = runtime.get("dependency_receipt_sha256")
    lease = runtime.get("gpu_lease_digest")
    identity = sha256_json(server_state)
    source = lifecycle.get("server_source_generation_sha256")
    generation = lifecycle.get("server_generation_sha256")
    quiescence = lifecycle.get("quiescence_receipt")
    expected = {
        "framework": framework,
        "model": model,
        "client_config_sha256": config.digest,
        "server_source_generation_sha256": source,
        "server_generation_sha256": generation,
        "server_identity_sha256": identity,
        "dependency_receipt_sha256": dependency,
        "gpu_lease_digest": lease,
        "execution_attestation_sha256": attestation.digest,
        "local_runtime_receipt_sha256": sha256_json(runtime),
        "reward_eligible": event.payload.get("reward_eligible"),
        "cleanup_verified": (
            isinstance(quiescence, Mapping) and quiescence.get("verified") is True
        ),
        "cleanup_succeeded": (
            runtime.get("lifecycle") == "cleanup"
            and isinstance(quiescence, Mapping)
            and quiescence.get("verified") is True
        ),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        _reject("Server lineage differs from raw runtime evidence")


def _apply(
    state: ServerLineageReplay,
    value: Mapping[str, Any],
    event: EpisodeEvent,
) -> ServerLineageReplay:
    sequence = state.observation_count + 1
    if value["lineage_sequence"] != sequence:
        _reject("Server lineage sequence is duplicated or out of order")
    owner = value["owner"]
    if (
        state.anchor_id is None
        or owner["anchor_id"] != state.anchor_id
        or owner["anchor_generation"] != state.anchor_generation
    ):
        _reject("Server lineage is bound to a stale anchor")
    active = state.active
    if value["lifecycle"] == "reuse":
        if (
            value["cleanup_verified"]
            or value["cleanup_succeeded"]
            or active is not None and not _same(active, value)
        ):
            _reject("Server reuse drifted from the active generation")
        if active is None and (
            value["server_generation_sha256"] in state.retired_generations
            or value["server_identity_sha256"] in state.retired_identities
        ):
            _reject("A retired server generation was reused")
        return replace(
            state, observation_count=sequence,
            active=value if active is None else active,
        )
    if (
        active is None or value["reward_eligible"] is not False
        or value["cleanup_verified"] is not True
        or value["cleanup_succeeded"] is not True
        or not _same(active, value)
        or any(
            event.payload.get(key) is not None
            for key in ("attempt_id", "candidate_id", "opportunity_id")
        )
    ):
        _reject("Cleanup does not close the exact active server generation")
    cleanup = tuple(item.receipt.digest for item in event.artifacts)
    return replace(
        state,
        observation_count=sequence,
        active=None,
        retired_generations=(
            *state.retired_generations, str(active["server_generation_sha256"])
        ),
        retired_identities=(
            *state.retired_identities, str(active["server_identity_sha256"])
        ),
        cleanup_artifacts=(*state.cleanup_artifacts, *cleanup),
    )


def _decision(
    state: ServerLineageReplay, payload: Mapping[str, Any]
) -> ServerLineageReplay:
    active = state.active
    verdict = payload.get("verdict")
    if state.observation_count == 0:
        return state
    if active is None or active["owner"].get("kind") != "candidate":
        if verdict == "keep":
            new_anchor = payload.get("new_anchor_id")
            if not isinstance(new_anchor, str) or not new_anchor:
                _reject("Kept candidate has no successor anchor")
            return replace(
                state,
                anchor_id=new_anchor,
                anchor_generation=state.anchor_generation + 1,
            )
        return state
    if active["owner"].get("id") != payload.get("candidate_id"):
        _reject("Candidate decision targets another server generation")
    if verdict != "keep":
        _reject("Reverted candidate server was not cleaned up")
    new_anchor = payload.get("new_anchor_id")
    if not isinstance(new_anchor, str) or not new_anchor:
        _reject("Kept candidate server has no successor anchor")
    owner = dict(active["owner"])
    owner.update(
        {
            "kind": "anchor", "id": new_anchor, "anchor_id": new_anchor,
            "anchor_generation": owner["anchor_generation"] + 1,
        }
    )
    return replace(
        state,
        active={**active, "owner": owner},
        anchor_id=new_anchor,
        anchor_generation=state.anchor_generation + 1,
    )


def _validate_reference(
    event: EpisodeEvent, value: Mapping[str, Any], receipt: ArtifactReceipt
) -> None:
    owner = value["owner"]
    expected = {
        "schema": _REF_SCHEMA,
        "receipt_sha256": receipt.digest,
        "lineage_sequence": value["lineage_sequence"],
        "lifecycle": value["lifecycle"],
        "server_source_generation_sha256": value["server_source_generation_sha256"],
        "server_generation_sha256": value["server_generation_sha256"],
        "server_identity_sha256": value["server_identity_sha256"],
        "owner": owner,
        "reward_eligible": value["reward_eligible"],
        "cleanup_succeeded": value["cleanup_succeeded"],
    }
    if event.payload.get("server_lineage") != expected:
        _reject("Measurement server lineage reference drifted")
    if (
        event.payload.get("action_id") != value["action_id"]
        or event.payload.get("config_sha256") != value["client_config_sha256"]
        or event.payload.get("reward_eligible") is not value["reward_eligible"]
        or owner["kind"] == "candidate"
        and event.payload.get("candidate_id") != owner["id"]
    ):
        _reject("Measurement event targets another server generation")


def _document_from_reference(
    run_id: str, event: EpisodeEvent, ref: object, receipt: ArtifactReceipt
) -> Mapping[str, Any]:
    if not isinstance(ref, Mapping):
        _reject("Exported server lineage reference is invalid")
    owner = _mapping(ref.get("owner"), "owner")
    unknown = "0" * 64
    return {
        "schema": _LINEAGE_SCHEMA, "run_id": run_id,
        "action_id": event.payload.get("action_id"),
        "lineage_sequence": ref.get("lineage_sequence"),
        "lifecycle": ref.get("lifecycle"), "framework": "unloaded",
        "model": "unloaded", "owner": owner,
        "client_config_sha256": event.payload.get("config_sha256"),
        "server_source_generation_sha256": ref.get("server_source_generation_sha256"),
        "server_generation_sha256": ref.get("server_generation_sha256"),
        "server_identity_sha256": ref.get("server_identity_sha256"),
        "dependency_receipt_sha256": unknown, "gpu_lease_digest": unknown,
        "execution_attestation_sha256": unknown,
        "local_runtime_receipt_sha256": unknown,
        "reward_eligible": ref.get("reward_eligible"),
        "cleanup_verified": ref.get("lifecycle") == "cleanup",
        "cleanup_succeeded": ref.get("cleanup_succeeded"),
    }


def _validate_document(run_id: str, value: Mapping[str, Any]) -> None:
    owner = value.get("owner")
    digests = (
        "client_config_sha256", "server_source_generation_sha256",
        "server_generation_sha256", "server_identity_sha256",
        "dependency_receipt_sha256", "gpu_lease_digest",
        "execution_attestation_sha256", "local_runtime_receipt_sha256",
    )
    valid = (
        frozenset(value) == _KEYS and value.get("schema") == _LINEAGE_SCHEMA
        and value.get("run_id") == run_id and value.get("lifecycle") in _MODES
        and isinstance(value.get("action_id"), str) and bool(value["action_id"])
        and isinstance(value.get("lineage_sequence"), int)
        and not isinstance(value.get("lineage_sequence"), bool)
        and value["lineage_sequence"] > 0
        and isinstance(value.get("framework"), str) and bool(value["framework"])
        and isinstance(value.get("model"), str) and bool(value["model"])
        and isinstance(owner, Mapping) and frozenset(owner) == _OWNER_KEYS
        and owner.get("kind") in {"anchor", "candidate"}
        and isinstance(owner.get("id"), str) and bool(owner["id"])
        and isinstance(owner.get("anchor_id"), str) and bool(owner["anchor_id"])
        and isinstance(owner.get("anchor_generation"), int)
        and not isinstance(owner.get("anchor_generation"), bool)
        and owner["anchor_generation"] >= 0
        and all(_DIGEST.fullmatch(str(value.get(key, ""))) for key in digests)
        and isinstance(value.get("reward_eligible"), bool)
        and isinstance(value.get("cleanup_verified"), bool)
        and isinstance(value.get("cleanup_succeeded"), bool)
    )
    if not valid:
        _reject("Server lineage document is malformed")


def _same(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "framework", "model", "server_source_generation_sha256",
        "server_generation_sha256", "server_identity_sha256",
        "dependency_receipt_sha256", "gpu_lease_digest", "owner",
    )
    return all(left[key] == right[key] for key in keys)


def _reject_cleanup_reward(events: tuple[EpisodeEvent, ...], cleanup: set[str]) -> None:
    if not cleanup:
        return
    for event in events:
        if event.semantic_role is not SemanticRole.REWARD:
            continue
        if cleanup.intersection(_nested_digests(event.payload)) or cleanup.intersection(
            item.receipt.digest for item in event.artifacts
        ):
            _reject("Cleanup lifecycle evidence was used as reward evidence")


def _nested_digests(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            digest
            for item in value.values()
            for digest in _nested_digests(item)
        }
    if isinstance(value, (list, tuple)):
        return {digest for item in value for digest in _nested_digests(item)}
    return {value} if isinstance(value, str) and _DIGEST.fullmatch(value) else set()


def _local_runtime(attestation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    runtime = attestation.get("runtime")
    receipt = runtime.get("serving_runtime_receipt") if isinstance(runtime, Mapping) else None
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "apex.magpie-local-runtime-observation/v2":
        return None
    if receipt.get("verified") is not True or receipt.get("errors") != []:
        return None
    return receipt


def _single(event: EpisodeEvent, role: str) -> ArtifactReceipt | None:
    found = tuple(item.receipt for item in event.artifacts if item.role == role)
    if len(found) > 1:
        _reject(f"Measurement has duplicate {role} artifacts")
    return found[0] if found else None


def _required_single(event: EpisodeEvent, role: str) -> ArtifactReceipt:
    receipt = _single(event, role)
    if receipt is None:
        _reject(f"Measurement is missing {role}")
    return receipt


def _read(
    store: ArtifactStore, receipt: ArtifactReceipt, *, canonical: bool
) -> Mapping[str, Any]:
    raw = store.read_bytes(receipt)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Server lineage artifact is invalid", "e2e_server_lineage_mismatch"
        ) from error
    if not isinstance(value, Mapping) or canonical and canonical_json_bytes(value) != raw:
        _reject("Server lineage artifact is not a canonical object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _reject(f"Server lineage {label} is invalid")
    return value


def _reject(message: str) -> None:
    raise IntegrityError(message, "e2e_server_lineage_mismatch")


__all__ = ["ServerLineageReplay", "validate_e2e_server_lineage"]
