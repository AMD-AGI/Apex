"""Canonical local persistent-server lineage and replayable lifecycle state."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from apex.benchmark import NormalizedBenchmarkResult
from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_json
from apex.storage import ArtifactReceipt, ArtifactStore, EventRecord

from .benchmark_artifacts import BenchmarkEvidenceReceipts


LINEAGE_SCHEMA = "apex.e2e-local-server-lineage/v1"
REF_SCHEMA = "apex.e2e-local-server-lineage-ref/v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PERSISTENT_MODES = frozenset({"reuse", "cleanup"})
_LINEAGE_KEYS = frozenset(
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
_REF_KEYS = frozenset(
    {
        "schema", "receipt_sha256", "lineage_sequence", "lifecycle",
        "server_source_generation_sha256", "server_generation_sha256",
        "server_identity_sha256", "owner", "reward_eligible",
        "cleanup_succeeded",
    }
)


@dataclass(frozen=True, slots=True)
class LocalServerLifecycleProjection:
    """State rebuilt only from canonical measurement events and CAS artifacts."""

    observation_count: int = 0
    active: Mapping[str, Any] | None = None
    retired_generations: tuple[str, ...] = ()
    retired_identities: tuple[str, ...] = ()
    anchor_id: str | None = None
    anchor_generation: int = 0


@dataclass(frozen=True, slots=True)
class LocalServerLineageEvidence:
    document: Mapping[str, Any]
    receipt: ArtifactReceipt

    @property
    def reference(self) -> Mapping[str, Any]:
        value = self.document
        return {
            "schema": REF_SCHEMA,
            "receipt_sha256": self.receipt.digest,
            "lineage_sequence": value["lineage_sequence"],
            "lifecycle": value["lifecycle"],
            "server_source_generation_sha256": value[
                "server_source_generation_sha256"
            ],
            "server_generation_sha256": value["server_generation_sha256"],
            "server_identity_sha256": value["server_identity_sha256"],
            "owner": value["owner"],
            "reward_eligible": value["reward_eligible"],
            "cleanup_succeeded": value["cleanup_succeeded"],
        }

    @property
    def binding(self) -> dict[str, object]:
        return {"role": "local_server_lineage", "receipt": self.receipt.to_dict()}


def capture_local_server_lineage(
    *,
    store: ArtifactStore,
    events: Iterable[EventRecord],
    result: NormalizedBenchmarkResult,
    evidence: BenchmarkEvidenceReceipts,
    run_id: str,
    action_id: str,
    owner_kind: str,
    owner_id: str,
    anchor_id: str,
    anchor_generation: int,
) -> LocalServerLineageEvidence | None:
    """Derive a server generation without conflating it with client config bytes."""

    attestation_receipt = _single_binding(
        evidence.bindings, "benchmark_execution_attestation", required=False
    )
    if attestation_receipt is None:
        return None
    attestation = _read_object(store, attestation_receipt)
    runtime = _local_runtime(attestation)
    if runtime is None or runtime.get("lifecycle") not in _PERSISTENT_MODES:
        return None
    projection = replay_local_server_lineage(events, store)
    document = _derive_document(
        runtime=runtime,
        result=result,
        run_id=run_id,
        action_id=action_id,
        sequence=projection.observation_count + 1,
        owner_kind=owner_kind,
        owner_id=owner_id,
        anchor_id=anchor_id,
        anchor_generation=anchor_generation,
        config_sha256=evidence.config.digest,
        attestation_sha256=attestation_receipt.digest,
    )
    _apply_lineage(projection, document)
    receipt = store.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    return LocalServerLineageEvidence(document, receipt)


def replay_local_server_lineage(
    events: Iterable[EventRecord], store: ArtifactStore
) -> LocalServerLifecycleProjection:
    """Rebuild lifecycle state and reject missing, duplicated, or drifting lineage."""

    projection = LocalServerLifecycleProjection()
    for event in events:
        if event.event_type == "run.started":
            projection = replace(
                projection,
                anchor_id=str(event.payload.get("initial_anchor_id", "")),
            )
        elif event.event_type == "measurement_result":
            projection = _replay_measurement(projection, event, store)
        elif event.event_type == "e2e.candidate_decided":
            projection = _replay_decision(projection, event.payload)
    return projection


def require_resumable_server_lineage(
    events: Iterable[EventRecord], store: ArtifactStore, lease_digest: str
) -> None:
    """A persistent server from an interrupted lease cannot cross resume authority."""

    projection = replay_local_server_lineage(events, store)
    active = projection.active
    if active is not None and active["gpu_lease_digest"] != lease_digest:
        raise ContractError(
            "Resume cannot reuse a persistent server from the interrupted GPU lease",
            "resume_server_cleanup_required",
        )


def _derive_document(
    *, runtime: Mapping[str, Any], result: NormalizedBenchmarkResult,
    run_id: str, action_id: str, sequence: int, owner_kind: str,
    owner_id: str, anchor_id: str, anchor_generation: int,
    config_sha256: str, attestation_sha256: str,
) -> Mapping[str, Any]:
    lifecycle = _mapping(runtime.get("lifecycle_receipt"), "lifecycle receipt")
    server_state = _mapping(lifecycle.get("server_state"), "server state")
    dependency = _digest(runtime.get("dependency_receipt_sha256"), "dependency")
    lease = _digest(runtime.get("gpu_lease_digest"), "GPU lease")
    identity = sha256_json(server_state)
    source = _digest(
        lifecycle.get("server_source_generation_sha256"), "server source generation"
    )
    generation = _digest(
        lifecycle.get("server_generation_sha256"), "server generation"
    )
    quiescence = lifecycle.get("quiescence_receipt")
    cleanup = (
        isinstance(quiescence, Mapping) and quiescence.get("verified") is True
    )
    mode = str(runtime.get("lifecycle", ""))
    reward_eligible = result.reward_eligible
    local = result.local_runtime
    if (
        not local.required or not local.passed or local.lifecycle != mode
        or local.gpu_lease_digest != lease
        or local.server_source_generation_sha256 != source
        or local.server_generation_sha256 != generation
        or local.quiescence_verified is not (True if mode == "cleanup" else None)
    ):
        _reject("Normalized local runtime differs from its raw receipt")
    document = {
        "schema": LINEAGE_SCHEMA,
        "run_id": run_id,
        "action_id": action_id,
        "lineage_sequence": sequence,
        "lifecycle": mode,
        "framework": result.framework,
        "model": result.model,
        "owner": {
            "kind": owner_kind, "id": owner_id, "anchor_id": anchor_id,
            "anchor_generation": anchor_generation,
        },
        "client_config_sha256": config_sha256,
        "server_source_generation_sha256": source,
        "server_generation_sha256": generation,
        "server_identity_sha256": identity,
        "dependency_receipt_sha256": dependency,
        "gpu_lease_digest": lease,
        "execution_attestation_sha256": attestation_sha256,
        "local_runtime_receipt_sha256": sha256_json(runtime),
        "reward_eligible": reward_eligible,
        "cleanup_verified": cleanup,
        "cleanup_succeeded": mode == "cleanup" and cleanup,
    }
    _validate_document(document)
    return document


def _replay_measurement(
    projection: LocalServerLifecycleProjection,
    event: EventRecord,
    store: ArtifactStore,
) -> LocalServerLifecycleProjection:
    bindings = event.payload.get("artifacts")
    values = bindings if isinstance(bindings, list) else []
    lineage_receipt = _single_binding(values, "local_server_lineage", required=False)
    attestation_receipt = _single_binding(
        values, "benchmark_execution_attestation", required=False
    )
    normalized_receipt = _single_binding(
        values, "normalized_benchmark", required=False
    )
    ref = event.payload.get("server_lineage")
    local_persistent = False
    if attestation_receipt is not None:
        runtime = _local_runtime(_read_object(store, attestation_receipt))
        local_persistent = bool(
            runtime is not None and runtime.get("lifecycle") in _PERSISTENT_MODES
        )
    if normalized_receipt is not None:
        normalized = _read_object(store, normalized_receipt)
        local = normalized.get("local_runtime")
        local_persistent = local_persistent or bool(
            isinstance(local, Mapping)
            and local.get("required") is True
            and local.get("passed") is True
            and local.get("lifecycle") in _PERSISTENT_MODES
        )
    if not local_persistent and lineage_receipt is None and ref is None:
        return projection
    if lineage_receipt is None or attestation_receipt is None:
        _reject("Local persistent measurement is missing server lineage")
    document = _read_object(store, lineage_receipt)
    _validate_document(document)
    _validate_event_binding(event.payload, document, lineage_receipt, attestation_receipt)
    return _apply_lineage(projection, document)


def _apply_lineage(
    projection: LocalServerLifecycleProjection, document: Mapping[str, Any]
) -> LocalServerLifecycleProjection:
    expected = projection.observation_count + 1
    if document["lineage_sequence"] != expected:
        _reject("Local server lineage sequence is duplicated or out of order")
    owner = document["owner"]
    if (
        projection.anchor_id is None
        or owner["anchor_id"] != projection.anchor_id
        or owner["anchor_generation"] != projection.anchor_generation
    ):
        _reject("Local server lineage is bound to a stale anchor")
    active = projection.active
    mode = document["lifecycle"]
    if mode == "reuse":
        if (
            document["cleanup_verified"]
            or document["cleanup_succeeded"]
            or active is not None and not _same(active, document)
        ):
            _reject("Local server reuse drifted from the active generation")
        if active is None and (
            document["server_generation_sha256"] in projection.retired_generations
            or document["server_identity_sha256"] in projection.retired_identities
        ):
            _reject("A retired local server generation was reused")
        return replace(
            projection, observation_count=expected,
            active=document if active is None else active,
        )
    if (
        active is None or document["reward_eligible"] is not False
        or document["cleanup_verified"] is not True
        or document["cleanup_succeeded"] is not True
        or not _same(active, document)
    ):
        _reject("Local server cleanup does not close the exact active generation")
    return replace(
        projection,
        observation_count=expected,
        active=None,
        retired_generations=(
            *projection.retired_generations,
            str(active["server_generation_sha256"]),
        ),
        retired_identities=(
            *projection.retired_identities,
            str(active["server_identity_sha256"]),
        ),
    )


def _same(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "framework", "model", "server_source_generation_sha256",
        "server_generation_sha256", "server_identity_sha256",
        "dependency_receipt_sha256", "gpu_lease_digest", "owner",
    )
    return all(left[key] == right[key] for key in keys)


def _replay_decision(
    projection: LocalServerLifecycleProjection, payload: Mapping[str, Any]
) -> LocalServerLifecycleProjection:
    active = projection.active
    candidate = payload.get("candidate_id")
    verdict = payload.get("verdict")
    if projection.observation_count == 0:
        return projection
    if active is None or active["owner"].get("kind") != "candidate":
        if verdict == "keep":
            new_anchor = payload.get("new_anchor_id")
            if not isinstance(new_anchor, str) or not new_anchor:
                _reject("Kept candidate has no successor anchor")
            return replace(
                projection,
                anchor_id=new_anchor,
                anchor_generation=projection.anchor_generation + 1,
            )
        return projection
    if active["owner"].get("id") != candidate:
        _reject("Candidate decision targets another active server generation")
    if verdict != "keep":
        _reject("A reverted candidate server was not cleaned up")
    owner = dict(active["owner"])
    owner.update(
        {
            "kind": "anchor", "id": payload.get("new_anchor_id"),
            "anchor_id": payload.get("new_anchor_id"),
            "anchor_generation": owner["anchor_generation"] + 1,
        }
    )
    return replace(
        projection,
        active={**active, "owner": owner},
        anchor_id=str(payload.get("new_anchor_id")),
        anchor_generation=projection.anchor_generation + 1,
    )


def _validate_event_binding(
    payload: Mapping[str, Any], document: Mapping[str, Any],
    receipt: ArtifactReceipt, attestation: ArtifactReceipt,
) -> None:
    ref = payload.get("server_lineage")
    if not isinstance(ref, Mapping) or frozenset(ref) != _REF_KEYS:
        _reject("Measurement server lineage reference is invalid")
    expected = LocalServerLineageEvidence(document, receipt).reference
    if dict(ref) != expected:
        _reject("Measurement server lineage reference drifted")
    checks = {
        "action_id": document["action_id"],
        "config_sha256": document["client_config_sha256"],
        "reward_eligible": document["reward_eligible"],
    }
    if (
        any(payload.get(key) != value for key, value in checks.items())
        or document["execution_attestation_sha256"] != attestation.digest
        or document["lifecycle"] == "cleanup"
        and any(
            payload.get(key) is not None
            for key in ("attempt_id", "candidate_id", "opportunity_id")
        )
    ):
        _reject("Measurement and server lineage identities differ")


def _validate_document(value: Mapping[str, Any]) -> None:
    owner = value.get("owner")
    valid = (
        frozenset(value) == _LINEAGE_KEYS
        and value.get("schema") == LINEAGE_SCHEMA
        and value.get("lifecycle") in _PERSISTENT_MODES
        and isinstance(value.get("run_id"), str) and bool(value["run_id"])
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
        and all(_DIGEST.fullmatch(str(value.get(key, ""))) for key in (
            "client_config_sha256", "server_source_generation_sha256",
            "server_generation_sha256", "server_identity_sha256",
            "dependency_receipt_sha256", "gpu_lease_digest",
            "execution_attestation_sha256", "local_runtime_receipt_sha256",
        ))
        and isinstance(value.get("reward_eligible"), bool)
        and isinstance(value.get("cleanup_verified"), bool)
        and isinstance(value.get("cleanup_succeeded"), bool)
    )
    if not valid:
        _reject("Local server lineage document is invalid")


def _local_runtime(attestation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    runtime = attestation.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    receipt = runtime.get("serving_runtime_receipt")
    if not isinstance(receipt, Mapping):
        return None
    if receipt.get("schema") != "apex.magpie-local-runtime-observation/v2":
        return None
    if (
        receipt.get("execution_mode") != "local"
    ):
        _reject("Local server runtime observation is not verified")
    if (
        receipt.get("verified") is not True
        or receipt.get("process_succeeded") is not True
        or receipt.get("errors") != []
    ):
        return None
    return receipt


def _single_binding(
    bindings: Iterable[Mapping[str, Any]], role: str, *, required: bool
) -> ArtifactReceipt | None:
    found = tuple(
        ArtifactReceipt.from_dict(dict(item["receipt"]))
        for item in bindings
        if isinstance(item, Mapping) and item.get("role") == role
        and isinstance(item.get("receipt"), Mapping)
    )
    if len(found) > 1 or required and len(found) != 1:
        _reject(f"Server lineage requires exactly one {role} artifact")
    return found[0] if found else None


def _read_object(store: ArtifactStore, receipt: ArtifactReceipt) -> Mapping[str, Any]:
    raw = store.read_bytes(receipt)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Server lineage evidence is not JSON", "e2e_server_lineage_mismatch"
        ) from error
    if not isinstance(value, Mapping):
        _reject("Server lineage evidence is not an object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _reject(f"Local server {label} is invalid")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        _reject(f"Local server {label} digest is invalid")
    return value


def _reject(message: str) -> None:
    raise IntegrityError(message, "e2e_server_lineage_mismatch")


__all__ = [
    "LINEAGE_SCHEMA",
    "LocalServerLifecycleProjection",
    "LocalServerLineageEvidence",
    "capture_local_server_lineage",
    "replay_local_server_lineage",
    "require_resumable_server_lineage",
]
