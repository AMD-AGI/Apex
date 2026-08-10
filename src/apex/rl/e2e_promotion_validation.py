"""Offline replay of multi-window paired E2E promotion evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EPairedMeasurement,
    E2EPairedVerdict,
    E2EPairedWindow,
    evaluate_paired_current_anchor,
)
from apex.storage import ArtifactReceipt, ArtifactStore

from .e2e_benchmark_validation import (
    BenchmarkBundle,
    DeliveryEvidence,
    load_benchmark_bundle,
    mapping,
    read_json,
    reject,
    single_child_receipt,
    single_event_receipt,
    validate_candidate_runtime,
)
from .e2e_gpu_lease_validation import (
    validate_gpu_lease,
    validate_measurement_bracket,
)
from .models import CandidateEpisode, EpisodeEvent


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ORDER = ("anchor", "candidate", "candidate", "anchor")
_SLOTS = ("ab-anchor", "ab-candidate", "ba-candidate", "ba-anchor")
_PAIR_FIELDS = frozenset(
    {
        "schema",
        "pair_id",
        "window_ids",
        "attempt_id",
        "candidate_id",
        "opportunity_id",
        "anchor_id",
        "anchor_generation",
        "gpu_lease_digest",
        "gpu_device_scope",
        "window_order",
        "anchor_config_sha256",
        "candidate_config_sha256",
        "anchor_image",
        "candidate_image",
        "observations",
        "measurement",
        "verdict",
    }
)


@dataclass(frozen=True, slots=True)
class MatchedPromotionReplay:
    """A paired promotion reconstructed independently from all raw observations."""

    receipt: ArtifactReceipt
    measurement: E2EPairedMeasurement
    verdict: E2EPairedVerdict


def replay_matched_promotion(
    *,
    run_id: str,
    child: CandidateEpisode,
    artifacts: ArtifactStore,
    protocol_hash: str,
    acceptance_policy: E2EAcceptancePolicy,
    delivery: DeliveryEvidence,
    decision: Mapping[str, Any],
) -> MatchedPromotionReplay:
    """Validate every paired edge and recompute the point/confidence verdict."""

    pair_event = _pair_event(child)
    pair_receipt = single_event_receipt(pair_event, "paired_promotion")
    pair = read_json(artifacts, pair_receipt, canonical=True)
    window_ids = _validate_pair_identity(pair_event, pair, pair_receipt, child)
    observations = _load_observations(
        run_id, child, pair_event, pair, window_ids, artifacts, protocol_hash
    )
    _validate_side_identity(pair, observations, delivery)
    validate_gpu_lease(run_id=run_id, event=pair_event, pair=pair, artifacts=artifacts)
    measurement = _measurement(window_ids, observations, acceptance_policy)
    verdict = evaluate_paired_current_anchor(measurement, acceptance_policy)
    _validate_derived_pair(pair_event, pair, measurement, verdict)
    _validate_decision(child, decision, delivery, pair_receipt, verdict)
    return MatchedPromotionReplay(pair_receipt, measurement, verdict)


def _pair_event(child: CandidateEpisode) -> EpisodeEvent:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and event.payload.get("measurement_kind") == "paired_promotion_abba"
    )
    if len(events) != 1:
        reject("E2E attempt has no unique paired-promotion event")
    return events[0]


def _validate_pair_identity(
    event: EpisodeEvent,
    pair: Mapping[str, Any],
    receipt: ArtifactReceipt,
    child: CandidateEpisode,
) -> tuple[str, ...]:
    if child.candidate_id is None or child.opportunity_id is None:
        reject("Paired promotion has incomplete child lineage")
    window_ids = _window_ids(pair.get("window_ids"))
    expected = {
        "attempt_id": child.attempt_id,
        "candidate_id": child.candidate_id,
        "opportunity_id": child.opportunity_id,
    }
    if (
        pair.get("schema") != "apex.e2e-paired-promotion/v1"
        or set(pair) != _PAIR_FIELDS
        or pair.get("window_order") != list(_ORDER)
        or any(pair.get(key) != value for key, value in expected.items())
        or pair.get("anchor_generation") != child.anchor_generation
        or not _text(pair.get("pair_id"))
        or not _text(pair.get("anchor_id"))
        or not _digest(pair.get("gpu_lease_digest"))
        or not _text(pair.get("gpu_device_scope"))
        or not _digest(pair.get("anchor_config_sha256"))
        or not _digest(pair.get("candidate_config_sha256"))
    ):
        reject("Paired-promotion identity differs")
    measurement = mapping(pair.get("measurement"), "paired measurement")
    event_expected = {
        **expected,
        "anchor_id": pair.get("anchor_id"),
        "anchor_generation": pair.get("anchor_generation"),
        "measurement_kind": "paired_promotion_abba",
        "pair_id": pair.get("pair_id"),
        "window_ids": list(window_ids),
        "paired_measurement_id": _json_digest(measurement),
        "gpu_lease_digest": pair.get("gpu_lease_digest"),
        "window_order": list(_ORDER),
    }
    if any(event.payload.get(key) != value for key, value in event_expected.items()):
        reject("Paired-promotion event targets another measurement")
    if single_event_receipt(event, "paired_promotion") != receipt:
        reject("Paired-promotion receipt differs")
    _validate_aggregate_roles(event, window_ids)
    return window_ids


def _validate_aggregate_roles(event: EpisodeEvent, window_ids: tuple[str, ...]) -> None:
    expected = {"paired_promotion", "promotion_gpu_lease"}
    expected.update(
        f"promotion_{position}_{_ORDER[position % 4]}_{kind}"
        for position in range(len(window_ids) * len(_ORDER))
        for kind in ("normalized", "quality", "config")
    )
    roles = tuple(item.role for item in event.artifacts)
    if len(roles) != len(expected) or set(roles) != expected:
        reject("Paired-promotion aggregate artifact roles differ")


def _load_observations(
    run_id: str,
    child: CandidateEpisode,
    pair_event: EpisodeEvent,
    pair: Mapping[str, Any],
    window_ids: tuple[str, ...],
    artifacts: ArtifactStore,
    protocol_hash: str,
) -> tuple[BenchmarkBundle, ...]:
    raw = pair.get("observations")
    expected_count = len(window_ids) * len(_ORDER)
    if not isinstance(raw, list) or len(raw) != expected_count:
        reject("Paired-promotion observations are incomplete")
    bundles: list[BenchmarkBundle] = []
    sequences: list[int] = []
    for position in range(expected_count):
        side = _ORDER[position % len(_ORDER)]
        observation = mapping(raw[position], "promotion observation")
        action_id = _action_id(pair, window_ids, position)
        event = _leg_event(child, action_id)
        _validate_leg_lineage(event, pair, action_id)
        validate_measurement_bracket(
            run_id=run_id,
            action_id=action_id,
            lease_digest=str(pair.get("gpu_lease_digest")),
            event=event,
            artifacts=artifacts,
        )
        bundle = load_benchmark_bundle(event, artifacts, protocol_hash)
        if dict(observation) != _observation_document(position, side, action_id, bundle):
            reject("Paired-promotion observation differs from raw CAS evidence")
        _validate_pair_bindings(pair_event, position, side, bundle)
        bundles.append(bundle)
        sequences.append(event.sequence)
    if sequences != sorted(sequences) or len(set(sequences)) != expected_count:
        reject("Paired-promotion leg order is ambiguous")
    if sequences[-1] >= pair_event.sequence:
        reject("Paired-promotion was recorded before its final leg")
    return tuple(bundles)


def _action_id(
    pair: Mapping[str, Any], window_ids: tuple[str, ...], position: int
) -> str:
    attempt_id = _text(pair.get("attempt_id"))
    if attempt_id is None:
        reject("Paired-promotion action identity is invalid")
    local = position % len(_ORDER)
    window_id = window_ids[position // len(_ORDER)]
    return f"promotion-{attempt_id}-{window_id}-{_SLOTS[local]}"


def _leg_event(child: CandidateEpisode, action_id: str) -> EpisodeEvent:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and event.payload.get("action_id") == action_id
    )
    if len(events) != 1:
        reject("Paired-promotion leg is missing or duplicated")
    return events[0]


def _validate_leg_lineage(
    event: EpisodeEvent, pair: Mapping[str, Any], action_id: str
) -> None:
    expected = {
        "action_id": action_id,
        "attempt_id": pair.get("attempt_id"),
        "candidate_id": pair.get("candidate_id"),
        "opportunity_id": pair.get("opportunity_id"),
        "anchor_generation": pair.get("anchor_generation"),
    }
    if any(event.payload.get(key) != value for key, value in expected.items()):
        reject("Paired-promotion leg lineage differs")


def _observation_document(
    position: int, side: str, action_id: str, bundle: BenchmarkBundle
) -> dict[str, Any]:
    serving = mapping(bundle.normalized.get("serving_runtime"), "serving runtime")
    return {
        "position": position,
        "side": side,
        "action_id": action_id,
        "measurement": bundle.measurement.to_dict(),
        "normalized_receipt": bundle.normalized_receipt.digest,
        "quality_receipt": bundle.quality_receipt.digest,
        "config_receipt": bundle.config.digest,
        "requested_image": serving.get("requested_image"),
        "resolved_image_id": serving.get("resolved_image_id"),
    }


def _validate_pair_bindings(
    event: EpisodeEvent,
    position: int,
    side: str,
    bundle: BenchmarkBundle,
) -> None:
    prefix = f"promotion_{position}_{side}"
    expected = {
        f"{prefix}_normalized": bundle.normalized_receipt,
        f"{prefix}_quality": bundle.quality_receipt,
        f"{prefix}_config": bundle.config,
    }
    if any(single_event_receipt(event, role) != value for role, value in expected.items()):
        reject("Paired-promotion aggregate bindings differ from a leg")


def _validate_side_identity(
    pair: Mapping[str, Any],
    observations: Sequence[BenchmarkBundle],
    delivery: DeliveryEvidence,
) -> None:
    anchors = tuple(item for index, item in enumerate(observations) if _ORDER[index % 4] == "anchor")
    candidates = tuple(
        item for index, item in enumerate(observations) if _ORDER[index % 4] == "candidate"
    )
    if (
        len({item.config.digest for item in anchors}) != 1
        or len({item.config.digest for item in candidates}) != 1
        or pair.get("anchor_config_sha256") != anchors[0].config.digest
        or pair.get("candidate_config_sha256") != candidates[0].config.digest
        or pair.get("anchor_image") != _image(anchors[0])
        or pair.get("candidate_image") != _image(candidates[0])
        or len({_image_tuple(item) for item in anchors}) != 1
        or len({_image_tuple(item) for item in candidates}) != 1
    ):
        reject("Paired-promotion side config or image identity differs")
    for candidate in candidates:
        validate_candidate_runtime(candidate, delivery)


def _measurement(
    window_ids: tuple[str, ...],
    observations: tuple[BenchmarkBundle, ...],
    policy: E2EAcceptancePolicy,
) -> E2EPairedMeasurement:
    windows = tuple(
        E2EPairedWindow(
            window_id,
            observations[offset].measurement,
            observations[offset + 1].measurement,
            observations[offset + 2].measurement,
            observations[offset + 3].measurement,
        )
        for offset, window_id in zip(
            range(0, len(observations), len(_ORDER)), window_ids, strict=True
        )
    )
    return E2EPairedMeasurement(windows, policy.digest, policy.min_paired_windows)


def _validate_derived_pair(
    event: EpisodeEvent,
    pair: Mapping[str, Any],
    measurement: E2EPairedMeasurement,
    verdict: E2EPairedVerdict,
) -> None:
    if (
        pair.get("measurement") != measurement.to_dict()
        or pair.get("verdict") != verdict.to_dict()
        or event.payload.get("paired_measurement_id") != measurement.digest
        or event.payload.get("verdict") != verdict.to_dict()
    ):
        reject("Paired-promotion verdict differs from raw replay")


def _validate_decision(
    child: CandidateEpisode,
    decision: Mapping[str, Any],
    delivery: DeliveryEvidence,
    pair_receipt: ArtifactReceipt,
    verdict: E2EPairedVerdict,
) -> None:
    expected = {
        "micro_receipt": single_child_receipt(child, "micro_qualification").digest,
        "safety_receipt": single_child_receipt(child, "safety_qualification").digest,
        "delivery_receipt": delivery.receipt.digest,
        "paired_promotion_receipt": pair_receipt.digest,
        "measurement_verdict": verdict.to_dict(),
    }
    if (
        any(decision.get(key) != value for key, value in expected.items())
        or "benchmark_receipt" in decision
        or single_child_receipt(child, "paired_promotion") != pair_receipt
    ):
        reject("Decision does not bind the replayed paired promotion")
    rewards = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "reward_committed"
    )
    if len(rewards) != 1 or single_event_receipt(rewards[0], "paired_promotion") != pair_receipt:
        reject("Reward does not bind the replayed paired promotion")


def _image(bundle: BenchmarkBundle) -> dict[str, Any]:
    serving = mapping(bundle.normalized.get("serving_runtime"), "serving runtime")
    return {
        "requested_image": serving.get("requested_image"),
        "resolved_image_id": serving.get("resolved_image_id"),
    }


def _image_tuple(bundle: BenchmarkBundle) -> tuple[Any, Any]:
    value = _image(bundle)
    return value["requested_image"], value["resolved_image_id"]


def _window_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < 3:
        reject("Paired-promotion window IDs are invalid")
    result = tuple(item for item in value if _text(item) is not None)
    if len(result) != len(value) or len(set(result)) != len(result):
        reject("Paired-promotion window IDs are invalid")
    return result


def _json_digest(value: Mapping[str, Any]) -> str:
    from apex.core import sha256_json

    return sha256_json(dict(value))


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _digest(value: Any) -> str | None:
    return value if isinstance(value, str) and _DIGEST.fullmatch(value) else None


__all__ = ["MatchedPromotionReplay", "replay_matched_promotion"]
