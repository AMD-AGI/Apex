"""Offline replay of four-leg matched E2E promotion evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EVerdict,
    e2e_comparison_selection_policy,
    evaluate_current_anchor,
    select_conservative_e2e_verdict,
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
from .e2e_gpu_lease_validation import validate_gpu_lease
from .models import CandidateEpisode, EpisodeEvent


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ORDER = ("anchor", "candidate", "candidate", "anchor")
_SLOTS = ("ab-anchor", "ab-candidate", "ba-candidate", "ba-anchor")
_PAIR_FIELDS = frozenset(
    {
        "schema",
        "pair_id",
        "window_id",
        "attempt_id",
        "candidate_id",
        "opportunity_id",
        "anchor_id",
        "anchor_generation",
        "gpu_lease_digest",
        "gpu_device_scope",
        "order",
        "anchor_config_sha256",
        "candidate_config_sha256",
        "anchor_image",
        "candidate_image",
        "observations",
        "comparisons",
        "selection_policy",
        "selected_comparison",
        "verdict",
    }
)


@dataclass(frozen=True, slots=True)
class MatchedPromotionReplay:
    """A v2 pair reconstructed independently from its four raw measurements."""

    receipt: ArtifactReceipt
    comparisons: tuple[E2EVerdict, E2EVerdict]
    selected_comparison: int
    verdict: E2EVerdict


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
    """Validate every v2 pair edge and recompute its conservative verdict."""

    pair_event = _pair_event(child)
    pair_receipt = single_event_receipt(pair_event, "matched_promotion_pair")
    pair = read_json(artifacts, pair_receipt, canonical=True)
    _validate_pair_identity(pair_event, pair, pair_receipt, child)
    observations = _load_observations(
        child,
        pair_event,
        pair,
        artifacts,
        protocol_hash,
    )
    _validate_side_identity(pair, observations, delivery)
    validate_gpu_lease(
        run_id=run_id,
        event=pair_event,
        pair=pair,
        artifacts=artifacts,
    )
    comparisons = (
        evaluate_current_anchor(
            observations[0].measurement,
            observations[1].measurement,
            acceptance_policy,
        ),
        evaluate_current_anchor(
            observations[3].measurement,
            observations[2].measurement,
            acceptance_policy,
        ),
    )
    selected = select_conservative_e2e_verdict(comparisons)
    verdict = comparisons[selected]
    _validate_derived_pair(pair_event, pair, comparisons, selected, verdict)
    _validate_decision(child, decision, delivery, pair_receipt, verdict)
    return MatchedPromotionReplay(pair_receipt, comparisons, selected, verdict)


def _pair_event(child: CandidateEpisode) -> EpisodeEvent:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and event.payload.get("measurement_kind") == "matched_promotion_ab_ba"
    )
    if len(events) != 1:
        reject("E2E attempt has no unique matched-promotion v2 event")
    return events[0]


def _validate_pair_identity(
    event: EpisodeEvent,
    pair: Mapping[str, Any],
    receipt: ArtifactReceipt,
    child: CandidateEpisode,
) -> None:
    if child.candidate_id is None or child.opportunity_id is None:
        reject("Matched promotion has incomplete child lineage")
    expected = {
        "attempt_id": child.attempt_id,
        "candidate_id": child.candidate_id,
        "opportunity_id": child.opportunity_id,
    }
    if (
        pair.get("schema") != "apex.e2e-matched-promotion/v2"
        or set(pair) != _PAIR_FIELDS
        or pair.get("order") != list(_ORDER)
        or any(pair.get(key) != value for key, value in expected.items())
        or pair.get("anchor_generation") != child.anchor_generation
        or pair.get("selection_policy") != e2e_comparison_selection_policy()
        or not _text(pair.get("pair_id"))
        or not _text(pair.get("window_id"))
        or not _text(pair.get("anchor_id"))
        or not _digest(pair.get("gpu_lease_digest"))
        or not _text(pair.get("gpu_device_scope"))
        or not _digest(pair.get("anchor_config_sha256"))
        or not _digest(pair.get("candidate_config_sha256"))
    ):
        reject("Matched-promotion v2 identity or policy differs")
    event_expected = {
        **expected,
        "anchor_id": pair.get("anchor_id"),
        "anchor_generation": pair.get("anchor_generation"),
        "measurement_kind": "matched_promotion_ab_ba",
        "pair_id": pair.get("pair_id"),
        "window_id": pair.get("window_id"),
        "gpu_lease_digest": pair.get("gpu_lease_digest"),
        "order": list(_ORDER),
    }
    if any(event.payload.get(key) != value for key, value in event_expected.items()):
        reject("Matched-promotion event targets another pair")
    if single_event_receipt(event, "matched_promotion_pair") != receipt:
        reject("Matched-promotion pair receipt differs")
    expected_roles = {"matched_promotion_pair", "promotion_gpu_lease"}
    expected_roles.update(
        f"promotion_{position}_{side}_{kind}"
        for position, side in enumerate(_ORDER)
        for kind in ("normalized", "quality", "config")
    )
    roles = tuple(item.role for item in event.artifacts)
    if len(roles) != len(expected_roles) or set(roles) != expected_roles:
        reject("Matched-promotion aggregate artifact roles differ")


def _load_observations(
    child: CandidateEpisode,
    pair_event: EpisodeEvent,
    pair: Mapping[str, Any],
    artifacts: ArtifactStore,
    protocol_hash: str,
) -> tuple[BenchmarkBundle, ...]:
    raw = pair.get("observations")
    if not isinstance(raw, list) or len(raw) != 4:
        reject("Matched-promotion observations are incomplete")
    prefix = f"promotion-{pair.get('attempt_id')}-{pair.get('window_id')}-"
    window_events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and isinstance(event.payload.get("action_id"), str)
        and event.payload["action_id"].startswith(prefix)
    )
    if len(window_events) != 4:
        reject("Matched-promotion window does not contain exactly four legs")
    bundles: list[BenchmarkBundle] = []
    sequences: list[int] = []
    for position, side in enumerate(_ORDER):
        observation = mapping(raw[position], "promotion observation")
        action_id = _action_id(pair, position)
        event = _leg_event(child, action_id)
        _validate_leg_lineage(event, pair, action_id)
        bundle = load_benchmark_bundle(event, artifacts, protocol_hash)
        expected = _observation_document(position, side, action_id, bundle)
        if dict(observation) != expected:
            reject("Matched-promotion observation differs from raw CAS evidence")
        _validate_pair_bindings(pair_event, position, side, bundle)
        bundles.append(bundle)
        sequences.append(event.sequence)
    if sequences != sorted(sequences) or len(set(sequences)) != 4:
        reject("Matched-promotion leg order is ambiguous")
    if sequences[-1] >= pair_event.sequence:
        reject("Matched-promotion pair was recorded before its final leg")
    return tuple(bundles)


def _action_id(pair: Mapping[str, Any], position: int) -> str:
    attempt_id = _text(pair.get("attempt_id"))
    window_id = _text(pair.get("window_id"))
    if attempt_id is None or window_id is None:
        reject("Matched-promotion action identity is invalid")
    return f"promotion-{attempt_id}-{window_id}-{_SLOTS[position]}"


def _leg_event(child: CandidateEpisode, action_id: str) -> EpisodeEvent:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and event.payload.get("action_id") == action_id
    )
    if len(events) != 1:
        reject("Matched-promotion leg is missing or duplicated")
    return events[0]


def _validate_leg_lineage(
    event: EpisodeEvent,
    pair: Mapping[str, Any],
    action_id: str,
) -> None:
    expected = {
        "action_id": action_id,
        "attempt_id": pair.get("attempt_id"),
        "candidate_id": pair.get("candidate_id"),
        "opportunity_id": pair.get("opportunity_id"),
        "anchor_generation": pair.get("anchor_generation"),
    }
    if any(event.payload.get(key) != value for key, value in expected.items()):
        reject("Matched-promotion leg lineage differs")


def _observation_document(
    position: int,
    side: str,
    action_id: str,
    bundle: BenchmarkBundle,
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
    if any(single_event_receipt(event, role) != receipt for role, receipt in expected.items()):
        reject("Matched-promotion aggregate bindings differ from a leg")


def _validate_side_identity(
    pair: Mapping[str, Any],
    observations: Sequence[BenchmarkBundle],
    delivery: DeliveryEvidence,
) -> None:
    anchors = (observations[0], observations[3])
    candidates = (observations[1], observations[2])
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
        reject("Matched-promotion side config or image identity differs")
    for candidate in candidates:
        validate_candidate_runtime(candidate, delivery)


def _image(bundle: BenchmarkBundle) -> dict[str, Any]:
    serving = mapping(bundle.normalized.get("serving_runtime"), "serving runtime")
    return {
        "requested_image": serving.get("requested_image"),
        "resolved_image_id": serving.get("resolved_image_id"),
    }


def _image_tuple(bundle: BenchmarkBundle) -> tuple[Any, Any]:
    value = _image(bundle)
    return value["requested_image"], value["resolved_image_id"]


def _validate_derived_pair(
    event: EpisodeEvent,
    pair: Mapping[str, Any],
    comparisons: tuple[E2EVerdict, E2EVerdict],
    selected: int,
    verdict: E2EVerdict,
) -> None:
    expected_comparisons = [item.to_dict() for item in comparisons]
    if (
        pair.get("comparisons") != expected_comparisons
        or pair.get("selection_policy") != e2e_comparison_selection_policy()
        or pair.get("selected_comparison") != selected
        or pair.get("verdict") != verdict.to_dict()
        or event.payload.get("verdict") != verdict.to_dict()
    ):
        reject("Matched-promotion selected verdict differs from conservative replay")


def _validate_decision(
    child: CandidateEpisode,
    decision: Mapping[str, Any],
    delivery: DeliveryEvidence,
    pair_receipt: ArtifactReceipt,
    verdict: E2EVerdict,
) -> None:
    expected = {
        "micro_receipt": single_child_receipt(child, "micro_qualification").digest,
        "safety_receipt": single_child_receipt(child, "safety_qualification").digest,
        "delivery_receipt": delivery.receipt.digest,
        "promotion_pair_receipt": pair_receipt.digest,
        "measurement_verdict": verdict.to_dict(),
    }
    if (
        any(decision.get(key) != value for key, value in expected.items())
        or "benchmark_receipt" in decision
        or single_child_receipt(child, "matched_promotion_pair") != pair_receipt
    ):
        reject("Decision does not bind the replayed matched-promotion pair")
    rewards = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "reward_committed"
    )
    if (
        len(rewards) != 1
        or single_event_receipt(rewards[0], "matched_promotion_pair") != pair_receipt
    ):
        reject("Reward does not bind the replayed matched-promotion pair")


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _digest(value: Any) -> str | None:
    return value if isinstance(value, str) and _DIGEST.fullmatch(value) else None


__all__ = ["MatchedPromotionReplay", "replay_matched_promotion"]
