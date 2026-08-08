"""Journal/CAS reconstruction for matched E2E promotion windows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError, sha256_file, sha256_json
from apex.evaluation import E2EAcceptancePolicy, E2EVerdict, evaluate_current_anchor
from apex.storage import ArtifactReceipt, EventRecord

from .promotion import MatchedPromotion, PromotionObservation
from .recovery_artifacts import read_json_object, recover_measurement
from .recovery_bindings import unique_role, verify_benchmark_event
from .run_record import E2ERunRecord
from .services import CandidateDeployment


_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{16}")
_ORDER = ("anchor", "candidate", "candidate", "anchor")
_SLOTS = ("ab-anchor", "ab-candidate", "ba-candidate", "ba-anchor")


def recover_matched_promotion(
    record: E2ERunRecord,
    *,
    pair_event: EventRecord,
    events_by_key: Mapping[str, EventRecord],
    protocol_hash: str,
    policy: E2EAcceptancePolicy,
    attempt_id: str,
    candidate_id: str,
    opportunity_id: str,
) -> MatchedPromotion:
    """Recover one complete pair; partial windows deliberately have no pair event."""

    receipt = _required_role(pair_event, "matched_promotion_pair")
    lease_receipt = _required_role(pair_event, "promotion_gpu_lease")
    value = read_json_object(record, receipt, label="matched promotion")
    identity = _identity(
        value,
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        opportunity_id=opportunity_id,
    )
    _verify_pair_event(pair_event, value, receipt)
    _verify_lease(record, lease_receipt, value)
    observations = _observations(
        record,
        value,
        events_by_key=events_by_key,
        pair_event=pair_event,
        protocol_hash=protocol_hash,
    )
    comparisons = _comparisons(observations, policy)
    selected = _selected(comparisons)
    verdict = comparisons[selected]
    _verify_derived(value, observations, comparisons, selected, verdict)
    return MatchedPromotion(
        *identity,
        observations,
        comparisons,
        selected,
        verdict,
        receipt,
    )


def validate_promotion_context(
    promotion: MatchedPromotion,
    *,
    anchor_id: str,
    anchor_generation: int,
    anchor_config: Path,
    anchor_image_id: str | None,
    deployment: CandidateDeployment,
) -> None:
    """Bind an intrinsic pair to the then-current live anchor and deployment."""

    expected_candidate = deployment.deployed_image_id
    if (
        promotion.anchor_id != anchor_id
        or promotion.anchor_generation != anchor_generation
        or promotion.anchor_config_sha256 != sha256_file(anchor_config)
        or promotion.candidate_config_sha256
        != sha256_file(deployment.measurement_config)
        or expected_candidate is None
        or promotion.candidate_image.get("requested_image") != expected_candidate
        or promotion.candidate_image.get("resolved_image_id") != expected_candidate
        or (
            anchor_image_id is not None
            and (
                promotion.anchor_image.get("requested_image") != anchor_image_id
                or promotion.anchor_image.get("resolved_image_id") != anchor_image_id
            )
        )
    ):
        raise IntegrityError(
            "Matched pair targets another live anchor or runtime",
            "promotion_anchor_lineage_mismatch",
        )


def verify_promotion_reward_binding(
    event: EventRecord, promotion: MatchedPromotion
) -> None:
    receipt = _required_role(event, "matched_promotion_pair")
    vector = event.payload.get("reward_vector")
    metrics = vector.get("metrics") if isinstance(vector, Mapping) else None
    if (
        receipt.digest != promotion.receipt.digest
        or not isinstance(metrics, Mapping)
        or metrics.get("anchor_measurement_id")
        != promotion.verdict.anchor_measurement_id
        or metrics.get("candidate_measurement_id")
        != promotion.verdict.candidate_measurement_id
    ):
        raise IntegrityError("Reward uses another matched pair", "promotion_reward_mismatch")


def _identity(
    value: Mapping[str, Any],
    *,
    attempt_id: str,
    candidate_id: str,
    opportunity_id: str,
) -> tuple[Any, ...]:
    if (
        value.get("schema") != "apex.e2e-matched-promotion/v1"
        or value.get("attempt_id") != attempt_id
        or value.get("candidate_id") != candidate_id
        or value.get("opportunity_id") != opportunity_id
        or value.get("order") != list(_ORDER)
    ):
        raise IntegrityError("Matched pair identity differs", "promotion_lineage_mismatch")
    anchor_generation = _integer(value.get("anchor_generation"), "anchor generation")
    return (
        _text(value.get("pair_id"), "pair id"),
        _text(value.get("window_id"), "window id"),
        attempt_id,
        candidate_id,
        opportunity_id,
        _text(value.get("anchor_id"), "anchor id"),
        anchor_generation,
        _sha256(value.get("gpu_lease_digest"), "GPU lease"),
        _text(value.get("gpu_device_scope"), "GPU scope"),
        _sha256(value.get("anchor_config_sha256"), "anchor config"),
        _sha256(value.get("candidate_config_sha256"), "candidate config"),
        dict(_mapping(value.get("anchor_image"), "anchor image")),
        dict(_mapping(value.get("candidate_image"), "candidate image")),
    )


def _observations(
    record: E2ERunRecord,
    value: Mapping[str, Any],
    *,
    events_by_key: Mapping[str, EventRecord],
    pair_event: EventRecord,
    protocol_hash: str,
) -> tuple[PromotionObservation, ...]:
    raw = value.get("observations")
    if not isinstance(raw, list) or len(raw) != 4:
        raise IntegrityError("Matched observations are incomplete", "promotion_lineage_mismatch")
    observations = tuple(
        _observation(
            record,
            _mapping(item, "promotion observation"),
            position=position,
            pair=value,
            events_by_key=events_by_key,
            protocol_hash=protocol_hash,
        )
        for position, item in enumerate(raw)
    )
    measurement_events = tuple(
        events_by_key[f"benchmark.{item.action_id}.measurement"] for item in observations
    )
    sequences = tuple(item.sequence for item in measurement_events)
    if tuple(sorted(sequences)) != sequences or sequences[-1] >= pair_event.sequence:
        raise IntegrityError("Matched window order differs", "promotion_order_mismatch")
    return observations


def _observation(
    record: E2ERunRecord,
    item: Mapping[str, Any],
    *,
    position: int,
    pair: Mapping[str, Any],
    events_by_key: Mapping[str, EventRecord],
    protocol_hash: str,
) -> PromotionObservation:
    side = _ORDER[position]
    window_id = _text(pair.get("window_id"), "window id")
    attempt_id = _text(pair.get("attempt_id"), "attempt id")
    action_id = f"promotion-{attempt_id}-{window_id}-{_SLOTS[position]}"
    if (
        item.get("position") != position
        or item.get("side") != side
        or item.get("action_id") != action_id
    ):
        raise IntegrityError("Matched observation order differs", "promotion_order_mismatch")
    event = events_by_key.get(f"benchmark.{action_id}.measurement")
    if event is None:
        raise IntegrityError("Matched benchmark event is missing", "promotion_lineage_mismatch")
    _verify_measurement_lineage(event, pair, action_id)
    receipts = _observation_receipts(event, item)
    verify_benchmark_event(
        (event,), normalized=receipts[0], quality=receipts[1], config=receipts[2]
    )
    measurement = recover_measurement(
        record,
        receipts[0],
        protocol_hash=protocol_hash,
        quality_receipt=receipts[1],
    )
    if item.get("measurement") != measurement.to_dict():
        raise IntegrityError("Matched metrics differ", "promotion_measurement_mismatch")
    runtime = _runtime_identity(record, receipts[0], receipts[2])
    if (
        item.get("requested_image") != runtime[0]
        or item.get("resolved_image_id") != runtime[1]
    ):
        raise IntegrityError("Matched image binding differs", "promotion_image_mismatch")
    return PromotionObservation(
        position,
        side,
        action_id,
        measurement,
        receipts[0],
        receipts[1],
        receipts[2],
        runtime[0],
        runtime[1],
    )


def _observation_receipts(
    event: EventRecord, item: Mapping[str, Any]
) -> tuple[ArtifactReceipt, ArtifactReceipt, ArtifactReceipt]:
    values = tuple(
        _required_role(event, role)
        for role in ("normalized_benchmark", "quality_evidence", "benchmark_config")
    )
    expected = (
        item.get("normalized_receipt"),
        item.get("quality_receipt"),
        item.get("config_receipt"),
    )
    if tuple(receipt.digest for receipt in values) != expected:
        raise IntegrityError("Matched receipts were exchanged", "promotion_receipt_mismatch")
    return values


def _verify_measurement_lineage(
    event: EventRecord, pair: Mapping[str, Any], action_id: str
) -> None:
    expected = {
        "action_id": action_id,
        "attempt_id": pair.get("attempt_id"),
        "candidate_id": pair.get("candidate_id"),
        "opportunity_id": pair.get("opportunity_id"),
        "anchor_generation": pair.get("anchor_generation"),
    }
    if event.event_type != "measurement_result" or any(
        event.payload.get(name) != observed for name, observed in expected.items()
    ):
        raise IntegrityError("Matched measurement lineage differs", "promotion_lineage_mismatch")


def _runtime_identity(
    record: E2ERunRecord,
    normalized: ArtifactReceipt,
    config: ArtifactReceipt,
) -> tuple[str | None, str | None]:
    value = read_json_object(record, normalized, label="matched benchmark")
    runtime = _mapping(value.get("serving_runtime"), "serving runtime")
    required = runtime.get("required") is True
    requested = runtime.get("requested_image")
    resolved = runtime.get("resolved_image_id")
    if required:
        if (
            runtime.get("passed") is not True
            or runtime.get("process_succeeded") is not True
            or runtime.get("input_config_sha256") != config.digest
            or not isinstance(requested, str)
            or not requested
            or not isinstance(resolved, str)
            or not _IMAGE_ID.fullmatch(resolved)
        ):
            raise IntegrityError("Matched runtime proof is invalid", "promotion_runtime_mismatch")
    elif any(item is not None for item in (requested, resolved, runtime.get("input_config_sha256"))):
        raise IntegrityError("Non-container runtime claims image proof", "promotion_runtime_mismatch")
    return requested, resolved


def _comparisons(
    observations: tuple[PromotionObservation, ...], policy: E2EAcceptancePolicy
) -> tuple[E2EVerdict, E2EVerdict]:
    return (
        evaluate_current_anchor(
            observations[0].measurement, observations[1].measurement, policy
        ),
        evaluate_current_anchor(
            observations[3].measurement, observations[2].measurement, policy
        ),
    )


def _selected(comparisons: tuple[E2EVerdict, E2EVerdict]) -> int:
    failures = tuple(index for index, item in enumerate(comparisons) if not item.keep)
    return failures[0] if failures else min(
        range(2), key=lambda index: comparisons[index].throughput_gain_pct
    )


def _verify_derived(
    value: Mapping[str, Any],
    observations: tuple[PromotionObservation, ...],
    comparisons: tuple[E2EVerdict, E2EVerdict],
    selected: int,
    verdict: E2EVerdict,
) -> None:
    anchor = tuple(item for item in observations if item.side == "anchor")
    candidate = tuple(item for item in observations if item.side == "candidate")
    if (
        value.get("comparisons") != [item.to_dict() for item in comparisons]
        or value.get("selected_comparison") != selected
        or value.get("verdict") != verdict.to_dict()
        or value.get("anchor_config_sha256") != anchor[0].config.digest
        or value.get("candidate_config_sha256") != candidate[0].config.digest
        or len({item.config.digest for item in anchor}) != 1
        or len({item.config.digest for item in candidate}) != 1
        or value.get("anchor_image") != _image(anchor[0])
        or value.get("candidate_image") != _image(candidate[0])
        or len({_image_tuple(item) for item in anchor}) != 1
        or len({_image_tuple(item) for item in candidate}) != 1
    ):
        raise IntegrityError("Matched pair derivation differs", "promotion_replay_mismatch")


def _verify_pair_event(
    event: EventRecord, value: Mapping[str, Any], receipt: ArtifactReceipt
) -> None:
    expected = {
        "attempt_id": value.get("attempt_id"),
        "candidate_id": value.get("candidate_id"),
        "opportunity_id": value.get("opportunity_id"),
        "anchor_id": value.get("anchor_id"),
        "anchor_generation": value.get("anchor_generation"),
        "pair_id": value.get("pair_id"),
        "window_id": value.get("window_id"),
        "gpu_lease_digest": value.get("gpu_lease_digest"),
        "order": value.get("order"),
        "verdict": value.get("verdict"),
    }
    if any(event.payload.get(name) != observed for name, observed in expected.items()):
        raise IntegrityError("Matched pair event differs", "promotion_lineage_mismatch")
    if _required_role(event, "matched_promotion_pair").digest != receipt.digest:
        raise IntegrityError("Matched pair receipt differs", "promotion_receipt_mismatch")
    _verify_pair_observation_bindings(event, value)


def _verify_pair_observation_bindings(
    event: EventRecord, value: Mapping[str, Any]
) -> None:
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) != 4:
        raise IntegrityError("Matched pair bindings are absent", "promotion_receipt_mismatch")
    for position, raw in enumerate(observations):
        item = _mapping(raw, "promotion observation")
        side = _ORDER[position]
        for kind, field in (
            ("normalized", "normalized_receipt"),
            ("quality", "quality_receipt"),
            ("config", "config_receipt"),
        ):
            role = f"promotion_{position}_{side}_{kind}"
            if _required_role(event, role).digest != item.get(field):
                raise IntegrityError(
                    "Matched pair receipts were exchanged", "promotion_receipt_mismatch"
                )


def _verify_lease(
    record: E2ERunRecord, receipt: ArtifactReceipt, value: Mapping[str, Any]
) -> None:
    lease = read_json_object(record, receipt, label="promotion GPU lease")
    ownership = _mapping(lease.get("ownership"), "GPU ownership")
    if (
        receipt.digest != value.get("gpu_lease_digest")
        or sha256_json(dict(lease)) != receipt.digest
        or lease.get("schema_version") != 2
        or lease.get("run_id") != record.run_id
        or lease.get("execution_scope") != value.get("gpu_device_scope")
        or not isinstance(lease.get("physical_scope"), str)
        or not isinstance(lease.get("owner_pid"), int)
        or not isinstance(lease.get("acquired_unix_seconds"), (int, float))
        or not isinstance(lease.get("lock_path"), str)
        or ownership.get("schema_version") != 2
        or ownership.get("policy_id")
        != "clean_hsa_kfd_rsmi_process_gpu_map_v2"
        or ownership.get("foreign_owners") != []
        or not _valid_gpu_inventory_binding(ownership)
        or _execution_scope(ownership) != lease.get("execution_scope")
        or _physical_scope(ownership) != lease.get("physical_scope")
    ):
        raise IntegrityError("Matched GPU lease differs", "promotion_lease_mismatch")


def _physical_scope(ownership: Mapping[str, Any]) -> str:
    devices = ownership.get("selected_devices")
    if not isinstance(devices, list) or not devices:
        return ""
    identities = []
    for raw in devices:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("unique_id"), str):
            return ""
        identities.append(raw["unique_id"])
    return "amd-gpu-unique-id-set=" + ",".join(sorted(identities))


def _execution_scope(ownership: Mapping[str, Any]) -> str:
    devices = ownership.get("selected_devices")
    if not isinstance(devices, list) or not devices:
        return ""
    identities = []
    for raw in devices:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("unique_id"), str):
            return ""
        identities.append(raw["unique_id"])
    return "amd-gpu-set=" + ",".join(identities)


def _valid_gpu_inventory_binding(ownership: Mapping[str, Any]) -> bool:
    monitors = ownership.get("rsmi_monitor_inventory")
    joined = ownership.get("device_inventory")
    selected = ownership.get("selected_devices")
    hsa_inventory = ownership.get("hsa_inventory")
    hsa = hsa_inventory.get("devices") if isinstance(hsa_inventory, Mapping) else None
    if not all(isinstance(value, list) and value for value in (monitors, joined, selected, hsa)):
        return False
    assert isinstance(monitors, list)
    assert isinstance(joined, list)
    assert isinstance(selected, list)
    assert isinstance(hsa, list)
    if len(joined) != len(hsa):
        return False
    monitor_map = _valid_monitor_inventory(monitors)
    if monitor_map is None:
        return False
    joined_rsmi_indices = [
        item.get("rsmi_index") for item in joined if isinstance(item, Mapping)
    ]
    if (
        len(joined_rsmi_indices) != len(joined)
        or any(not _strict_int(index) for index in joined_rsmi_indices)
        or len(set(joined_rsmi_indices)) != len(joined)
    ):
        return False
    for expected_index, (raw_joined, raw_hsa) in enumerate(zip(joined, hsa, strict=True)):
        if not isinstance(raw_joined, Mapping) or not isinstance(raw_hsa, Mapping):
            return False
        rsmi_index = raw_joined.get("rsmi_index")
        monitor = monitor_map.get(rsmi_index) if _strict_int(rsmi_index) else None
        if not _joined_identity_matches(raw_joined, raw_hsa, monitor, expected_index):
            return False
    selected_indices = [
        item.get("hsa_gpu_index") for item in selected if isinstance(item, Mapping)
    ]
    return (
        len(selected_indices) == len(selected)
        and all(_strict_int(index) for index in selected_indices)
        and len(set(selected_indices)) == len(selected)
        and all(isinstance(item, Mapping) and item in joined for item in selected)
    )


def _valid_monitor_inventory(
    monitors: list[Any],
) -> dict[int, Mapping[str, Any]] | None:
    result: dict[int, Mapping[str, Any]] = {}
    for expected_index, raw in enumerate(monitors):
        if not isinstance(raw, Mapping) or raw.get("rsmi_index") != expected_index:
            return None
        if (
            not _strict_int(raw.get("node_id"))
            or not _strict_int(raw.get("pci_id"))
            or not _strict_int(raw.get("render_minor"))
            or raw.get("render_minor", 0) < 128
            or not isinstance(raw.get("unique_id"), str)
            or not _GPU_UUID.fullmatch(raw["unique_id"])
        ):
            return None
        result[expected_index] = raw
    for field in ("unique_id", "node_id", "pci_id", "render_minor"):
        if len({item[field] for item in result.values()}) != len(result):
            return None
    return result


def _joined_identity_matches(
    joined: Mapping[str, Any],
    hsa: Mapping[str, Any],
    monitor: Mapping[str, Any] | None,
    expected_index: int,
) -> bool:
    if monitor is None:
        return False
    domain = hsa.get("domain")
    bdf_id = hsa.get("bdf_id")
    render_node = joined.get("render_node")
    if not _strict_int(domain) or not _strict_int(bdf_id) or not isinstance(render_node, str):
        return False
    render = render_node.removeprefix("/dev/dri/renderD")
    return (
        joined.get("hsa_gpu_index") == expected_index == hsa.get("hsa_gpu_index")
        and joined.get("kfd_node_id") == hsa.get("node_id") == monitor.get("node_id")
        and joined.get("unique_id") == hsa.get("unique_id") == monitor.get("unique_id")
        and render.isdecimal()
        and int(render) == monitor.get("render_minor")
        and (domain << 32) | bdf_id == monitor.get("pci_id")
    )


def _strict_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _required_role(event: EventRecord, role: str) -> ArtifactReceipt:
    receipt = unique_role((event,), role)
    if receipt is None:
        raise IntegrityError(f"{role} is missing", "promotion_lineage_mismatch")
    return receipt


def _image(item: PromotionObservation) -> dict[str, str | None]:
    return {
        "requested_image": item.requested_image,
        "resolved_image_id": item.resolved_image_id,
    }


def _image_tuple(item: PromotionObservation) -> tuple[str | None, str | None]:
    return item.requested_image, item.resolved_image_id


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} is invalid", "promotion_lineage_mismatch")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{label} is invalid", "promotion_lineage_mismatch")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrityError(f"{label} is invalid", "promotion_lineage_mismatch")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise IntegrityError(f"{label} is invalid", "promotion_lineage_mismatch")
    return text


__all__ = [
    "recover_matched_promotion",
    "validate_promotion_context",
    "verify_promotion_reward_binding",
]
