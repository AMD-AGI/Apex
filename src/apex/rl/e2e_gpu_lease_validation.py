"""Semantic reconstruction of GPU lease evidence used by E2E promotion."""

from __future__ import annotations

import math
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError
from apex.runtime import (
    GpuDeviceIdentity,
    GpuLeaseReceipt,
    GpuLeaseHeartbeatReceipt,
    GpuLeaseOwnerIdentity,
    GpuMeasurementBracketReceipt,
    GpuOwnershipReceipt,
    GpuProcessIdentity,
    GpuSelectorRequest,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    RsmiDeviceIdentity,
    load_gpu_doctor_receipt,
)
from apex.storage import ArtifactStore

from .e2e_benchmark_validation import (
    mapping,
    read_json,
    reject,
    single_event_receipt,
)
from .models import EpisodeEvent


def validate_gpu_lease(
    *,
    run_id: str,
    event: EpisodeEvent,
    pair: Mapping[str, Any],
    artifacts: ArtifactStore,
) -> None:
    """Rebuild the public runtime receipt and require byte-level field identity."""

    receipt = single_event_receipt(event, "promotion_gpu_lease")
    document = read_json(artifacts, receipt, canonical=True)
    try:
        lease = _lease(document)
    except (ContractError, KeyError, TypeError, ValueError) as error:
        raise _lease_error() from error
    if (
        lease.to_dict() != dict(document)
        or lease.digest != receipt.digest
        or receipt.digest != pair.get("gpu_lease_digest")
        or lease.run_id != run_id
        or lease.execution_scope != pair.get("gpu_device_scope")
    ):
        reject("Matched-promotion GPU lease differs from its declared scope")


def validate_measurement_bracket(
    *,
    run_id: str,
    action_id: str,
    lease_digest: str,
    event: EpisodeEvent,
    artifacts: ArtifactStore,
) -> None:
    """Rebuild one formal pre/post lease bracket from its leg event."""

    receipt = single_event_receipt(event, "gpu_measurement_bracket")
    document = read_json(artifacts, receipt, canonical=True)
    try:
        bracket = load_measurement_bracket(document)
    except (ContractError, KeyError, TypeError, ValueError) as error:
        raise _lease_error() from error
    if (
        bracket.to_dict() != dict(document)
        or bracket.digest != receipt.digest
        or bracket.run_id != run_id
        or bracket.action_id != action_id
        or bracket.lease_digest != lease_digest
        or event.payload.get("gpu_measurement_bracket_digest") != receipt.digest
    ):
        reject("Formal measurement GPU bracket differs from its action")


def _lease(value: Mapping[str, Any]) -> GpuLeaseReceipt:
    ownership = _ownership(mapping(value["ownership"], "GPU ownership"))
    doctor = load_gpu_doctor_receipt(
        mapping(value["doctor"], "GPU doctor"), ownership=ownership
    )
    return GpuLeaseReceipt(
        _integer(value["schema_version"], positive=True),
        _text(value["run_id"]),
        _text(value["execution_scope"]),
        _text(value["physical_scope"]),
        _integer(value["owner_pid"], positive=True),
        _number(value["acquired_unix_seconds"]),
        _text(value["lock_path"]),
        ownership,
        doctor,
        _text_tuple(value["lock_paths"]),
    )


def load_measurement_bracket(
    value: Mapping[str, Any],
) -> GpuMeasurementBracketReceipt:
    return GpuMeasurementBracketReceipt(
        _integer(value["schema_version"], positive=True),
        _text(value["run_id"]),
        _text(value["action_id"]),
        _text(value["lease_digest"]),
        _number(value["started_unix_seconds"]),
        _number(value["finished_unix_seconds"]),
        load_gpu_heartbeat(mapping(value["pre"], "pre GPU lease heartbeat")),
        load_gpu_heartbeat(mapping(value["post"], "post GPU lease heartbeat")),
    )


def load_gpu_heartbeat(value: Mapping[str, Any]) -> GpuLeaseHeartbeatReceipt:
    ownership = _ownership(mapping(value["ownership"], "GPU ownership"))
    doctor = load_gpu_doctor_receipt(
        mapping(value["doctor"], "GPU doctor"), ownership=ownership
    )
    owner = mapping(value["owner"], "GPU lease owner")
    return GpuLeaseHeartbeatReceipt(
        _integer(value["schema_version"], positive=True),
        _text(value["run_id"]),
        _text(value["lease_digest"]),
        _integer(value["sequence"], positive=True),
        _text(value["reason"]),
        _number(value["observed_unix_seconds"]),
        _number(value["valid_until_unix_seconds"]),
        _number(value["ttl_seconds"]),
        GpuLeaseOwnerIdentity(
            _integer(owner["pid"], positive=True),
            _integer(owner["uid"]),
            _integer(owner["start_time_ticks"], positive=True),
            _text(owner["cmdline_sha256"]),
        ),
        ownership,
        doctor,
    )
def _ownership(value: Mapping[str, Any]) -> GpuOwnershipReceipt:
    selector = mapping(value["selector_inputs"], "GPU selector inputs")
    return GpuOwnershipReceipt(
        _integer(value["schema_version"], positive=True),
        _text(value["policy_id"]),
        GpuSelectorRequest(
            requested=_optional_text_tuple(selector["requested"]),
            rocr_visible_devices=_optional_text_tuple(selector["ROCR_VISIBLE_DEVICES"]),
            hip_visible_devices=_optional_text_tuple(selector["HIP_VISIBLE_DEVICES"]),
            cuda_visible_devices=_optional_text_tuple(selector["CUDA_VISIBLE_DEVICES"]),
            gpu_device_ordinal=_optional_text_tuple(selector["GPU_DEVICE_ORDINAL"]),
        ),
        _integer(value["observed_unix_ns"], positive=True),
        _text(value["library_path"]),
        _text(value["library_sha256"]),
        _text(value["topology_root"]),
        _hsa_inventory(mapping(value["hsa_inventory"], "HSA inventory")),
        _objects(value["rsmi_monitor_inventory"], _rsmi_device),
        _objects(value["device_inventory"], _gpu_device),
        _objects(value["selected_devices"], _gpu_device),
        _objects(value["allowed_owners"], _process),
        _objects(value["foreign_owners"], _process),
    )


def _hsa_inventory(value: Mapping[str, Any]) -> HsaInventoryEvidence:
    return HsaInventoryEvidence(
        _integer(value["schema_version"], positive=True),
        _text(value["policy_id"]),
        _text(value["helper_path"]),
        _text(value["helper_sha256"]),
        _text(value["library_path"]),
        _text(value["library_sha256"]),
        _objects(value["devices"], _hsa_device),
    )


def _hsa_device(value: Mapping[str, Any]) -> HsaGpuIdentity:
    return HsaGpuIdentity(
        _integer(value["hsa_gpu_index"]),
        _integer(value["node_id"]),
        _integer(value["generic_node_id"]),
        _integer(value["bdf_id"]),
        _integer(value["domain"]),
        _text(value["unique_id"]),
    )


def _rsmi_device(value: Mapping[str, Any]) -> RsmiDeviceIdentity:
    return RsmiDeviceIdentity(
        _integer(value["rsmi_index"]),
        _integer(value["node_id"]),
        _integer(value["pci_id"]),
        _text(value["unique_id"]),
        _integer(value["render_minor"]),
    )


def _gpu_device(value: Mapping[str, Any]) -> GpuDeviceIdentity:
    return GpuDeviceIdentity(
        _integer(value["hsa_gpu_index"]),
        _integer(value["kfd_node_id"]),
        _integer(value["rsmi_index"]),
        _text(value["unique_id"]),
        _text(value["render_node"]),
    )


def _process(value: Mapping[str, Any]) -> GpuProcessIdentity:
    return GpuProcessIdentity(
        _integer(value["pid"], positive=True),
        _integer(value["uid"]),
        _integer(value["start_time_ticks"], positive=True),
        _text(value["cmdline_sha256"]),
        _integer_tuple(value["rsmi_device_indices"]),
    )


def _objects(value: Any, constructor: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise TypeError("expected list")
    return tuple(constructor(mapping(item, "GPU lease item")) for item in value)


def _optional_text_tuple(value: Any) -> tuple[str, ...] | None:
    return None if value is None else _text_tuple(value)


def _text_tuple(value: Any) -> tuple[str, ...]:
    return _tuple(value, _text)


def _integer_tuple(value: Any) -> tuple[int, ...]:
    return _tuple(value, _integer)


def _tuple(value: Any, converter: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise TypeError("expected list")
    return tuple(converter(item) for item in value)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("expected text")
    return value


def _integer(value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError("expected integer")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TypeError("expected finite nonnegative number")
    return result


def _lease_error():
    return IntegrityError(
        "Matched-promotion GPU lease is invalid",
        "e2e_measurement_evidence_mismatch",
    )


__all__ = [
    "load_measurement_bracket",
    "load_gpu_heartbeat",
    "validate_gpu_lease",
    "validate_measurement_bracket",
]
