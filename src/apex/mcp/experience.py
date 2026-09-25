"""Read-only exact-identity retrieval of canonical measured experience."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ContractError
from apex.knowledge import ExperienceIdentity, ExperienceView
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    CapabilitySideEffect,
)
from apex.storage import EventJournal

from .scope import CapabilityScope


def experience_retrieve_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="experience.retrieve",
        title="Retrieve compatible measured experience",
        summary=(
            "Verify one canonical journal and return only records with an exact "
            "task/operator/GPU/framework/version/shape/source/harness/policy identity."
        ),
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {
                "run_path": {"type": "string", "minLength": 1},
                "run_id": {"type": "string", "minLength": 1},
                "identity": _identity_schema(),
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["run_path", "run_id", "identity"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "identity": {"type": "object"},
                "records": {"type": "array", "items": {"type": "object"}},
                "record_count": {"type": "integer", "minimum": 0},
                "experience_view_digest": {"type": "string"},
                "event_journal": {"type": "string"},
                "evidence_only": {"const": True},
            },
            "required": [
                "run_id",
                "identity",
                "records",
                "record_count",
                "experience_view_digest",
                "event_journal",
                "evidence_only",
            ],
            "additionalProperties": False,
        },
        side_effects=(CapabilitySideEffect.READ_RESULTS,),
        required_authority=CapabilityAuthority.WORKSPACE_USER,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=10,
        artifact_classes=("measured_experience_projection",),
        reward_role=CapabilityRewardRole.EVIDENCE_ONLY,
    )


class ExperienceRetrieveHandler:
    def __init__(self, scope: CapabilityScope) -> None:
        self._scope = scope

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        arguments = request.arguments
        run_path = _required_text(arguments, "run_path")
        run_id = _required_text(arguments, "run_id")
        identity_value = arguments.get("identity")
        if not isinstance(identity_value, Mapping):
            raise ContractError(
                "Experience identity is required", "invalid_capability_arguments"
            )
        identity = ExperienceIdentity.from_mapping(identity_value)
        limit = arguments.get("limit", 8)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ContractError(
                "Experience limit is invalid", "invalid_capability_arguments"
            )
        journal_path = self._scope.read_results(f"{run_path}/events/run.db")
        journal = EventJournal.open_read_only(journal_path)
        view = ExperienceView.from_events(journal.iter_events(run_id))
        records = view.compatible(identity, limit=limit)
        _, locator = self._scope.locator(journal_path)
        return CapabilityResult(
            capability_id=request.capability_id,
            content={
                "run_id": run_id,
                "identity": identity.to_dict(),
                "records": [item.to_dict() for item in records],
                "record_count": len(records),
                "experience_view_digest": view.digest,
                "event_journal": locator,
                "evidence_only": True,
            },
        )


def _identity_schema() -> dict[str, Any]:
    required = [
        "task_id",
        "operator",
        "gpu_arch",
        "framework",
        "versions",
        "shape_hash",
        "source_hash",
        "harness_hash",
        "policy_hash",
    ]
    digest = {"type": "string", "pattern": "^(?:sha256:)?[0-9a-f]{64}$"}
    return {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "operator": {"type": "string", "minLength": 1},
            "gpu_arch": {"type": "string", "minLength": 1},
            "framework": {"type": "string", "minLength": 1},
            "versions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "shape_hash": digest,
            "source_hash": digest,
            "harness_hash": digest,
            "policy_hash": digest,
        },
        "required": required,
        "additionalProperties": False,
    }


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ContractError(f"{key} is required", "invalid_capability_arguments")
    return item.strip()


__all__ = ["ExperienceRetrieveHandler", "experience_retrieve_descriptor"]
