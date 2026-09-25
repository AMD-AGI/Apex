"""Canonical descriptors for planned but not yet executable capabilities."""

from __future__ import annotations

from typing import Any, Mapping

from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityKind,
    CapabilityRewardRole,
    CapabilitySideEffect,
)


def planned_capability_descriptors() -> tuple[CapabilityDescriptor, ...]:
    """Return honest inventory entries; unavailable entries never become MCP tools."""

    return (
        *_guidance_descriptors(),
        *_acquisition_descriptors(),
        *_evaluation_descriptors(),
        *_state_descriptors(),
    )


def _guidance_descriptors() -> tuple[CapabilityDescriptor, ...]:
    return (
        _descriptor(
            "amd-hip-kernel-optimization",
            "AMD HIP kernel optimization method",
            "Attributed HIP-specific guidance; no executable HIP evaluator.",
            CapabilityKind.SKILL,
            {},
            {},
        ),
        _descriptor(
            "amd-kernel-optimization",
            "AMD kernel optimization method",
            "Attributed optimization workflow guidance; no executable sanitizer step.",
            CapabilityKind.SKILL,
            {},
            {},
        ),
        _descriptor(
            "amd-kernel-debugging",
            "AMD kernel debugging method",
            "ROCm/gfx950 debugging guidance without measurement claims.",
            CapabilityKind.SKILL,
            {},
            {},
        ),
    )


def _acquisition_descriptors() -> tuple[CapabilityDescriptor, ...]:
    return (
        _descriptor(
            "benchmark.run",
            "Run a Magpie benchmark",
            "Execute a frozen normal-runtime benchmark and return artifact receipts.",
            CapabilityKind.TOOL,
            {
                "config_path": _string(),
                "benchmark_pass": {"type": "string", "enum": ["measurement"]},
                "gpu_devices": _string(),
            },
            {"receipt": _object()},
            required=("config_path", "benchmark_pass"),
            effects=(CapabilitySideEffect.WRITE_RESULTS, CapabilitySideEffect.RUN_PROCESS),
            authority=CapabilityAuthority.WORKSPACE_USER,
            gpu=CapabilityGpuRequirement.REQUIRED,
            timeout=7200,
            artifacts=("benchmark_receipt",),
            reward=CapabilityRewardRole.EVIDENCE_ONLY,
        ),
        _descriptor(
            "profile.capture",
            "Capture a Magpie profile",
            "Capture diagnostic-only profiler evidence for a frozen workload.",
            CapabilityKind.TOOL,
            {
                "config_path": _string(),
                "profile_mode": {"type": "string", "enum": ["magpie_config"]},
                "gpu_devices": _string(),
            },
            {"receipt": _object(), "artifacts": _array()},
            required=("config_path", "profile_mode"),
            effects=(CapabilitySideEffect.WRITE_RESULTS, CapabilitySideEffect.RUN_PROCESS),
            authority=CapabilityAuthority.WORKSPACE_USER,
            gpu=CapabilityGpuRequirement.REQUIRED,
            timeout=7200,
            artifacts=("diagnostic_trace",),
            reward=CapabilityRewardRole.INELIGIBLE,
        ),
    )


def _evaluation_descriptors() -> tuple[CapabilityDescriptor, ...]:
    return (
        _descriptor(
            "kernel.compile",
            "Compile a frozen kernel",
            "Consume injected one-shot authority, freeze the isolated editable projection, and compile only after baseline revalidation.",
            CapabilityKind.TOOL,
            {
                "run_locator": _string(),
                "confirmed_draft_digest": _string(),
                "gpu_devices": _string(),
            },
            {"receipt": _object()},
            required=("run_locator", "confirmed_draft_digest"),
            effects=(CapabilitySideEffect.WRITE_RESULTS, CapabilitySideEffect.RUN_PROCESS),
            authority=CapabilityAuthority.FORMAL_EVALUATOR,
            gpu=CapabilityGpuRequirement.REQUIRED,
            timeout=3600,
            artifacts=("compile_receipt",),
            reward=CapabilityRewardRole.EVIDENCE_ONLY,
        ),
        _attempt_evaluator_descriptor(
            "kernel.correctness",
            "Evaluate frozen-kernel correctness",
            "correctness_receipt",
            gpu=CapabilityGpuRequirement.REQUIRED,
        ),
        _attempt_evaluator_descriptor(
            "kernel.measure",
            "Measure frozen kernel invocations",
            "raw_measurement",
            gpu=CapabilityGpuRequirement.REQUIRED,
        ),
        _descriptor(
            "kernel.grade",
            "Grade trusted kernel measurements",
            "Recompute a grade from evaluator-captured raw evidence; missing authority or evidence returns unverified without reward.",
            CapabilityKind.TOOL,
            {
                "run_locator": _string(),
                "attempt_id": _string(),
                "contract_digest": _string(),
                "candidate_digest": _string(),
            },
            {"receipt": _object()},
            required=("run_locator",),
            effects=(CapabilitySideEffect.WRITE_RESULTS,),
            authority=CapabilityAuthority.FORMAL_EVALUATOR,
            artifacts=("kernel_grade",),
            reward=CapabilityRewardRole.EVALUATOR_OWNED,
        ),
    )


def _state_descriptors() -> tuple[CapabilityDescriptor, ...]:
    return (
        _campaign_start_descriptor(),
        _campaign_descriptor(
            "campaign.status", "Inspect formal campaign status",
            effects=(CapabilitySideEffect.READ_RESULTS,),
        ),
        _stop_descriptor(),
        _campaign_descriptor("campaign.checkpoint", "Checkpoint formal campaign state"),
        _descriptor(
            "campaign.resume",
            "Resume a formal E2E campaign",
            "Resume canonical E2E state only after current execution identity, dependency, provenance, and GPU preflight.",
            CapabilityKind.CAMPAIGN,
            {"run_locator": _string()},
            {"campaign": _object()},
            required=("run_locator",),
            effects=(
                CapabilitySideEffect.READ_RESULTS,
                CapabilitySideEffect.WRITE_RESULTS,
                CapabilitySideEffect.RUN_PROCESS,
            ),
            authority=CapabilityAuthority.WORKSPACE_USER,
            gpu=CapabilityGpuRequirement.REQUIRED,
            timeout=7200,
            artifacts=("campaign_state",),
            reward=CapabilityRewardRole.EVIDENCE_ONLY,
        ),
        _descriptor(
            "bundle.build",
            "Build a verified unapplied bundle",
            "Build and CAS-bind an immutable source bundle only for a verified improving attempt.",
            CapabilityKind.DELIVERY,
            {
                "run_locator": _string(),
                "attempt_id": _string(),
                "contract_digest": _string(),
                "candidate_digest": _string(),
            },
            {"verification": _object()},
            required=(
                "run_locator",
                "attempt_id",
                "contract_digest",
                "candidate_digest",
            ),
            effects=(CapabilitySideEffect.READ_RESULTS, CapabilitySideEffect.WRITE_RESULTS),
            authority=CapabilityAuthority.FORMAL_EVALUATOR,
            timeout=7200,
            artifacts=("delivery_bundle", "bundle_verification"),
            reward=CapabilityRewardRole.EVIDENCE_ONLY,
        ),
        _delivery_descriptor(
            "bundle.verify",
            "Verify an immutable source bundle",
            authority=CapabilityAuthority.WORKSPACE_USER,
            effects=(
                CapabilitySideEffect.READ_WORKSPACE,
                CapabilitySideEffect.READ_RESULTS,
            ),
        ),
    )


def _campaign_start_descriptor() -> CapabilityDescriptor:
    return _descriptor(
        "campaign.start",
        "Start a formal kernel campaign draft",
        "Freeze an unverified task plus Apex execution identity; run no agent, GPU, or evaluator.",
        CapabilityKind.CAMPAIGN,
        {"task": _kernel_task_schema()},
        {"campaign": _object()},
        required=("task",),
        effects=(
            CapabilitySideEffect.READ_WORKSPACE,
            CapabilitySideEffect.WRITE_RESULTS,
        ),
        authority=CapabilityAuthority.WORKSPACE_USER,
        artifacts=("campaign_state", "evaluation_contract_draft"),
        reward=CapabilityRewardRole.EVIDENCE_ONLY,
    )


def _kernel_task_schema() -> dict[str, Any]:
    command = _command_schema()
    return _schema(
        {
            "schema_version": {"type": "integer", "const": 1},
            "task_id": _string(),
            "instructions": _string(),
            "language": {"type": "string", "enum": ["python", "triton"]},
            "editable_files": _string_array(),
            "target_functions": _string_array(),
            "commands": _schema(
                {
                    "compile": command,
                    "correctness": command,
                    "performance": command,
                },
                ("compile", "correctness", "performance"),
            ),
            "measurement": _measurement_schema(),
            "gpu_arch": _string(),
            "mode": {"type": "string", "const": "optimize_existing"},
            "agent_backend": {
                "type": "string",
                "enum": ["codex", "claude", "cursor"],
            },
            "agent_options": _object(),
            "budget": _object(),
            "scope": _object(),
            "delivery": _object(),
            "dataset_split": {
                "type": "string",
                "enum": ["train", "validation", "heldout"],
            },
            "data_visibility": {
                "type": "string",
                "enum": ["public", "private", "heldout_private"],
            },
        },
        (
            "task_id",
            "instructions",
            "language",
            "editable_files",
            "target_functions",
            "commands",
        ),
    )


def _command_schema() -> dict[str, Any]:
    return _schema(
        {
            "argv": _string_array(),
            "timeout_seconds": {"type": "integer", "minimum": 1},
            "cwd": _string(),
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        ("argv",),
    )


def _measurement_schema() -> dict[str, Any]:
    return _schema(
        {
            "schema": {"type": "string", "const": "apex.kernel-measurement/v1"},
            "adapter_id": _string(),
            "harness_files": _string_array(),
            "measurement_method_sha256": {
                "type": "string",
                "pattern": "^(sha256:)?[0-9a-f]{64}$",
            },
            "runner": _command_schema(),
            "aggregation": {
                "type": "string",
                "enum": ["equal_case", "workload_weighted"],
            },
        },
        (
            "schema",
            "adapter_id",
            "harness_files",
            "measurement_method_sha256",
            "runner",
        ),
    )


def _stop_descriptor() -> CapabilityDescriptor:
    return _descriptor(
        "campaign.stop",
        "Stop a standalone formal kernel campaign",
        "Close standalone attempts and derive terminal reward from recorded evidence.",
        CapabilityKind.CAMPAIGN,
        {
            "run_locator": _string(),
            "reason": {"type": "string", "enum": ["user_requested"]},
        },
        {"campaign": _object()},
        required=("run_locator",),
        effects=(
            CapabilitySideEffect.READ_RESULTS,
            CapabilitySideEffect.WRITE_RESULTS,
        ),
        authority=CapabilityAuthority.WORKSPACE_USER,
        artifacts=("campaign_state", "kernel_terminal_result"),
        reward=CapabilityRewardRole.EVIDENCE_ONLY,
    )


def _attempt_evaluator_descriptor(
    capability_id: str,
    title: str,
    artifact: str,
    *,
    gpu: CapabilityGpuRequirement,
) -> CapabilityDescriptor:
    return _descriptor(
        capability_id,
        title,
        "Evaluator-owned operation over a frozen candidate and contract.",
        CapabilityKind.TOOL,
        {
            "run_locator": _string(),
            "attempt_id": _string(),
            "contract_digest": _string(),
            "candidate_digest": _string(),
            "gpu_devices": _string(),
        },
        {"receipt": _object()},
        required=(
            "run_locator",
            "attempt_id",
            "contract_digest",
            "candidate_digest",
        ),
        effects=(CapabilitySideEffect.WRITE_RESULTS, CapabilitySideEffect.RUN_PROCESS),
        authority=CapabilityAuthority.FORMAL_EVALUATOR,
        gpu=gpu,
        timeout=3600,
        artifacts=(artifact,),
        reward=CapabilityRewardRole.EVIDENCE_ONLY,
    )


def _campaign_descriptor(
    capability_id: str,
    title: str,
    *,
    effects: tuple[CapabilitySideEffect, ...] = (CapabilitySideEffect.WRITE_RESULTS,),
) -> CapabilityDescriptor:
    return _descriptor(
        capability_id,
        title,
        "Operate canonical event/CAS campaign state, never backend chat history.",
        CapabilityKind.CAMPAIGN,
        {"run_locator": _string()},
        {"campaign": _object()},
        required=("run_locator",),
        effects=effects,
        authority=CapabilityAuthority.WORKSPACE_USER,
        artifacts=("campaign_state",),
        reward=CapabilityRewardRole.EVIDENCE_ONLY,
    )


def _delivery_descriptor(
    capability_id: str,
    title: str,
    *,
    authority: CapabilityAuthority,
    effects: tuple[CapabilitySideEffect, ...] = (
        CapabilitySideEffect.READ_WORKSPACE,
        CapabilitySideEffect.WRITE_RESULTS,
    ),
) -> CapabilityDescriptor:
    return _descriptor(
        capability_id,
        title,
        "Build or independently verify immutable delivery evidence.",
        CapabilityKind.DELIVERY,
        {"bundle_path": _string()},
        {"verification": _object()},
        required=("bundle_path",),
        effects=effects,
        authority=authority,
        timeout=7200,
        artifacts=("delivery_bundle", "bundle_verification"),
        reward=CapabilityRewardRole.EVIDENCE_ONLY,
    )


def _descriptor(
    capability_id: str,
    title: str,
    summary: str,
    kind: CapabilityKind,
    input_properties: Mapping[str, Any],
    output_properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
    effects: tuple[CapabilitySideEffect, ...] = (),
    authority: CapabilityAuthority = CapabilityAuthority.NONE,
    gpu: CapabilityGpuRequirement = CapabilityGpuRequirement.NONE,
    timeout: int = 30,
    artifacts: tuple[str, ...] = (),
    reward: CapabilityRewardRole = CapabilityRewardRole.INELIGIBLE,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id,
        title,
        summary,
        kind,
        _schema(input_properties, required),
        _schema(output_properties, tuple(output_properties)),
        effects,
        authority,
        gpu,
        timeout,
        artifacts,
        reward,
    )


def _schema(
    properties: Mapping[str, Any], required: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _integer(minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _object() -> dict[str, Any]:
    return {"type": "object"}


def _array() -> dict[str, Any]:
    return {"type": "array"}


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": _string(), "minItems": 1}


__all__ = ["planned_capability_descriptors"]
