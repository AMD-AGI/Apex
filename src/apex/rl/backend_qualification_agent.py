"""Strict typed loaders for backend-qualification agent receipts."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ContractError
from apex.ports import (
    AgentExecutionAuthorityReceipt,
    AgentInvocationReceipt,
    AgentProcessContainmentReceipt,
)


def load_agent_invocation(value: object) -> AgentInvocationReceipt:
    raw = _mapping(value, "agent invocation")
    expected = {
        "schema", "cli_name", "cli_version", "executable_path",
        "resolved_executable_path", "entrypoint_sha256", "runtime_closure_sha256",
        "argv", "workspace", "prompt_transport", "execution_authority",
        "execution_authority_sha256", "credential_environment_key",
        "requested_allowed_files", "allowed_files_enforced_by_cli", "max_turns",
        "turn_policy", "process_containment_policy_id", "isolation",
    }
    if set(raw) != expected or raw.get("schema") != "apex.agent-invocation/v4":
        _reject("Agent invocation receipt fields differ")
    authority = _load_agent_authority(raw["execution_authority"])
    try:
        receipt = AgentInvocationReceipt(
            cli_name=_text(raw["cli_name"]),
            cli_version=_text(raw["cli_version"]),
            executable_path=_text(raw["executable_path"]),
            resolved_executable_path=_text(raw["resolved_executable_path"]),
            entrypoint_sha256=_text(raw["entrypoint_sha256"]),
            runtime_closure_sha256=_optional_text(raw["runtime_closure_sha256"]),
            argv=_texts(raw["argv"]),
            workspace=_text(raw["workspace"]),
            prompt_transport=_text(raw["prompt_transport"]),
            execution_authority=authority,
            credential_environment_key=_text(raw["credential_environment_key"]),
            requested_allowed_files=_texts(raw["requested_allowed_files"]),
            allowed_files_enforced_by_cli=_boolean(raw["allowed_files_enforced_by_cli"]),
            max_turns=_integer(raw["max_turns"], positive=True),
            turn_policy=_text(raw["turn_policy"]),
            process_containment_policy_id=_text(raw["process_containment_policy_id"]),
            isolation=tuple(sorted(_string_mapping(raw["isolation"]).items())),
        )
    except ContractError as error:
        raise ContractError(
            "Agent invocation receipt is invalid", "qualification_artifacts_invalid"
        ) from error
    if raw.get("execution_authority_sha256") != authority.digest or raw != receipt.to_dict():
        _reject("Agent invocation receipt digest or fields differ")
    return receipt


def load_agent_containment(value: object) -> AgentProcessContainmentReceipt:
    raw = _mapping(value, "agent process containment")
    fields = set(AgentProcessContainmentReceipt.__dataclass_fields__)
    if set(raw) != fields | {"schema", "namespace_empty_verified"}:
        _reject("Agent process containment fields differ")
    try:
        values = {field: raw[field] for field in fields}
        values["live_namespace_members_after"] = tuple(
            _integers(values["live_namespace_members_after"])
        )
        receipt = AgentProcessContainmentReceipt(**values)
    except (ContractError, TypeError, ValueError) as error:
        raise ContractError(
            "Agent process containment is invalid", "qualification_artifacts_invalid"
        ) from error
    if raw != receipt.to_dict():
        _reject("Agent process containment receipt differs")
    return receipt


def _load_agent_authority(value: object) -> AgentExecutionAuthorityReceipt:
    raw = _mapping(value, "agent execution authority")
    expected = {
        "schema", "authority_id", "authority_kind", "run_id", "attempt_id",
        "backend", "workspace", "allowed_files", "requested_environment_keys",
        "parent_receipt_sha256", "source_anchor_sha256", "permission_policy_id",
        "credential_policy_id", "prompt_transport_policy_id",
    }
    if set(raw) != expected or raw.get("schema") != "apex.agent-execution-authority/v1":
        _reject("Agent execution authority fields differ")
    try:
        receipt = AgentExecutionAuthorityReceipt(
            authority_id=_text(raw["authority_id"]),
            authority_kind=_text(raw["authority_kind"]),
            run_id=_text(raw["run_id"]),
            attempt_id=_text(raw["attempt_id"]),
            backend=_text(raw["backend"]),
            workspace=_text(raw["workspace"]),
            allowed_files=_texts(raw["allowed_files"]),
            requested_environment_keys=_texts(raw["requested_environment_keys"]),
            parent_receipt_sha256=_text(raw["parent_receipt_sha256"]),
            source_anchor_sha256=_text(raw["source_anchor_sha256"]),
            permission_policy_id=_text(raw["permission_policy_id"]),
            credential_policy_id=_text(raw["credential_policy_id"]),
            prompt_transport_policy_id=_text(raw["prompt_transport_policy_id"]),
        )
    except ContractError as error:
        raise ContractError(
            "Agent execution authority is invalid", "qualification_artifacts_invalid"
        ) from error
    if raw != receipt.to_dict():
        _reject("Agent execution authority receipt differs")
    return receipt


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _reject(f"{label.title()} must be an object")
    return value


def _string_mapping(value: object) -> dict[str, str]:
    raw = _mapping(value, "string mapping")
    if any(not isinstance(item, str) or not item for item in raw.values()):
        _reject("String mapping values are invalid")
    return dict(raw)


def _texts(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        _reject("Expected a string list")
    return tuple(_text(item) for item in value)


def _integers(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        _reject("Expected an integer list")
    return tuple(_integer(item, positive=True) for item in value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        _reject("Expected non-empty text")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _integer(value: object, *, positive: bool) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        _reject("Expected an integer")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        _reject("Expected a boolean")
    return value


def _reject(message: str) -> None:
    raise ContractError(message, "qualification_artifacts_invalid")


__all__ = ["load_agent_containment", "load_agent_invocation"]
