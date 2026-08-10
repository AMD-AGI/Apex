"""Evaluator-issued permission receipt for one formal agent process."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from apex.core import ContractError, sha256_json, validate_identifier


_SHA256 = re.compile(r"[0-9a-f]{64}")
FORMAL_AGENT_PERMISSION_POLICY = "sealed_editable_projection_v1"
BACKEND_CREDENTIAL_POLICY = "backend_scoped_environment_redaction_v1"
FORMAL_PROMPT_TRANSPORT_POLICY = "stdin_only_v1"
_AUTHORITY_KINDS = frozenset({"evaluation_contract", "e2e_controller"})
_BACKENDS = frozenset({"codex", "claude", "cursor"})


@dataclass(frozen=True, slots=True)
class AgentExecutionAuthorityReceipt:
    """Bind one backend invocation to one formal run and writable projection."""

    authority_id: str
    authority_kind: str
    run_id: str
    attempt_id: str
    backend: str
    workspace: str
    allowed_files: tuple[str, ...]
    requested_environment_keys: tuple[str, ...]
    parent_receipt_sha256: str
    source_anchor_sha256: str
    permission_policy_id: str = FORMAL_AGENT_PERMISSION_POLICY
    credential_policy_id: str = BACKEND_CREDENTIAL_POLICY
    prompt_transport_policy_id: str = FORMAL_PROMPT_TRANSPORT_POLICY

    def __post_init__(self) -> None:
        validate_identifier(self.authority_id, field_name="agent execution authority ID")
        if self.authority_kind not in _AUTHORITY_KINDS:
            raise ContractError(
                "Formal agent authority kind is invalid", "invalid_agent_execution_authority"
            )
        if not self.run_id or not self.attempt_id or self.backend not in _BACKENDS:
            raise ContractError(
                "Formal agent authority identity is incomplete",
                "invalid_agent_execution_authority",
            )
        if not Path(self.workspace).is_absolute():
            raise ContractError(
                "Formal agent authority workspace must be absolute",
                "invalid_agent_execution_authority",
            )
        _validate_allowed_files(self.allowed_files)
        if any(
            not isinstance(key, str) or not key
            for key in self.requested_environment_keys
        ):
            raise ContractError(
                "Formal agent environment key is invalid", "invalid_agent_execution_authority"
            )
        if tuple(sorted(set(self.requested_environment_keys))) != self.requested_environment_keys:
            raise ContractError(
                "Formal agent environment keys are not canonical",
                "invalid_agent_execution_authority",
            )
        if not _SHA256.fullmatch(self.parent_receipt_sha256) or not _SHA256.fullmatch(
            self.source_anchor_sha256
        ):
            raise ContractError(
                "Formal agent authority digest is invalid", "invalid_agent_execution_authority"
            )
        expected = (
            FORMAL_AGENT_PERMISSION_POLICY,
            BACKEND_CREDENTIAL_POLICY,
            FORMAL_PROMPT_TRANSPORT_POLICY,
        )
        observed = (
            self.permission_policy_id,
            self.credential_policy_id,
            self.prompt_transport_policy_id,
        )
        if observed != expected:
            raise ContractError(
                "Formal agent authority policy is invalid", "invalid_agent_execution_authority"
            )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.agent-execution-authority/v1",
            "authority_id": self.authority_id,
            "authority_kind": self.authority_kind,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "backend": self.backend,
            "workspace": self.workspace,
            "allowed_files": list(self.allowed_files),
            "requested_environment_keys": list(self.requested_environment_keys),
            "parent_receipt_sha256": self.parent_receipt_sha256,
            "source_anchor_sha256": self.source_anchor_sha256,
            "permission_policy_id": self.permission_policy_id,
            "credential_policy_id": self.credential_policy_id,
            "prompt_transport_policy_id": self.prompt_transport_policy_id,
        }


def _validate_allowed_files(values: tuple[str, ...]) -> None:
    if (
        not values
        or any(not isinstance(value, str) or not value for value in values)
        or tuple(sorted(set(values))) != values
    ):
        raise ContractError(
            "Formal agent editable files are not canonical",
            "invalid_agent_execution_authority",
        )
    for value in values:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ContractError(
                "Formal agent editable file is invalid", "invalid_agent_execution_authority"
            )


__all__ = [
    "AgentExecutionAuthorityReceipt",
    "BACKEND_CREDENTIAL_POLICY",
    "FORMAL_AGENT_PERMISSION_POLICY",
    "FORMAL_PROMPT_TRANSPORT_POLICY",
]
