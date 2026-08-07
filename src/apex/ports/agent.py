"""Agent execution protocol; implementations live in :mod:`apex.execution`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from apex.core import AgentBackendName, ContractError


_SEMANTIC_EVENT_KINDS = {"agent_message", "tool_called", "tool_result"}
_CURRENCY = re.compile(r"[A-Z]{3,8}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class AgentTranscriptEvent:
    """One normalized stream event emitted by an agent process."""

    kind: str
    text: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentSemanticEvent:
    """Provider-neutral action extracted only from one structured JSON event."""

    index: int
    source_event_index: int
    source_kind: str
    kind: str
    role: str | None = None
    text: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    succeeded: bool | None = None

    def __post_init__(self) -> None:
        indexes = (self.index, self.source_event_index)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indexes):
            raise ContractError("Agent event indexes must be nonnegative", "invalid_agent_event")
        if (
            self.kind not in _SEMANTIC_EVENT_KINDS
            or not isinstance(self.source_kind, str)
            or not self.source_kind
        ):
            raise ContractError("Agent semantic event kind is invalid", "invalid_agent_event")
        strings = (self.role, self.text, self.tool_name, self.tool_call_id)
        if any(value is not None and not isinstance(value, str) for value in strings):
            raise ContractError("Agent semantic event field is invalid", "invalid_agent_event")
        if self.succeeded is not None and not isinstance(self.succeeded, bool):
            raise ContractError("Agent tool outcome is invalid", "invalid_agent_event")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "source_event_index": self.source_event_index,
            "source_kind": self.source_kind,
            "kind": self.kind,
            "role": self.role,
            "text": self.text,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "succeeded": self.succeeded,
        }


@dataclass(frozen=True, slots=True)
class AgentUsage:
    """Provider-neutral usage obtained exclusively from structured stream data."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    turn_count: int = 0
    tool_call_count: int = 0
    source_event_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        counts = (
            self.input_tokens,
            self.cached_input_tokens,
            self.cache_creation_input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.total_tokens,
            self.turn_count,
            self.tool_call_count,
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in counts
        ):
            raise ContractError("Agent usage counts must be nonnegative integers", "invalid_agent_usage")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.source_event_indices
        ):
            raise ContractError("Agent usage source index is invalid", "invalid_agent_usage")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "turn_count": self.turn_count,
            "tool_call_count": self.tool_call_count,
            "source_event_indices": list(self.source_event_indices),
        }


@dataclass(frozen=True, slots=True)
class AgentCost:
    """One explicit provider cost total, retained without binary-float rounding."""

    amount: str
    currency: str
    source_event_index: int
    source_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.amount, str) or not isinstance(self.currency, str):
            raise ContractError("Agent cost fields are invalid", "invalid_agent_cost")
        try:
            value = Decimal(self.amount)
        except InvalidOperation as error:
            raise ContractError("Agent cost amount is invalid", "invalid_agent_cost") from error
        currency = self.currency.upper()
        if not value.is_finite() or value < 0 or not _CURRENCY.fullmatch(currency):
            raise ContractError("Agent cost is invalid", "invalid_agent_cost")
        if (
            isinstance(self.source_event_index, bool)
            or not isinstance(self.source_event_index, int)
            or self.source_event_index < 0
            or not isinstance(self.source_key, str)
            or not self.source_key
        ):
            raise ContractError("Agent cost source is invalid", "invalid_agent_cost")
        object.__setattr__(self, "amount", format(value.normalize(), "f"))
        object.__setattr__(self, "currency", currency)

    def to_dict(self) -> dict[str, object]:
        return {
            "amount": self.amount,
            "currency": self.currency,
            "source_event_index": self.source_event_index,
            "source_key": self.source_key,
        }


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Frozen invocation presented to one stateless candidate worker."""

    run_id: str
    attempt_id: str
    backend: AgentBackendName
    prompt: str
    workspace: Path
    allowed_files: tuple[str, ...]
    model: str | None = None
    effort: str | None = None
    max_turns: int = 25
    timeout_seconds: int = 3600
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns <= 0
        ):
            raise ContractError("Agent max_turns must be positive", "invalid_agent_budget")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise ContractError("Agent timeout must be positive", "invalid_agent_budget")


@dataclass(frozen=True, slots=True)
class AgentInvocationReceipt:
    """Exact CLI entrypoint and invocation policy used for one agent process."""

    cli_name: str
    cli_version: str
    executable_path: str
    resolved_executable_path: str
    entrypoint_sha256: str
    argv: tuple[str, ...]
    workspace: str
    prompt_transport: str
    requested_allowed_files: tuple[str, ...]
    allowed_files_enforced_by_cli: bool
    max_turns: int
    turn_policy: str
    isolation: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        strings = (
            self.cli_name,
            self.cli_version,
            self.executable_path,
            self.resolved_executable_path,
            self.workspace,
            self.prompt_transport,
            self.turn_policy,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ContractError("Agent invocation identity is invalid", "invalid_agent_invocation")
        paths = (self.executable_path, self.resolved_executable_path, self.workspace)
        if any(not Path(value).is_absolute() for value in paths):
            raise ContractError("Agent invocation paths must be absolute", "invalid_agent_invocation")
        if not _SHA256.fullmatch(self.entrypoint_sha256):
            raise ContractError("Agent entrypoint digest is invalid", "invalid_agent_invocation")
        if not self.argv or any(not isinstance(value, str) or not value for value in self.argv):
            raise ContractError("Agent argv is invalid", "invalid_agent_invocation")
        if any(
            not isinstance(value, str) or not value for value in self.requested_allowed_files
        ) or not isinstance(self.allowed_files_enforced_by_cli, bool):
            raise ContractError("Agent file scope receipt is invalid", "invalid_agent_invocation")
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns <= 0
        ):
            raise ContractError("Agent invocation budget is invalid", "invalid_agent_invocation")
        keys = [key for key, _ in self.isolation]
        if (
            len(keys) != len(set(keys))
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in self.isolation
            )
        ):
            raise ContractError("Agent isolation receipt is invalid", "invalid_agent_invocation")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.agent-invocation/v1",
            "cli_name": self.cli_name,
            "cli_version": self.cli_version,
            "executable_path": self.executable_path,
            "resolved_executable_path": self.resolved_executable_path,
            "entrypoint_sha256": self.entrypoint_sha256,
            "argv": list(self.argv),
            "workspace": self.workspace,
            "prompt_transport": self.prompt_transport,
            "requested_allowed_files": list(self.requested_allowed_files),
            "allowed_files_enforced_by_cli": self.allowed_files_enforced_by_cli,
            "max_turns": self.max_turns,
            "turn_policy": self.turn_policy,
            "isolation": dict(self.isolation),
        }


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Normalized completion from a backend process."""

    backend: AgentBackendName
    model: str | None
    exit_code: int | None
    timed_out: bool
    events: Sequence[AgentTranscriptEvent]
    stdout: str
    stderr: str
    duration_seconds: float
    semantic_events: Sequence[AgentSemanticEvent] = ()
    usage: AgentUsage | None = None
    cost: AgentCost | None = None
    effort: str | None = None
    invocation: AgentInvocationReceipt | None = None
    budget_exceeded: bool = False
    budget_enforcement_failed: bool = False
    budget_reason: str | None = None
    observed_turns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.budget_exceeded, bool):
            raise ContractError("Agent budget outcome is invalid", "invalid_agent_result")
        if not isinstance(self.budget_enforcement_failed, bool):
            raise ContractError("Agent budget enforcement is invalid", "invalid_agent_result")
        if self.budget_reason is not None and (
            not isinstance(self.budget_reason, str) or not self.budget_reason
        ):
            raise ContractError("Agent budget reason is invalid", "invalid_agent_result")
        if (self.budget_exceeded or self.budget_enforcement_failed) != (
            self.budget_reason is not None
        ):
            raise ContractError("Agent budget reason is inconsistent", "invalid_agent_result")
        if (
            isinstance(self.observed_turns, bool)
            or not isinstance(self.observed_turns, int)
            or self.observed_turns < 0
        ):
            raise ContractError("Agent observed turn count is invalid", "invalid_agent_result")

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and not self.budget_exceeded
            and not self.budget_enforcement_failed
            and self.exit_code == 0
        )


class AgentBackend(Protocol):
    """Run one fresh stateless worker and return its complete transcript."""

    name: AgentBackendName

    def run(self, request: AgentRequest) -> AgentResult: ...
