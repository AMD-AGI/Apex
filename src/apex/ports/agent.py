"""Agent execution protocol; implementations live in :mod:`apex.execution`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from apex.core import AgentBackendName, ContractError


_SEMANTIC_EVENT_KINDS = {"agent_message", "tool_called", "tool_result"}
_CURRENCY = re.compile(r"[A-Z]{3,8}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
STRUCTURED_TURN_CHECKPOINT_POLICY = "structured_agent_turn_checkpoint_v2"
BOUNDARY_QUIESCENCE_POLICY = "sigstop_process_group_snapshot_v1"


class AgentTerminationKind(str, Enum):
    """Evaluator-visible reason an agent process stopped producing candidates."""

    COMPLETED = "completed"
    EXACT_TURN_BOUNDARY = "exact_turn_boundary"
    TIMEOUT = "timeout"
    INVALID_STREAM = "invalid_stream"
    TURN_OVERRUN = "turn_overrun"
    PROCESS_FAILED = "process_failed"


class AgentCaptureStatus(str, Enum):
    """Completeness of the stream and process-group capture after agent exit."""

    COMPLETE = "complete"
    OUTPUT_TRUNCATED = "output_truncated"
    CLEANUP_FAILED = "cleanup_failed"


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
    boundary_quiescence_policy_id: str
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
            self.boundary_quiescence_policy_id,
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
            "schema": "apex.agent-invocation/v2",
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
            "boundary_quiescence_policy_id": self.boundary_quiescence_policy_id,
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
    termination_kind: AgentTerminationKind = AgentTerminationKind.COMPLETED
    capture_status: AgentCaptureStatus = AgentCaptureStatus.COMPLETE
    termination_reason: str | None = None
    observed_turns: int = 0
    observer_stop_sent: bool = False
    observer_suspend_sent: bool = False
    suspension_verified: bool = False
    discarded_stdout_lines: int = 0
    discarded_stdout_bytes: int = 0
    discarded_stdout_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.termination_kind, AgentTerminationKind):
            raise ContractError("Agent termination kind is invalid", "invalid_agent_result")
        if not isinstance(self.capture_status, AgentCaptureStatus):
            raise ContractError("Agent capture status is invalid", "invalid_agent_result")
        if self.termination_reason is not None and (
            not isinstance(self.termination_reason, str) or not self.termination_reason
        ):
            raise ContractError("Agent termination reason is invalid", "invalid_agent_result")
        if (self.termination_kind is AgentTerminationKind.COMPLETED) != (
            self.termination_reason is None
        ):
            raise ContractError("Agent termination reason is inconsistent", "invalid_agent_result")
        if (
            isinstance(self.observed_turns, bool)
            or not isinstance(self.observed_turns, int)
            or self.observed_turns < 0
        ):
            raise ContractError("Agent observed turn count is invalid", "invalid_agent_result")
        if self.timed_out != (self.termination_kind is AgentTerminationKind.TIMEOUT):
            raise ContractError("Agent timeout evidence is inconsistent", "invalid_agent_result")
        suspension_flags = (self.observer_stop_sent, self.observer_suspend_sent, self.suspension_verified)
        if any(not isinstance(value, bool) for value in suspension_flags) or (
            self.suspension_verified and not self.observer_suspend_sent
        ):
            raise ContractError("Agent suspension evidence is invalid", "invalid_agent_result")
        discarded_counts = (self.discarded_stdout_lines, self.discarded_stdout_bytes)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in discarded_counts
        ):
            raise ContractError("Agent discarded-tail counts are invalid", "invalid_agent_result")
        has_tail = self.discarded_stdout_lines > 0 or self.discarded_stdout_bytes > 0
        if (
            has_tail
            and (
                self.discarded_stdout_lines == 0
                or self.discarded_stdout_bytes == 0
                or self.discarded_stdout_sha256 is None
                or not _SHA256.fullmatch(self.discarded_stdout_sha256)
            )
        ) or (not has_tail and self.discarded_stdout_sha256 is not None):
            raise ContractError("Agent discarded-tail digest is invalid", "invalid_agent_result")
        if self.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY:
            self._validate_exact_boundary()

    def _validate_exact_boundary(self) -> None:
        invocation = self.invocation
        if (
            invocation is None
            or invocation.turn_policy != STRUCTURED_TURN_CHECKPOINT_POLICY
            or invocation.boundary_quiescence_policy_id != BOUNDARY_QUIESCENCE_POLICY
            or self.observed_turns != invocation.max_turns
            or self.termination_reason != "max_turns_exact_boundary"
        ):
            raise ContractError("Exact-boundary evidence is incomplete", "invalid_agent_result")

    @property
    def candidate_capture_allowed(self) -> bool:
        """Whether trusted code may freeze source bytes from this process."""

        exact_boundary = (
            self.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY
            and self.observer_suspend_sent
            and self.suspension_verified
            and (self.exit_code == 0 or self.observer_stop_sent)
        )
        return self.capture_status is AgentCaptureStatus.COMPLETE and (
            self.succeeded or exact_boundary
        )

    @property
    def candidate_rejection_reason(self) -> str | None:
        """Stable reason a workspace may not cross the source-freeze boundary."""

        if self.capture_status is AgentCaptureStatus.CLEANUP_FAILED:
            return "agent_process_cleanup_failed"
        if self.capture_status is AgentCaptureStatus.OUTPUT_TRUNCATED:
            return "agent_output_truncated"
        if self.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY and (
            not self.observer_suspend_sent or not self.suspension_verified
        ):
            return "agent_boundary_suspension_unverified"
        if (
            self.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY
            and self.exit_code != 0
            and not self.observer_stop_sent
        ):
            return "agent_boundary_stop_unverified"
        return {
            AgentTerminationKind.COMPLETED: None if self.succeeded else "agent_failed",
            AgentTerminationKind.EXACT_TURN_BOUNDARY: None,
            AgentTerminationKind.TIMEOUT: "agent_timeout",
            AgentTerminationKind.INVALID_STREAM: "agent_turn_stream_invalid",
            AgentTerminationKind.TURN_OVERRUN: "agent_turn_budget_overrun",
            AgentTerminationKind.PROCESS_FAILED: "agent_failed",
        }[self.termination_kind]

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and self.termination_kind is AgentTerminationKind.COMPLETED
            and self.capture_status is AgentCaptureStatus.COMPLETE
            and self.exit_code == 0
        )


class AgentBackend(Protocol):
    """Run one fresh stateless worker and return its complete transcript."""

    name: AgentBackendName

    def run(self, request: AgentRequest) -> AgentResult: ...
