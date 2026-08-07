"""Small immutable enums and identifier helpers shared across Apex."""

from __future__ import annotations

import re
import uuid
from enum import Enum

from .errors import ContractError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AgentBackendName(str, Enum):
    """Agent backends supported by the generic execution port."""

    CODEX = "codex"
    CLAUDE = "claude"
    CURSOR = "cursor"


class TaskStatus(str, Enum):
    """Terminal status vocabulary shared by kernel and E2E tasks."""

    SUCCEEDED = "succeeded"
    CANDIDATE_READY = "candidate_ready"
    NO_GAIN = "no_gain"
    REJECTED = "rejected"
    NEEDS_INPUT = "needs_input"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    BASELINE_INVALID = "baseline_invalid"
    PROVENANCE_UNRESOLVED = "provenance_unresolved"
    NO_MEASUREMENT = "no_measurement"
    VERIFICATION_FAILED = "verification_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class ValidationLevel(str, Enum):
    """Strength of implementation validation; distinct from task status."""

    NONE = "none"
    RUNTIME_OVERLAY_VERIFIED = "runtime_overlay_verified"
    SOURCE_REBUILD_VERIFIED = "source_rebuild_verified"


def validate_identifier(value: str, *, field_name: str = "identifier") -> str:
    """Validate a compact identifier suitable for paths and event keys."""

    if not _IDENTIFIER.fullmatch(value):
        raise ContractError(
            f"Invalid {field_name}: {value!r}",
            reason_code="invalid_identifier",
            details={"field": field_name},
        )
    return value


def new_identifier(prefix: str) -> str:
    """Generate a validated, sortable-enough opaque identifier."""

    validate_identifier(prefix, field_name="prefix")
    return f"{prefix}-{uuid.uuid4().hex}"
