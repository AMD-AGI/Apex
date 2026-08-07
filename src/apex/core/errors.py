"""Typed failures surfaced by Apex use cases and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(eq=False)
class ApexError(Exception):
    """Base error with a stable reason code and safe structured details."""

    message: str
    reason_code: str = "apex_error"
    details: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class ConfigurationError(ApexError):
    """Configuration is malformed, ambiguous, or incomplete."""


class ContractError(ApexError):
    """A caller or adapter violated a typed Apex boundary."""


class DependencyError(ApexError):
    """A pinned dependency cannot be resolved or verified."""


class IntegrityError(ApexError):
    """Content, provenance, or workspace integrity validation failed."""


class StateTransitionError(ApexError):
    """An action is not permitted from the current controller state."""
