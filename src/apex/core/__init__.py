"""Stable, standard-library-only primitives shared by every Apex module."""

from .errors import (
    ApexError,
    ConfigurationError,
    ContractError,
    DependencyError,
    IntegrityError,
    StateTransitionError,
)
from .hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from .types import (
    AgentBackendName,
    TaskStatus,
    ValidationLevel,
    new_identifier,
    validate_identifier,
)

__all__ = [
    "AgentBackendName",
    "ApexError",
    "ConfigurationError",
    "ContractError",
    "DependencyError",
    "IntegrityError",
    "StateTransitionError",
    "TaskStatus",
    "ValidationLevel",
    "canonical_json_bytes",
    "new_identifier",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_identifier",
]
