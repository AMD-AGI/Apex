"""Task profiling and capability contracts for safety evaluation.

These types describe what *can* be checked.  They intentionally do not say
whether a check is required; that decision belongs to the trusted
``VerificationPolicy`` in :mod:`apex.evaluation.safety.policy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Mapping, Sequence

from apex.core import ContractError, canonical_json_bytes, sha256_json


PROFILE_SCHEMA_VERSION = "apex.task-safety-profile/v1"


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class KernelLanguage(_ValueEnum):
    TRITON = "triton"
    HIP = "hip"
    PYTHON = "python"
    FLYDSL = "flydsl"
    UNKNOWN = "unknown"


class ArtifactKind(_ValueEnum):
    PYTHON_JIT = "python_jit"
    SOURCE_AOT = "source_aot"
    PRECOMPILED = "precompiled"
    UNKNOWN = "unknown"


class InstrumentationControl(_ValueEnum):
    COMPILER_CONTROLLED = "compiler_controlled"
    RECOMPILE = "recompile"
    NONE = "none"
    UNKNOWN = "unknown"


class CapabilityStatus(_ValueEnum):
    READY = "ready"
    ADAPTER_REQUIRED = "adapter_required"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE_RUNTIME = "unavailable_runtime"


_CAPABILITY_PRECEDENCE = (
    CapabilityStatus.NOT_APPLICABLE,
    CapabilityStatus.UNSUPPORTED,
    CapabilityStatus.ADAPTER_REQUIRED,
    CapabilityStatus.UNAVAILABLE_RUNTIME,
)


def _enum(enum_type: type[_ValueEnum], value: object, field: str) -> _ValueEnum:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise ContractError(
            f"invalid {field}: {value!r}",
            reason_code="invalid_safety_profile",
            details={"field": field},
        ) from exc


def normalize_relative_path(value: str, *, field: str = "submission_path") -> str:
    """Return a canonical workspace-relative POSIX path or fail closed."""

    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ContractError(
            f"{field} must be a non-empty POSIX path",
            reason_code="unsafe_safety_path",
            details={"field": field},
        )
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractError(
            f"{field} must stay inside the frozen candidate: {value!r}",
            reason_code="unsafe_safety_path",
            details={"field": field, "path": value},
        )
    normalized = path.as_posix()
    if normalized != value:
        raise ContractError(
            f"{field} must already be canonical: {value!r}",
            reason_code="unsafe_safety_path",
            details={"field": field, "path": value},
        )
    return normalized


@dataclass(frozen=True, slots=True)
class CapabilityCheck:
    """One independently observed capability dimension."""

    status: CapabilityStatus
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(CapabilityStatus, self.status, "capability"))
        if self.status is not CapabilityStatus.READY and not self.reason_code:
            raise ContractError(
                "non-ready capability requires a reason_code",
                reason_code="invalid_safety_capability",
            )
        if self.status is CapabilityStatus.READY and self.reason_code:
            raise ContractError(
                "ready capability cannot carry a blocking reason",
                reason_code="invalid_safety_capability",
            )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"status": self.status.value}
        if self.reason_code is not None:
            value["reason_code"] = self.reason_code
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CapabilityCheck":
        return cls(
            status=_enum(CapabilityStatus, value.get("status"), "capability"),
            reason_code=str(value["reason_code"]) if value.get("reason_code") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """Engine, adapter, runtime, and effective capability for one tool."""

    tool: str
    engine: CapabilityCheck
    adapter: CapabilityCheck
    runtime: CapabilityCheck
    effective: CapabilityCheck | None = None

    def __post_init__(self) -> None:
        tool = self.tool.strip().lower().replace("-", "_")
        if not tool or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in tool):
            raise ContractError("invalid safety tool name", "invalid_safety_capability")
        object.__setattr__(self, "tool", tool)
        resolved = self._resolve_effective()
        if self.effective is None:
            object.__setattr__(self, "effective", resolved)
        elif self.effective != resolved:
            raise ContractError(
                "effective capability disagrees with its independent dimensions",
                reason_code="forged_safety_capability",
                details={"tool": tool},
            )

    def _resolve_effective(self) -> CapabilityCheck:
        dimensions = (self.engine, self.adapter, self.runtime)
        for status in _CAPABILITY_PRECEDENCE:
            for check in dimensions:
                if check.status is status:
                    return check
        return CapabilityCheck(CapabilityStatus.READY)

    @property
    def ready(self) -> bool:
        assert self.effective is not None
        return self.effective.status is CapabilityStatus.READY

    def to_dict(self) -> dict[str, object]:
        assert self.effective is not None
        return {
            "tool": self.tool,
            "engine": self.engine.to_dict(),
            "adapter": self.adapter.to_dict(),
            "runtime": self.runtime.to_dict(),
            "effective": self.effective.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ToolCapability":
        def check(name: str) -> CapabilityCheck:
            raw = value.get(name)
            if not isinstance(raw, Mapping):
                raise ContractError(f"missing capability dimension {name}", "invalid_safety_capability")
            return CapabilityCheck.from_dict(raw)

        return cls(
            tool=str(value.get("tool", "")),
            engine=check("engine"),
            adapter=check("adapter"),
            runtime=check("runtime"),
            effective=check("effective"),
        )


@dataclass(frozen=True, slots=True)
class TaskSafetyProfile:
    """Immutable, caller-neutral boundary of a candidate safety task."""

    language: KernelLanguage
    artifact_kind: ArtifactKind
    instrumentation_control: InstrumentationControl
    submission_paths: tuple[str, ...]
    target_symbols: tuple[str, ...] = ()
    adapter_capabilities: tuple[str, ...] = ()
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ContractError("unsupported safety profile schema", "unsupported_safety_schema")
        object.__setattr__(self, "language", _enum(KernelLanguage, self.language, "language"))
        object.__setattr__(self, "artifact_kind", _enum(ArtifactKind, self.artifact_kind, "artifact_kind"))
        object.__setattr__(
            self,
            "instrumentation_control",
            _enum(InstrumentationControl, self.instrumentation_control, "instrumentation_control"),
        )
        paths = tuple(normalize_relative_path(path) for path in self.submission_paths)
        if not paths or len(set(paths)) != len(paths) or paths != tuple(sorted(paths)):
            raise ContractError(
                "submission_paths must be non-empty, unique, and sorted",
                reason_code="invalid_safety_profile",
            )
        object.__setattr__(self, "submission_paths", paths)
        symbols = _normalized_strings(self.target_symbols, "target_symbols")
        capabilities = _normalized_strings(self.adapter_capabilities, "adapter_capabilities")
        object.__setattr__(self, "target_symbols", symbols)
        object.__setattr__(self, "adapter_capabilities", capabilities)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "language": self.language.value,
            "artifact_kind": self.artifact_kind.value,
            "instrumentation_control": self.instrumentation_control.value,
            "submission_paths": list(self.submission_paths),
            "target_symbols": list(self.target_symbols),
            "adapter_capabilities": list(self.adapter_capabilities),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TaskSafetyProfile":
        return cls(
            schema_version=str(value.get("schema_version", "")),
            language=_enum(KernelLanguage, value.get("language"), "language"),
            artifact_kind=_enum(ArtifactKind, value.get("artifact_kind"), "artifact_kind"),
            instrumentation_control=_enum(
                InstrumentationControl,
                value.get("instrumentation_control"),
                "instrumentation_control",
            ),
            submission_paths=_sequence_of_strings(value.get("submission_paths"), "submission_paths"),
            target_symbols=_sequence_of_strings(value.get("target_symbols", ()), "target_symbols"),
            adapter_capabilities=_sequence_of_strings(
                value.get("adapter_capabilities", ()), "adapter_capabilities"
            ),
        )


def _sequence_of_strings(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{field} must be a sequence", "invalid_safety_profile")
    if any(not isinstance(item, str) for item in value):
        raise ContractError(f"{field} must contain strings", "invalid_safety_profile")
    return tuple(value)


def _normalized_strings(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized = tuple(value.strip() for value in values)
    if any(not value or "\x00" in value for value in normalized):
        raise ContractError(f"invalid {field}", "invalid_safety_profile")
    if normalized != tuple(sorted(set(normalized))):
        raise ContractError(f"{field} must be unique and sorted", "invalid_safety_profile")
    return normalized


__all__ = [
    "ArtifactKind",
    "CapabilityCheck",
    "CapabilityStatus",
    "InstrumentationControl",
    "KernelLanguage",
    "PROFILE_SCHEMA_VERSION",
    "TaskSafetyProfile",
    "ToolCapability",
    "normalize_relative_path",
]
