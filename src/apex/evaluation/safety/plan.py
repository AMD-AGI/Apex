"""Immutable safety verification plans and candidate-freeze receipts."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from apex.core import ContractError, canonical_json_bytes, sha256_file, sha256_json, validate_identifier

from .policy import VerificationPolicy
from .profile import TaskSafetyProfile, ToolCapability, normalize_relative_path


PLAN_SCHEMA_VERSION = "apex.safety-verification-plan/v1"
ISOLATION_SCHEMA_VERSION = "apex.safety-phase-isolation/v1"
CANDIDATE_MANIFEST_SCHEMA_VERSION = "apex.frozen-safety-candidate/v1"
_SHELL_EXECUTABLES = {
    "bash",
    "csh",
    "cmd",
    "dash",
    "fish",
    "ksh",
    "powershell",
    "pwsh",
    "sh",
    "tcsh",
    "zsh",
}


def validate_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(
            f"{field} must be a lowercase SHA-256 digest",
            reason_code="invalid_safety_digest",
            details={"field": field},
        )
    return value


def validate_immutable_image_id(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ContractError(
            "safety runtime must use an immutable sha256 image ID, not a tag",
            reason_code="mutable_safety_runtime",
        )
    validate_sha256(value.removeprefix("sha256:"), field="runtime_image_id")
    return value


@dataclass(frozen=True, slots=True)
class ToolRuntimeIdentity:
    """Exact evaluator-owned tool and runtime identity."""

    tool: str
    version: str
    plugin_digest: str
    runtime_image_id: str
    helper_digest: str
    dispatch_digest: str

    def __post_init__(self) -> None:
        tool = self.tool.strip().lower().replace("-", "_")
        if not tool or not self.version.strip() or "\x00" in self.version:
            raise ContractError("invalid safety tool identity", "invalid_safety_plan")
        object.__setattr__(self, "tool", tool)
        object.__setattr__(self, "version", self.version.strip())
        validate_sha256(self.plugin_digest, field="plugin_digest")
        validate_immutable_image_id(self.runtime_image_id)
        validate_sha256(self.helper_digest, field="helper_digest")
        validate_sha256(self.dispatch_digest, field="dispatch_digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "version": self.version,
            "plugin_digest": self.plugin_digest,
            "runtime_image_id": self.runtime_image_id,
            "helper_digest": self.helper_digest,
            "dispatch_digest": self.dispatch_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ToolRuntimeIdentity":
        return cls(
            tool=str(value.get("tool", "")),
            version=str(value.get("version", "")),
            plugin_digest=str(value.get("plugin_digest", "")),
            runtime_image_id=str(value.get("runtime_image_id", "")),
            helper_digest=str(value.get("helper_digest", "")),
            dispatch_digest=str(value.get("dispatch_digest", "")),
        )


@dataclass(frozen=True, slots=True)
class ToolVerificationPlan:
    """Evaluator-owned argv and limits for one tool/runtime pair."""

    identity: ToolRuntimeIdentity
    capability: ToolCapability
    argv: tuple[str, ...]
    cases: tuple[str, ...]
    positive_control_digest: str
    timeout_seconds: int = 300
    output_limit_bytes: int = 16 * 1024 * 1024
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "environment", _environment_tuple(self.environment))
        if self.capability.tool != self.identity.tool:
            raise ContractError("tool capability/identity mismatch", "invalid_safety_plan")
        _validate_argv(self.argv)
        cases = tuple(case.strip() for case in self.cases)
        if not cases or any(not case or "\x00" in case for case in cases):
            raise ContractError("safety cases must be non-empty", "invalid_safety_plan")
        if cases != tuple(sorted(set(cases))):
            raise ContractError("safety cases must be unique and sorted", "invalid_safety_plan")
        object.__setattr__(self, "cases", cases)
        validate_sha256(self.positive_control_digest, field="positive_control_digest")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 86_400:
            raise ContractError("invalid safety timeout", "invalid_safety_plan")
        if self.output_limit_bytes <= 0 or self.output_limit_bytes > 256 * 1024 * 1024:
            raise ContractError("invalid safety output bound", "invalid_safety_plan")
        normalized_environment = tuple(sorted(self.environment))
        if normalized_environment != self.environment:
            raise ContractError("safety environment must be sorted", "invalid_safety_plan")
        keys: set[str] = set()
        for key, value in self.environment:
            if (
                not key
                or key in keys
                or "=" in key
                or "\x00" in key
                or "\x00" in value
                or key.startswith("APEX_SAFETY_")
            ):
                raise ContractError("invalid safety environment", "invalid_safety_plan")
            keys.add(key)

    @property
    def tool(self) -> str:
        return self.identity.tool

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "capability": self.capability.to_dict(),
            "argv": list(self.argv),
            "cases": list(self.cases),
            "positive_control_digest": self.positive_control_digest,
            "timeout_seconds": self.timeout_seconds,
            "output_limit_bytes": self.output_limit_bytes,
            "environment": [[key, value] for key, value in self.environment],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ToolVerificationPlan":
        raw_identity = value.get("identity")
        raw_capability = value.get("capability")
        if not isinstance(raw_identity, Mapping) or not isinstance(raw_capability, Mapping):
            raise ContractError("invalid tool verification plan", "invalid_safety_plan")
        return cls(
            identity=ToolRuntimeIdentity.from_dict(raw_identity),
            capability=ToolCapability.from_dict(raw_capability),
            argv=_string_tuple(value.get("argv"), "argv"),
            cases=_string_tuple(value.get("cases"), "cases"),
            positive_control_digest=str(value.get("positive_control_digest", "")),
            timeout_seconds=_integer(value.get("timeout_seconds"), "timeout_seconds"),
            output_limit_bytes=_integer(value.get("output_limit_bytes"), "output_limit_bytes"),
            environment=_environment_tuple(value.get("environment", ())),
        )


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    """Frozen plan bound to one source, candidate, deployment, and policy."""

    run_id: str
    candidate_id: str
    anchor_generation: int
    profile: TaskSafetyProfile
    policy_fingerprint: str
    source_digest: str
    candidate_digest: str
    deployed_digest: str
    tools: tuple[ToolVerificationPlan, ...]
    schema_version: str = PLAN_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        candidate_id: str,
        anchor_generation: int,
        profile: TaskSafetyProfile,
        policy: VerificationPolicy,
        source_digest: str,
        candidate_digest: str,
        deployed_digest: str,
        tools: Sequence[ToolVerificationPlan] = (),
    ) -> "VerificationPlan":
        """Bind an evaluator-owned policy fingerprint without caller duplication."""

        return cls(
            run_id=run_id,
            candidate_id=candidate_id,
            anchor_generation=anchor_generation,
            profile=profile,
            policy_fingerprint=policy.fingerprint,
            source_digest=source_digest,
            candidate_digest=candidate_digest,
            deployed_digest=deployed_digest,
            tools=tuple(tools),
        )

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise ContractError("unsupported safety plan schema", "unsupported_safety_schema")
        validate_identifier(self.run_id, field_name="run_id")
        validate_identifier(self.candidate_id, field_name="candidate_id")
        if self.anchor_generation < 0:
            raise ContractError("anchor_generation cannot be negative", "invalid_safety_plan")
        validate_sha256(self.policy_fingerprint, field="policy_fingerprint")
        validate_sha256(self.source_digest, field="source_digest")
        validate_sha256(self.candidate_digest, field="candidate_digest")
        validate_sha256(self.deployed_digest, field="deployed_digest")
        object.__setattr__(self, "tools", tuple(self.tools))
        tool_names = tuple(tool.tool for tool in self.tools)
        if tool_names != tuple(sorted(set(tool_names))):
            raise ContractError("safety tools must be unique and sorted", "invalid_safety_plan")

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._body())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "anchor_generation": self.anchor_generation,
            "profile": self.profile.to_dict(),
            "policy_fingerprint": self.policy_fingerprint,
            "source_digest": self.source_digest,
            "candidate_digest": self.candidate_digest,
            "deployed_digest": self.deployed_digest,
            "tools": [tool.to_dict() for tool in self.tools],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "plan_fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VerificationPlan":
        raw_profile = value.get("profile")
        raw_tools = value.get("tools")
        if not isinstance(raw_profile, Mapping):
            raise ContractError("invalid safety profile in plan", "invalid_safety_plan")
        if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
            raise ContractError("invalid safety tools in plan", "invalid_safety_plan")
        tools: list[ToolVerificationPlan] = []
        for raw in raw_tools:
            if not isinstance(raw, Mapping):
                raise ContractError("invalid safety tool plan", "invalid_safety_plan")
            tools.append(ToolVerificationPlan.from_dict(raw))
        plan = cls(
            schema_version=str(value.get("schema_version", "")),
            run_id=str(value.get("run_id", "")),
            candidate_id=str(value.get("candidate_id", "")),
            anchor_generation=_integer(value.get("anchor_generation"), "anchor_generation"),
            profile=TaskSafetyProfile.from_dict(raw_profile),
            policy_fingerprint=str(value.get("policy_fingerprint", "")),
            source_digest=str(value.get("source_digest", "")),
            candidate_digest=str(value.get("candidate_digest", "")),
            deployed_digest=str(value.get("deployed_digest", "")),
            tools=tuple(tools),
        )
        if value.get("plan_fingerprint") != plan.fingerprint:
            raise ContractError("safety plan fingerprint mismatch", "safety_plan_tampered")
        return plan


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    """Controller-owned, read-only candidate snapshot."""

    root: Path
    submission_paths: tuple[str, ...]
    candidate_digest: str

    def __post_init__(self) -> None:
        if not self.root.is_absolute() or self.root != self.root.resolve(strict=False):
            raise ContractError("frozen candidate root must be absolute", "invalid_frozen_candidate")
        paths = tuple(normalize_relative_path(path) for path in self.submission_paths)
        if paths != tuple(sorted(set(paths))):
            raise ContractError(
                "frozen candidate paths must be unique and sorted",
                "invalid_frozen_candidate",
            )
        object.__setattr__(self, "submission_paths", paths)
        validate_sha256(self.candidate_digest, field="candidate_digest")

    @classmethod
    def capture(cls, root: Path, profile: TaskSafetyProfile) -> "FrozenCandidate":
        digest = fingerprint_frozen_candidate(root, profile.submission_paths)
        return cls(root=root, submission_paths=profile.submission_paths, candidate_digest=digest)

    def verify(self) -> None:
        observed = fingerprint_frozen_candidate(self.root, self.submission_paths)
        if observed != self.candidate_digest:
            raise ContractError(
                "frozen candidate bytes no longer match their digest",
                reason_code="candidate_digest_mismatch",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
            "root": str(self.root),
            "submission_paths": list(self.submission_paths),
            "candidate_digest": self.candidate_digest,
        }


def fingerprint_frozen_candidate(root: Path, submission_paths: Sequence[str]) -> str:
    """Hash exact declared files while rejecting links, escapes, and mutable files."""

    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ContractError("invalid frozen candidate root", "invalid_frozen_candidate")
    lexical_root = root
    resolved_root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    for relative_value in submission_paths:
        relative = normalize_relative_path(relative_value)
        path = lexical_root.joinpath(*relative.split("/"))
        cursor = lexical_root
        for part in relative.split("/"):
            cursor = cursor / part
            try:
                metadata = cursor.lstat()
            except FileNotFoundError as exc:
                raise ContractError(
                    f"declared safety source is missing: {relative}",
                    reason_code="missing_candidate_source",
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(
                    f"symlinks are forbidden in frozen candidate paths: {relative}",
                    reason_code="candidate_path_escape",
                )
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not path.is_file():
            raise ContractError(
                f"candidate path escapes or is not a file: {relative}",
                reason_code="candidate_path_escape",
            )
        metadata = path.stat()
        if metadata.st_nlink != 1:
            raise ContractError(
                f"hard-linked candidate sources are not frozen: {relative}",
                reason_code="mutable_frozen_candidate",
            )
        if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ContractError(
                f"candidate source is writable: {relative}",
                reason_code="mutable_frozen_candidate",
            )
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )
    if not entries:
        raise ContractError("frozen candidate has no declared files", "invalid_frozen_candidate")
    return sha256_json(
        {"schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION, "files": entries}
    )


@dataclass(frozen=True, slots=True)
class PhaseIsolationReceipt:
    """Controller attestation that the agent phase ended before verification."""

    run_id: str
    plan_fingerprint: str
    anchor_generation: int
    candidate_digest: str
    frozen_root: str
    evaluator_artifact_root: str
    agent_process_tree_terminated: bool
    credentials_revoked: bool
    tool_channels_revoked: bool
    report_directory_hidden_from_agent: bool
    candidate_read_only: bool
    producer: str = "apex-controller"
    schema_version: str = ISOLATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ISOLATION_SCHEMA_VERSION:
            raise ContractError("unsupported phase isolation schema", "unsupported_safety_schema")
        validate_identifier(self.run_id, field_name="run_id")
        validate_sha256(self.plan_fingerprint, field="plan_fingerprint")
        validate_sha256(self.candidate_digest, field="candidate_digest")
        if self.anchor_generation < 0 or self.producer != "apex-controller":
            raise ContractError("invalid phase isolation receipt", "invalid_phase_isolation")
        frozen_path = Path(self.frozen_root)
        artifact_path = Path(self.evaluator_artifact_root)
        if (
            not frozen_path.is_absolute()
            or not artifact_path.is_absolute()
            or frozen_path != frozen_path.resolve(strict=False)
            or artifact_path != artifact_path.resolve(strict=False)
        ):
            raise ContractError("phase isolation paths must be absolute", "invalid_phase_isolation")

    @property
    def complete(self) -> bool:
        return all(
            (
                self.agent_process_tree_terminated,
                self.credentials_revoked,
                self.tool_channels_revoked,
                self.report_directory_hidden_from_agent,
                self.candidate_read_only,
            )
        )

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._body())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_fingerprint": self.plan_fingerprint,
            "anchor_generation": self.anchor_generation,
            "candidate_digest": self.candidate_digest,
            "frozen_root": self.frozen_root,
            "evaluator_artifact_root": self.evaluator_artifact_root,
            "agent_process_tree_terminated": self.agent_process_tree_terminated,
            "credentials_revoked": self.credentials_revoked,
            "tool_channels_revoked": self.tool_channels_revoked,
            "report_directory_hidden_from_agent": self.report_directory_hidden_from_agent,
            "candidate_read_only": self.candidate_read_only,
            "producer": self.producer,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "receipt_fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PhaseIsolationReceipt":
        receipt = cls(
            schema_version=str(value.get("schema_version", "")),
            run_id=str(value.get("run_id", "")),
            plan_fingerprint=str(value.get("plan_fingerprint", "")),
            anchor_generation=_integer(value.get("anchor_generation"), "anchor_generation"),
            candidate_digest=str(value.get("candidate_digest", "")),
            frozen_root=str(value.get("frozen_root", "")),
            evaluator_artifact_root=str(value.get("evaluator_artifact_root", "")),
            agent_process_tree_terminated=value.get("agent_process_tree_terminated") is True,
            credentials_revoked=value.get("credentials_revoked") is True,
            tool_channels_revoked=value.get("tool_channels_revoked") is True,
            report_directory_hidden_from_agent=value.get("report_directory_hidden_from_agent") is True,
            candidate_read_only=value.get("candidate_read_only") is True,
            producer=str(value.get("producer", "")),
        )
        if value.get("receipt_fingerprint") != receipt.fingerprint:
            raise ContractError("phase isolation receipt fingerprint mismatch", "phase_isolation_tampered")
        return receipt


def _validate_argv(argv: Sequence[str]) -> None:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ContractError("safety command must be an argv vector", "invalid_safety_argv")
    if any(not isinstance(item, str) or not item or "\x00" in item for item in argv):
        raise ContractError("invalid safety argv", "invalid_safety_argv")
    executable = os.path.basename(argv[0]).lower()
    if executable in _SHELL_EXECUTABLES:
        raise ContractError(
            "safety tools must be invoked directly without a shell",
            reason_code="shell_forbidden_for_safety_tool",
        )


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{field} must be a sequence", "invalid_safety_plan")
    if any(not isinstance(item, str) for item in value):
        raise ContractError(f"{field} must contain strings", "invalid_safety_plan")
    return tuple(value)


def _environment_tuple(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError("environment must be a sequence", "invalid_safety_plan")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ContractError("invalid safety environment entry", "invalid_safety_plan")
        key, item_value = item
        if not isinstance(key, str) or not isinstance(item_value, str):
            raise ContractError("invalid safety environment entry", "invalid_safety_plan")
        result.append((key, item_value))
    return tuple(result)


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} must be an integer", "invalid_safety_plan")
    return value


__all__ = [
    "CANDIDATE_MANIFEST_SCHEMA_VERSION",
    "FrozenCandidate",
    "ISOLATION_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "PhaseIsolationReceipt",
    "ToolRuntimeIdentity",
    "ToolVerificationPlan",
    "VerificationPlan",
    "fingerprint_frozen_candidate",
    "validate_immutable_image_id",
    "validate_sha256",
]
