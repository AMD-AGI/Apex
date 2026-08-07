"""Typed contracts for source-rebuilt E2E patch delivery.

These values intentionally describe evidence, not aspirations.  A caller may
construct an unverified bundle, but the terminal policy in ``e2e_verify`` is
the only code allowed to turn it into a source-rebuild-verified result.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from apex.core import ContractError, IntegrityError, sha256_json, validate_identifier


_GIT_OBJECT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REGULAR_MODES = frozenset({"100644", "100755"})
_CHANGE_KINDS = frozenset({"added", "modified", "deleted", "renamed"})
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "fish", "csh", "tcsh", "ksh"})


def safe_bundle_path(value: str, *, field: str = "path") -> str:
    """Return a normalized relative POSIX path or fail closed."""

    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise IntegrityError(f"Unsafe {field}: {value!r}", "unsafe_bundle_path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.casefold() == ".git" for part in path.parts)
        or path.as_posix() != value
    ):
        raise IntegrityError(f"Unsafe {field}: {value!r}", "unsafe_bundle_path")
    return value


def validate_sha256(value: str, *, field: str) -> str:
    value = str(value).removeprefix("sha256:")
    if not _SHA256.fullmatch(value):
        raise ContractError(f"{field} must be a lowercase SHA-256", "invalid_bundle_contract")
    return value


def _git_object(value: str, *, field: str) -> str:
    if not _GIT_OBJECT.fullmatch(value):
        raise ContractError(f"{field} must be an exact Git object id", "invalid_source_lock")
    return value


def validate_image_digest(value: str, *, field: str) -> str:
    if not _IMAGE_DIGEST.fullmatch(value):
        raise ContractError(f"{field} must be an immutable image digest", "invalid_image_identity")
    return value


def _locator_image_digest(value: str) -> str | None:
    if _IMAGE_DIGEST.fullmatch(value):
        return value
    if "@" in value:
        digest = value.rsplit("@", 1)[1]
        return digest if _IMAGE_DIGEST.fullmatch(digest) else None
    return None


def _mode(value: str | None, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if value not in _REGULAR_MODES:
        reason = "submodule_boundary" if value == "160000" else "unsupported_source_mode"
        raise IntegrityError(f"Unsupported source mode: {value!r}", reason)
    return value


def _matches_allowlist(path: str, allowlist: Sequence[str]) -> bool:
    for allowed in allowlist:
        if allowed.endswith("/"):
            if path.startswith(allowed):
                return True
        elif path == allowed:
            return True
    return False


@dataclass(frozen=True, slots=True)
class SourceFileChange:
    """Exact before/after identity for one Git source change."""

    kind: str
    old_path: str | None
    new_path: str | None
    before_blob: str | None
    after_blob: str | None
    before_sha256: str | None
    after_sha256: str | None
    old_mode: str | None
    new_mode: str | None

    def __post_init__(self) -> None:
        if self.kind not in _CHANGE_KINDS:
            raise ContractError("Unknown source change kind", "invalid_source_change")
        old_required = self.kind != "added"
        new_required = self.kind != "deleted"
        if old_required != (self.old_path is not None) or new_required != (self.new_path is not None):
            raise ContractError("Source change paths do not match kind", "invalid_source_change")
        if self.old_path is not None:
            safe_bundle_path(self.old_path, field="old_path")
        if self.new_path is not None:
            safe_bundle_path(self.new_path, field="new_path")
        if self.kind == "renamed" and self.old_path == self.new_path:
            raise ContractError("Rename must change the path", "invalid_source_change")
        if self.kind == "modified" and self.old_path != self.new_path:
            raise ContractError("Modified path identities must match", "invalid_source_change")
        for value, required, field in (
            (self.before_blob, old_required, "before_blob"),
            (self.after_blob, new_required, "after_blob"),
        ):
            if required:
                if value is None:
                    raise ContractError(f"Missing {field}", "invalid_source_change")
                _git_object(value, field=field)
            elif value is not None:
                raise ContractError(f"Unexpected {field}", "invalid_source_change")
        for value, required, field in (
            (self.before_sha256, old_required, "before_sha256"),
            (self.after_sha256, new_required, "after_sha256"),
        ):
            if required:
                if value is None:
                    raise ContractError(f"Missing {field}", "invalid_source_change")
                validate_sha256(value, field=field)
            elif value is not None:
                raise ContractError(f"Unexpected {field}", "invalid_source_change")
        _mode(self.old_mode, required=old_required)
        _mode(self.new_mode, required=new_required)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceFileChange":
        try:
            return cls(
                kind=str(value["kind"]),
                old_path=str(value["old_path"]) if value.get("old_path") is not None else None,
                new_path=str(value["new_path"]) if value.get("new_path") is not None else None,
                before_blob=str(value["before_blob"]) if value.get("before_blob") is not None else None,
                after_blob=str(value["after_blob"]) if value.get("after_blob") is not None else None,
                before_sha256=str(value["before_sha256"]) if value.get("before_sha256") is not None else None,
                after_sha256=str(value["after_sha256"]) if value.get("after_sha256") is not None else None,
                old_mode=str(value["old_mode"]) if value.get("old_mode") is not None else None,
                new_mode=str(value["new_mode"]) if value.get("new_mode") is not None else None,
            )
        except KeyError as error:
            raise ContractError("Source change is incomplete", "invalid_source_change") from error


@dataclass(frozen=True, slots=True)
class SourceRepositoryLock:
    """An ordered patch rooted at one exact, clean source repository."""

    repository_id: str
    url: str
    base_commit: str
    base_tree: str
    patched_tree: str
    patch_path: str
    patch_sha256: str
    order: int
    dependencies: tuple[str, ...]
    editable_allowlist: tuple[str, ...]
    changes: tuple[SourceFileChange, ...]
    build_recipe_sha256: str
    accepted_candidate_id: str
    anchor_generation: int
    clean_base: bool
    license_id: str
    runtime_component: str

    def __post_init__(self) -> None:
        validate_identifier(self.repository_id, field_name="repository_id")
        validate_identifier(self.accepted_candidate_id, field_name="accepted_candidate_id")
        if not self.url.strip():
            raise ContractError("Repository URL is required", "invalid_source_lock")
        _git_object(self.base_commit, field="base_commit")
        _git_object(self.base_tree, field="base_tree")
        _git_object(self.patched_tree, field="patched_tree")
        safe_bundle_path(self.patch_path, field="patch_path")
        validate_sha256(self.patch_sha256, field="patch_sha256")
        validate_sha256(self.build_recipe_sha256, field="build_recipe_sha256")
        if self.order < 0 or self.anchor_generation < 0 or not self.changes:
            raise ContractError("Source lock order/generation/changes are invalid", "invalid_source_lock")
        if self.clean_base is not True:
            raise ContractError("Formal source locks require a clean base", "dirty_source_base")
        if not self.license_id.strip() or not self.runtime_component.strip():
            raise ContractError("Source license/runtime provenance is missing", "invalid_source_lock")
        if self.repository_id in self.dependencies or len(set(self.dependencies)) != len(self.dependencies):
            raise ContractError("Repository dependencies are invalid", "invalid_source_lock")
        for dependency in self.dependencies:
            validate_identifier(dependency, field_name="repository_dependency")
        if not self.editable_allowlist:
            raise ContractError("Editable allowlist cannot be empty", "invalid_source_lock")
        normalized: list[str] = []
        for path in self.editable_allowlist:
            is_prefix = path.endswith("/")
            base = path[:-1] if is_prefix else path
            safe_bundle_path(base, field="editable_allowlist")
            normalized.append(base + ("/" if is_prefix else ""))
        if tuple(normalized) != self.editable_allowlist or len(set(normalized)) != len(normalized):
            raise ContractError("Editable allowlist is not canonical", "invalid_source_lock")
        for change in self.changes:
            for path in (change.old_path, change.new_path):
                if path is not None and not _matches_allowlist(path, normalized):
                    raise IntegrityError(f"Changed path is outside editable allowlist: {path}", "change_outside_allowlist")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dependencies"] = list(self.dependencies)
        value["editable_allowlist"] = list(self.editable_allowlist)
        value["changes"] = [item.to_dict() for item in self.changes]
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceRepositoryLock":
        try:
            changes = value["changes"]
            if not isinstance(changes, Sequence) or isinstance(changes, (str, bytes)):
                raise TypeError
            return cls(
                repository_id=str(value["repository_id"]),
                url=str(value["url"]),
                base_commit=str(value["base_commit"]),
                base_tree=str(value["base_tree"]),
                patched_tree=str(value["patched_tree"]),
                patch_path=str(value["patch_path"]),
                patch_sha256=str(value["patch_sha256"]),
                order=int(value["order"]),
                dependencies=tuple(str(item) for item in value.get("dependencies", ())),
                editable_allowlist=tuple(str(item) for item in value["editable_allowlist"]),
                changes=tuple(SourceFileChange.from_mapping(item) for item in changes),
                build_recipe_sha256=str(value["build_recipe_sha256"]),
                accepted_candidate_id=str(value["accepted_candidate_id"]),
                anchor_generation=int(value["anchor_generation"]),
                clean_base=value["clean_base"] is True,
                license_id=str(value["license_id"]),
                runtime_component=str(value["runtime_component"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Source repository lock is malformed", "invalid_source_lock") from error


@dataclass(frozen=True, slots=True)
class BuildStep:
    """One fixed argv build step.  Shell command strings are forbidden."""

    argv: tuple[str, ...]
    repository_id: str
    cwd: str = "."
    environment: tuple[tuple[str, str], ...] = ()
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        if not self.argv or any(not isinstance(part, str) or not part or "\x00" in part for part in self.argv):
            raise ContractError("Build step argv is invalid", "invalid_build_recipe")
        validate_identifier(self.repository_id, field_name="repository_id")
        if self.cwd != ".":
            safe_bundle_path(self.cwd, field="build cwd")
        executable = PurePosixPath(self.argv[0]).name.lower()
        if executable in _SHELLS and any(part in {"-c", "--command"} for part in self.argv[1:]):
            raise ContractError("Build recipe may not execute shell command strings", "shell_build_step_forbidden")
        if self.timeout_seconds <= 0:
            raise ContractError("Build timeout must be positive", "invalid_build_recipe")
        keys = [key for key, _ in self.environment]
        if len(set(keys)) != len(keys) or any(not key or "=" in key or "\x00" in key for key in keys):
            raise ContractError("Build environment is invalid", "invalid_build_recipe")

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "repository_id": self.repository_id,
            "cwd": self.cwd,
            "environment": {key: value for key, value in self.environment},
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BuildStep":
        try:
            environment = value.get("environment", {})
            if not isinstance(environment, Mapping):
                raise TypeError
            return cls(
                argv=tuple(str(item) for item in value["argv"]),
                repository_id=str(value["repository_id"]),
                cwd=str(value.get("cwd", ".")),
                environment=tuple(sorted((str(key), str(item)) for key, item in environment.items())),
                timeout_seconds=int(value.get("timeout_seconds", 3600)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Build step is malformed", "invalid_build_recipe") from error


@dataclass(frozen=True, slots=True)
class BuildRecipeLock:
    """Controller-trusted fixed build recipe bound to a parent image."""

    recipe_id: str
    parent_image_digest: str
    output_image_locator: str
    steps: tuple[BuildStep, ...]
    recipe_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.recipe_id, field_name="recipe_id")
        validate_image_digest(self.parent_image_digest, field="parent_image_digest")
        if not self.output_image_locator.strip() or not self.steps:
            raise ContractError("Build recipe is incomplete", "invalid_build_recipe")
        if self.recipe_sha256 is not None and self.recipe_sha256 != self.computed_sha256:
            raise IntegrityError("Build recipe digest drift", "build_recipe_drift")

    @property
    def computed_sha256(self) -> str:
        return sha256_json(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "recipe_id": self.recipe_id,
            "parent_image_digest": self.parent_image_digest,
            "output_image_locator": self.output_image_locator,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "recipe_sha256": self.computed_sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BuildRecipeLock":
        try:
            steps = value["steps"]
            if value.get("schema_version") != 1 or not isinstance(steps, Sequence):
                raise TypeError
            return cls(
                recipe_id=str(value["recipe_id"]),
                parent_image_digest=str(value["parent_image_digest"]),
                output_image_locator=str(value["output_image_locator"]),
                steps=tuple(BuildStep.from_mapping(item) for item in steps),
                recipe_sha256=str(value["recipe_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Build recipe lock is malformed", "invalid_build_recipe") from error


@dataclass(frozen=True, slots=True)
class DerivedImageIdentity:
    """Expected immutable output of the source rebuild."""

    locator: str
    parent_digest: str
    image_digest: str
    sbom_sha256: str
    build_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.locator.strip():
            raise ContractError("Derived image locator is required", "invalid_image_identity")
        validate_image_digest(self.parent_digest, field="parent_digest")
        validate_image_digest(self.image_digest, field="image_digest")
        validate_sha256(self.sbom_sha256, field="sbom_sha256")
        if _locator_image_digest(self.locator) != self.image_digest:
            raise ContractError("Derived image locator does not bind its digest", "invalid_image_identity")
        if len(set(self.build_ids)) != len(self.build_ids) or any(not item for item in self.build_ids):
            raise ContractError("Derived image build IDs are invalid", "invalid_image_identity")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "build_ids": list(self.build_ids)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DerivedImageIdentity":
        try:
            return cls(
                locator=str(value["locator"]),
                parent_digest=str(value["parent_digest"]),
                image_digest=str(value["image_digest"]),
                sbom_sha256=str(value["sbom_sha256"]),
                build_ids=tuple(str(item) for item in value.get("build_ids", ())),
            )
        except (KeyError, TypeError) as error:
            raise ContractError("Derived image identity is malformed", "invalid_image_identity") from error


@dataclass(frozen=True, slots=True)
class BundleProvenanceLock:
    """Exact workload, policy, image, model, and agent identity for delivery."""

    primary_run_id: str
    framework: str
    model_id: str
    model_revision: str
    gpu_arch: str
    baseline_image_digest: str
    original_config_sha256: str
    workload_semantics_sha256: str
    accuracy_policy_sha256: str
    performance_policy_sha256: str
    safety_policy_sha256: str | None
    agent_backend: str
    agent_model: str

    def __post_init__(self) -> None:
        validate_identifier(self.primary_run_id, field_name="primary_run_id")
        if any(not value.strip() for value in (
            self.framework,
            self.model_id,
            self.gpu_arch,
            self.agent_backend,
            self.agent_model,
        )):
            raise ContractError("Delivery provenance identity is incomplete", "invalid_delivery_provenance")
        _git_object(self.model_revision, field="model_revision")
        validate_image_digest(self.baseline_image_digest, field="baseline_image_digest")
        for field, value in (
            ("original_config_sha256", self.original_config_sha256),
            ("workload_semantics_sha256", self.workload_semantics_sha256),
            ("accuracy_policy_sha256", self.accuracy_policy_sha256),
            ("performance_policy_sha256", self.performance_policy_sha256),
        ):
            validate_sha256(value, field=field)
        if self.safety_policy_sha256 is not None:
            validate_sha256(self.safety_policy_sha256, field="safety_policy_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BundleProvenanceLock":
        try:
            return cls(
                primary_run_id=str(value["primary_run_id"]),
                framework=str(value["framework"]),
                model_id=str(value["model_id"]),
                model_revision=str(value["model_revision"]),
                gpu_arch=str(value["gpu_arch"]),
                baseline_image_digest=str(value["baseline_image_digest"]),
                original_config_sha256=str(value["original_config_sha256"]),
                workload_semantics_sha256=str(value["workload_semantics_sha256"]),
                accuracy_policy_sha256=str(value["accuracy_policy_sha256"]),
                performance_policy_sha256=str(value["performance_policy_sha256"]),
                safety_policy_sha256=str(value["safety_policy_sha256"]) if value.get("safety_policy_sha256") is not None else None,
                agent_backend=str(value["agent_backend"]),
                agent_model=str(value["agent_model"]),
            )
        except KeyError as error:
            raise ContractError("Delivery provenance is malformed", "invalid_delivery_provenance") from error


def replay_semantics(document: Mapping[str, Any]) -> dict[str, Any]:
    """Project config semantics while ignoring only deployment metadata/locator."""

    projected = copy.deepcopy(dict(document))
    projected.pop("apex", None)
    benchmark = projected.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ContractError("Replay config lacks benchmark mapping", "invalid_replay_config")
    benchmark["docker_image"] = "<APEX_DERIVED_IMAGE>"
    return projected


__all__ = [
    "BuildRecipeLock",
    "BuildStep",
    "BundleProvenanceLock",
    "DerivedImageIdentity",
    "SourceFileChange",
    "SourceRepositoryLock",
    "replay_semantics",
    "safe_bundle_path",
    "validate_image_digest",
    "validate_sha256",
]
