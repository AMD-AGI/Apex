"""Build, engagement, replay, and terminal delivery receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from apex.core import ContractError, TaskStatus, ValidationLevel, sha256_json

from .e2e_models import validate_image_digest, validate_sha256
from .git_patch import RepositoryApplyReceipt


@dataclass(frozen=True, slots=True)
class PrimaryVerificationEvidence:
    """Evidence produced by the promotion environment before clean replay."""

    environment_id: str
    runtime_identity_sha256: str
    source_stack_sha256: str
    build_receipt_sha256: str
    engagement_receipt_sha256: str
    benchmark_receipt_sha256: str
    safety_source_sha256: str | None
    performance_source_sha256: str
    deployed_source_sha256: str
    engagement_verified: bool
    normal_runtime_measurement: bool
    accuracy_passed: bool
    latency_gates_passed: bool
    objective_improved: bool
    overlay_verified: bool = False
    overlay_source_sha256: str | None = None
    overlay_rebuild_parity_passed: bool | None = None
    safety_certified: bool = False
    safety_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.environment_id:
            raise ContractError("Primary verification environment is required", "invalid_primary_evidence")
        for field, value in (
            ("runtime_identity_sha256", self.runtime_identity_sha256),
            ("source_stack_sha256", self.source_stack_sha256),
            ("build_receipt_sha256", self.build_receipt_sha256),
            ("engagement_receipt_sha256", self.engagement_receipt_sha256),
            ("benchmark_receipt_sha256", self.benchmark_receipt_sha256),
            ("performance_source_sha256", self.performance_source_sha256),
            ("deployed_source_sha256", self.deployed_source_sha256),
        ):
            validate_sha256(value, field=field)
        if self.safety_source_sha256 is not None:
            validate_sha256(self.safety_source_sha256, field="safety_source_sha256")
            if self.safety_receipt_sha256 is None:
                raise ContractError("Safety source lacks its evidence receipt", "candidate_lineage_mismatch")
            validate_sha256(self.safety_receipt_sha256, field="safety_receipt_sha256")
        elif self.safety_receipt_sha256 is not None:
            raise ContractError("Safety receipt lacks a source binding", "candidate_lineage_mismatch")
        if self.overlay_source_sha256 is not None:
            validate_sha256(self.overlay_source_sha256, field="overlay_source_sha256")
        if self.performance_source_sha256 != self.source_stack_sha256 or self.deployed_source_sha256 != self.source_stack_sha256:
            raise ContractError("Primary source lineage is inconsistent", "candidate_lineage_mismatch")
        if self.safety_source_sha256 is not None and self.safety_source_sha256 != self.source_stack_sha256:
            raise ContractError("Safety source differs from deployed source", "candidate_lineage_mismatch")
        if self.safety_certified and self.safety_source_sha256 is None:
            raise ContractError("Safety certification lacks source binding", "candidate_lineage_mismatch")
        if self.overlay_verified:
            if self.overlay_source_sha256 != self.source_stack_sha256 or self.overlay_rebuild_parity_passed is not True:
                raise ContractError("Overlay evidence does not match source rebuild", "overlay_rebuild_mismatch")
        elif self.overlay_source_sha256 is not None or self.overlay_rebuild_parity_passed is not None:
            raise ContractError("Unexpected overlay evidence", "invalid_primary_evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrimaryVerificationEvidence":
        try:
            return cls(
                environment_id=str(value["environment_id"]),
                runtime_identity_sha256=str(value["runtime_identity_sha256"]),
                source_stack_sha256=str(value["source_stack_sha256"]),
                build_receipt_sha256=str(value["build_receipt_sha256"]),
                engagement_receipt_sha256=str(value["engagement_receipt_sha256"]),
                benchmark_receipt_sha256=str(value["benchmark_receipt_sha256"]),
                safety_source_sha256=str(value["safety_source_sha256"]) if value.get("safety_source_sha256") is not None else None,
                performance_source_sha256=str(value["performance_source_sha256"]),
                deployed_source_sha256=str(value["deployed_source_sha256"]),
                engagement_verified=bool(value["engagement_verified"]),
                normal_runtime_measurement=bool(value["normal_runtime_measurement"]),
                accuracy_passed=bool(value["accuracy_passed"]),
                latency_gates_passed=bool(value["latency_gates_passed"]),
                objective_improved=bool(value["objective_improved"]),
                overlay_verified=bool(value.get("overlay_verified", False)),
                overlay_source_sha256=str(value["overlay_source_sha256"]) if value.get("overlay_source_sha256") is not None else None,
                overlay_rebuild_parity_passed=value.get("overlay_rebuild_parity_passed"),
                safety_certified=bool(value.get("safety_certified", False)),
                safety_receipt_sha256=str(value["safety_receipt_sha256"]) if value.get("safety_receipt_sha256") is not None else None,
            )
        except (KeyError, TypeError) as error:
            raise ContractError("Primary verification evidence is malformed", "invalid_primary_evidence") from error


@dataclass(frozen=True, slots=True)
class BuiltArtifact:
    component: str
    runtime_path: str
    sha256: str
    build_id: str | None
    source_stack_sha256: str

    def __post_init__(self) -> None:
        if not self.component or not self.runtime_path.startswith("/"):
            raise ContractError("Built artifact identity is invalid", "invalid_build_receipt")
        validate_sha256(self.sha256, field="artifact_sha256")
        validate_sha256(self.source_stack_sha256, field="source_stack_sha256")
        if self.build_id is not None and not self.build_id:
            raise ContractError("Build ID cannot be empty", "invalid_build_receipt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BuiltArtifact":
        try:
            return cls(
                component=str(value["component"]),
                runtime_path=str(value["runtime_path"]),
                sha256=str(value["sha256"]),
                build_id=str(value["build_id"]) if value.get("build_id") is not None else None,
                source_stack_sha256=str(value["source_stack_sha256"]),
            )
        except KeyError as error:
            raise ContractError("Built artifact is malformed", "invalid_build_receipt") from error


@dataclass(frozen=True, slots=True)
class BuildStepReceipt:
    index: int
    repository_id: str
    cwd: str
    argv_sha256: str
    exit_code: int | None
    timed_out: bool
    stdout_sha256: str
    stderr_sha256: str

    def __post_init__(self) -> None:
        if self.index < 0 or not self.repository_id or not self.cwd:
            raise ContractError("Build step receipt identity is invalid", "invalid_build_receipt")
        for field, value in (
            ("argv_sha256", self.argv_sha256),
            ("stdout_sha256", self.stdout_sha256),
            ("stderr_sha256", self.stderr_sha256),
        ):
            validate_sha256(value, field=field)

    @property
    def verified(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "verified": self.verified}


@dataclass(frozen=True, slots=True)
class SourceBuildReceipt:
    bundle_digest: str
    recipe_sha256: str
    expected_parent_digest: str
    observed_parent_digest: str
    expected_image_digest: str
    observed_image_digest: str
    expected_sbom_sha256: str
    observed_sbom_sha256: str
    source_stack_sha256: str
    clean_worktrees: bool
    artifacts: tuple[BuiltArtifact, ...]
    step_receipts: tuple[BuildStepReceipt, ...] = ()

    def __post_init__(self) -> None:
        for field, value in (
            ("bundle_digest", self.bundle_digest),
            ("recipe_sha256", self.recipe_sha256),
            ("source_stack_sha256", self.source_stack_sha256),
            ("expected_sbom_sha256", self.expected_sbom_sha256),
            ("observed_sbom_sha256", self.observed_sbom_sha256),
        ):
            validate_sha256(value, field=field)
        for field, value in (
            ("expected_parent_digest", self.expected_parent_digest),
            ("observed_parent_digest", self.observed_parent_digest),
            ("expected_image_digest", self.expected_image_digest),
            ("observed_image_digest", self.observed_image_digest),
        ):
            validate_image_digest(value, field=field)
        if not self.artifacts:
            raise ContractError("Build receipt must identify deployed artifacts", "invalid_build_receipt")
        artifact_keys = [(item.component, item.runtime_path) for item in self.artifacts]
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ContractError("Build receipt contains duplicate artifacts", "invalid_build_receipt")

    @property
    def steps_succeeded(self) -> bool:
        return (
            bool(self.step_receipts)
            and tuple(item.index for item in self.step_receipts)
            == tuple(range(len(self.step_receipts)))
            and all(item.verified for item in self.step_receipts)
        )

    @property
    def verified(self) -> bool:
        return (
            self.clean_worktrees
            and self.steps_succeeded
            and self.expected_parent_digest == self.observed_parent_digest
            and self.expected_image_digest == self.observed_image_digest
            and self.expected_sbom_sha256 == self.observed_sbom_sha256
            and all(item.source_stack_sha256 == self.source_stack_sha256 for item in self.artifacts)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "step_receipts": [item.to_dict() for item in self.step_receipts],
            "steps_succeeded": self.steps_succeeded,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    component: str
    runtime_path: str
    expected_sha256: str
    observed_sha256: str
    expected_build_id: str | None
    observed_build_id: str | None
    engagement_kind: str
    runtime_identity: str
    actually_loaded: bool

    def __post_init__(self) -> None:
        validate_sha256(self.expected_sha256, field="expected_loaded_sha256")
        validate_sha256(self.observed_sha256, field="observed_loaded_sha256")
        if self.engagement_kind not in {"python_import", "process_map", "linker_build_id"}:
            raise ContractError("Loaded-byte engagement kind is invalid", "invalid_engagement_receipt")
        if not self.runtime_identity:
            raise ContractError("Loaded-byte runtime identity is missing", "invalid_engagement_receipt")

    @property
    def verified(self) -> bool:
        return (
            self.actually_loaded
            and self.expected_sha256 == self.observed_sha256
            and self.expected_build_id == self.observed_build_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "verified": self.verified}


@dataclass(frozen=True, slots=True)
class LoadedByteEngagementReceipt:
    bundle_digest: str
    image_digest: str
    source_stack_sha256: str
    runtime_started_from_image: bool
    artifacts: tuple[LoadedArtifact, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.bundle_digest, field="bundle_digest")
        validate_image_digest(self.image_digest, field="image_digest")
        validate_sha256(self.source_stack_sha256, field="source_stack_sha256")
        if not self.artifacts:
            raise ContractError("Engagement receipt has no artifacts", "invalid_engagement_receipt")
        artifact_keys = [(item.component, item.runtime_path) for item in self.artifacts]
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ContractError("Engagement receipt contains duplicate artifacts", "invalid_engagement_receipt")

    @property
    def verified(self) -> bool:
        return self.runtime_started_from_image and all(item.verified for item in self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "artifacts": [item.to_dict() for item in self.artifacts], "verified": self.verified}


@dataclass(frozen=True, slots=True)
class ReplayConfigInvariantReceipt:
    measurement_config_sha256: str
    replay_config_sha256: str
    workload_semantics_sha256: str
    replay_image_locator: str
    unchanged_except_image_locator: bool

    def __post_init__(self) -> None:
        validate_sha256(self.measurement_config_sha256, field="measurement_config_sha256")
        validate_sha256(self.replay_config_sha256, field="replay_config_sha256")
        validate_sha256(self.workload_semantics_sha256, field="workload_semantics_sha256")
        if not self.replay_image_locator:
            raise ContractError("Replay image locator is missing", "invalid_replay_receipt")

    @property
    def verified(self) -> bool:
        return self.unchanged_except_image_locator

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "verified": self.verified}


@dataclass(frozen=True, slots=True)
class CleanReplayReceipt:
    bundle_digest: str
    primary_environment_id: str
    replay_environment_id: str
    image_digest: str
    replay_config_sha256: str
    benchmark_receipt_sha256: str
    source_stack_sha256: str
    source_materialization_sha256: str
    primary_runtime_identity_sha256: str
    replay_runtime_identity_sha256s: tuple[str, ...]
    normal_runtime_measurement: bool
    quality_passed: bool
    accuracy_passed: bool
    latency_gates_passed: bool
    objective_improved: bool
    paired_measurement: Mapping[str, Any]
    paired_verdict: Mapping[str, Any]
    raw_artifacts: tuple["ReplayArtifactReceipt", ...]

    def __post_init__(self) -> None:
        for field, value in (
            ("bundle_digest", self.bundle_digest),
            ("replay_config_sha256", self.replay_config_sha256),
            ("benchmark_receipt_sha256", self.benchmark_receipt_sha256),
            ("source_stack_sha256", self.source_stack_sha256),
            ("source_materialization_sha256", self.source_materialization_sha256),
            ("primary_runtime_identity_sha256", self.primary_runtime_identity_sha256),
        ):
            validate_sha256(value, field=field)
        for value in self.replay_runtime_identity_sha256s:
            validate_sha256(value, field="replay_runtime_identity_sha256")
        validate_image_digest(self.image_digest, field="image_digest")
        if not self.primary_environment_id or not self.replay_environment_id:
            raise ContractError("Replay environments are missing", "invalid_replay_receipt")
        if not _valid_paired_replay(self.paired_measurement, self.paired_verdict):
            raise ContractError("Paired replay evidence is invalid", "invalid_replay_receipt")
        raw_ids = tuple(self.paired_measurement["raw_measurement_receipts"])
        report_ids = tuple(
            item.measurement_receipt
            for item in self.raw_artifacts
            if item.role == "benchmark_report"
        )
        quality_ids = {
            item.quality_receipt
            for item in self.raw_artifacts
            if item.role == "quality_result"
        }
        attestation_ids = tuple(
            item.measurement_receipt
            for item in self.raw_artifacts
            if item.role == "execution_attestation"
        )
        if (
            tuple(sorted(report_ids)) != tuple(sorted(raw_ids))
            or tuple(sorted(attestation_ids)) != tuple(sorted(raw_ids))
            or len(self.replay_runtime_identity_sha256s) != len(raw_ids)
            or len(set((item.role, item.relative_path, item.measurement_receipt) for item in self.raw_artifacts))
            != len(self.raw_artifacts)
            or any(item.quality_receipt not in quality_ids for item in self.raw_artifacts)
        ):
            raise ContractError("Raw replay artifacts are incomplete", "invalid_replay_receipt")

    @property
    def fresh_source_materialization(self) -> bool:
        return bool(self.source_materialization_sha256)

    @property
    def fresh_runtime(self) -> bool:
        identities = self.replay_runtime_identity_sha256s
        return bool(
            identities
            and len(set(identities)) == len(identities)
            and self.primary_runtime_identity_sha256 not in identities
        )

    @property
    def verified(self) -> bool:
        return all(
            (
                self.primary_environment_id != self.replay_environment_id,
                self.fresh_source_materialization,
                self.fresh_runtime,
                self.normal_runtime_measurement,
                self.quality_passed,
                self.accuracy_passed,
                self.latency_gates_passed,
                self.objective_improved,
                self.paired_verdict.get("keep") is True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "replay_runtime_identity_sha256s": list(
                self.replay_runtime_identity_sha256s
            ),
            "fresh_source_materialization": self.fresh_source_materialization,
            "fresh_runtime": self.fresh_runtime,
            "raw_artifacts": [item.to_dict() for item in self.raw_artifacts],
            "verified": self.verified,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CleanReplayReceipt":
        try:
            return cls(
                bundle_digest=str(value["bundle_digest"]),
                primary_environment_id=str(value["primary_environment_id"]),
                replay_environment_id=str(value["replay_environment_id"]),
                image_digest=str(value["image_digest"]),
                replay_config_sha256=str(value["replay_config_sha256"]),
                benchmark_receipt_sha256=str(value["benchmark_receipt_sha256"]),
                source_stack_sha256=str(value["source_stack_sha256"]),
                source_materialization_sha256=str(
                    value["source_materialization_sha256"]
                ),
                primary_runtime_identity_sha256=str(
                    value["primary_runtime_identity_sha256"]
                ),
                replay_runtime_identity_sha256s=tuple(
                    str(item) for item in value["replay_runtime_identity_sha256s"]
                ),
                normal_runtime_measurement=bool(value["normal_runtime_measurement"]),
                quality_passed=bool(value["quality_passed"]),
                accuracy_passed=bool(value["accuracy_passed"]),
                latency_gates_passed=bool(value["latency_gates_passed"]),
                objective_improved=bool(value["objective_improved"]),
                paired_measurement=dict(value["paired_measurement"]),
                paired_verdict=dict(value["paired_verdict"]),
                raw_artifacts=tuple(
                    ReplayArtifactReceipt.from_mapping(dict(item))
                    for item in value["raw_artifacts"]
                ),
            )
        except (KeyError, TypeError) as error:
            raise ContractError("Clean replay receipt is malformed", "invalid_replay_receipt") from error


@dataclass(frozen=True, slots=True)
class ReplayArtifactReceipt:
    """Portable locator and digest for one raw clean-replay artifact."""

    role: str
    run_id: str
    measurement_receipt: str
    quality_receipt: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            self.role not in {
                "benchmark_report",
                "execution_attestation",
                "quality_result",
                "quality_sample",
                "quality_raw_artifact",
            }
            or not self.run_id
            or not self.measurement_receipt
            or not self.quality_receipt
            or path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or self.size_bytes < 0
            or not self.media_type
        ):
            raise ContractError("Replay artifact identity is invalid", "invalid_replay_receipt")
        validate_sha256(self.sha256, field="replay_artifact_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayArtifactReceipt":
        try:
            return cls(
                role=str(value["role"]),
                run_id=str(value["run_id"]),
                measurement_receipt=str(value["measurement_receipt"]),
                quality_receipt=str(value["quality_receipt"]),
                relative_path=str(value["relative_path"]),
                sha256=str(value["sha256"]),
                size_bytes=int(value["size_bytes"]),
                media_type=str(value["media_type"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Replay artifact is malformed", "invalid_replay_receipt") from error


def _valid_paired_replay(
    measurement: Mapping[str, Any], verdict: Mapping[str, Any]
) -> bool:
    windows = measurement.get("windows")
    raw = measurement.get("raw_measurement_receipts")
    return bool(
        measurement.get("schema") == "apex.e2e-paired-measurement/v1"
        and isinstance(windows, list)
        and len(windows) >= 3
        and isinstance(raw, list)
        and len(raw) == 4 * len(windows)
        and verdict.get("measurement_id") == sha256_json(dict(measurement))
    )


@dataclass(frozen=True, slots=True)
class DeliveryVerificationResult:
    bundle_digest: str
    verified: bool
    status: TaskStatus
    validation_level: ValidationLevel
    reason_code: str
    repository_receipts: tuple[RepositoryApplyReceipt, ...]
    build_receipt: SourceBuildReceipt | None
    engagement_receipt: LoadedByteEngagementReceipt | None
    config_receipt: ReplayConfigInvariantReceipt | None
    replay_receipt: CleanReplayReceipt | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "bundle_digest": self.bundle_digest,
            "verified": self.verified,
            "status": self.status.value,
            "validation_level": self.validation_level.value,
            "reason_code": self.reason_code,
            "repository_receipts": [item.to_dict() for item in self.repository_receipts],
            "build_receipt": self.build_receipt.to_dict() if self.build_receipt else None,
            "engagement_receipt": self.engagement_receipt.to_dict() if self.engagement_receipt else None,
            "config_receipt": self.config_receipt.to_dict() if self.config_receipt else None,
            "replay_receipt": self.replay_receipt.to_dict() if self.replay_receipt else None,
        }


def source_stack_digest(locks: Sequence[object]) -> str:
    """Hash ordered source identities without importing bundle serialization."""

    from apex.core import sha256_json

    values = [item.to_dict() for item in locks]  # type: ignore[attr-defined]
    return sha256_json({"schema_version": 1, "repositories": values})


__all__ = [
    "BuiltArtifact",
    "BuildStepReceipt",
    "CleanReplayReceipt",
    "DeliveryVerificationResult",
    "LoadedArtifact",
    "LoadedByteEngagementReceipt",
    "PrimaryVerificationEvidence",
    "ReplayConfigInvariantReceipt",
    "ReplayArtifactReceipt",
    "SourceBuildReceipt",
    "source_stack_digest",
]
