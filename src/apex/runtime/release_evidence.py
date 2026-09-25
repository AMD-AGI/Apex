"""Path-free release claims parsed before independent authority verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from apex.core import ContractError, sha256_json
from .magpie_config import RESULT_SCHEMA
from .release_qualification import QualificationEvidence
from .release_showcase import ShowcaseEvidence


_GIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_DIGEST = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")


def _strict(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError(f"{name} field set differs", "invalid_release_evidence")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be non-empty", "invalid_release_evidence")
    return value.strip()


def _match(value: object, pattern: re.Pattern[str], field: str) -> str:
    text = _text(value, field)
    if not pattern.fullmatch(text):
        raise ContractError(f"{field} has invalid identity", "invalid_release_evidence")
    return text


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be boolean", "invalid_release_evidence")
    return value


def _count(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{field} must be non-negative", "invalid_release_evidence")
    return value


def _argv(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be argv", "invalid_release_evidence")
    result = tuple(_text(item, field) for item in value)
    for item in result:
        path = PurePosixPath(item)
        if path.is_absolute() or ".." in path.parts:
            raise ContractError(f"{field} is not portable", "invalid_release_evidence")
    return result


@dataclass(frozen=True, slots=True)
class BaselineAuditEvidence:
    """Externally observed latest-remote audit for one clean Git source."""

    component: str
    repository: str
    branch: str
    commit: str
    tree: str
    remote_tip: str
    fetched: bool
    ancestry_reviewed: bool
    clean: bool

    SCHEMA = "apex.baseline-audit-evidence/v1"

    def __post_init__(self) -> None:
        for field in ("component", "repository", "branch"):
            _text(getattr(self, field), field)
        for field in ("commit", "tree", "remote_tip"):
            _match(getattr(self, field), _GIT, field)
        for field in ("fetched", "ancestry_reviewed", "clean"):
            _boolean(getattr(self, field), field)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }}

    @classmethod
    def from_dict(cls, value: object) -> BaselineAuditEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "baseline audit")
        if raw["schema"] != cls.SCHEMA:
            raise ContractError("baseline audit schema differs", "invalid_release_evidence")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class VerifiedComponentEvidence:
    """One exact, clean dependency or source checkout."""

    name: str
    repository: str
    commit: str
    tree: str
    clean: bool

    def __post_init__(self) -> None:
        _text(self.name, "component name")
        _text(self.repository, "component repository")
        _match(self.commit, _GIT, "component commit")
        _match(self.tree, _GIT, "component tree")
        _boolean(self.clean, "component clean")

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> VerifiedComponentEvidence:
        raw = _strict(value, set(cls.__dataclass_fields__), "verified component")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class DependencyVerificationEvidence:
    """Fresh-environment output bound to every checked-in runtime lock."""

    apex_tree: str
    dependencies_lock_sha256: str
    e2e_source_lock_sha256: str
    lm_eval_runtime_lock_sha256: str
    evaluator_policy_lock_sha256: str
    agent_templates_lock_sha256: str
    lm_eval_runtime_sha256: str
    all_imports_exact: bool
    components: tuple[VerifiedComponentEvidence, ...]

    SCHEMA = "apex.release-dependency-verification/v2"

    def __post_init__(self) -> None:
        _match(self.apex_tree, _GIT, "dependency Apex tree")
        for field in (
            "dependencies_lock_sha256", "e2e_source_lock_sha256",
            "lm_eval_runtime_lock_sha256", "evaluator_policy_lock_sha256",
            "agent_templates_lock_sha256",
            "lm_eval_runtime_sha256",
        ):
            _match(getattr(self, field), _SHA256, field)
        _boolean(self.all_imports_exact, "all_imports_exact")
        names = tuple(item.name for item in self.components)
        if names != tuple(sorted(set(names))):
            raise ContractError("verified components are not unique/sorted", "invalid_release_evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            **{
                field: getattr(self, field)
                for field in self.__dataclass_fields__
                if field != "components"
            },
            "components": [item.to_dict() for item in self.components],
        }

    @classmethod
    def from_dict(cls, value: object) -> DependencyVerificationEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "dependency verification")
        if raw["schema"] != cls.SCHEMA or not isinstance(raw["components"], list):
            raise ContractError("dependency verification schema differs", "invalid_release_evidence")
        values = {field: raw[field] for field in cls.__dataclass_fields__}
        values["components"] = tuple(
            VerifiedComponentEvidence.from_dict(item) for item in raw["components"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CpuGateEvidence:
    """Exact complete CPU/static gate result for one Apex tree."""

    apex_tree: str
    dependencies_lock_sha256: str
    e2e_source_lock_sha256: str
    corpus_manifest_sha256: str
    compatibility_ledger_sha256: str
    pytest_argv: tuple[str, ...]
    pytest_exit_code: int
    passed_count: int
    failed_count: int
    compileall_argv: tuple[str, ...]
    compileall_exit_code: int
    forbidden_scan_argv: tuple[str, ...]
    forbidden_scan_exit_code: int
    forbidden_scan_clean: bool

    SCHEMA = "apex.release-cpu-gate/v1"

    def __post_init__(self) -> None:
        _match(self.apex_tree, _GIT, "CPU gate Apex tree")
        for field in (
            "dependencies_lock_sha256", "e2e_source_lock_sha256",
            "corpus_manifest_sha256", "compatibility_ledger_sha256",
        ):
            _match(getattr(self, field), _SHA256, field)
        for field in ("pytest_argv", "compileall_argv", "forbidden_scan_argv"):
            if not isinstance(getattr(self, field), tuple):
                raise ContractError(f"{field} must be immutable argv", "invalid_release_evidence")
            _argv(list(getattr(self, field)), field)
        for field in (
            "pytest_exit_code", "compileall_exit_code", "forbidden_scan_exit_code"
        ):
            if not isinstance(getattr(self, field), int) or isinstance(getattr(self, field), bool):
                raise ContractError(f"{field} must be integer", "invalid_release_evidence")
        _count(self.passed_count, "passed_count")
        _count(self.failed_count, "failed_count")
        _boolean(self.forbidden_scan_clean, "forbidden_scan_clean")

    def to_dict(self) -> dict[str, Any]:
        result = {"schema": self.SCHEMA}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            result[field] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, value: object) -> CpuGateEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "CPU gate")
        if raw["schema"] != cls.SCHEMA:
            raise ContractError("CPU gate schema differs", "invalid_release_evidence")
        values = {field: raw[field] for field in cls.__dataclass_fields__}
        for field in ("pytest_argv", "compileall_argv", "forbidden_scan_argv"):
            values[field] = _argv(raw[field], field)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CliIdentityEvidence:
    """Fresh-shell installed entrypoint/import identity without host paths."""

    apex_tree: str
    project_version: str
    entrypoint: str
    import_module: str
    executable_sha256: str
    import_file_sha256: str

    SCHEMA = "apex.release-cli-identity/v1"

    def __post_init__(self) -> None:
        _match(self.apex_tree, _GIT, "CLI Apex tree")
        for field in ("project_version", "entrypoint", "import_module"):
            _text(getattr(self, field), field)
        _match(self.executable_sha256, _SHA256, "executable_sha256")
        _match(self.import_file_sha256, _SHA256, "import_file_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{
            field: getattr(self, field) for field in self.__dataclass_fields__
        }}

    @classmethod
    def from_dict(cls, value: object) -> CliIdentityEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "CLI identity")
        if raw["schema"] != cls.SCHEMA:
            raise ContractError("CLI identity schema differs", "invalid_release_evidence")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class MagpieConfigResolutionEntryEvidence:
    """One Apex resolution result bound to a frozen published Magpie config."""

    path: str
    config_sha256: str
    plan_sha256: str
    capability_receipt_sha256: str
    status: str
    run_mode: str
    lifecycle: str

    def __post_init__(self) -> None:
        path = PurePosixPath(_text(self.path, "Magpie config path"))
        if (
            path.is_absolute()
            or ".." in path.parts
            or not self.path.startswith("examples/benchmarks/")
        ):
            raise ContractError(
                "Magpie config path is unsafe", "invalid_release_evidence"
            )
        for field in (
            "config_sha256",
            "plan_sha256",
            "capability_receipt_sha256",
        ):
            _match(getattr(self, field), _SHA256, field)
        if self.status not in {
            "config_compatible",
            "capability_upgrade_required",
        }:
            raise ContractError(
                "Magpie config resolution status is invalid", "invalid_release_evidence"
            )
        if self.run_mode not in {"docker", "local", "ray"}:
            raise ContractError(
                "Magpie config resolution run mode is invalid", "invalid_release_evidence"
            )
        if self.lifecycle not in {"one_shot", "reuse", "cleanup"}:
            raise ContractError(
                "Magpie config resolution lifecycle is invalid",
                "invalid_release_evidence",
            )

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> MagpieConfigResolutionEntryEvidence:
        raw = _strict(value, set(cls.__dataclass_fields__), "Magpie config resolution entry")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class MagpieConfigResolutionEvidence:
    """All-corpus Apex projection over the pinned published Magpie main model."""

    magpie_commit: str
    corpus_manifest_sha256: str
    plan_schema: str
    capability_schema: str
    result_schema: str
    entries: tuple[MagpieConfigResolutionEntryEvidence, ...]
    resolved_manifest_sha256: str

    SCHEMA = "apex.release-magpie-config-resolution-evidence/v2"

    def __post_init__(self) -> None:
        _match(self.magpie_commit, _GIT, "Magpie config resolution commit")
        _match(self.corpus_manifest_sha256, _SHA256, "Magpie config resolution corpus manifest")
        if (
            self.plan_schema != "apex.magpie-main-resolved-plan/v1"
            or self.capability_schema
            != "apex.magpie-main-capability-receipt/v1"
            or self.result_schema != RESULT_SCHEMA
            or not self.entries
        ):
            raise ContractError(
                "Magpie config resolution schemas or entries differ",
                "invalid_release_evidence",
            )
        paths = tuple(item.path for item in self.entries)
        if paths != tuple(sorted(set(paths))):
            raise ContractError(
                "Magpie config resolution entries are not unique/sorted",
                "invalid_release_evidence",
            )
        _match(self.resolved_manifest_sha256, _SHA256, "Magpie resolved manifest")
        if self.resolved_manifest_sha256 != sha256_json(self.manifest_payload()):
            raise ContractError(
                "Magpie resolved manifest digest differs",
                "invalid_release_evidence",
            )

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "magpie_commit": self.magpie_commit,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "plan_schema": self.plan_schema,
            "capability_schema": self.capability_schema,
            "result_schema": self.result_schema,
            "entries": [item.to_dict() for item in self.entries],
        }

    def run_mode_manifest_payload(self, run_mode: str) -> dict[str, Any]:
        """Bind every resolved entry for one orthogonal execution mode."""

        if run_mode not in {"docker", "local", "ray"}:
            raise ContractError(
                "Magpie config resolution run mode is invalid", "invalid_release_evidence"
            )
        return {
            "schema": "apex.release-magpie-config-run-mode-manifest/v1",
            "resolved_manifest_sha256": self.resolved_manifest_sha256,
            "run_mode": run_mode,
            "entries": [
                item.to_dict() for item in self.entries if item.run_mode == run_mode
            ],
        }

    def run_mode_manifest_sha256(self, run_mode: str) -> str:
        """Return a path-free digest over the exact selected plan receipts."""

        return sha256_json(self.run_mode_manifest_payload(run_mode))

    def e2e_v2_entries(self) -> tuple[MagpieConfigResolutionEntryEvidence, ...]:
        """Return the exact Docker one-shot product scope."""

        return tuple(
            item
            for item in self.entries
            if item.run_mode == "docker" and item.lifecycle == "one_shot"
        )

    def e2e_v2_manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": "apex.release-magpie-e2e-v2-scope-manifest/v1",
            "resolved_manifest_sha256": self.resolved_manifest_sha256,
            "product_scope": "docker_one_shot",
            "entries": [item.to_dict() for item in self.e2e_v2_entries()],
        }

    def e2e_v2_manifest_sha256(self) -> str:
        return sha256_json(self.e2e_v2_manifest_payload())

    def e2e_v2_rejection_entries(self) -> tuple[MagpieConfigResolutionEntryEvidence, ...]:
        """Return the exact complement that V2 rejects before execution."""

        selected = set(self.e2e_v2_entries())
        return tuple(item for item in self.entries if item not in selected)

    def e2e_v2_rejection_manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": "apex.release-magpie-e2e-v2-rejection-manifest/v1",
            "resolved_manifest_sha256": self.resolved_manifest_sha256,
            "reason_code": "e2e_docker_only",
            "entries": [item.to_dict() for item in self.e2e_v2_rejection_entries()],
        }

    def e2e_v2_rejection_manifest_sha256(self) -> str:
        return sha256_json(self.e2e_v2_rejection_manifest_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.manifest_payload(),
                "resolved_manifest_sha256": self.resolved_manifest_sha256}

    @classmethod
    def from_dict(cls, value: object) -> MagpieConfigResolutionEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "Magpie config resolution evidence")
        if raw["schema"] != cls.SCHEMA or not isinstance(raw["entries"], list):
            raise ContractError(
                "Magpie config resolution evidence schema differs",
                "invalid_release_evidence",
            )
        values = {field: raw[field] for field in cls.__dataclass_fields__}
        values["entries"] = tuple(
            MagpieConfigResolutionEntryEvidence.from_dict(item) for item in raw["entries"]
        )
        return cls(**values)


def build_magpie_config_resolution_evidence(
    *,
    magpie_commit: str,
    corpus_manifest_sha256: str,
    plan_schema: str,
    capability_schema: str,
    result_schema: str,
    entries: Sequence[MagpieConfigResolutionEntryEvidence],
) -> MagpieConfigResolutionEvidence:
    """Sort and self-digest Apex per-config resolution results."""

    ordered = tuple(sorted(entries, key=lambda item: item.path))
    payload = {
        "schema": MagpieConfigResolutionEvidence.SCHEMA,
        "magpie_commit": magpie_commit,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "plan_schema": plan_schema,
        "capability_schema": capability_schema,
        "result_schema": result_schema,
        "entries": [item.to_dict() for item in ordered],
    }
    return MagpieConfigResolutionEvidence(
        magpie_commit,
        corpus_manifest_sha256,
        plan_schema,
        capability_schema,
        result_schema,
        ordered,
        sha256_json(payload),
    )


@dataclass(frozen=True, slots=True)
class ImageIdentityEvidence:
    """Immutable runtime image plus the exact in-image source digest."""

    name: str
    apex_tree: str
    image_id: str
    repo_digest: str
    source_digest: str

    SCHEMA = "apex.release-image-identity/v1"

    def __post_init__(self) -> None:
        _text(self.name, "image name")
        _match(self.apex_tree, _GIT, "image Apex tree")
        _match(self.image_id, _IMAGE_ID, "image_id")
        _match(self.repo_digest, _REPO_DIGEST, "repo_digest")
        _match(self.source_digest, _SHA256, "source_digest")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{
            field: getattr(self, field) for field in self.__dataclass_fields__
        }}

    @classmethod
    def from_dict(cls, value: object) -> ImageIdentityEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields, "image identity")
        if raw["schema"] != cls.SCHEMA:
            raise ContractError("image identity schema differs", "invalid_release_evidence")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class ReleaseEvidence:
    """Dynamic claims; omissions and unverified qualifications remain blockers."""

    apex_baseline: BaselineAuditEvidence | None = None
    magpie_baseline: BaselineAuditEvidence | None = None
    dependencies: DependencyVerificationEvidence | None = None
    magpie_config_resolution: MagpieConfigResolutionEvidence | None = None
    cpu_gate: CpuGateEvidence | None = None
    cli_identity: CliIdentityEvidence | None = None
    images: tuple[ImageIdentityEvidence, ...] = ()
    showcases: tuple[ShowcaseEvidence, ...] = ()
    qualifications: tuple[QualificationEvidence, ...] = ()

    def __post_init__(self) -> None:
        for field, values, key in (
            ("images", self.images, lambda item: item.name),
            ("showcases", self.showcases, lambda item: item.showcase_id),
            ("qualifications", self.qualifications, lambda item: item.qualification_id),
        ):
            names = tuple(key(item) for item in values)
            if names != tuple(sorted(set(names))):
                raise ContractError(f"{field} are not unique/sorted", "invalid_release_evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "apex_baseline": self.apex_baseline.to_dict() if self.apex_baseline else None,
            "magpie_baseline": self.magpie_baseline.to_dict() if self.magpie_baseline else None,
            "dependencies": self.dependencies.to_dict() if self.dependencies else None,
            "magpie_config_resolution": (
                self.magpie_config_resolution.to_dict()
                if self.magpie_config_resolution else None
            ),
            "cpu_gate": self.cpu_gate.to_dict() if self.cpu_gate else None,
            "cli_identity": self.cli_identity.to_dict() if self.cli_identity else None,
            "images": [item.to_dict() for item in self.images],
            "showcases": [item.to_dict() for item in self.showcases],
            "qualifications": [item.to_dict() for item in self.qualifications],
        }

    @classmethod
    def from_dict(cls, value: object) -> ReleaseEvidence:
        raw = _strict(value, set(cls.__dataclass_fields__), "release evidence")
        parsers = {
            "apex_baseline": BaselineAuditEvidence.from_dict,
            "magpie_baseline": BaselineAuditEvidence.from_dict,
            "dependencies": DependencyVerificationEvidence.from_dict,
            "magpie_config_resolution": MagpieConfigResolutionEvidence.from_dict,
            "cpu_gate": CpuGateEvidence.from_dict,
            "cli_identity": CliIdentityEvidence.from_dict,
        }
        values: dict[str, Any] = {}
        for field, parser in parsers.items():
            values[field] = None if raw[field] is None else parser(raw[field])
        sequence_parsers: Sequence[tuple[str, Any]] = (
            ("images", ImageIdentityEvidence.from_dict),
            ("showcases", ShowcaseEvidence.from_dict),
            ("qualifications", QualificationEvidence.from_dict),
        )
        for field, parser in sequence_parsers:
            if not isinstance(raw[field], list):
                raise ContractError(f"{field} must be a list", "invalid_release_evidence")
            values[field] = tuple(parser(item) for item in raw[field])
        return cls(**values)


__all__ = [
    "BaselineAuditEvidence", "CliIdentityEvidence", "CpuGateEvidence",
    "DependencyVerificationEvidence", "ImageIdentityEvidence",
    "MagpieConfigResolutionEntryEvidence", "MagpieConfigResolutionEvidence",
    "QualificationEvidence", "ReleaseEvidence", "ShowcaseEvidence",
    "VerifiedComponentEvidence", "build_magpie_config_resolution_evidence",
]
