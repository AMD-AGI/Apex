"""Pinned, offline GEAK source archive and manifest construction."""

from __future__ import annotations

import io
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_json


GEAK_REPOSITORY = "https://github.com/AMD-AGI/GEAK"
GEAK_GIT_SHA = "6fa40c36b68bad9d543ae551b95bd3d169865744"
GEAK_LICENSE = "Apache-2.0"
GEAK_LICENSE_SHA256 = "cc09a983d7b46105587a34d759bbc9ed4e37d9857e5367766710d3f55e603315"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SourceEstatePin:
    """Expected aggregate identity for one upstream source subtree."""

    estate_id: str
    path: str
    expected_files: int
    expected_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if self.expected_files < 1 or self.expected_bytes < 1:
            raise ContractError("Source estate pin is empty", "invalid_source_estate_pin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "estate_id": self.estate_id,
            "path": self.path,
            "expected_files": self.expected_files,
            "expected_bytes": self.expected_bytes,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SourceEstatePin":
        try:
            return cls(
                estate_id=str(value["estate_id"]),
                path=str(value["path"]),
                expected_files=int(value["expected_files"]),
                expected_bytes=int(value["expected_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Malformed source estate pin", "invalid_source_estate_pin"
            ) from error


@dataclass(frozen=True, slots=True)
class PinnedSourceManifest:
    """Repository and exact revision from which normalized cards may be built."""

    repository: str
    git_sha: str
    license: str
    license_path: str
    license_sha256: str
    transform_version: str
    estates: tuple[SourceEstatePin, ...]

    def __post_init__(self) -> None:
        if not _GIT_SHA.fullmatch(self.git_sha) or not _SHA256.fullmatch(self.license_sha256):
            raise ContractError("Invalid source pin digest", "invalid_source_pin")
        _validate_relative_path(self.license_path)
        if not all(
            (self.repository.strip(), self.license.strip(), self.transform_version.strip())
        ):
            raise ContractError("Incomplete pinned source manifest", "invalid_source_pin")
        if not self.estates:
            raise ContractError("Pinned source manifest has no estates", "invalid_source_pin")
        if len({item.estate_id for item in self.estates}) != len(self.estates):
            raise ContractError("Duplicate source estate id", "duplicate_source_estate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "repository": self.repository,
            "git_sha": self.git_sha,
            "license": self.license,
            "license_path": self.license_path,
            "license_sha256": self.license_sha256,
            "transform_version": self.transform_version,
            "estates": [item.to_dict() for item in self.estates],
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "PinnedSourceManifest":
        try:
            if int(value.get("schema_version", 1)) != 1:
                raise ValueError("unsupported schema")
            return cls(
                repository=str(value["repository"]),
                git_sha=str(value["git_sha"]),
                license=str(value["license"]),
                license_path=str(value["license_path"]),
                license_sha256=str(value["license_sha256"]),
                transform_version=str(value["transform_version"]),
                estates=tuple(SourceEstatePin.from_mapping(item) for item in value["estates"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed pinned source manifest", "invalid_source_pin") from error

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Exact committed upstream bytes and their release disposition."""

    path: str
    estate_id: str
    content: bytes
    content_sha256: str
    card_eligible: bool
    exclusion_reason: str | None

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "estate_id": self.estate_id,
            "size_bytes": len(self.content),
            "content_sha256": self.content_sha256,
            "card_eligible": self.card_eligible,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Verified archive projection; runtime retrieval never reads this raw layer."""

    pin: PinnedSourceManifest
    files: tuple[SourceFile, ...]
    license_content: bytes

    def to_manifest(self) -> dict[str, Any]:
        entries = [item.to_manifest_entry() for item in self.files]
        value = {
            "schema_version": 1,
            "source_pin": self.pin.to_dict(),
            "files": entries,
            "summary": {
                "file_count": len(entries),
                "total_bytes": sum(item["size_bytes"] for item in entries),
                "card_eligible_files": sum(bool(item["card_eligible"]) for item in entries),
            },
        }
        return {**value, "manifest_sha256": sha256_json(value)}


def default_geak_source_pin() -> PinnedSourceManifest:
    """Return the audited GEAK knowledge-estate pin."""

    return PinnedSourceManifest(
        repository=GEAK_REPOSITORY,
        git_sha=GEAK_GIT_SHA,
        license=GEAK_LICENSE,
        license_path="LICENSE.md",
        license_sha256=GEAK_LICENSE_SHA256,
        transform_version="geak_markdown_to_card_v1",
        estates=(
            SourceEstatePin("perf_knowledge", "perf_knowledge", 689, 4_207_132),
            SourceEstatePin("kernel_workflow", "kernel_workflow/knowledge", 8, 63_988),
            SourceEstatePin("e2e_workflow", "e2e_workflow/knowledge", 36, 249_043),
        ),
    )


def archive_pinned_sources(root: Path, pin: PinnedSourceManifest) -> SourceSnapshot:
    """Read only committed bytes at the pin via ``git archive`` and validate them."""

    checkout = Path(root).resolve(strict=True)
    observed = _git(checkout, "rev-parse", "HEAD").decode("ascii").strip()
    if observed != pin.git_sha:
        raise IntegrityError(
            "GEAK checkout does not match the source pin", "source_revision_mismatch"
        )
    paths = (pin.license_path, *(estate.path for estate in pin.estates))
    archive = _git(checkout, "archive", "--format=tar", pin.git_sha, *paths)
    members = _read_regular_members(archive)
    license_content = members.pop(pin.license_path, None)
    if license_content is None or sha256_bytes(license_content) != pin.license_sha256:
        raise IntegrityError("Pinned upstream license does not match", "source_license_mismatch")
    files = tuple(_source_file(path, content, pin) for path, content in sorted(members.items()))
    _validate_estate_aggregates(files, pin.estates)
    return SourceSnapshot(pin=pin, files=files, license_content=license_content)


def _git(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise IntegrityError(
            "Unable to read pinned source archive", "source_archive_failed"
        ) from error
    return result.stdout


def _read_regular_members(content: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise IntegrityError(
                        "Archive contains a link or special file", "unsafe_source_archive"
                    )
                _validate_relative_path(member.name)
                handle = archive.extractfile(member)
                if handle is None:
                    raise IntegrityError("Archive member has no content", "invalid_source_archive")
                files[member.name] = handle.read()
    except tarfile.TarError as error:
        raise IntegrityError(
            "Pinned source archive is malformed", "invalid_source_archive"
        ) from error
    return files


def _source_file(path: str, content: bytes, pin: PinnedSourceManifest) -> SourceFile:
    estate_id = _estate_for_path(path, pin.estates)
    eligible, reason = _card_disposition(path)
    return SourceFile(
        path=path,
        estate_id=estate_id,
        content=content,
        content_sha256=sha256_bytes(content),
        card_eligible=eligible,
        exclusion_reason=reason,
    )


def _estate_for_path(path: str, estates: Iterable[SourceEstatePin]) -> str:
    for estate in estates:
        if path == estate.path or path.startswith(f"{estate.path}/"):
            return estate.estate_id
    raise IntegrityError("Archive contains an unpinned source path", "unexpected_source_path")


def _card_disposition(path: str) -> tuple[bool, str | None]:
    if path.startswith("perf_knowledge/expert_skills/"):
        return False, "nested_expert_skill_requires_separate_audit"
    if path.startswith("e2e_workflow/knowledge/analysis_skills/"):
        return False, "nested_analysis_tool_requires_separate_audit"
    if "/_templates/" in path:
        return False, "source_template_not_advisory_knowledge"
    if path.endswith("/_archive.md"):
        return False, "archived_source"
    if path.endswith(".md"):
        return True, None
    if path.endswith((".py", ".sh")):
        return False, "executable_source_requires_separate_audit"
    if path.endswith((".yaml", ".yml", ".json")):
        return False, "registry_metadata_not_prompt_card"
    return False, "unsupported_source_format"


def _validate_estate_aggregates(
    files: tuple[SourceFile, ...], estates: tuple[SourceEstatePin, ...]
) -> None:
    for estate in estates:
        selected = tuple(item for item in files if item.estate_id == estate.estate_id)
        observed = (len(selected), sum(len(item.content) for item in selected))
        expected = (estate.expected_files, estate.expected_bytes)
        if observed != expected:
            raise IntegrityError(
                "Pinned source estate aggregate does not match",
                "source_estate_mismatch",
                {"estate_id": estate.estate_id, "expected": expected, "observed": observed},
            )


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ContractError("Unsafe source manifest path", "invalid_source_manifest_path")


__all__ = [
    "GEAK_GIT_SHA",
    "GEAK_LICENSE",
    "GEAK_REPOSITORY",
    "PinnedSourceManifest",
    "SourceEstatePin",
    "SourceFile",
    "SourceSnapshot",
    "archive_pinned_sources",
    "default_geak_source_pin",
]
