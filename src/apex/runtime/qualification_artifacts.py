"""Read-only resolution of release qualifications from formal CAS artifacts."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from apex.core import ContractError, sha256_bytes, sha256_json
from apex.ports import (
    QualificationAuthorityPort,
    QualificationAuthorityReceipt,
    build_qualification_authority_receipt,
)
from apex.storage import ArtifactReceipt

from .formal_results import FormalResultsRootValidator
from .release_qualification import (
    QUALIFICATION_IDS,
    QualificationEvidence,
)


INDEX_NAME = "qualification-artifacts.json"
INDEX_SCHEMA = "apex.formal-qualification-artifact-index/v1"
COLLECTION_SCHEMA = "apex.formal-qualification-artifact-resolution/v1"
AUTHORITY_ID = "apex-formal-qualification-artifact-resolver/v1"
_JSON_LIMIT = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class QualificationArtifactSet:
    """One content-addressed campaign manifest plus its read-only CAS."""

    qualification_id: str
    manifest_receipt: ArtifactReceipt
    manifest: Mapping[str, Any]
    artifacts: ReadOnlyQualificationArtifactStore


class QualificationCampaignArtifactVerifier(Protocol):
    """Trusted kind-specific logic that recomputes evidence from raw artifacts."""

    qualification_id: str
    verifier_identity_sha256: str

    def recompute(self, artifacts: QualificationArtifactSet) -> QualificationEvidence:
        """Recompute one exact qualification; never trust a supplied claim."""


@dataclass(frozen=True, slots=True)
class QualificationArtifactResolution:
    """Path-free outcome for one required qualification kind."""

    qualification_id: str
    status: str
    reason_code: str
    artifact_manifest_sha256: str | None
    evidence: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.qualification_id not in QUALIFICATION_IDS:
            _invalid("qualification artifact result id differs")
        if self.status not in {"verified", "unavailable", "invalid"}:
            _invalid("qualification artifact result status differs")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            _invalid("qualification artifact result reason is invalid")
        if self.artifact_manifest_sha256 is not None:
            _digest(self.artifact_manifest_sha256, "artifact manifest")
        if (self.status == "verified") != (self.evidence is not None):
            _invalid("qualification artifact result evidence differs")
        if self.evidence is not None:
            parsed = QualificationEvidence.from_dict(self.evidence)
            if parsed.qualification_id != self.qualification_id:
                _invalid("qualification artifact result evidence id differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification_id": self.qualification_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "evidence": dict(self.evidence) if self.evidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class QualificationArtifactCollection:
    """Self-digested, path-free inspection report; not itself authority."""

    artifact_index_sha256: str | None
    entries: tuple[QualificationArtifactResolution, ...]
    collection_sha256: str

    def __post_init__(self) -> None:
        if self.artifact_index_sha256 is not None:
            _digest(self.artifact_index_sha256, "artifact index")
        names = tuple(item.qualification_id for item in self.entries)
        if names != tuple(sorted(QUALIFICATION_IDS)):
            _invalid("qualification artifact collection coverage differs")
        if self.collection_sha256 != sha256_json(self.payload()):
            _invalid("qualification artifact collection digest differs")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": COLLECTION_SCHEMA,
            "artifact_index_sha256": self.artifact_index_sha256,
            "entries": [item.to_dict() for item in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "collection_sha256": self.collection_sha256}


class ReadOnlyQualificationArtifactStore:
    """Strict reader for the existing ``artifacts/sha256`` CAS layout."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def read_bytes(self, receipt: ArtifactReceipt) -> bytes:
        """Verify one canonical receipt and return its immutable bytes."""

        _validate_receipt(receipt)
        path = self.root / receipt.relative_path
        content = _read_regular(path, root=self.root, limit=_JSON_LIMIT)
        if len(content) != receipt.size or sha256_bytes(content) != receipt.digest:
            _invalid("qualification CAS artifact digest differs")
        return content

    def read_json(self, receipt: ArtifactReceipt) -> Mapping[str, Any]:
        """Verify one canonical receipt and return a strict JSON object."""

        content = self.read_bytes(receipt)
        return _json_mapping(content, "qualification CAS artifact")

    def verify(self, receipt: ArtifactReceipt) -> None:
        """Verify one receipt without exposing a caller-selected path."""

        self.read_bytes(receipt)


class EvaluatorQualificationArtifactAuthority(QualificationAuthorityPort):
    """Recompute claims using only an existing formal root and trusted verifiers."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        results_policy: FormalResultsRootValidator,
        verifiers: Sequence[QualificationCampaignArtifactVerifier] = (),
    ) -> None:
        self.root = results_policy.validate(artifact_root, require_new=False)
        if not self.root.is_dir():
            _unavailable("Qualification artifact root does not exist")
        self.artifacts = ReadOnlyQualificationArtifactStore(self.root / "artifacts")
        self._verifiers = _verifier_registry(verifiers)

    def verify(
        self, evidence: Mapping[str, Any]
    ) -> QualificationAuthorityReceipt:
        """Bind only byte-identical evidence independently recomputed from CAS."""

        supplied = QualificationEvidence.from_dict(evidence)
        index, index_bytes = self._load_index()
        artifact_set = self._artifact_set(index, supplied.qualification_id)
        verifier = self._verifiers.get(supplied.qualification_id)
        if verifier is None:
            _unavailable("No kind-specific qualification artifact verifier is installed")
        recomputed = _run_verifier(verifier, artifact_set)
        if recomputed.to_dict() != supplied.to_dict():
            _invalid("qualification claim differs from recomputed artifacts")
        if recomputed.apex_tree != index["apex_tree"]:
            _invalid("qualification artifact Apex tree differs")
        if self._index_bytes() != index_bytes:
            _invalid("qualification artifact index changed during verification")
        return build_qualification_authority_receipt(
            qualification_id=supplied.qualification_id,
            evidence_receipt_sha256=supplied.receipt_sha256,
            artifact_manifest_sha256=artifact_set.manifest_receipt.digest,
            verifier_identity_sha256=verifier.verifier_identity_sha256,
            authority_id=AUTHORITY_ID,
        )

    def collect(self) -> QualificationArtifactCollection:
        """Inspect every required kind without promoting unavailable artifacts."""

        try:
            index, index_bytes = self._load_index()
        except ContractError as error:
            status = (
                "unavailable"
                if error.reason_code == "qualification_artifacts_unavailable"
                else "invalid"
            )
            return _collection(None, tuple(
                QualificationArtifactResolution(
                    name, status, error.reason_code, None, None
                )
                for name in sorted(QUALIFICATION_IDS)
            ))
        entries = tuple(self._collect_one(index, name) for name in sorted(QUALIFICATION_IDS))
        if self._index_bytes() != index_bytes:
            _invalid("qualification artifact index changed during collection")
        return _collection(sha256_bytes(index_bytes), entries)

    def _collect_one(
        self, index: Mapping[str, Any], qualification_id: str
    ) -> QualificationArtifactResolution:
        digest: str | None = None
        try:
            artifact_set = self._artifact_set(index, qualification_id)
            digest = artifact_set.manifest_receipt.digest
            verifier = self._verifiers.get(qualification_id)
            if verifier is None:
                _unavailable("No kind-specific qualification artifact verifier is installed")
            evidence = _run_verifier(verifier, artifact_set)
            if evidence.apex_tree != index["apex_tree"]:
                _invalid("qualification artifact Apex tree differs")
            return QualificationArtifactResolution(
                qualification_id, "verified", "verified", digest, evidence.to_dict()
            )
        except ContractError as error:
            status = (
                "unavailable"
                if error.reason_code == "qualification_artifacts_unavailable"
                else "invalid"
            )
            return QualificationArtifactResolution(
                qualification_id, status, error.reason_code, digest, None
            )

    def _load_index(self) -> tuple[Mapping[str, Any], bytes]:
        content = self._index_bytes()
        raw = _json_mapping(content, "qualification artifact index")
        expected = {"schema", "apex_tree", "entries", "manifest_sha256"}
        if set(raw) != expected or raw.get("schema") != INDEX_SCHEMA:
            _invalid("qualification artifact index fields differ")
        apex_tree = raw.get("apex_tree")
        if not isinstance(apex_tree, str) or _GIT.fullmatch(apex_tree) is None:
            _invalid("qualification artifact index Apex tree is invalid")
        entries = _index_entries(raw.get("entries"))
        payload = {"schema": INDEX_SCHEMA, "apex_tree": apex_tree, "entries": entries}
        if raw.get("manifest_sha256") != sha256_json(payload):
            _invalid("qualification artifact index digest differs")
        return {**payload, "manifest_sha256": raw["manifest_sha256"]}, content

    def _index_bytes(self) -> bytes:
        path = self.root / INDEX_NAME
        if not path.exists():
            _unavailable("Formal result root has no qualification artifact index")
        return _read_regular(path, root=self.root, limit=_JSON_LIMIT)

    def _artifact_set(
        self, index: Mapping[str, Any], qualification_id: str
    ) -> QualificationArtifactSet:
        entry = next(
            (
                item for item in index["entries"]
                if item["qualification_id"] == qualification_id
            ),
            None,
        )
        if entry is None:
            _unavailable("Formal result root has no artifact for this qualification")
        receipt = _parse_receipt(entry["manifest_receipt"])
        manifest = self.artifacts.read_json(receipt)
        return QualificationArtifactSet(
            qualification_id, receipt, manifest, self.artifacts
        )


def _run_verifier(
    verifier: QualificationCampaignArtifactVerifier,
    artifacts: QualificationArtifactSet,
) -> QualificationEvidence:
    try:
        result = verifier.recompute(artifacts)
        evidence = QualificationEvidence.from_dict(result.to_dict())
    except ContractError as error:
        raise ContractError(
            "Kind-specific qualification artifact verification failed",
            "qualification_artifacts_invalid",
            {"cause": error.reason_code},
        ) from error
    if evidence.qualification_id != artifacts.qualification_id:
        _invalid("kind-specific verifier returned a different qualification")
    return evidence


def _collection(
    index_digest: str | None,
    entries: tuple[QualificationArtifactResolution, ...],
) -> QualificationArtifactCollection:
    payload = {
        "schema": COLLECTION_SCHEMA,
        "artifact_index_sha256": index_digest,
        "entries": [item.to_dict() for item in entries],
    }
    return QualificationArtifactCollection(index_digest, entries, sha256_json(payload))


def _verifier_registry(
    values: Sequence[QualificationCampaignArtifactVerifier],
) -> dict[str, QualificationCampaignArtifactVerifier]:
    result: dict[str, QualificationCampaignArtifactVerifier] = {}
    for item in values:
        if item.qualification_id not in QUALIFICATION_IDS:
            raise ContractError(
                "Qualification artifact verifier id is invalid",
                "qualification_artifacts_invalid",
            )
        _digest(item.verifier_identity_sha256, "qualification verifier identity")
        if item.qualification_id in result:
            raise ContractError(
                "Qualification artifact verifier id is duplicated",
                "qualification_artifacts_invalid",
            )
        result[item.qualification_id] = item
    return result


def _index_entries(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        _invalid("qualification artifact index entries are invalid")
    result: list[dict[str, Any]] = []
    names: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "qualification_id", "manifest_receipt"
        }:
            _invalid("qualification artifact index entry fields differ")
        name = item["qualification_id"]
        if name not in QUALIFICATION_IDS:
            _invalid("qualification artifact index entry id differs")
        receipt = _parse_receipt(item["manifest_receipt"])
        _validate_receipt(receipt)
        names.append(str(name))
        result.append({"qualification_id": name, "manifest_receipt": receipt.to_dict()})
    if names != sorted(set(names)):
        _invalid("qualification artifact index entries are not unique and sorted")
    return result


def _validate_receipt(receipt: ArtifactReceipt) -> None:
    _digest(receipt.digest, "qualification artifact receipt")
    if type(receipt.size) is not int or not 0 < receipt.size <= _JSON_LIMIT:
        _invalid("qualification artifact receipt size is invalid")
    if receipt.media_type != "application/json":
        _invalid("qualification artifact receipt media type differs")
    expected = PurePosixPath("sha256") / receipt.digest[:2] / receipt.digest
    path = PurePosixPath(receipt.relative_path)
    if path != expected or path.is_absolute() or ".." in path.parts:
        _invalid("qualification artifact receipt path differs")


def _parse_receipt(value: object) -> ArtifactReceipt:
    fields = {"digest", "size", "media_type", "relative_path"}
    if not isinstance(value, Mapping) or set(value) != fields:
        _invalid("qualification artifact receipt fields differ")
    if (
        not isinstance(value["digest"], str)
        or type(value["size"]) is not int
        or not isinstance(value["media_type"], str)
        or not isinstance(value["relative_path"], str)
    ):
        _invalid("qualification artifact receipt types differ")
    receipt = ArtifactReceipt(
        value["digest"], value["size"], value["media_type"], value["relative_path"]
    )
    _validate_receipt(receipt)
    return receipt


def _read_regular(path: Path, *, root: Path, limit: int) -> bytes:
    _inside(path, root)
    _reject_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(
            "Qualification artifact cannot be opened",
            "qualification_artifacts_invalid",
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _invalid("qualification artifact is not an isolated regular file")
        if before.st_size < 1 or before.st_size > limit:
            _invalid("qualification artifact size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _invalid("qualification artifact was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = os.lstat(path)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
    )
    if identity(before) != identity(after) or identity(after) != identity(observed):
        _invalid("qualification artifact changed during read")
    return b"".join(chunks)


def _inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        _invalid("qualification artifact path escapes its formal root")


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError as error:
            raise ContractError(
                "Qualification artifact path cannot be inspected",
                "qualification_artifacts_invalid",
            ) from error
        if stat.S_ISLNK(info.st_mode):
            _invalid("qualification artifact path contains a symlink")


def _json_mapping(content: bytes, label: str) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                _invalid(f"{label} contains duplicate keys")
            result[key] = value
        return result

    def constant(value: str) -> None:
        _invalid(f"{label} contains a non-finite value")

    try:
        value = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(
            f"{label} is not strict JSON",
            "qualification_artifacts_invalid",
        ) from error
    if not isinstance(value, Mapping):
        _invalid(f"{label} root is not an object")
    _json_depth(value, 0, label)
    return value


def _json_depth(value: object, depth: int, label: str) -> None:
    if depth > 64:
        _invalid(f"{label} nesting is too deep")
    if isinstance(value, Mapping):
        for item in value.values():
            _json_depth(item, depth + 1, label)
    elif isinstance(value, list):
        for item in value:
            _json_depth(item, depth + 1, label)


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _invalid(f"{label} digest is invalid")


def _invalid(message: str) -> None:
    raise ContractError(message, "qualification_artifacts_invalid")


def _unavailable(message: str) -> None:
    raise ContractError(message, "qualification_artifacts_unavailable")


__all__ = [
    "AUTHORITY_ID",
    "COLLECTION_SCHEMA",
    "INDEX_NAME",
    "INDEX_SCHEMA",
    "EvaluatorQualificationArtifactAuthority",
    "QualificationArtifactCollection",
    "QualificationArtifactResolution",
    "QualificationArtifactSet",
    "QualificationCampaignArtifactVerifier",
    "ReadOnlyQualificationArtifactStore",
]
