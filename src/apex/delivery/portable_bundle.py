"""CAS-bound portable bundle capture and offline official-loader replay."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from apex.core import ContractError, IntegrityError, canonical_json_bytes
from apex.storage import ArtifactReceipt, ArtifactStore

from .e2e_bundle import load_and_verify_e2e_bundle
from .kernel_bundle import load_and_verify_kernel_bundle


_EVIDENCE_SCHEMA = "apex.portable-bundle-evidence/v1"
_VERIFICATION_SCHEMA = "apex.portable-bundle-verification/v1"
_KINDS = {"kernel", "e2e"}


class ArtifactReader(Protocol):
    def read_bytes(self, receipt: ArtifactReceipt) -> bytes: ...


@dataclass(frozen=True, slots=True)
class PortableBundleEvidence:
    bundle_kind: str
    bundle_digest: str
    evidence_receipt: ArtifactReceipt
    verification_receipt: ArtifactReceipt
    files: tuple[tuple[str, ArtifactReceipt], ...]

    def artifact_bindings(self) -> tuple[dict[str, object], ...]:
        return (
            _binding("winner_bundle", self.evidence_receipt),
            _binding("bundle_verification", self.verification_receipt),
            *(
                _binding(f"winner_bundle_file_{index:04d}", receipt)
                for index, (_, receipt) in enumerate(self.files)
            ),
        )


@dataclass(frozen=True, slots=True)
class PortableBundleVerification:
    bundle_kind: str
    bundle_digest: str
    file_count: int


def capture_portable_bundle(
    artifacts: ArtifactStore,
    bundle_path: Path,
    *,
    bundle_kind: str,
    expected_digest: str,
) -> PortableBundleEvidence:
    """Verify a real bundle, then bind every declared byte to the run CAS."""

    kind = _bundle_kind(bundle_kind)
    root, digest, verified = _load_official(bundle_path, kind, expected_digest)
    if kind == "e2e" and not verified:
        raise IntegrityError(
            "Formal E2E delivery bundle is not independently verified",
            "e2e_bundle_not_verified",
        )
    files = tuple(
        (relative, artifacts.put_file(path, media_type=_media_type(path)))
        for relative, path in _regular_files(root)
    )
    evidence_document = {
        "schema": _EVIDENCE_SCHEMA,
        "bundle_kind": kind,
        "bundle_digest": digest,
        "files": [
            {"path": relative, "receipt": receipt.to_dict()}
            for relative, receipt in files
        ],
    }
    evidence = artifacts.put_bytes(
        canonical_json_bytes(evidence_document), media_type="application/json"
    )
    verification_document = {
        "schema": _VERIFICATION_SCHEMA,
        "verified": True,
        "verifier_id": _verifier_id(kind),
        "bundle_kind": kind,
        "bundle_digest": digest,
        "bundle_evidence_receipt": evidence.to_dict(),
        "file_count": len(files),
    }
    verification = artifacts.put_bytes(
        canonical_json_bytes(verification_document), media_type="application/json"
    )
    return PortableBundleEvidence(kind, digest, evidence, verification, files)


def verify_portable_bundle(
    artifacts: ArtifactReader,
    evidence_receipt: ArtifactReceipt,
    verification_receipt: ArtifactReceipt,
) -> PortableBundleVerification:
    """Reconstruct CAS bytes and rerun the official bundle loader offline."""

    evidence = _read_canonical_json(artifacts, evidence_receipt)
    verification = _read_canonical_json(artifacts, verification_receipt)
    kind, digest, files = _parse_evidence(evidence)
    expected_verification = {
        "schema": _VERIFICATION_SCHEMA,
        "verified": True,
        "verifier_id": _verifier_id(kind),
        "bundle_kind": kind,
        "bundle_digest": digest,
        "bundle_evidence_receipt": evidence_receipt.to_dict(),
        "file_count": len(files),
    }
    if verification != expected_verification:
        raise IntegrityError(
            "Portable bundle verification receipt is invalid",
            "portable_bundle_verification_mismatch",
        )
    with tempfile.TemporaryDirectory(prefix="apex-portable-bundle-") as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        for relative, receipt in files:
            target = root.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(artifacts.read_bytes(receipt))
        _, replayed_digest, replayed_verified = _load_official(root, kind, digest)
    if kind == "e2e" and not replayed_verified:
        raise IntegrityError(
            "Portable E2E bundle lacks a verified terminal receipt",
            "e2e_bundle_not_verified",
        )
    return PortableBundleVerification(kind, replayed_digest, len(files))


def _parse_evidence(
    value: Mapping[str, Any],
) -> tuple[str, str, tuple[tuple[str, ArtifactReceipt], ...]]:
    if set(value) != {"schema", "bundle_kind", "bundle_digest", "files"} or value.get(
        "schema"
    ) != _EVIDENCE_SCHEMA:
        raise IntegrityError(
            "Portable bundle evidence schema is invalid", "invalid_portable_bundle"
        )
    kind = _bundle_kind(str(value.get("bundle_kind", "")))
    digest = str(value.get("bundle_digest", "")).removeprefix("sha256:")
    raw_files = value.get("files")
    if len(digest) != 64 or not isinstance(raw_files, list) or not raw_files:
        raise IntegrityError(
            "Portable bundle evidence is incomplete", "invalid_portable_bundle"
        )
    files: list[tuple[str, ArtifactReceipt]] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "receipt"}:
            raise IntegrityError(
                "Portable bundle file entry is invalid", "invalid_portable_bundle"
            )
        relative = _safe_relative(str(item.get("path", "")))
        receipt_value = item.get("receipt")
        if relative in seen or not isinstance(receipt_value, dict) or set(
            receipt_value
        ) != {"digest", "size", "media_type", "relative_path"}:
            raise IntegrityError(
                "Portable bundle file entry is invalid", "invalid_portable_bundle"
            )
        seen.add(relative)
        files.append((relative, ArtifactReceipt.from_dict(receipt_value)))
    if [relative for relative, _ in files] != sorted(seen):
        raise IntegrityError(
            "Portable bundle files are not canonical", "invalid_portable_bundle"
        )
    return kind, digest, tuple(files)


def _load_official(
    bundle_path: Path, kind: str, expected_digest: str
) -> tuple[Path, str, bool]:
    if kind == "kernel":
        loaded = load_and_verify_kernel_bundle(
            bundle_path, expected_digest=expected_digest
        )
        return loaded.path, loaded.digest, True
    loaded = load_and_verify_e2e_bundle(bundle_path, expected_digest=expected_digest)
    return loaded.path, loaded.digest, loaded.verified


def _regular_files(root: Path) -> tuple[tuple[str, Path], ...]:
    values: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IntegrityError("Bundle contains a symlink", "bundle_symlink")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_nlink != 1:
            raise IntegrityError("Bundle file is unsafe", "bundle_hardlink")
        values.append((path.relative_to(root).as_posix(), path))
    return tuple(sorted(values))


def _read_canonical_json(
    artifacts: ArtifactReader, receipt: ArtifactReceipt
) -> Mapping[str, Any]:
    content = artifacts.read_bytes(receipt)
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IntegrityError(
            "Portable bundle document cannot be decoded", "invalid_portable_bundle"
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise IntegrityError(
            "Portable bundle document is not canonical", "invalid_portable_bundle"
        )
    return value


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise IntegrityError("Portable bundle path is unsafe", "unsafe_bundle_path")
    return path.as_posix()


def _bundle_kind(value: str) -> str:
    if value not in _KINDS:
        raise ContractError("Unsupported portable bundle kind", "invalid_bundle_kind")
    return value


def _verifier_id(kind: str) -> str:
    return f"apex_{kind}_bundle_official_loader_v1"


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".patch": "text/x-diff",
        ".py": "text/x-python",
    }.get(path.suffix.lower(), "application/octet-stream")


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = [
    "PortableBundleEvidence",
    "PortableBundleVerification",
    "capture_portable_bundle",
    "verify_portable_bundle",
]
