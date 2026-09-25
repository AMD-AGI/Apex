"""Content-addressed artifact storage with durable publication."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file

from ._atomic import FaultHook, fsync_directory


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Immutable evidence needed to locate and verify one artifact."""

    digest: str
    size: int
    media_type: str
    relative_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactReceipt":
        try:
            return cls(
                digest=str(value["digest"]),
                size=int(value["size"]),
                media_type=str(value["media_type"]),
                relative_path=str(value["relative_path"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Malformed artifact receipt",
                reason_code="invalid_artifact_receipt",
            ) from error


class ArtifactStore:
    """A SHA-256 CAS whose receipts are checked on every read."""

    def __init__(self, root: Path, *, fault_hook: FaultHook | None = None) -> None:
        self.root = Path(root)
        self._fault_hook = fault_hook

    def put_bytes(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactReceipt:
        if not isinstance(content, bytes):
            raise ContractError("Artifact content must be bytes", "invalid_artifact_content")
        digest = sha256_bytes(content)
        destination = self._path_for_digest(digest)
        receipt = self._receipt(digest, len(content), media_type, destination)
        if destination.exists():
            self._verify_path(destination, receipt)
            return receipt
        self._publish_bytes(destination, content)
        self._verify_path(destination, receipt)
        return receipt

    def put_file(
        self,
        source: Path,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactReceipt:
        source = Path(source)
        if not source.is_file():
            raise ContractError("Artifact source is not a file", "artifact_source_missing")
        temporary, digest, size = self._stage_file(source)
        try:
            destination = self._path_for_digest(digest)
            receipt = self._receipt(digest, size, media_type, destination)
            if destination.exists():
                self._verify_path(destination, receipt)
                return receipt
            self._publish_staged(temporary, destination)
            self._verify_path(destination, receipt)
            return receipt
        finally:
            temporary.unlink(missing_ok=True)

    def read_bytes(self, receipt: ArtifactReceipt) -> bytes:
        destination = self._resolve_receipt(receipt)
        try:
            content = destination.read_bytes()
        except OSError as error:
            raise IntegrityError("Artifact is missing", "artifact_missing") from error
        if len(content) != receipt.size or sha256_bytes(content) != receipt.digest:
            raise IntegrityError("Artifact failed receipt verification", "artifact_digest_mismatch")
        return content

    def verify(self, receipt: ArtifactReceipt) -> None:
        self._verify_path(self._resolve_receipt(receipt), receipt)

    def _receipt(
        self,
        digest: str,
        size: int,
        media_type: str,
        destination: Path,
    ) -> ArtifactReceipt:
        if not media_type:
            raise ContractError("Artifact media type is required", "invalid_media_type")
        return ArtifactReceipt(digest, size, media_type, str(destination.relative_to(self.root)))

    def _path_for_digest(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / digest

    def _resolve_receipt(self, receipt: ArtifactReceipt) -> Path:
        expected = self._path_for_digest(receipt.digest)
        supplied = self.root / receipt.relative_path
        if supplied != expected or len(receipt.digest) != 64:
            raise IntegrityError("Artifact receipt path is invalid", "artifact_receipt_path_mismatch")
        return expected

    def _publish_bytes(self, destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".artifact.")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self._fault("after_temp_fsync")
            self._fault("before_replace")
            os.replace(temporary, destination)
            self._fault("after_replace")
            fsync_directory(destination.parent)
            self._fault("after_parent_fsync")
        finally:
            temporary.unlink(missing_ok=True)

    def _stage_file(self, source: Path) -> tuple[Path, str, int]:
        staging = self.root / ".staging"
        staging.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=staging, prefix=".artifact.")
        temporary = Path(name)
        try:
            digest = hashlib.sha256()
            size = 0
            with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
                while chunk := input_stream.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            self._fault("after_temp_fsync")
            return temporary, digest.hexdigest(), size
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _publish_staged(self, temporary: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._fault("before_replace")
        os.replace(temporary, destination)
        self._fault("after_replace")
        fsync_directory(destination.parent)
        fsync_directory(self.root / ".staging")
        self._fault("after_parent_fsync")

    def _verify_path(self, path: Path, receipt: ArtifactReceipt) -> None:
        if not path.is_file():
            raise IntegrityError("Artifact is missing", "artifact_missing")
        if path.stat().st_size != receipt.size or sha256_file(path) != receipt.digest:
            raise IntegrityError("Artifact failed receipt verification", "artifact_digest_mismatch")

    def _fault(self, stage: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage)


__all__ = ["ArtifactReceipt", "ArtifactStore"]
