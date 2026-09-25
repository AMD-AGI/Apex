"""Typed offline dataset CAS identity for evaluator execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import sha256_json, validate_identifier

from .evaluator_artifact_receipt import EvaluatorArtifactReceipt


DATASET_SCHEMA = "apex.evaluator-dataset-receipt/v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class EvaluatorDatasetFile:
    """One exact offline dataset file assigned to a named split."""

    split: str
    artifact: EvaluatorArtifactReceipt

    def __post_init__(self) -> None:
        validate_identifier(self.split, field_name="evaluator dataset split")

    def to_dict(self) -> dict[str, object]:
        return {"split": self.split, **self.artifact.to_dict()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluatorDatasetFile:
        if set(value) != {"split", "path", "size_bytes", "sha256"}:
            raise ValueError("Evaluator dataset file fields are invalid")
        artifact = EvaluatorArtifactReceipt.from_mapping(
            {key: value[key] for key in ("path", "size_bytes", "sha256")}
        )
        return cls(str(value["split"]), artifact)


@dataclass(frozen=True, slots=True)
class EvaluatorDatasetReceipt:
    """Offline dataset CAS identity consumed by the evaluator authority."""

    repository: str
    path: str
    name: str
    revision: str
    files: tuple[EvaluatorDatasetFile, ...]

    def __post_init__(self) -> None:
        if (
            not self.repository.startswith("https://")
            or any(character.isspace() for character in self.repository)
            or not self.path
            or not self.name
            or not _COMMIT.fullmatch(self.revision)
            or not self.files
            or len({item.artifact.path for item in self.files}) != len(self.files)
            or tuple(sorted(self.files, key=_dataset_file_key)) != self.files
        ):
            raise ValueError("Evaluator dataset receipt is invalid")

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": DATASET_SCHEMA,
            "repository": self.repository,
            "path": self.path,
            "name": self.name,
            "revision": self.revision,
            "offline": True,
            "files": [item.to_dict() for item in self.files],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluatorDatasetReceipt:
        fields = {
            "schema", "repository", "path", "name", "revision", "offline",
            "files", "sha256",
        }
        if (
            set(value) != fields
            or value.get("schema") != DATASET_SCHEMA
            or value.get("offline") is not True
            or not isinstance(value.get("files"), list)
        ):
            raise ValueError("Evaluator dataset receipt envelope is invalid")
        receipt = cls(
            repository=str(value["repository"]),
            path=str(value["path"]),
            name=str(value["name"]),
            revision=str(value["revision"]),
            files=tuple(_dataset_file(item) for item in value["files"]),
        )
        if value.get("sha256") != receipt.sha256:
            raise ValueError("Evaluator dataset receipt digest is invalid")
        return receipt


def _dataset_file(value: object) -> EvaluatorDatasetFile:
    if not isinstance(value, Mapping):
        raise ValueError("Evaluator dataset file declaration is invalid")
    return EvaluatorDatasetFile.from_mapping(value)


def _dataset_file_key(value: EvaluatorDatasetFile) -> tuple[str, str]:
    return value.split, value.artifact.path


__all__ = [
    "DATASET_SCHEMA",
    "EvaluatorDatasetFile",
    "EvaluatorDatasetReceipt",
]
