"""Shared content-addressed artifact receipt for evaluator authorities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping


_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EvaluatorArtifactReceipt:
    """One bounded, non-empty authority artifact at a safe relative locator."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if (
            not self.path
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or not _DIGEST.fullmatch(self.sha256)
        ):
            raise ValueError("Evaluator artifact receipt is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluatorArtifactReceipt:
        if set(value) != {"path", "size_bytes", "sha256"}:
            raise ValueError("Evaluator artifact receipt fields are invalid")
        return cls(str(value["path"]), value["size_bytes"], str(value["sha256"]))


__all__ = ["EvaluatorArtifactReceipt"]
