"""Repository identity boundary used before a formal campaign acquires a GPU."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, sha256_bytes


_GIT_OBJECT = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True, slots=True)
class WorkspaceRepositoryIdentity:
    root: str
    status: str
    remote: str | None
    commit: str | None
    tree: str | None
    dirty_paths: tuple[str, ...]
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not Path(self.root).is_absolute() or self.status not in {"resolved", "unresolved"}:
            raise ContractError("Repository identity is invalid", "invalid_repository_identity")
        resolved = self.status == "resolved"
        complete = bool(
            self.remote
            and self.commit
            and self.tree
            and _GIT_OBJECT.fullmatch(self.commit)
            and _GIT_OBJECT.fullmatch(self.tree)
        )
        if resolved != complete or resolved == bool(self.unavailable_reason):
            raise ContractError("Repository identity is incoherent", "invalid_repository_identity")

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def to_dict(self) -> dict[str, object]:
        return {
            "root_sha256": sha256_bytes(self.root.encode("utf-8")),
            "status": self.status,
            "remote": self.remote,
            "commit": self.commit,
            "tree": self.tree,
            "dirty_paths": list(self.dirty_paths),
            "unavailable_reason": self.unavailable_reason,
        }


class WorkspaceRepositoryIdentityPort(Protocol):
    def inspect(self, workspace: Path) -> WorkspaceRepositoryIdentity: ...


__all__ = ["WorkspaceRepositoryIdentity", "WorkspaceRepositoryIdentityPort"]
