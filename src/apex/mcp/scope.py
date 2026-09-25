"""Run-scoped path authority for local capability handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from apex.core import ContractError, validate_identifier


@dataclass(frozen=True, slots=True)
class CapabilityScope:
    """Confine capability reads and immutable outputs to caller-selected roots."""

    workspace: Path
    results: Path

    def __post_init__(self) -> None:
        if self.workspace.is_symlink() or self.results.is_symlink():
            raise ContractError("Capability scope roots cannot be symlinks", "unsafe_capability_path")
        workspace = self.workspace.resolve(strict=True)
        results = self.results.resolve()
        if not workspace.is_dir():
            raise ContractError("Capability workspace is not a directory", "invalid_capability_scope")
        if results == Path("/") or results == workspace:
            raise ContractError("Capability results root is too broad", "invalid_capability_scope")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "results", results)

    def read_workspace(self, relative_path: str) -> Path:
        return self._read_scoped(self.workspace, relative_path)

    def read_results(self, relative_path: str) -> Path:
        return self._read_scoped(self.results, relative_path)

    @staticmethod
    def _read_scoped(root: Path, relative_path: str) -> Path:
        relative = _relative_path(relative_path)
        candidate = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ContractError("Capability input cannot use symlinks", "unsafe_capability_path")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise ContractError("Capability input escapes its scope", "unsafe_capability_path") from error
        return resolved

    def claim_output(self, category: str, run_id: str) -> Path:
        validate_identifier(run_id, field_name="capability run_id")
        if not category or "/" in category or category in {".", ".."}:
            raise ContractError("Capability output category is invalid", "invalid_capability_scope")
        self.results.mkdir(parents=True, exist_ok=True)
        if self.results.is_symlink() or self.results.resolve() != self.results:
            raise ContractError("Capability results root is unsafe", "unsafe_capability_path")
        parent = self.results / category
        parent.mkdir(exist_ok=True)
        if parent.is_symlink() or parent.resolve() != parent:
            raise ContractError("Capability output parent is unsafe", "unsafe_capability_path")
        output = parent / run_id
        if output.exists() or output.is_symlink():
            raise ContractError("Capability output already exists", "capability_output_exists")
        return output

    def locator(self, path: Path) -> tuple[str, str]:
        resolved = path.resolve(strict=True)
        for label, root in (("results", self.results), ("workspace", self.workspace)):
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            return label, relative.as_posix()
        raise ContractError("Capability artifact escapes its scope", "unsafe_capability_path")


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or path in {PurePosixPath("."), PurePosixPath("..")}
        or ".." in path.parts
    ):
        raise ContractError("Capability path must be workspace-relative", "unsafe_capability_path")
    return path


__all__ = ["CapabilityScope"]
