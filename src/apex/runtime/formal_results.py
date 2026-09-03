"""Fail-closed placement policy for evaluator-owned formal results."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from apex.core import ContractError


@dataclass(frozen=True, slots=True)
class FormalResultsRootValidator:
    """Reject formal evidence roots that overlap protected source checkouts."""

    protected_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        roots = tuple(
            sorted(
                {_canonical_protected(path) for path in self.protected_roots},
                key=str,
            )
        )
        if not roots:
            raise ContractError(
                "Formal results validation requires a protected root",
                "formal_results_policy_missing",
            )
        object.__setattr__(self, "protected_roots", roots)

    def validate(
        self,
        path: Path,
        *,
        require_new: bool = False,
    ) -> Path:
        """Return one canonical safe root without creating it."""

        selected = path.expanduser()
        if not selected.is_absolute():
            raise ContractError(
                "Formal results root must be absolute",
                "formal_results_not_absolute",
            )
        _reject_symlink_components(selected)
        resolved = selected.resolve(strict=False)
        if resolved == Path("/"):
            raise ContractError(
                "Formal results root cannot be the filesystem root",
                "formal_results_too_broad",
            )
        for protected in self.protected_roots:
            if _overlaps(resolved, protected):
                raise ContractError(
                    "Formal results root overlaps a protected source checkout",
                    "formal_results_overlap",
                    {"results": str(resolved), "protected_root": str(protected)},
                )
        if resolved.exists() and not resolved.is_dir():
            raise ContractError(
                "Formal results root must be a directory",
                "invalid_formal_results_root",
            )
        if require_new and resolved.exists():
            raise ContractError(
                "Formal results root already exists",
                "formal_results_root_exists",
            )
        return resolved


def formal_results_validator(
    *,
    apex_root: Path,
    dependency_roots: Iterable[Path] = (),
    source_roots: Iterable[Path] = (),
    workspace_roots: Iterable[Path] = (),
) -> FormalResultsRootValidator:
    """Compose the placement policy from verified source authorities."""

    return FormalResultsRootValidator(
        (
            apex_root,
            *tuple(dependency_roots),
            *tuple(source_roots),
            *tuple(workspace_roots),
        )
    )


def _canonical_protected(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ContractError(
            "Protected source roots must be absolute",
            "formal_results_policy_invalid",
        )
    _reject_symlink_components(selected)
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise ContractError(
            "Protected source root does not exist",
            "formal_results_policy_invalid",
        ) from error
    if not resolved.is_dir():
        raise ContractError(
            "Protected source root is not a directory",
            "formal_results_policy_invalid",
        )
    return resolved


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ContractError(
                "Formal results path cannot be inspected",
                "unsafe_formal_results_root",
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(
                "Formal results paths cannot contain symlinks",
                "unsafe_formal_results_root",
                {"path": str(current)},
            )


def _overlaps(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


__all__ = ["FormalResultsRootValidator", "formal_results_validator"]
