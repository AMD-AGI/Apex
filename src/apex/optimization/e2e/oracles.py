"""Version-bound correctness-oracle routing for dynamically found kernels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from apex.core import ContractError, IntegrityError, sha256_json


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CorrectnessOracleBinding:
    """One reviewed source-to-test relationship inside an exact repository."""

    repository_id: str
    source_relative_path: str
    test_relative_path: str
    test_argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.repository_id:
            raise ContractError("Oracle repository is empty", "invalid_oracle_binding")
        _safe_relative(self.source_relative_path, "source_relative_path")
        _safe_relative(self.test_relative_path, "test_relative_path")
        if not self.test_argv or any(not item for item in self.test_argv):
            raise ContractError("Oracle argv is empty", "invalid_oracle_binding")
        expected = self.test_relative_path
        if expected not in self.test_argv:
            raise ContractError(
                "Oracle argv does not name its reviewed test",
                "invalid_oracle_binding",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_id": self.repository_id,
            "source_relative_path": self.source_relative_path,
            "test_relative_path": self.test_relative_path,
            "test_argv": list(self.test_argv),
        }


@dataclass(frozen=True, slots=True)
class ResolvedCorrectnessOracle:
    """A checked test file and its policy-bound execution description."""

    test_file: Path
    test_command: str
    binding_sha256: str
    policy_sha256: str


class CorrectnessOracleRegistry:
    """Resolve reviewed tests by repository-relative source path, never symbol name."""

    def __init__(
        self,
        *,
        source_roots: Mapping[str, Path],
        bindings: Sequence[CorrectnessOracleBinding],
        source_lock_sha256: str,
    ) -> None:
        if not _SHA256.fullmatch(source_lock_sha256):
            raise ContractError("Oracle source lock is invalid", "invalid_oracle_binding")
        self._roots = {
            name: _regular_directory(root, name) for name, root in source_roots.items()
        }
        indexed: dict[tuple[str, str], CorrectnessOracleBinding] = {}
        for binding in bindings:
            if binding.repository_id not in self._roots:
                raise ContractError(
                    f"Oracle repository is not source-locked: {binding.repository_id}",
                    "invalid_oracle_binding",
                )
            key = (binding.repository_id, binding.source_relative_path)
            if key in indexed:
                raise ContractError("Duplicate correctness oracle", "invalid_oracle_binding")
            self._validate_files(binding)
            indexed[key] = binding
        self._bindings = indexed
        self.policy_sha256 = sha256_json(
            {
                "schema": "apex.correctness-oracle-policy/v1",
                "source_lock_sha256": source_lock_sha256,
                "bindings": [
                    item.to_dict()
                    for item in sorted(
                        indexed.values(),
                        key=lambda value: (
                            value.repository_id,
                            value.source_relative_path,
                        ),
                    )
                ],
            }
        )

    def resolve(
        self,
        *,
        repository_id: str,
        source_root: Path,
        source_path: Path,
    ) -> ResolvedCorrectnessOracle | None:
        expected_root = self._roots.get(repository_id)
        if expected_root is None:
            return None
        observed_root = source_root.resolve(strict=True)
        if observed_root != expected_root:
            raise IntegrityError(
                "Kernel source root differs from the oracle source lock",
                "oracle_source_root_drift",
            )
        try:
            relative = source_path.resolve(strict=True).relative_to(observed_root).as_posix()
        except ValueError as error:
            raise IntegrityError(
                "Kernel source escapes the oracle source root",
                "source_outside_root",
            ) from error
        binding = self._bindings.get((repository_id, relative))
        if binding is None:
            return None
        self._validate_files(binding)
        test_file = observed_root.joinpath(*Path(binding.test_relative_path).parts)
        binding_sha256 = sha256_json(
            {
                "schema": "apex.correctness-oracle-binding/v1",
                "policy_sha256": self.policy_sha256,
                **binding.to_dict(),
            }
        )
        return ResolvedCorrectnessOracle(
            test_file=test_file,
            test_command=" ".join(binding.test_argv),
            binding_sha256=binding_sha256,
            policy_sha256=self.policy_sha256,
        )

    def _validate_files(self, binding: CorrectnessOracleBinding) -> None:
        root = self._roots[binding.repository_id]
        _regular_file(root, binding.source_relative_path, "oracle_source_missing")
        _regular_file(root, binding.test_relative_path, "oracle_test_missing")


def _safe_relative(value: str, field: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ContractError(
            f"Oracle {field} must be a normalized safe relative path",
            "invalid_oracle_binding",
        )
    return path


def _regular_directory(path: Path, repository_id: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ContractError(
            f"Oracle source root is unsafe: {repository_id}",
            "invalid_oracle_source_root",
        )
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ContractError(
            f"Oracle source root is unavailable: {repository_id}",
            "invalid_oracle_source_root",
        ) from error


def _regular_file(root: Path, relative: str, reason: str) -> Path:
    path = root.joinpath(*_safe_relative(relative, "path").parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise IntegrityError("Oracle path is unavailable", reason) from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise IntegrityError("Oracle path escapes source root", "oracle_path_escape") from error
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError("Oracle path is not a regular file", reason)
    return resolved


__all__ = [
    "CorrectnessOracleBinding",
    "CorrectnessOracleRegistry",
    "ResolvedCorrectnessOracle",
]
