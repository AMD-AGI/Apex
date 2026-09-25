"""Bind benchmark views to an Apex projection of published Magpie main."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import ConfigurationError, sha256_json
from apex.runtime import DependencyReceipt, MagpieConfigContract


def resolved_scoring_document(
    original: Mapping[str, Any], resolved: MagpieConfigContract
) -> dict[str, Any]:
    """Reconstruct the Apex scoring view while preserving frozen Magpie defaults."""

    raw_benchmark = original.get("benchmark")
    if not isinstance(raw_benchmark, Mapping):
        raise ConfigurationError(
            "Benchmark config lacks its raw benchmark mapping",
            "invalid_benchmark_config",
        )
    scoring = _restore_redactions(resolved.scoring_config, raw_benchmark)
    if sha256_json(scoring) != resolved.plan["scoring_config_sha256"]:
        raise ConfigurationError(
            "Magpie scoring view cannot be reconstructed from frozen input",
            "magpie_scoring_view_mismatch",
        )
    document = copy.deepcopy(dict(original))
    document["benchmark"] = scoring
    return document


def validate_resolved_binding(
    source: Path,
    original_sha256: str,
    receipt: DependencyReceipt,
    resolved: MagpieConfigContract,
) -> None:
    """Require the plan to bind the exact config, dependency lock, and Magpie pin."""

    if (
        resolved.config_path != source
        or resolved.config_sha256 != original_sha256
        or resolved.magpie_commit != receipt.commits.get("magpie")
        or resolved.dependency_lock_sha256 != receipt.lock_sha256
        or resolved.status != "config_compatible"
    ):
        raise ConfigurationError(
            "Magpie resolved contract does not authorize this benchmark view",
            "magpie_config_resolution_mismatch",
        )


def validated_source_roots(values: Sequence[Path]) -> tuple[Path, ...]:
    """Validate immutable diagnostic source-root inputs."""

    roots: list[Path] = []
    for value in values:
        if not value.is_absolute() or value.is_symlink():
            raise ConfigurationError(
                "Diagnostic source repository roots must be absolute directories",
                "invalid_source_repository",
            )
        resolved = value.resolve(strict=True)
        if not resolved.is_dir() or resolved in roots:
            raise ConfigurationError(
                "Diagnostic source repository roots must be unique directories",
                "invalid_source_repository",
            )
        roots.append(resolved)
    return tuple(roots)


def _restore_redactions(resolved: object, raw: object) -> Any:
    if resolved == "<redacted>":
        if raw is None or raw == "<redacted>":
            raise ConfigurationError(
                "Magpie redacted scoring value is unavailable",
                "magpie_scoring_view_mismatch",
            )
        return copy.deepcopy(raw)
    if isinstance(resolved, Mapping):
        raw_mapping = raw if isinstance(raw, Mapping) else {}
        return {
            str(key): _restore_redactions(value, raw_mapping.get(key))
            for key, value in resolved.items()
        }
    if isinstance(resolved, list):
        raw_values = raw if isinstance(raw, list) else []
        return [
            _restore_redactions(
                value, raw_values[index] if index < len(raw_values) else None
            )
            for index, value in enumerate(resolved)
        ]
    return copy.deepcopy(resolved)


__all__ = [
    "resolved_scoring_document",
    "validate_resolved_binding",
    "validated_source_roots",
]
