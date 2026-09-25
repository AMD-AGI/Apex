"""Small fail-closed merge helpers for immutable RL projection inputs."""

from __future__ import annotations

from typing import Any

from apex.core import IntegrityError


def merge_projected_int(
    target: Any,
    field_name: str,
    value: object,
) -> None:
    """Merge one integer while recording non-identity projection conflicts."""

    if value is None:
        return
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        target.validation_reasons.add(f"invalid_{field_name}")
        return
    existing = getattr(target, field_name)
    if existing is not None and existing != parsed:
        target.validation_reasons.add(f"conflicting_{field_name}")
    elif existing is None:
        setattr(target, field_name, parsed)


def merge_projected_identifier(
    target: Any,
    field_name: str,
    value: str | None,
) -> None:
    """Merge an identity field, rejecting any conflicting lineage."""

    if value is None:
        return
    existing = getattr(target, field_name)
    if existing is not None and existing != value:
        raise IntegrityError(
            f"Attempt events disagree on {field_name}",
            f"{field_name}_mismatch",
        )
    setattr(target, field_name, value)


__all__ = ["merge_projected_identifier", "merge_projected_int"]
