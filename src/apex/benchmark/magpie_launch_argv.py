"""Validate the only attestor-owned variation in a Magpie launch argv."""

from __future__ import annotations

from pathlib import Path

from apex.core import ContractError


def validated_magpie_launch_argv(
    canonical: tuple[str, ...], launch: tuple[str, ...]
) -> tuple[str, ...]:
    """Permit one immutable config projection and reject every other drift."""

    differing = tuple(
        index for index, pair in enumerate(zip(canonical, launch))
        if pair[0] != pair[1]
    )
    valid_shape = (
        isinstance(launch, tuple)
        and len(launch) == len(canonical) == 8
        and all(
            isinstance(item, str) and item and "\0" not in item
            for item in launch
        )
        and differing in {(), (5,)}
    )
    if not valid_shape:
        raise ContractError(
            "Magpie launch argv differs outside its config projection",
            "magpie_launch_argv_invalid",
        )
    config = Path(launch[5])
    if not config.is_absolute() or config.is_symlink() or not config.is_file():
        raise ContractError(
            "Magpie launch config is unsafe", "magpie_launch_argv_invalid"
        )
    if differing and (config.stat().st_nlink != 1 or config.stat().st_mode & 0o222):
        raise ContractError(
            "Magpie launch config is not immutable", "magpie_launch_argv_invalid"
        )
    return launch


__all__ = ["validated_magpie_launch_argv"]
