"""Private durable-file helpers shared by storage adapters."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable


FaultHook = Callable[[str], None]


def fsync_directory(path: Path) -> None:
    """Persist directory metadata after a rename or unlink."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    destination: Path,
    content: bytes,
    *,
    fault_hook: FaultHook | None = None,
) -> None:
    """Durably replace ``destination`` with ``content``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _fault(fault_hook, "after_temp_fsync")
        _fault(fault_hook, "before_replace")
        os.replace(temporary, destination)
        _fault(fault_hook, "after_replace")
        fsync_directory(destination.parent)
        _fault(fault_hook, "after_parent_fsync")
    finally:
        temporary.unlink(missing_ok=True)


def _fault(hook: FaultHook | None, stage: str) -> None:
    if hook is not None:
        hook(stage)


__all__ = ["FaultHook", "atomic_write_bytes", "fsync_directory"]
