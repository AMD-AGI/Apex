"""Durably publish disposable report projections."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Mapping

from .replication import ReplicationProjection
from .report import ReportProjection


def write_run_projections(
    output_dir: Path,
    *,
    report: ReportProjection,
    replication: ReplicationProjection,
) -> Mapping[str, Path]:
    """Atomically write report views; deleting them never loses run truth."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "report.json": report.json_bytes,
        "report.md": report.markdown_bytes,
        "replication_guide.json": replication.json_bytes,
        "replication_guide.md": replication.markdown_bytes,
    }
    paths: dict[str, Path] = {}
    for name, content in sorted(values.items()):
        destination = root / name
        descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix=f".{name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        paths[name] = destination
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return paths


__all__ = ["write_run_projections"]
