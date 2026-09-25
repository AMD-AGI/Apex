"""Shared terminal projection publishing for kernel and workload runs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from apex.orchestration import WorkloadState
from apex.reporting import build_replication_guide, build_report, write_run_projections
from apex.rl import EpisodeGraphMaterializer
from apex.storage import ArtifactStore, EventJournal


def publish_terminal_projections(
    *,
    root: Path,
    run_id: str,
    artifacts: ArtifactStore,
    workload_state: WorkloadState | None = None,
) -> Mapping[str, Path]:
    """Rebuild reports from canonical state without appending another truth."""

    graph = EpisodeGraphMaterializer(
        EventJournal(root / "events" / "run.db"),
        artifacts,
    ).materialize(run_id, workload_state=workload_state)
    return write_run_projections(
        root,
        report=build_report(graph),
        replication=build_replication_guide(graph),
    )


__all__ = ["publish_terminal_projections"]
