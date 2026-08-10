"""Read-only CLI services for rebuilding report and RL projections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from apex.reporting import (
    RunEvidenceSource,
    ShowcaseExporter,
    build_replication_guide,
    build_report,
    materialize_run_graph,
    resolve_projection_output,
    resolve_run_source,
    write_run_projections,
)
from apex.rl import (
    DatasetExportConfig,
    DatasetExporter,
)


RunProjectionSource = RunEvidenceSource


def rebuild_report(
    run_root: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> Mapping[str, Any]:
    """Rebuild disposable report views without appending canonical events."""

    source = resolve_run_source(run_root, run_id=run_id)
    destination = resolve_projection_output(source.root, output_dir)
    graph = materialize_run_graph(source)
    report = build_report(graph)
    replication = build_replication_guide(graph)
    paths = write_run_projections(
        destination,
        report=report,
        replication=replication,
    )
    return {
        "schema_version": 1,
        "status": "reported",
        "run_id": source.run_id,
        "episode_graph_id": graph.graph_id,
        "journal_high_water_mark": graph.high_water_mark,
        "report_sha256": report.digest,
        "replication_sha256": replication.digest,
        "outputs": {name: str(path) for name, path in sorted(paths.items())},
    }


def export_rl_dataset(
    run_root: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
    config: DatasetExportConfig | None = None,
) -> Mapping[str, Any]:
    """Export one canonical run through the sole EpisodeGraph exporter."""

    source = resolve_run_source(run_root, run_id=run_id)
    destination = resolve_projection_output(source.root, output_dir)
    graph = materialize_run_graph(source)
    result = DatasetExporter(source.artifacts).export(
        graph,
        destination,
        config=config,
    )
    return {
        "schema_version": 1,
        "status": "exported",
        "run_id": source.run_id,
        "episode_graph_id": graph.graph_id,
        "journal_high_water_mark": graph.high_water_mark,
        "record_count": result.record_count,
        "sft_count": result.sft_count,
        "skipped": list(result.skipped),
        "dataset_sha256": result.dataset_sha256,
        "manifest_sha256": result.manifest_sha256,
        "output_dir": str(result.output_dir),
    }


def export_showcase_projection(
    run_root: Path,
    output_dir: Path,
    *,
    showcase_id: str,
    run_id: str | None = None,
) -> Mapping[str, Any]:
    """Export a sanitized showcase only through canonical graph/CAS replay."""

    source = resolve_run_source(run_root, run_id=run_id)
    destination = resolve_projection_output(source.root, output_dir)
    graph = materialize_run_graph(source)
    result = ShowcaseExporter(source.artifacts).export(
        graph, destination, showcase_id=showcase_id
    )
    return {
        "schema": "apex.showcase-export-result/v1",
        "showcase_id": result.showcase_id,
        "status": result.status,
        "qualification_blockers": list(result.blockers),
        "checksums_sha256": result.checksums_sha256,
        "output_dir": str(result.output_dir),
        "run_id": source.run_id,
        "episode_graph_id": graph.graph_id,
    }


__all__ = [
    "RunProjectionSource",
    "export_rl_dataset",
    "export_showcase_projection",
    "rebuild_report",
    "resolve_projection_output",
    "resolve_run_source",
]
