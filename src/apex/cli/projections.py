"""Read-only CLI services for rebuilding report and RL projections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, validate_identifier
from apex.orchestration.replay import replay_workload_state
from apex.reporting import build_replication_guide, build_report, write_run_projections
from apex.rl import (
    DatasetExportConfig,
    DatasetExporter,
    EpisodeGraph,
    EpisodeGraphMaterializer,
)
from apex.storage import ArtifactStore, EventJournal


@dataclass(frozen=True, slots=True)
class RunProjectionSource:
    """Resolved canonical journal/CAS pair for one immutable run ID."""

    root: Path
    run_id: str
    journal: EventJournal
    artifacts: ArtifactStore


def rebuild_report(
    run_root: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> Mapping[str, Any]:
    """Rebuild disposable report views without appending canonical events."""

    source = resolve_run_source(run_root, run_id=run_id)
    destination = resolve_projection_output(source.root, output_dir)
    graph = _materialize_graph(source)
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
    graph = _materialize_graph(source)
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


def resolve_run_source(
    run_root: Path,
    *,
    run_id: str | None = None,
) -> RunProjectionSource:
    """Fail closed unless a pre-existing canonical run layout is complete."""

    supplied_root = Path(run_root).expanduser()
    if not supplied_root.exists():
        raise ContractError("Run root does not exist", "projection_run_root_missing")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise ContractError("Run root is not a directory", "projection_run_root_invalid")
    journal_path = root / "events" / "run.db"
    artifact_root = root / "artifacts"
    if not journal_path.is_file() or journal_path.is_symlink():
        raise ContractError("Canonical event journal is missing", "projection_journal_missing")
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ContractError("Canonical artifact store is missing", "projection_cas_missing")
    resolved_id = _resolve_run_id(root, run_id)
    journal = EventJournal(journal_path)
    if not journal.iter_events(resolved_id, verify=True):
        raise ContractError("Run has no canonical events", "projection_run_empty")
    return RunProjectionSource(
        root=root,
        run_id=resolved_id,
        journal=journal,
        artifacts=ArtifactStore(artifact_root),
    )


def resolve_projection_output(run_root: Path, output_dir: Path) -> Path:
    """Keep disposable output out of the journal and artifact CAS."""

    supplied = Path(output_dir).expanduser()
    if supplied.exists() and supplied.is_symlink():
        raise ContractError("Projection output cannot be a symlink", "projection_output_symlink")
    destination = supplied.resolve()
    protected = (run_root / "events", run_root / "artifacts")
    if any(destination == path or destination.is_relative_to(path) for path in protected):
        raise ContractError(
            "Projection output overlaps canonical run evidence",
            "projection_output_overlaps_evidence",
        )
    return destination


def _materialize_graph(source: RunProjectionSource) -> EpisodeGraph:
    events = tuple(source.journal.iter_events(source.run_id, verify=True))
    state = replay_workload_state(source.run_id, events)
    return EpisodeGraphMaterializer(source.journal, source.artifacts).materialize(
        source.run_id, workload_state=state
    )


def _resolve_run_id(root: Path, supplied: str | None) -> str:
    declared = _result_run_id(root / "result.json")
    if supplied is not None:
        resolved = validate_identifier(supplied, field_name="run_id")
        if declared is not None and declared != resolved:
            raise ContractError("Run ID conflicts with result.json", "projection_run_id_conflict")
        return resolved
    if declared is not None:
        return declared
    if root.name.startswith(("run-", "e2e-")):
        return validate_identifier(root.name, field_name="run_id")
    raise ContractError(
        "Run ID is required when it cannot be derived from result.json or the run directory",
        "projection_run_id_required",
    )


def _result_run_id(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ContractError("Run result is not a safe regular JSON file", "projection_result_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("Run result is not valid JSON", "projection_result_invalid") from error
    if not isinstance(value, Mapping) or not isinstance(value.get("run_id"), str):
        return None
    return validate_identifier(str(value["run_id"]), field_name="run_id")


__all__ = [
    "RunProjectionSource",
    "export_rl_dataset",
    "rebuild_report",
    "resolve_projection_output",
    "resolve_run_source",
]
