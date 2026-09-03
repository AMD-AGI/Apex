"""Shared read-only resolution of canonical run journals and artifact stores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.core import ContractError, validate_identifier
from apex.orchestration.replay import replay_workload_state
from apex.rl import EpisodeGraph, EpisodeGraphMaterializer
from apex.storage import ArtifactStore, EventJournal


@dataclass(frozen=True, slots=True)
class RunEvidenceSource:
    root: Path
    run_id: str
    journal: EventJournal
    artifacts: ArtifactStore


def resolve_run_source(
    run_root: Path, *, run_id: str | None = None
) -> RunEvidenceSource:
    """Fail closed unless a pre-existing canonical run layout is complete."""

    supplied = Path(run_root).expanduser()
    if not supplied.exists():
        raise ContractError("Run root does not exist", "projection_run_root_missing")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ContractError("Run root is not a directory", "projection_run_root_invalid")
    journal_path, artifact_root = root / "events" / "run.db", root / "artifacts"
    if not journal_path.is_file() or journal_path.is_symlink():
        raise ContractError("Canonical event journal is missing", "projection_journal_missing")
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise ContractError("Canonical artifact store is missing", "projection_cas_missing")
    selected = _resolve_run_id(root, run_id)
    journal = EventJournal(journal_path)
    if not journal.iter_events(selected, verify=True):
        raise ContractError("Run has no canonical events", "projection_run_empty")
    return RunEvidenceSource(root, selected, journal, ArtifactStore(artifact_root))


def materialize_run_graph(source: RunEvidenceSource) -> EpisodeGraph:
    events = tuple(source.journal.iter_events(source.run_id, verify=True))
    state = replay_workload_state(source.run_id, events)
    return EpisodeGraphMaterializer(source.journal, source.artifacts).materialize(
        source.run_id, workload_state=state
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


def _resolve_run_id(root: Path, supplied: str | None) -> str:
    declared = _result_run_id(root / "result.json")
    if supplied is not None:
        selected = validate_identifier(supplied, field_name="run_id")
        if declared is not None and declared != selected:
            raise ContractError("Run ID conflicts with result.json", "projection_run_id_conflict")
        return selected
    if declared is not None:
        return declared
    if root.name.startswith(("run-", "e2e-", "campaign-")):
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
    "RunEvidenceSource",
    "materialize_run_graph",
    "resolve_projection_output",
    "resolve_run_source",
]
