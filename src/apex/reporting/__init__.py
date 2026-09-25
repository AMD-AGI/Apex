"""Rebuildable reports and exact replication projections."""

from .replication import ReplicationProjection, build_replication_guide
from .report import ReportProjection, build_report
from .run_source import (
    RunEvidenceSource,
    materialize_run_graph,
    resolve_projection_output,
    resolve_run_source,
)
from .writer import write_run_projections
from .showcase import (
    ShowcaseExportResult,
    ShowcaseExporter,
    ShowcaseVerification,
    verify_showcase,
)

__all__ = [
    "ReplicationProjection",
    "ReportProjection",
    "RunEvidenceSource",
    "build_replication_guide",
    "build_report",
    "materialize_run_graph",
    "resolve_projection_output",
    "resolve_run_source",
    "write_run_projections",
    "ShowcaseExportResult",
    "ShowcaseExporter",
    "ShowcaseVerification",
    "verify_showcase",
]
