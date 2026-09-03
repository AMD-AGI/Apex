from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from apex.reporting import (
    build_replication_guide,
    build_report,
    write_run_projections,
)
from apex.rl import EpisodeArtifact, EpisodeGraphMaterializer

from .conftest import append_event


def _graph(canonical_run):
    return EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])


def _reproducible_kernel_graph(canonical_run):
    graph = _graph(canonical_run)
    start = graph.parent.events[0]
    receipt = canonical_run["packet_receipt"]
    declaration = {
        "schema": "apex.replication-declaration/v1",
        "task_kind": "single_kernel",
        "dependency_receipts": [{"name": "contract", "digest": "b" * 64}],
        "source_commits": [
            {"name": "workspace", "commit": "2" * 40, "tree": "3" * 40}
        ],
        "parent_image_digest": None,
        "derived_image_digest": None,
        "commands": [
            {"name": name, "argv": argv}
            for name, argv in (
                ("verify_bundle", ["apex", "bundle", "verify", "--bundle", "./bundle"]),
                ("apply_bundle", ["apex", "bundle", "apply", "--bundle", "./bundle"]),
                ("compile", ["python", "compile.py"]),
                ("correctness", ["python", "correctness.py"]),
                ("performance", ["python", "performance.py"]),
            )
        ],
        "benchmark_config_receipts": [],
        "bundle_receipt": {
            "kind": "kernel",
            "digest": "4" * 64,
            "evidence_receipt": receipt.digest,
            "verification_receipt": receipt.digest,
        },
    }
    changed = replace(
        start,
        payload={**start.payload, "replication": declaration},
        artifacts=(
            *start.artifacts,
            EpisodeArtifact("winner_bundle", receipt, start.event_id),
            EpisodeArtifact("bundle_verification", receipt, start.event_id),
        ),
    )
    parent = replace(
        graph.parent,
        kind="single_kernel",
        workload_id=None,
        events=(changed, *graph.parent.events[1:]),
    )
    return replace(graph, parent=parent)


def test_report_is_stable_and_headline_is_measured_only(canonical_run):
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "e2e_result",
        {
            "attempt_id": "attempt-1",
            "evidence_class": "self_reported",
            "metrics": {"throughput": 999999},
        },
        "attempt-1-claimed-e2e",
    )
    graph = _graph(canonical_run)
    first = build_report(graph)
    second = build_report(graph)
    assert first.json_bytes == second.json_bytes
    assert first.markdown_bytes == second.markdown_bytes
    assert b"999999" not in first.json_bytes
    measured = first.document["headline_measured_results"]
    assert measured == [
        {
            "attempt_id": "attempt-1",
            "event_id": next(
                event.event_id
                for event in graph.children[0].events
                if event.event_type == "measurement_result"
            ),
            "metrics": {"s50": 1.2, "s99": 1.1, "srobust": 1.1},
        }
    ]
    assert "attempt-2" in first.markdown
    assert "infrastructure_error" in first.markdown
    assert first.document["terminal_reward"] == {
        "task_reward": None,
        "reward_vector": None,
        "policy_id": None,
        "policy_digest": None,
        "source_receipt": None,
        "raw_measurement_receipts": [],
        "trainability": "unscored",
        "untrainable_reason": None,
    }
    assert "Task reward: `null`" in first.markdown


def test_report_redacts_provenance_secrets(canonical_run):
    graph = replace(
        _graph(canonical_run),
        provenance={"gpu": "gfx950", "api_key": "sk-ant-supersecretvalue12345"},
    )
    report = build_report(graph)
    assert b"supersecretvalue" not in report.json_bytes
    assert report.document["provenance"]["api_key"] == "[REDACTED]"


def test_replication_guide_renders_only_committed_argv(canonical_run):
    guide = build_replication_guide(_reproducible_kernel_graph(canonical_run))
    assert guide.document["reproducible"] is True
    assert "apex bundle verify --bundle ./bundle" in guide.markdown
    assert "python correctness.py" in guide.markdown
    assert "performance" in guide.markdown


def test_e2e_replication_requires_images_configs_and_clean_replay(canonical_run):
    graph = _graph(canonical_run)
    start = graph.parent.events[0]
    receipt = canonical_run["packet_receipt"]
    declaration = {
        "schema": "apex.replication-declaration/v1",
        "task_kind": "e2e_kernel_only",
        "dependency_receipts": [{"name": "recipe", "digest": "1" * 64}],
        "source_commits": [
            {"name": "vllm", "commit": "2" * 40, "tree": "3" * 40}
        ],
        "parent_image_digest": "sha256:" + "4" * 64,
        "derived_image_digest": "sha256:" + "5" * 64,
        "commands": [
            {"name": "verify_bundle", "argv": ["apex", "bundle", "verify"]},
            {"name": "build_image", "argv": ["python", "build.py"]},
            {"name": "clean_replay", "argv": ["apex", "bundle", "verify"]},
        ],
        "benchmark_config_receipts": [
            {"name": "benchmark_replay", "digest": "6" * 64}
        ],
        "bundle_receipt": {
            "kind": "e2e",
            "digest": "7" * 64,
            "evidence_receipt": receipt.digest,
            "verification_receipt": receipt.digest,
        },
    }
    changed = replace(
        start,
        payload={**start.payload, "replication": declaration},
        artifacts=(
            *start.artifacts,
            EpisodeArtifact("winner_bundle", receipt, start.event_id),
            EpisodeArtifact("bundle_verification", receipt, start.event_id),
        ),
    )
    parent = replace(graph.parent, events=(changed, *graph.parent.events[1:]))

    guide = build_replication_guide(replace(graph, parent=parent))

    assert guide.document["reproducible"] is True
    assert guide.document["derived_image_digest"] == "sha256:" + "5" * 64


def test_replication_guide_reports_missing_evidence_without_guessing(canonical_run):
    graph = _graph(canonical_run)
    parent = replace(
        graph.parent,
        events=tuple(
            event for event in graph.parent.events if "replication" not in event.payload
        ),
    )
    guide = build_replication_guide(replace(graph, parent=parent))
    assert guide.document["reproducible"] is False
    assert "replication_declaration_missing" in guide.document["validation_reasons"]
    assert "No executable replication argv was committed" in guide.markdown


def test_replication_guide_redacts_secret_argv_and_refuses_claim(canonical_run):
    graph = _graph(canonical_run)
    start = graph.parent.events[0]
    declaration = dict(start.payload["replication"])
    declaration["commands"] = [
        {
            "name": "apply_bundle",
            "argv": ["tool", "--authorization", "Bearer abcdefghijklmnopqrstuvwxyz"],
        }
    ]
    changed_start = replace(start, payload={**start.payload, "replication": declaration})
    parent = replace(
        graph.parent, events=(changed_start, *graph.parent.events[1:])
    )
    guide = build_replication_guide(replace(graph, parent=parent))
    assert guide.document["reproducible"] is False
    assert "abcdefghijklmnopqrstuvwxyz" not in guide.markdown
    assert "replication_command_contains_secret" in guide.document["validation_reasons"]


def test_projection_writer_is_rebuildable(canonical_run, tmp_path: Path):
    graph = _graph(canonical_run)
    report = build_report(graph)
    replication = build_replication_guide(graph)
    first = write_run_projections(
        tmp_path / "views", report=report, replication=replication
    )
    before = {name: path.read_bytes() for name, path in first.items()}
    for path in first.values():
        path.unlink()
    second = write_run_projections(
        tmp_path / "views",
        report=build_report(graph),
        replication=build_replication_guide(graph),
    )
    assert before == {name: path.read_bytes() for name, path in second.items()}
